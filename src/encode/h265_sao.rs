//! Sample adaptive offset decision for H.265 — choosing the offsets, then
//! applying the decoder's own filter.
//!
//! The third and last member of the loop-filter family here, and built the
//! way the second one was: `hevc::sao::sao_ctb_row` is the decoder's
//! filter, conformance-proven, and this module does not reimplement it.
//! What the encoder contributes is the *parameters* — per CTB, per
//! component, which of the two offset kinds to apply and with what values
//! — and the state the filter reads them out of.
//!
//! # Where it sits, and why the order is not negotiable
//!
//! SAO runs **after deblocking**, on the deblocked samples, and its output
//! is what the picture becomes: what a decoder emits, what SELF compares
//! against, and what the next picture predicts from. So the encoder's
//! sequence per picture is reconstruct, deblock, decide SAO, apply SAO —
//! and only then may the reconstruction be cropped or kept as a reference.
//!
//! One consequence shapes this whole module: the parameters cannot be
//! serialised as each CTU is coded, because they are not known until the
//! whole picture has been reconstructed *and* deblocked, and the reader
//! takes them at the **start** of each CTU, ahead of its coding quadtree.
//! The picture writer therefore decides in one pass and serialises in
//! another. That costs nothing — the decisions never depended on the
//! bitstream — but it is why `code_picture` looks the way it does.
//!
//! # The classification is the reader's, read off the filter
//!
//! Nothing here re-derives what a category means; the two classifiers are
//! mirrored line for line from `hevc::sao::sao_ctb`, which is the code
//! that will apply them:
//!
//! - **Band** (`type_idx` 1): a sample's band is `v >> (bit_depth - 5)`,
//!   0..=31, and only the four consecutive bands starting at
//!   `sao_band_position` carry an offset (`table[(k + pos) & 31]`), so the
//!   window wraps.
//! - **Edge** (`type_idx` 2): `edgeIdx = 2 + sign(v - a) + sign(v - b)`
//!   over the two neighbours the class picks, and `off_tab = [o0, o1, 0,
//!   o2, o3]` — category 2 is "no change" and has no offset at all. That
//!   is also why offsets 0 and 1 are positive and 2 and 3 negative: they
//!   apply to local minima and valleys, and to ridges and local maxima,
//!   and the syntax codes no sign for them.
//!
//! A sample whose classifying neighbours are not usable is **left
//! untouched** by the filter, so the statistics must skip it too, or the
//! offsets are computed from samples that will never receive them. In the
//! one-slice, one-tile pictures this encoder writes, "not usable" reduces
//! to "outside the picture" — but it is written as the filter's own
//! question rather than as that shortcut.
//!
//! # What is decided, and how
//!
//! Per CTB and component: the sum of squared error against the source, for
//! leaving it alone and for every candidate — 32 band positions and four
//! edge classes. Within a candidate each category's best offset is the
//! rounded mean of `src - rec` over its samples, clamped to the syntax's
//! range (cMax, and to the sign the syntax forces in edge mode). The
//! change in SSD from applying offset `o` to a category is then
//! `count * o^2 - 2 * o * sum`, exactly, so no filtering is needed to
//! score a candidate.
//!
//! The winner is the smallest `distortion + lambda * bins`, with `lambda`
//! and the bin counts the same placeholder policy the other decision
//! modules use — one function, replaced wholesale when a real bit count
//! exists. Then each CTB is offered its left and upper neighbour's
//! parameters, and merges when they cost less overall: a merge is two bins
//! against forty or so, so it wins often on flat content and is not a
//! rarely-taken path.
//!
//! # Scope
//!
//! Lossless is refused by name upstream rather than handled: every CU of a
//! lossless picture is transquant-bypass, every bypass sample is exempt
//! from both loop filters, and SAO would therefore be a no-op declared in
//! every slice header. Shipping that is worse than refusing it.

use crate::encode::h265_intra::IntraCtx;
use crate::hevc::ctu::SaoMerge;
use crate::hevc::frame::{Frame, Plane16};
use crate::hevc::pic::{PicInfo, SaoParams};
use crate::hevc::pps::Pps;
use crate::hevc::sao::{SaoBand, sao_ctb_row};
use crate::hevc::sps::Sps;
use crate::sample::Sample;

/// What the picture writer needs after the filter has run: the parameters
/// each CTB carries and, where one was taken, the merge that spells them
/// in two bins instead of forty.
pub struct SaoPlan {
    /// Per CTB in raster order, the parameters a decoder will hold — the
    /// same arrays this module put in `PicInfo::sao` before filtering.
    pub params: Vec<[SaoParams; 3]>,
    /// Per CTB, the merge the writer should spell, if any. A merged CTB's
    /// `params` entry equals the neighbour's, because that is what the
    /// reader will copy.
    pub merges: Vec<Option<SaoMerge>>,
}

/// Per-category statistics for one candidate: how many samples fell into
/// each category, and the total error `src - rec` over them.
#[derive(Clone, Copy, Default)]
struct Cat {
    count: i64,
    sum: i64,
}

impl Cat {
    /// The change in SSD from adding `o` to every sample of this category:
    /// `sum((r + o) - s)^2 - sum(r - s)^2` = `count*o^2 - 2*o*sum`, where
    /// `sum` is `sum(s - r)`. Negative is an improvement.
    fn delta(&self, o: i64) -> i64 {
        self.count * o * o - 2 * o * self.sum
    }

    /// The offset this category would like, before the syntax's limits:
    /// the rounded mean error, which is the minimiser of `delta`.
    fn best(&self) -> i64 {
        if self.count == 0 {
            return 0;
        }
        let (n, s) = (self.count, self.sum);
        if s >= 0 { (2 * s + n) / (2 * n) } else { -((-2 * s + n) / (2 * n)) }
    }
}

/// The Lagrangian the SAO decision prices bins with — the intra module's
/// constant, and the same placeholder standing.
///
/// # This is the first thing in this encoder that is not a mirror
///
/// Everywhere else, an encoder-side derivation has a decoder-side
/// counterpart it must agree with, and the rule has been to call that
/// counterpart rather than reimplement it: the intra decision predicts
/// through `hevc::intra::predict`, the inter decision derives candidates
/// through `hevc::mvpred`, both loop filters *are* the decoder's. Drift
/// is impossible rather than tested against, because there is only one
/// copy.
///
/// SAO has no such counterpart. A decoder applies the offsets it is
/// handed and has no opinion about which ones it should have been handed;
/// the standard fixes the classification and the syntax, and leaves the
/// choice entirely to the encoder. So this module is split down that
/// line, deliberately:
///
/// - **The arithmetic stays theirs.** Classification is mirrored from
///   `hevc::sao::sao_ctb` line for line, and the offsets are *applied* by
///   `sao_ctb_row` itself — never by code here.
/// - **Only the choice is ours**, and it is priced by this function and
///   [`bins_of`]: a Lagrangian times an approximate bin count, the same
///   single-function placeholder policy as `h265_intra`'s
///   `mode_signalling_cost` and `h265_me`'s `lambda`, to be replaced
///   wholesale when a real bit count exists. Nothing about it is derived
///   from the standard and nothing in a decoder can contradict it.
///
/// The consequence worth stating: a mistake here cannot produce an
/// illegal stream, and neither SELF nor CROSS can see it. It produces a
/// legal stream and a worse picture. That is why `sao_picture` checks its
/// own predicted distortion against the filter's actual result rather
/// than trusting the two to agree.
fn lambda(qp: i32) -> f32 {
    0.57 * 2f32.powf((qp - 12) as f32 / 3.0)
}

/// Approximate bins for one component's parameters, in the shape
/// `write_sao` will actually emit: the type, four truncated-unary
/// magnitudes, the signs band mode carries, and the position or class.
fn bins_of(p: &SaoParams, cmax: u32, first_two: bool) -> u32 {
    // `sao_type_idx` exists only for components 0 and 1; Cr inherits.
    let mut n = if first_two { 1 + u32::from(p.type_idx != 0) } else { 0 };
    if p.type_idx == 0 {
        return n;
    }
    for o in p.offsets {
        let a = o.unsigned_abs() as u32;
        n += a + u32::from(a < cmax);
        if p.type_idx == 1 && a != 0 {
            n += 1; // sao_offset_sign
        }
    }
    n += if p.type_idx == 1 { 5 } else { u32::from(first_two) * 2 };
    n
}

/// One component of one CTB, as the decision needs to see it.
struct Comp<'a, S: Sample> {
    rec: &'a Plane16<S>,
    src: &'a [S],
    src_stride: usize,
    /// Component-plane origin of the CTB and its size, already clipped to
    /// the picture.
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
    /// Component-plane picture size, for the filter's usability question.
    pw: usize,
    ph: usize,
    max: i32,
    shift: i32,
}

impl<S: Sample> Comp<'_, S> {
    #[inline]
    fn rec_at(&self, x: usize, y: usize) -> i32 {
        self.rec.data[self.rec.offset(x as isize, y as isize)].to_i32()
    }
    #[inline]
    fn err_at(&self, x: usize, y: usize) -> i64 {
        // src - rec: the direction an offset must move the sample.
        self.src[y * self.src_stride + x].to_i32() as i64 - self.rec_at(x, y) as i64
    }

    /// SSD of leaving this CTB alone — the baseline every candidate must
    /// beat.
    fn ssd_off(&self) -> i64 {
        let mut d = 0i64;
        for y in self.y0..self.y0 + self.h {
            for x in self.x0..self.x0 + self.w {
                let e = self.err_at(x, y);
                d += e * e;
            }
        }
        d
    }

    /// Band statistics: one bucket per band, `v >> shift` exactly as
    /// `sao_ctb`'s table lookup indexes it.
    fn band_stats(&self) -> [Cat; 32] {
        let mut cats = [Cat::default(); 32];
        for y in self.y0..self.y0 + self.h {
            for x in self.x0..self.x0 + self.w {
                let b = (self.rec_at(x, y) >> self.shift) as usize;
                let c = &mut cats[b & 31];
                c.count += 1;
                c.sum += self.err_at(x, y);
            }
        }
        cats
    }

    /// Edge statistics for one class: five buckets by `edgeIdx`, skipping
    /// samples the filter will not touch because a classifying neighbour
    /// is unusable. `usable` is the caller's mirror of the filter's own.
    fn edge_stats(&self, class: u8, usable: &dyn Fn(usize, usize, i32, i32) -> bool) -> [Cat; 5] {
        let (hp, vp): ([i32; 2], [i32; 2]) = match class {
            0 => ([-1, 1], [0, 0]),
            1 => ([0, 0], [-1, 1]),
            2 => ([-1, 1], [-1, 1]),
            _ => ([1, -1], [-1, 1]),
        };
        let mut cats = [Cat::default(); 5];
        for y in self.y0..self.y0 + self.h {
            for x in self.x0..self.x0 + self.w {
                let (xa, ya) = (x as i32 + hp[0], y as i32 + vp[0]);
                let (xb, yb) = (x as i32 + hp[1], y as i32 + vp[1]);
                if !usable(x, y, xa, ya) || !usable(x, y, xb, yb) {
                    continue;
                }
                let v = self.rec_at(x, y);
                let a = self.rec_at(xa as usize, ya as usize);
                let b = self.rec_at(xb as usize, yb as usize);
                let e = (2 + (v - a).signum() + (v - b).signum()) as usize;
                cats[e].count += 1;
                cats[e].sum += self.err_at(x, y);
            }
        }
        cats
    }

    /// The best parameters for this component, and the SSD they achieve —
    /// against `ssd_off` as the do-nothing baseline.
    fn decide(&self, cmax: i64, lam: f32, first_two: bool, usable: &dyn Fn(usize, usize, i32, i32) -> bool) -> (SaoParams, f32, i64) {
        let base = self.ssd_off();
        let mut best = SaoParams::default();
        let mut best_cost = base as f32 + lam * bins_of(&best, cmax as u32, first_two) as f32;
        let mut best_dist = base;

        // Band: every one of the 32 wrapping four-band windows.
        let bands = self.band_stats();
        let mut per_band = [(0i64, 0i64); 32]; // (offset, delta)
        for (b, c) in bands.iter().enumerate() {
            let o = c.best().clamp(-cmax, cmax);
            per_band[b] = (o, c.delta(o));
        }
        for pos in 0..32usize {
            let mut p = SaoParams { type_idx: 1, band_or_class: pos as u8, offsets: [0; 4] };
            let mut delta = 0i64;
            for k in 0..4 {
                let (o, d) = per_band[(pos + k) & 31];
                // A band whose offset would make things worse contributes
                // nothing; zero is always spellable.
                if d < 0 {
                    p.offsets[k] = o as i16;
                    delta += d;
                }
            }
            if p.offsets.iter().all(|&o| o == 0) {
                continue;
            }
            let cost = (base + delta) as f32 + lam * bins_of(&p, cmax as u32, first_two) as f32;
            if cost < best_cost {
                best_cost = cost;
                best_dist = base + delta;
                best = p;
            }
        }

        // Edge: four classes. The syntax forces the sign of each category,
        // so the clamp is one-sided and a category that wants the other
        // direction simply takes zero.
        for class in 0..4u8 {
            let cats = self.edge_stats(class, usable);
            let mut p = SaoParams { type_idx: 2, band_or_class: class, offsets: [0; 4] };
            let mut delta = 0i64;
            // off_tab order: categories 0, 1 take offsets 0, 1 (positive);
            // categories 3, 4 take offsets 2, 3 (negative). Category 2 has
            // no offset in the syntax at all.
            for (slot, cat) in [(0usize, 0usize), (1, 1), (2, 3), (3, 4)] {
                let c = &cats[cat];
                let want = c.best();
                let o = if slot < 2 { want.clamp(0, cmax) } else { want.clamp(-cmax, 0) };
                let d = c.delta(o);
                if o != 0 && d < 0 {
                    p.offsets[slot] = o as i16;
                    delta += d;
                }
            }
            if p.offsets.iter().all(|&o| o == 0) {
                continue;
            }
            let cost = (base + delta) as f32 + lam * bins_of(&p, cmax as u32, first_two) as f32;
            if cost < best_cost {
                best_cost = cost;
                best_dist = base + delta;
                best = p;
            }
        }
        (best, best_cost, best_dist)
    }

    /// The SSD this component would have under someone else's parameters —
    /// what a merge candidate has to be scored on, since a merged CTB
    /// applies the neighbour's offsets to its own samples.
    fn ssd_under(&self, p: &SaoParams, usable: &dyn Fn(usize, usize, i32, i32) -> bool) -> i64 {
        let base = self.ssd_off();
        match p.type_idx {
            0 => base,
            1 => {
                let mut table = [0i16; 32];
                for k in 0..4 {
                    table[(k + p.band_or_class as usize) & 31] = p.offsets[k];
                }
                let bands = self.band_stats();
                let mut d = 0i64;
                for (b, c) in bands.iter().enumerate() {
                    d += c.delta(table[b] as i64);
                }
                base + d
            }
            _ => {
                let cats = self.edge_stats(p.band_or_class, usable);
                let tab = [p.offsets[0] as i64, p.offsets[1] as i64, 0, p.offsets[2] as i64, p.offsets[3] as i64];
                let mut d = 0i64;
                for (i, c) in cats.iter().enumerate() {
                    d += c.delta(tab[i]);
                }
                base + d
            }
        }
    }

    /// Whether the filter's `max` and `shift` describe this component —
    /// a cheap guard that the caller passed matching geometry.
    fn sane(&self) -> bool {
        self.max > 0 && self.shift >= 0 && self.x0 + self.w <= self.pw && self.y0 + self.h <= self.ph
    }
}

/// Decide this picture's SAO, record it where the filter reads it, and
/// apply the decoder's own filter to the reconstruction.
///
/// `recon` must already be deblocked; afterwards it holds what a decoder
/// will emit. `info` must be the state the deblocker used — the CTB slice
/// marks and the filter-exempt map in particular — and comes back with
/// `sao` filled, which is what a decoder's `PicInfo` would hold.
///
/// The sources are the padded, coded-size planes the rest of the encoder
/// works in, so the decision compares like with like: the reconstruction
/// at coded size against the source at coded size, including the
/// replicated edge the conformance window will hide.
#[allow(clippy::too_many_arguments)]
pub fn sao_picture<S: Sample>(
    ctx: &IntraCtx<'_, S>,
    recon: &mut Frame<S>,
    info: &mut PicInfo,
    sps: &Sps,
    pps: &Pps,
    src_y: &[S],
    y_stride: usize,
    src_cb: &[S],
    src_cr: &[S],
    c_stride: usize,
) -> SaoPlan {
    let log2 = sps.log2_ctb_size;
    let ctb = 1usize << log2;
    let (wc, hc) = (info.wc, info.hc);
    let cat = sps.chroma_array_type();
    let ncomp = if cat != 0 { 3 } else { 1 };
    let (sw, sh) = if cat != 0 { sps.sub_wh() } else { (1, 1) };
    let cmax = ((1i64 << (sps.bit_depth_luma.min(10) - 5)) - 1) as i64;
    let lam = lambda(ctx.qp);

    let mut params = vec![[SaoParams::default(); 3]; wc * hc];
    let mut merges: Vec<Option<SaoMerge>> = vec![None; wc * hc];
    // What the decision believes the filtered picture's error will be,
    // per component. Every candidate is scored analytically, per category,
    // and never by filtering — which is sound only if this module
    // classifies samples exactly as `hevc::sao` does. The debug check at
    // the end holds it to that.
    #[cfg_attr(not(debug_assertions), allow(unused_mut, unused_variables))]
    let mut predicted = [0i64; 3];

    for ry in 0..hc {
        for rx in 0..wc {
            let addr = ry * wc + rx;
            // The filter's own neighbour question, for the geometry this
            // encoder writes: one slice, one tile, so a neighbour is
            // usable exactly when it is inside the picture. Written as a
            // closure per component below, because it is asked in
            // component coordinates.
            let mut own = [SaoParams::default(); 3];
            let mut own_cost = 0f32;
            let mut own_dist = [0i64; 3];
            let mut comps: Vec<Comp<'_, S>> = Vec::with_capacity(ncomp);
            for c in 0..ncomp {
                let (csw, csh) = if c == 0 { (1, 1) } else { (sw, sh) };
                let (plane, src, stride) = match c {
                    0 => (&recon.y, src_y, y_stride),
                    1 => (&recon.cb, src_cb, c_stride),
                    _ => (&recon.cr, src_cr, c_stride),
                };
                let (pw, ph) = (recon.width / csw, recon.height / csh);
                let x0 = rx * ctb / csw;
                let y0 = ry * ctb / csh;
                comps.push(Comp {
                    rec: plane,
                    src,
                    src_stride: stride,
                    x0,
                    y0,
                    w: (ctb / csw).min(pw - x0),
                    h: (ctb / csh).min(ph - y0),
                    pw,
                    ph,
                    max: (1i32 << if c == 0 { sps.bit_depth_luma } else { sps.bit_depth_chroma }) - 1,
                    shift: if c == 0 { sps.bit_depth_luma as i32 } else { sps.bit_depth_chroma as i32 } - 5,
                });
            }
            let usable = |_x: usize, _y: usize, xn: i32, yn: i32, pw: usize, ph: usize| -> bool {
                xn >= 0 && yn >= 0 && (xn as usize) < pw && (yn as usize) < ph
            };
            for (c, comp) in comps.iter().enumerate() {
                debug_assert!(comp.sane(), "SAO decision handed a component that does not fit its plane");
                let (pw, ph) = (comp.pw, comp.ph);
                let u = move |x: usize, y: usize, xn: i32, yn: i32| usable(x, y, xn, yn, pw, ph);
                if c == 2 {
                    // Cr cannot choose its own type or edge class — the
                    // syntax gives it Cb's. Decide its offsets under that
                    // constraint rather than picking a shape it cannot
                    // spell.
                    let fixed = own[1];
                    let (p, cost, dist) = comp.decide_constrained(fixed.type_idx, fixed.band_or_class, cmax, lam, &u);
                    own[2] = p;
                    own_cost += cost;
                    own_dist[2] = dist;
                    continue;
                }
                let (p, cost, dist) = comp.decide(cmax, lam, true, &u);
                own[c] = p;
                own_cost += cost;
                own_dist[c] = dist;
            }

            // Merge: the neighbour's parameters applied to these samples,
            // priced at the two bins a merge actually costs.
            let mut best = (own_cost, None::<SaoMerge>, own, own_dist);
            for (which, src_addr) in [(SaoMerge::Left, addr.checked_sub(1).filter(|_| rx > 0)), (SaoMerge::Up, addr.checked_sub(wc).filter(|_| ry > 0))] {
                let Some(na) = src_addr else { continue };
                let cand = params[na];
                let mut dist = [0i64; 3];
                for (c, comp) in comps.iter().enumerate() {
                    let (pw, ph) = (comp.pw, comp.ph);
                    let u = move |x: usize, y: usize, xn: i32, yn: i32| usable(x, y, xn, yn, pw, ph);
                    dist[c] = comp.ssd_under(&cand[c], &u);
                }
                // One or two merge bins, and nothing else.
                let cost = dist.iter().sum::<i64>() as f32 + lam * if which == SaoMerge::Left { 1.0 } else { 2.0 };
                if cost < best.0 {
                    best = (cost, Some(which), cand, dist);
                }
            }
            drop(comps);
            merges[addr] = best.1;
            params[addr] = best.2;
            info.sao[addr] = best.2;
            for c in 0..3 {
                predicted[c] += best.3[c];
            }
        }
    }

    // Apply, exactly as `decoder.rs` does it: a band of deblocked source
    // lines per CTB row so an already-filtered neighbour never feeds back,
    // then the decoder's own kernels over it.
    let mut band = SaoBand::<S>::new();
    let mut src = Frame::<S>::new(recon.width, ctb + 4, recon.chroma, recon.bit_depth);
    for ry in 0..hc {
        band.fill(recon, &mut src, ctb, ry);
        sao_ctb_row(ctx.dsp, recon, &src, &band, info, sps, pps, ry);
    }

    // The decision scored every candidate analytically, never by
    // filtering. That is only sound if this module classifies samples
    // exactly as the filter above does, so in a debug build the two are
    // compared for real: one pass over the picture, against the very
    // prediction the choices were made on. A mismatch means the decision
    // is choosing against a model of a filter that does not exist — a bug
    // that produces legal streams and worse pictures, and that nothing
    // else here would notice.
    #[cfg(debug_assertions)]
    {
        let mut actual = [0i64; 3];
        for c in 0..ncomp {
            let (csw, csh) = if c == 0 { (1, 1) } else { (sw, sh) };
            let (plane, src, stride) = match c {
                0 => (&recon.y, src_y, y_stride),
                1 => (&recon.cb, src_cb, c_stride),
                _ => (&recon.cr, src_cr, c_stride),
            };
            let (pw, ph) = (recon.width / csw, recon.height / csh);
            for y in 0..ph {
                for x in 0..pw {
                    let e = plane.data[plane.offset(x as isize, y as isize)].to_i32() as i64 - src[y * stride + x].to_i32() as i64;
                    actual[c] += e * e;
                }
            }
        }
        for c in 0..ncomp {
            debug_assert_eq!(
                actual[c], predicted[c],
                "SAO component {c}: the decision predicted {} and the filter produced {} - the two classify samples differently",
                predicted[c], actual[c]
            );
        }
    }

    SaoPlan { params, merges }
}

impl<S: Sample> Comp<'_, S> {
    /// [`Comp::decide`] with the type and class already fixed by another
    /// component — the Cr case, whose `sao_type_idx` and `sao_eo_class`
    /// come from Cb and whose four offsets are its own.
    fn decide_constrained(&self, type_idx: u8, class: u8, cmax: i64, lam: f32, usable: &dyn Fn(usize, usize, i32, i32) -> bool) -> (SaoParams, f32, i64) {
        let base = self.ssd_off();
        if type_idx == 0 {
            return (SaoParams::default(), base as f32, base);
        }
        let mut p = SaoParams { type_idx, band_or_class: class, offsets: [0; 4] };
        let mut delta = 0i64;
        if type_idx == 1 {
            let bands = self.band_stats();
            for k in 0..4 {
                let c = &bands[(class as usize + k) & 31];
                let o = c.best().clamp(-cmax, cmax);
                let d = c.delta(o);
                if o != 0 && d < 0 {
                    p.offsets[k] = o as i16;
                    delta += d;
                }
            }
        } else {
            let cats = self.edge_stats(class, usable);
            for (slot, cat) in [(0usize, 0usize), (1, 1), (2, 3), (3, 4)] {
                let c = &cats[cat];
                let want = c.best();
                let o = if slot < 2 { want.clamp(0, cmax) } else { want.clamp(-cmax, 0) };
                let d = c.delta(o);
                if o != 0 && d < 0 {
                    p.offsets[slot] = o as i16;
                    delta += d;
                }
            }
        }
        // Cr codes no type and no class, only its magnitudes and (in band
        // mode) their signs.
        let cost = (base + delta) as f32 + lam * bins_of(&p, cmax as u32, false) as f32;
        (p, cost, base + delta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::Cpu;
    use crate::dsp::distortion::DistortionDsp;
    use crate::dsp::hevc::HevcDsp;
    use crate::dsp::hevc_enc::HevcEncDsp;
    use crate::encode::Config;
    use crate::encode::h265_syntax::{Geometry as EncGeometry, write_pps, write_sps};
    use crate::picture::ChromaFormat;

    struct Kit {
        dsp: HevcDsp<u8>,
        enc: HevcEncDsp,
        dist: DistortionDsp<u8>,
    }

    impl Kit {
        fn new() -> Self {
            Kit { dsp: HevcDsp::new(Cpu::SCALAR), enc: HevcEncDsp::scalar(), dist: DistortionDsp::scalar() }
        }
        fn ctx(&self, qp: i32) -> IntraCtx<'_, u8> {
            IntraCtx { dsp: &self.dsp, enc: &self.enc, dist: &self.dist, qp, bit_depth: 8, strong_smoothing: false, bypass: false, free_to_trim: false }
        }
    }

    fn sets(w: u32, h: u32) -> (Sps, Pps) {
        let cfg = Config { width: w, height: h, chroma: ChromaFormat::Yuv420, bit_depth: 8, sao: true, ..Config::default() };
        let g = EncGeometry::new(&cfg);
        let sps = Sps::parse(&crate::nal::unescape_rbsp(&write_sps(&cfg, &g, 16, None))).unwrap();
        let mut pps = Pps::parse(&crate::nal::unescape_rbsp(&write_pps(30, false, true))).unwrap();
        pps.resolve_tiles(&sps).unwrap();
        (sps, pps)
    }

    /// A source picture and a reconstruction of it that is wrong in a way
    /// SAO is built to correct.
    ///
    /// This is the part of the test that has to be right, and the first
    /// attempt was not: seeding the reconstruction with zero-mean uniform
    /// noise produced a scene where SAO correctly chose to do nothing, and
    /// a test asserting it had done something would have been asserting a
    /// bug. SAO does not remove noise. It corrects error that is
    /// *conditional* on something the decoder can also see — the local
    /// shape (edge offsets) or the sample value (band offsets) — so the
    /// scenes below make exactly those errors and nothing else.
    struct Scene {
        recon: Frame<u8>,
        src_y: Vec<u8>,
        src_cb: Vec<u8>,
        src_cr: Vec<u8>,
        info: PicInfo,
        sps: Sps,
        pps: Pps,
        w: usize,
        h: usize,
    }

    /// What kind of error the reconstruction carries.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Err_ {
        /// Overshoot at local extrema — ringing. A local minimum comes
        /// back too low and a local maximum too high, which is precisely
        /// the pair edge categories 0 and 4 name.
        Ringing,
        /// A constant shift applied only to samples in one four-band
        /// window, which is what a band offset can undo and an edge offset
        /// cannot see at all.
        BandShift,
    }

    fn scene(w: usize, h: usize, kind: Err_, amount: i32) -> Scene {
        let (sps, pps) = sets(w as u32, h as u32);
        let geo = std::sync::Arc::new(crate::hevc::pic::Geometry::new(&sps, &pps));
        let mut info = PicInfo::new(geo);
        // One slice covering the picture, no exempt blocks: what the
        // deblocker leaves behind.
        info.ctb_slice.fill(0);
        info.ctb_slice_addr.fill(0);

        // The source. Ringing wants texture with real extrema; a band
        // shift wants values inside one window, bands being eight wide at
        // eight bits, so 96..=127 is exactly bands 12..=15.
        let src_at = |x: usize, y: usize| -> u8 {
            match kind {
                Err_::Ringing => (60 + ((x % 5) as i32 - 2) * 18 + ((y % 7) as i32 - 3) * 9).clamp(10, 245) as u8,
                Err_::BandShift => (96 + ((x / 8 + y / 8) % 32) as i32 % 32).clamp(96, 127) as u8,
            }
        };
        let mut src_y = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                src_y[y * w + x] = src_at(x, y);
            }
        }
        let (cw, chh) = (w / 2, h / 2);
        let mut src_cb = vec![0u8; cw * chh];
        let mut src_cr = vec![0u8; cw * chh];
        for y in 0..chh {
            for x in 0..cw {
                src_cb[y * cw + x] = src_at(x * 2, y * 2);
                src_cr[y * cw + x] = src_at(x * 2 + 1, y * 2);
            }
        }

        let perturb = |plane: &mut Plane16<u8>, src: &[u8], pw: usize, ph: usize| {
            let at = |x: usize, y: usize| src[y.min(ph - 1) * pw + x.min(pw - 1)] as i32;
            let (o, stride) = (plane.origin(), plane.stride);
            for y in 0..ph {
                for x in 0..pw {
                    let v = at(x, y);
                    let e = match kind {
                        Err_::Ringing => {
                            // Compare with the horizontal neighbours the
                            // source has; the picture edge simply gets no
                            // perturbation, which is also where the filter
                            // will decline to act.
                            if x == 0 || x + 1 >= pw {
                                0
                            } else {
                                let (l, r) = (at(x - 1, y), at(x + 1, y));
                                if v < l && v < r {
                                    -amount
                                } else if v > l && v > r {
                                    amount
                                } else {
                                    0
                                }
                            }
                        }
                        Err_::BandShift => {
                            if (96..=127).contains(&v) { amount } else { 0 }
                        }
                    };
                    plane.data[o + y * stride + x] = (v + e).clamp(0, 255) as u8;
                }
            }
        };
        let mut recon = Frame::<u8>::new(w, h, ChromaFormat::Yuv420, 8);
        perturb(&mut recon.y, &src_y, w, h);
        perturb(&mut recon.cb, &src_cb, cw, chh);
        perturb(&mut recon.cr, &src_cr, cw, chh);
        Scene { recon, src_y, src_cb, src_cr, info, sps, pps, w, h }
    }

    impl Scene {
        fn run(&mut self, ctx: &IntraCtx<'_, u8>) -> SaoPlan {
            let Scene { recon, info, sps, pps, src_y, src_cb, src_cr, w, .. } = self;
            sao_picture(ctx, recon, info, sps, pps, src_y, *w, src_cb, src_cr, *w / 2)
        }
        /// Luma SSD against the source, over the whole picture.
        fn ssd(&self) -> i64 {
            let (o, stride) = (self.recon.y.origin(), self.recon.y.stride);
            let mut d = 0i64;
            for y in 0..self.h {
                for x in 0..self.w {
                    let e = self.recon.y.data[o + y * stride + x] as i64 - self.src_y[y * self.w + x] as i64;
                    d += e * e;
                }
            }
            d
        }
    }

    /// The property that matters: SAO must leave the picture closer to the
    /// source than it found it.
    ///
    /// Not "the filter ran" and not "parameters were chosen" — a module
    /// that picked plausible offsets with the wrong sign, or that
    /// classified samples differently than the filter does, would still
    /// produce parameters and still filter, and would make the
    /// reconstruction worse. Both error shapes, because they exercise
    /// different halves of the decision.
    #[test]
    fn sao_moves_the_reconstruction_towards_the_source() {
        let kit = Kit::new();
        let ctx = kit.ctx(35);
        for (w, h) in [(64usize, 64usize), (128, 64), (64, 96)] {
            for (kind, amount, name) in [(Err_::Ringing, 5, "ringing"), (Err_::BandShift, 5, "band shift")] {
                let mut sc = scene(w, h, kind, amount);
                let before = sc.ssd();
                let plan = sc.run(&ctx);
                let after = sc.ssd();
                assert!(after < before, "{w}x{h} {name}: SAO made the picture worse ({before} -> {after})");
                assert!(
                    plan.params.iter().any(|p| p.iter().any(|c| c.type_idx != 0)),
                    "{w}x{h} {name}: nothing was chosen, so the improvement is not SAO's"
                );
            }
        }
    }

    /// Both offset kinds must be reachable, and the merge must be taken.
    /// Without this the module could lose a whole branch — the band search,
    /// say — and every other test here would still pass, because edge
    /// offsets alone fix the ringing scene.
    ///
    /// The two scenes are built so that each branch is the *only* one that
    /// can help: a band shift is invisible to the edge classifier (it moves
    /// whole value ranges, not local shapes), and ringing is invisible to
    /// the band classifier (its samples are spread across every band).
    #[test]
    fn every_branch_of_the_decision_is_reachable() {
        let kit = Kit::new();
        let (mut band, mut edge, mut merged, mut off) = (0usize, 0usize, 0usize, 0usize);
        for qp in [30i32, 35, 40] {
            let ctx = kit.ctx(qp);
            for (kind, amount) in [(Err_::Ringing, 6), (Err_::BandShift, 5)] {
                let mut sc = scene(128, 64, kind, amount);
                let plan = sc.run(&ctx);
                for (a, p) in plan.params.iter().enumerate() {
                    for c in p {
                        match c.type_idx {
                            0 => off += 1,
                            1 => band += 1,
                            _ => edge += 1,
                        }
                    }
                    if plan.merges[a].is_some() {
                        merged += 1;
                    }
                }
            }
        }
        assert!(band > 0, "no CTB component chose band offsets");
        assert!(edge > 0, "no CTB component chose edge offsets");
        assert!(off > 0, "no CTB component was left alone");
        assert!(merged > 0, "no CTB merged its neighbour's parameters");
    }

    /// Cr cannot spell its own `sao_type_idx`, nor its own `sao_eo_class` —
    /// the syntax gives it Cb's — so the decision must never produce a plan
    /// the writer would have to lie about. `write_sao` debug-asserts the
    /// same thing; this proves the decision satisfies it rather than
    /// relying on a debug build to notice, and the guard below keeps it
    /// from passing on a plan where every chroma component was simply off.
    #[test]
    fn cr_never_disagrees_with_cb_about_what_it_is() {
        let kit = Kit::new();
        let mut live = 0usize;
        for qp in [30i32, 35, 40] {
            let ctx = kit.ctx(qp);
            for (kind, amount) in [(Err_::Ringing, 6), (Err_::BandShift, 5)] {
                let mut sc = scene(128, 64, kind, amount);
                let plan = sc.run(&ctx);
                for (a, p) in plan.params.iter().enumerate() {
                    assert_eq!(p[2].type_idx, p[1].type_idx, "CTB {a}: Cr and Cb disagree about sao_type_idx");
                    if p[1].type_idx == 2 {
                        assert_eq!(p[2].band_or_class, p[1].band_or_class, "CTB {a}: Cr and Cb disagree about sao_eo_class");
                    }
                    if p[1].type_idx != 0 {
                        live += 1;
                    }
                }
            }
        }
        assert!(live > 0, "every chroma component was off; the agreement above was vacuous");
    }

    /// Offsets must be spellable: magnitudes within cMax, edge offsets in
    /// the sign the syntax forces, band positions inside five bits, edge
    /// classes inside two. `write_sao` cannot spell anything else, and an
    /// unspellable plan is a desync rather than a bad picture.
    #[test]
    fn every_chosen_parameter_is_within_what_the_syntax_can_carry() {
        let kit = Kit::new();
        let cmax = (1i16 << (8 - 5)) - 1;
        for qp in [28i32, 35, 45] {
            let ctx = kit.ctx(qp);
            for (kind, amount) in [(Err_::Ringing, 7), (Err_::BandShift, 7)] {
                let mut sc = scene(128, 64, kind, amount);
                let plan = sc.run(&ctx);
                for (a, p) in plan.params.iter().enumerate() {
                    for (c, comp) in p.iter().enumerate() {
                        for o in comp.offsets {
                            assert!(o.abs() <= cmax, "CTB {a} comp {c}: offset {o} above cMax {cmax}");
                        }
                        match comp.type_idx {
                            1 => assert!(comp.band_or_class < 32, "CTB {a} comp {c}: band position out of range"),
                            2 => {
                                assert!(comp.band_or_class < 4, "CTB {a} comp {c}: edge class out of range");
                                assert!(comp.offsets[0] >= 0 && comp.offsets[1] >= 0, "CTB {a} comp {c}: edge offsets 0/1 must be positive");
                                assert!(comp.offsets[2] <= 0 && comp.offsets[3] <= 0, "CTB {a} comp {c}: edge offsets 2/3 must be negative");
                            }
                            _ => assert_eq!(comp.offsets, [0; 4], "CTB {a} comp {c}: a component that is off carries offsets"),
                        }
                    }
                }
            }
        }
    }

    /// A merged CTB's stored parameters must equal the neighbour's, because
    /// that is what the reader copies — the writer sends no offsets at all.
    /// A plan whose merged entries differed would filter one picture and
    /// describe another, and SELF would fail far from here.
    #[test]
    fn a_merged_ctb_carries_exactly_its_neighbours_parameters() {
        let kit = Kit::new();
        let ctx = kit.ctx(35);
        let mut sc = scene(128, 64, Err_::Ringing, 6);
        let wc = sc.info.wc;
        let plan = sc.run(&ctx);
        let mut seen = 0usize;
        for (a, m) in plan.merges.iter().enumerate() {
            let Some(m) = m else { continue };
            seen += 1;
            let from = match m {
                SaoMerge::Left => a - 1,
                SaoMerge::Up => a - wc,
            };
            assert_eq!(plan.params[a], plan.params[from], "CTB {a}: merged parameters differ from the neighbour's");
        }
        assert!(seen > 0, "no merge was taken, so nothing was checked");
    }
}
