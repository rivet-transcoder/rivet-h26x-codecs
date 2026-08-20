//! Picture-level rate control: choosing a quantiser per picture to meet a
//! bitrate target.
//!
//! # Three kinds of property, and why this one is different
//!
//! Everything this encoder produces has been checkable against something.
//! Rate control is the first thing that is not, and it is worth naming the
//! three kinds so the difference is visible rather than felt:
//!
//! - **Conformance** — SELF and CROSS. The stream means what the encoder
//!   thinks it means: our decoder reproduces the encoder's own
//!   reconstruction byte for byte, and libavcodec agrees with our decoder.
//!   *Exact*, and it has a reference: the decoder.
//! - **Quality** — PSNR. Reported, never gated (except lossless, where it
//!   becomes exact and therefore conformance again). A *measurement*: it
//!   informs, and no particular value is required.
//! - **Control** — did the encoder achieve the objective it was handed?
//!   No ground truth exists at all. Unlike quality it is not informational:
//!   hitting the target *is* the feature. And unlike conformance, nothing
//!   in the bitstream is wrong when it fails.
//!
//! [`super::h265_sao`]'s predicted-versus-actual check was the first
//! instance of the third kind here, before it had a name. This module is
//! the second, and the sharper one: **a controller that ignores its target
//! entirely still produces a perfectly legal stream that passes SELF,
//! passes CROSS, and reports a fine PSNR.** Every check this project had
//! before today is blind to it.
//!
//! ## So what is actually checked
//!
//! Three things, at three different strengths, and the strengths are not
//! interchangeable:
//!
//! 1. **Exact — the ledger.** [`RateController::bits_spent`] must equal
//!    the bytes actually emitted, times eight. Not the *prediction* — a
//!    rate model is a heuristic and asserting it would be asserting a wish
//!    — but the *accounting*. This is what catches the silent class:
//!    forgetting start-code and NAL-header overhead, counting parameter
//!    sets once or three times, dropping the last picture from the
//!    accumulator, measuring the slice payload against an access-unit
//!    target. A ledger drifting eight percent low overshoots forever with
//!    every other check green. The encoder holds it to equality on every
//!    picture.
//! 2. **Gated, and deliberately loose — the band.** The gate asserts the
//!    achieved rate lands within [0.5x, 2.0x] of target. That is a wide
//!    band and it is wide on purpose: the corpus is six to twelve frames,
//!    which gives a controller almost no time to converge, and the opening
//!    IDR dominates a clip that short. A tighter tolerance would be flaky
//!    rather than rigorous, and a flaky row teaches people to re-run it.
//!    The tightness is bought back elsewhere: the gate's targets *bracket*
//!    each clip's natural constant-QP rate, so the controller is forced to
//!    move the quantiser in both directions and a target-ignoring
//!    controller fails the low row on every clip.
//! 3. **Ordered response — a test, not a gate row.** Encode one clip at
//!    several targets: the sizes must be strictly ordered with the targets
//!    and separated by a real margin. This is the strongest anti-vacuity
//!    property and it needs no convergence, so six frames are plenty — but
//!    it is a comparison *between* encodes, and the gate runs one cell at a
//!    time. It lives in this module's tests instead.
//!
//! The operating rule behind all three: **a rate-control check is
//! meaningful exactly when it fails for a controller that ignores its
//! target.** That is an experiment, not a judgement — replace this module's
//! output with a constant quantiser and see what goes red.
//!
//! ## What is deliberately not here
//!
//! - **No lookahead.** Every decision is made from the past only. A
//!   controller that has seen the next second of video can place bits far
//!   better; this one cannot see them.
//! - **No per-CTB adaptation.** The quantiser is picture-wide.
//!   `cu_qp_delta_enabled_flag` is 0 in our PPS, so per-CTB QP needs new
//!   syntax, a new writer and the reader's QP-prediction machinery — and it
//!   would stack a second control loop on top of this one before this one
//!   is proven. It is a follow-up, not an omission.
//! - **No VBV / HRD buffer model.** That one *does* have a decoder-side
//!   counterpart and therefore a conformance property, which makes it a
//!   task of its own rather than a corner of this one. Refused by name.
//! - **No bit allocation across a GOP beyond the intra/inter split below.**
//!
//! # The model, stated plainly
//!
//! One law, used everywhere: **bits halve for every six added to the
//! quantiser.** That is the quantiser's own definition — the step size
//! doubles every six QP — and at moderate rates the bit cost tracks it
//! closely enough to steer by. So for a picture of a given complexity,
//!
//! ```text
//!     bits(qp) ≈ k * 2^(-qp / 6)
//! ```
//!
//! and one observation `(qp, bits)` pins `k = bits * 2^(qp / 6)`. To ask
//! for `target` bits, invert it: `qp = 6 * log2(k / target)`.
//!
//! `k` is complexity, and it is tracked separately for intra and inter
//! pictures because they differ by an order of magnitude. Before any
//! observation exists it is seeded from the target's bits per pixel through
//! the same law, so the very first picture is a considered guess rather
//! than a fixed constant.
//!
//! Everything else is damping: the quantiser moves at most
//! [`MAX_QP_STEP`] per picture so quality does not visibly pulse, and the
//! bucket's correction is spread over [`CORRECTION_PICTURES`] rather than
//! taken out of the next picture alone.

/// The largest quantiser change between consecutive pictures. Rate control
/// that lurches is worse to watch than rate control that misses: a picture
/// noticeably softer than the one before it reads as a glitch, while a
/// steady small error reads as nothing at all.
const MAX_QP_STEP: i32 = 3;

/// How many pictures a bucket correction is spread over. Taking the whole
/// error out of the next picture makes the controller oscillate — it
/// overshoots, over-corrects, and rings.
const CORRECTION_PICTURES: f64 = 8.0;

/// How much more of the budget an intra picture may take than an inter one.
///
/// An IDR costs several times a P at the same quantiser, so splitting a
/// GOP's bits evenly starves everything after the keyframe — the classic
/// failure where picture two of every GOP is visibly worse than picture
/// one. Four is a round number in the right region rather than a measured
/// constant, and it is the first thing to replace with a measurement.
const INTRA_WEIGHT: f64 = 4.0;

/// Quantiser bounds. The syntax allows 0..=51 and both ends are legal;
/// these are the same bounds `ConstantQp` clamps to.
const QP_MIN: i32 = 0;
/// See [`QP_MIN`].
const QP_MAX: i32 = 51;

/// Bounds on the *seeded* quantiser — the one picked before any picture
/// has been measured. Deliberately narrower than the legal range.
///
/// A seed is a guess, and **the cost of guessing wrong is not symmetric**.
/// Guess too high and one picture is softer than it needed to be, which
/// the next measurement corrects. Guess too low and that picture can eat
/// the entire clip's budget, which nothing recovers — on a short clip
/// there is no later to make it back in. So the floor sits at the codec's
/// neutral quantiser: the estimate is allowed to say "compress harder than
/// neutral", and is not allowed to say "spend more freely than neutral"
/// before a single picture has been measured.
///
/// This is not hypothetical tuning. The bits-per-pixel estimate below has
/// one anchor — 0.1 bits per pixel at quantiser 32 — and an anchor cannot
/// be right for all content. Measured against the gate's own clips it was
/// **eighteen quantiser steps too low**, consistently, because a small
/// detailed picture is far harder per pixel than the anchor assumes. Its
/// *slope* was right (both it and the content move six steps per doubling)
/// and only its offset was wrong, which is exactly the error a single
/// measurement fixes and an open-loop guess cannot.
const SEED_QP_MIN: f64 = 26.0;
/// See [`SEED_QP_MIN`].
const SEED_QP_MAX: f64 = 45.0;

/// Which complexity estimate a picture draws on.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PicKind {
    /// An IDR or other intra-coded picture.
    Intra,
    /// A P or B picture.
    Inter,
}

/// Per-kind complexity: `k` in `bits ≈ k * 2^(-qp/6)`.
#[derive(Clone, Copy)]
struct Complexity {
    k: f64,
    /// Whether `k` has been pinned by a real observation yet, or is still
    /// the seed. The first real observation replaces the seed outright
    /// rather than being blended into it — a seed is a guess and deserves
    /// no weight once a fact exists.
    observed: bool,
}

/// Picture-level rate control against an average bitrate.
///
/// One per encoder, driven in coding order: [`RateController::pick_qp`]
/// before each picture, [`RateController::account`] after it, with the
/// size of the access unit the encoder actually emitted.
pub struct RateController {
    /// Bits this picture's share of a second works out to.
    per_picture: f64,
    /// Average weight across the GOP, so the intra/inter split redistributes
    /// bits without changing their total.
    avg_weight: f64,
    /// The leaky bucket: bits underspent so far. Negative means overspent.
    budget: f64,
    /// Per-kind complexity, indexed by [`PicKind`].
    complexity: [Complexity; 2],
    /// The last quantiser chosen **from a measured model**, per kind, for
    /// the step limit.
    ///
    /// Two refinements over "the last quantiser", both learned from
    /// watching an eight-picture clip converge:
    ///
    /// - **Per kind.** An intra picture is given several times an inter
    ///   picture's bits and therefore sits at a genuinely lower quantiser.
    ///   Damping one against the other is not damping, it is dragging two
    ///   unrelated quantities together.
    /// - **Measured only.** A limit is worth having between two considered
    ///   choices. Applying it against a *seed* means the first informed
    ///   correction — the single most valuable decision the controller
    ///   makes — is throttled to three steps, and on a short clip it never
    ///   arrives: the run below took five of eight pictures crawling from
    ///   a bad guess to the right answer, overspending the whole way.
    last_informed: [Option<u8>; 2],
    /// **The ledger.** Total bits emitted, as measured from the access
    /// units themselves — start codes, NAL headers, parameter sets and
    /// all. The encoder asserts this against the bytes it has produced;
    /// see the module documentation for why this is the one thing here
    /// held to equality.
    pub bits_spent: u64,
    /// Pictures accounted for, for the same reason.
    pub pictures: u64,
    /// What [`RateController::pick_qp`] chose for the picture currently
    /// being coded, so [`RateController::account`] can pin the model
    /// against the quantiser that actually produced the bits.
    pending: Option<(PicKind, u8)>,
}

impl RateController {
    /// A controller for `bps` bits per second at `fps` pictures per second
    /// over a `width` by `height` picture, with `gop` pictures between IDRs
    /// (0 meaning every picture is one).
    pub fn new(bps: u32, fps: u32, width: u32, height: u32, gop: u32) -> Self {
        let fps = fps.max(1) as f64;
        let per_picture = (bps as f64 / fps).max(1.0);
        // With `gop` pictures per keyframe, one of them carries
        // INTRA_WEIGHT and the rest carry 1.
        let n = if gop == 0 { 1.0 } else { gop as f64 };
        let avg_weight = (INTRA_WEIGHT + (n - 1.0)) / n;

        // Seed complexity from the target's bits per pixel, through the
        // same law the controller steers by: 0.1 bits per pixel is about
        // quantiser 32 on ordinary content, and every doubling of the rate
        // buys six quantiser steps. Turning that seed quantiser back into
        // a `k` keeps one law in the module instead of two.
        let pixels = (width.max(1) as f64) * (height.max(1) as f64);
        let seed_k = |target: f64| -> f64 {
            let bpp = (target / pixels).max(1e-6);
            let qp = (32.0 - 6.0 * (bpp / 0.1).log2()).clamp(SEED_QP_MIN, SEED_QP_MAX);
            target * 2f64.powf(qp / 6.0)
        };
        let intra_target = per_picture * INTRA_WEIGHT / avg_weight;
        let inter_target = per_picture / avg_weight;
        RateController {
            per_picture,
            avg_weight,
            budget: 0.0,
            complexity: [
                Complexity { k: seed_k(intra_target), observed: false },
                Complexity { k: seed_k(inter_target), observed: false },
            ],
            last_informed: [None; 2],
            bits_spent: 0,
            pictures: 0,
            pending: None,
        }
    }

    /// The bits this picture is aiming for: its share by kind, plus a
    /// slice of whatever the bucket has over- or under-spent so far.
    fn target_for(&self, kind: PicKind) -> f64 {
        let weight = match kind {
            PicKind::Intra => INTRA_WEIGHT,
            PicKind::Inter => 1.0,
        };
        let base = self.per_picture * weight / self.avg_weight;
        // Spread the correction, and never let it move the target by more
        // than half — a single expensive picture should bend the next few,
        // not flatten them.
        let correction = (self.budget / CORRECTION_PICTURES).clamp(-0.5 * base, 0.5 * base);
        (base + correction).max(16.0)
    }

    /// Choose the quantiser for the next picture. Call once per picture,
    /// in coding order, before coding it.
    pub fn pick_qp(&mut self, kind: PicKind) -> u8 {
        let target = self.target_for(kind);
        let c = self.complexity[kind as usize];
        // Invert bits(qp) = k * 2^(-qp/6).
        let want = 6.0 * (c.k / target).log2();
        let mut qp = want.round().clamp(QP_MIN as f64, QP_MAX as f64) as i32;
        // The step limit stops the quantiser pulsing between *considered*
        // choices, so it applies only between two of them — see
        // `last_informed`. The first informed pick of each kind is allowed
        // to be as large a correction as it needs to be, because the thing
        // it is correcting is a guess.
        let informed = c.observed;
        if let (Some(last), true) = (self.last_informed[kind as usize], informed) {
            let last = last as i32;
            qp = qp.clamp(last - MAX_QP_STEP, last + MAX_QP_STEP);
        }
        let qp = qp.clamp(QP_MIN, QP_MAX) as u8;
        if informed {
            self.last_informed[kind as usize] = Some(qp);
        }
        self.pending = Some((kind, qp));
        qp
    }

    /// Record what the picture actually cost, in **bytes of the access
    /// unit** — everything the encoder emitted for it, because that is
    /// what the target is measured against.
    ///
    /// Updates the ledger, the bucket and the complexity estimate. Must be
    /// called exactly once for every [`RateController::pick_qp`], or the
    /// ledger assertion in the encoder will say so.
    pub fn account(&mut self, bytes: usize) {
        let bits = (bytes as u64) * 8;
        self.bits_spent += bits;
        self.pictures += 1;
        let Some((kind, qp)) = self.pending.take() else {
            debug_assert!(false, "account() without a matching pick_qp()");
            return;
        };
        self.budget += self.per_picture - bits as f64;
        // Pin the model: one observation determines k exactly, given the
        // quantiser that produced it. A picture that coded to nothing says
        // nothing about complexity, so it is not allowed to zero the
        // estimate.
        if bits > 0 {
            let k_obs = bits as f64 * 2f64.powf(qp as f64 / 6.0);
            let c = &mut self.complexity[kind as usize];
            c.k = if c.observed { 0.5 * c.k + 0.5 * k_obs } else { k_obs };
            c.observed = true;
        }
    }

    /// The achieved rate in bits per second, given the frame rate the
    /// controller was built with — for reporting, never for deciding.
    pub fn achieved_bps(&self, fps: u32) -> f64 {
        if self.pictures == 0 {
            return 0.0;
        }
        self.bits_spent as f64 * fps.max(1) as f64 / self.pictures as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stand-in encoder: bits really do follow `k * 2^(-qp/6)` for a
    /// fixed complexity, so the controller's model is exactly right and
    /// what is under test is the *loop* — the bucket, the step limit, the
    /// intra split — rather than the model's fit to real video.
    fn synth_bits(k: f64, qp: u8) -> usize {
        ((k * 2f64.powf(-(qp as f64) / 6.0)) / 8.0).round().max(1.0) as usize
    }

    /// A realistic operating point, which the first version of these tests
    /// was not: 640x480 at a few hundred kilobits, and a scene whose cost
    /// puts the natural quantiser in the middle of the range.
    ///
    /// The first version drove a 64x64 picture at 200 kbps and upward,
    /// which works out to between four and forty *bits per pixel*. The
    /// seed correctly recommended quantiser 0 for it, the synthetic scene
    /// could not spend that many bits, and the tests failed — on the test's
    /// arithmetic, not the controller's. Worth recording, because a test
    /// whose scenario is impossible reads exactly like a broken feature.
    const W: u32 = 640;
    /// See [`W`].
    const H: u32 = 480;
    /// Complexity of the synthetic intra picture, chosen so that its
    /// natural quantiser at the middle target below is around 26.
    const K_INTRA: f64 = 1.17e6;
    /// See [`K_INTRA`].
    const K_INTER: f64 = 2.93e5;

    /// The ledger is the one exact property here: whatever bytes are
    /// handed to `account`, `bits_spent` is eight times their sum. Trivial
    /// arithmetic, deliberately pinned, because every silent rate-control
    /// bug this module warns about shows up as this number drifting from
    /// the bytes actually emitted.
    #[test]
    fn the_ledger_counts_every_byte_exactly_once() {
        let mut rc = RateController::new(500_000, 30, 64, 64, 8);
        let sizes = [900usize, 120, 140, 95, 210, 88, 400, 3];
        for (i, &b) in sizes.iter().enumerate() {
            let kind = if i == 0 { PicKind::Intra } else { PicKind::Inter };
            let _ = rc.pick_qp(kind);
            rc.account(b);
        }
        assert_eq!(rc.pictures, sizes.len() as u64);
        assert_eq!(rc.bits_spent, sizes.iter().sum::<usize>() as u64 * 8);
    }

    /// **The anti-vacuity property**, and the reason it is a test rather
    /// than a gate row: it compares whole encodes against each other, and
    /// the gate runs one cell at a time.
    ///
    /// Ask for more bits and more bits must come out — strictly ordered
    /// across a wide range of targets, and separated by a real margin, not
    /// by rounding. A controller that ignored its target would return the
    /// same quantiser every time and produce identical totals; this is the
    /// assertion that catches exactly that, and it needs no convergence,
    /// so a handful of pictures is enough.
    #[test]
    fn asking_for_more_bits_produces_more_bits() {
        // One fixed-complexity scene, coded at a spread of targets.
        let (ki, kp) = (K_INTRA, K_INTER);
        let mut totals = Vec::new();
        for bps in [200_000u32, 600_000, 1_800_000, 5_400_000] {
            let mut rc = RateController::new(bps, 30, W, H, 8);
            let mut total = 0usize;
            for i in 0..12 {
                let kind = if i % 8 == 0 { PicKind::Intra } else { PicKind::Inter };
                let qp = rc.pick_qp(kind);
                let b = synth_bits(if kind == PicKind::Intra { ki } else { kp }, qp);
                rc.account(b);
                total += b;
            }
            totals.push((bps, total));
        }
        for w in totals.windows(2) {
            let ((lo_bps, lo), (hi_bps, hi)) = (w[0], w[1]);
            assert!(hi > lo, "target {hi_bps} produced {hi} bytes, not more than target {lo_bps}'s {lo}");
            // A real margin, not rounding: tripling the target must move
            // the total by more than a quarter.
            assert!(
                hi as f64 > lo as f64 * 1.25,
                "target {hi_bps} produced {hi} bytes against {lo_bps}'s {lo} — ordered, but barely responsive"
            );
        }
    }

    /// Against a scene whose cost really does follow the model, the
    /// controller should land near its target rather than merely in the
    /// right direction. This is the accuracy claim, made where it is
    /// honest to make it: a synthetic scene with no content shocks and no
    /// convergence excuse.
    #[test]
    fn a_scene_that_matches_the_model_lands_near_the_target() {
        let (ki, kp) = (K_INTRA, K_INTER);
        for bps in [200_000u32, 600_000, 1_800_000] {
            let mut rc = RateController::new(bps, 30, W, H, 8);
            for i in 0..40 {
                let kind = if i % 8 == 0 { PicKind::Intra } else { PicKind::Inter };
                let qp = rc.pick_qp(kind);
                let b = synth_bits(if kind == PicKind::Intra { ki } else { kp }, qp);
                rc.account(b);
            }
            let ratio = rc.achieved_bps(30) / bps as f64;
            assert!(
                (0.75..=1.35).contains(&ratio),
                "target {bps}: achieved {:.0} bps, ratio {ratio:.2}",
                rc.achieved_bps(30)
            );
        }
    }

    /// The quantiser may not lurch. A controller that jumps from 20 to 45
    /// to meet a budget produces a visible pulse, which is worse to watch
    /// than a steady small miss.
    #[test]
    fn the_quantiser_never_moves_more_than_the_step_limit() {
        let mut rc = RateController::new(600_000, 30, W, H, 8);
        // The limit applies between measured picks of the same kind, so
        // the comparison tracks the previous inter quantiser specifically
        // and starts once two informed inter picks exist.
        let mut prev_inter: Option<u8> = None;
        let mut inter_informed = 0u32;
        // Alternating cheap and ruinously expensive pictures: the budget
        // swings hard, and the step limit is what stops the quantiser from
        // swinging with it.
        for i in 0..30 {
            let kind = if i % 8 == 0 { PicKind::Intra } else { PicKind::Inter };
            let qp = rc.pick_qp(kind);
            if kind == PicKind::Inter {
                inter_informed += 1;
                if inter_informed > 2 {
                    let p = prev_inter.expect("an earlier inter pick");
                    let d = (qp as i32 - p as i32).abs();
                    assert!(d <= MAX_QP_STEP, "picture {i}: quantiser moved {d}, from {p} to {qp}");
                }
                prev_inter = Some(qp);
            }
            rc.account(if i % 2 == 0 { 4000 } else { 20 });
        }
    }

    /// An intra picture gets a larger share than an inter one at the same
    /// complexity — which shows up as a *lower* quantiser being affordable
    /// for it. Without the split, the picture after every keyframe is
    /// starved, and this is the assertion that would fail if
    /// `INTRA_WEIGHT` were quietly dropped to 1.
    #[test]
    fn an_intra_picture_is_given_more_bits_than_an_inter_one() {
        let rc = RateController::new(500_000, 30, W, H, 8);
        let i = rc.target_for(PicKind::Intra);
        let p = rc.target_for(PicKind::Inter);
        assert!(i > p * 2.0, "intra target {i:.0} is not meaningfully above inter {p:.0}");
        // And the split must not invent bits: the GOP's weighted average
        // is still one picture's worth.
        let gop_total = i + p * 7.0;
        let plain = rc.per_picture * 8.0;
        assert!(
            (gop_total - plain).abs() < plain * 0.01,
            "the intra/inter split changed the GOP's total: {gop_total:.0} against {plain:.0}"
        );
    }
}
