//! Picture-level rate control: choosing a quantiser per picture to meet a
//! bitrate target. **Shared by both codecs.**
//!
//! It began as H.265's and moved here whole when H.264 wanted one, which is
//! not what anybody expected — the guess going in was that the budget and
//! the damping would be common and the quantiser-to-bits relationship would
//! not. It is the other way round, and emphatically: H.264 and H.265 define
//! the quantiser *identically*, as a step size that doubles every six, and
//! both decoders in this crate index their dequantisation tables by
//! `qp % 6` and shift by `qp / 6` (`h264::transform`, `hevc::residual`). The
//! one law this module steers by is therefore the most shared thing in it,
//! not the least.
//!
//! What is genuinely per-codec turned out to be only wiring, and little of
//! it: mapping that encoder's picture kind to [`PicKind`], and calling
//! [`RateController::pick_qp`] and [`RateController::account`] in its own
//! loop. Both encoders already carried a per-picture quantiser against a
//! fixed one in the parameter set — `slice_qp_delta` in both — so neither
//! needed new syntax to vary it.
//!
//! The one thing sharing *did* surface: H.264's constant-quantiser path
//! already believed B pictures could afford to be worse, coding them at
//! `qp + 2`. This module had no such belief and lumped them in with P. That
//! was a gap in the H.265 controller rather than a difference between the
//! codecs — H.265 has B pictures too — so [`B_WEIGHT`] fixes it for both.
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
//! output with a constant quantiser and see what goes red. Doing exactly
//! that is what caught the gate's failure tally not listing `RATE-FAIL`:
//! ten cells printed a failure and were counted as passes.
//!
//! ## What the corpus can and cannot ask for
//!
//! A target outside a clip's achievable range cannot be hit by *any*
//! controller, so the gate's targets have to sit inside every clip's range
//! at once. Measured, in bits per second, between quantiser 51 and 0:
//!
//! ```text
//!     clip                        floor      ceiling    range
//!     src_detail_64x64_420       21_900      933_930    42.6x
//!     src_motion_64x64_420       13_440      361_020    26.9x
//!     src_grad_64x64_420          5_910      107_970    18.3x
//!     src_odd_50x34_420          21_280       98_840     4.6x
//! ```
//!
//! The **common** range is only `[21_900, 98_840]` — about 4.5x — because
//! `src_odd_50x34_420` has almost no headroom. That is what caps the gate's
//! bracket at 64k/96k, and it is a property of the corpus rather than of
//! any controller: a tighter band wants a clip with more range, not better
//! code. At 32k the transient alone pushes two clips past 2.0x.
//!
//! ## The error has a shape, and it is not noise
//!
//! Across the gate's sixteen H.265 rate cells the achieved rate averages
//! about 1.17x of target, eleven of sixteen land above it, and every cell
//! is tighter at the higher target than the lower one. That is the seeded
//! first picture of each kind: on a six-to-twelve-frame clip one overspent
//! keyframe is a large share of the whole budget and there is no later to
//! recover it in, and the same absolute overspend matters less as the
//! target rises. A second pass or a lookahead is what fixes it; neither is
//! here, so the bias is reported rather than averaged away.
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
//! - **No guarantee of buffer conformance, only aim.** The controller
//!   knows the coded picture buffer when one is declared and caps each
//!   picture's target at [`CPB_AIM`] of what the buffer can afford, which
//!   keeps a well-behaved stream inside it. It cannot *promise* to: the
//!   size of a picture is not known until it is coded, so the cap is a
//!   target and not a limit. Promising would need the ability to re-code a
//!   picture that came out too large — panic mode — which is structurally
//!   possible now that both picture writers decide in one pass and
//!   serialise in another, and is deliberately not built until a clip
//!   exists that can demonstrate it working. `encode::hrd` is the
//!   instrument that would demonstrate it.
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

/// The largest quantiser change the *first measured* pick of a kind may
/// make, correcting the seed it inherited.
///
/// Wider than [`MAX_QP_STEP`], because that first correction is the most
/// valuable decision the controller makes and throttling it to three steps
/// wastes a short clip — the measured seed error on this project's own
/// corpus was eighteen steps. But not unbounded, which is what it used to
/// be: a single observation of content the model does not describe can
/// then recommend the extreme, and the controller takes it in one move.
/// Sixteen closes the measured eighteen almost entirely in one move and
/// leaves a step or two of ordinary correction. It was twelve first, which
/// stopped the runaway just as well but cost real accuracy in the other
/// direction — cheap content needing a large *downward* correction could
/// not reach its target inside a short clip, and the gate's smooth-ramp
/// clip fell from 0.75x to 0.59x of its target, uncomfortably close to the
/// band's floor. The bound exists to stop one catastrophic move, not to
/// slow every large one.
const MAX_FIRST_STEP: i32 = 16;

/// How much more of the budget an intra picture may take than an inter one.
///
/// An IDR costs several times a P at the same quantiser, so splitting a
/// GOP's bits evenly starves everything after the keyframe — the classic
/// failure where picture two of every GOP is visibly worse than picture
/// one. Four is a round number in the right region rather than a measured
/// constant, and it is the first thing to replace with a measurement.
const INTRA_WEIGHT: f64 = 4.0;

/// How much of an inter picture's share a **B** picture gets.
///
/// Nothing references a B picture here, so spending fewer bits on it costs
/// only itself. Both constant-quantiser paths already encode that belief —
/// H.264's codes B pictures at `qp + 2` — and this is the same statement in
/// budget terms rather than quantiser terms: two quantiser steps is a
/// factor of `2^(-2/6)`, which is 0.79.
///
/// The H.265 controller shipped without it and treated B pictures as P,
/// which was a gap rather than a codec difference; sharing this module with
/// H.264 is what exposed it.
const B_WEIGHT: f64 = 0.79;

/// The most one picture's observation may move the complexity estimate, as
/// a factor either way.
///
/// A single picture cannot legitimately reveal that content is sixteen
/// times cheaper than the last measurement said. When it appears to,
/// something other than complexity has changed — and the loop's response
/// is to lower the quantiser, observe the same bits again, lower it
/// further, and run away.
///
/// That is not hypothetical: wiring this controller to H.264 produced
/// exactly it. Its pictures were falling outside the transform envelope
/// and coding as all-skip at a *fixed* size, so every observation implied
/// a cheaper picture, and the quantiser walked 32, 24, 21, 18, 15, 12, 9
/// while the bits never moved. The envelope was the real bug and is fixed,
/// but a controller that diverges when its model does not apply is a
/// controller with a sharp edge, and content whose cost genuinely ignores
/// the quantiser — a held frame, a black frame — can present the same way.
/// Bounding the step does not make the model right; it makes being wrong
/// survivable.
const MAX_K_RATIO: f64 = 4.0;

/// How little the bits may change, across a quantiser move of at least
/// [`MAX_QP_STEP`], before this module concludes that the content is not
/// responding to the quantiser at all.
///
/// Bounding the complexity step (above) slows a runaway; it cannot stop
/// one, because a picture whose cost never moves keeps implying a cheaper
/// picture forever and the quantiser keeps walking. The only way out is to
/// notice. This is the noticing: move the quantiser meaningfully, see the
/// bits stay inside this band, and conclude that the model does not apply
/// here — then stop lowering the quantiser, because lowering it is what
/// the model recommends and the model is the thing that is wrong.
///
/// It is deliberately a tight band. Content that responds even weakly
/// still gets steered; only content that does not respond at all is
/// frozen, and the flag is recomputed on every observation, so content
/// that starts responding again is steered again.
const INSENSITIVE_BAND: f64 = 0.06;

/// How much of what the buffer can afford a picture is allowed to aim at.
///
/// Not 1.0, because the controller *aims*: it chooses a quantiser from a
/// model and finds out what the picture cost afterwards. Aiming exactly at
/// the limit means missing it half the time, and every miss is a
/// non-conforming stream rather than a slightly-off rate. Three quarters
/// leaves room for the model to be wrong in the direction that matters.
///
/// This buys *aim*, not a guarantee. A guarantee needs the ability to
/// re-code a picture that came out too large — panic mode — which is
/// deliberately not here: see the module header.
const CPB_AIM: f64 = 0.75;

/// How many times a picture may be coded before the encoder gives up on
/// fitting it into the buffer — the first attempt plus this many more.
///
/// Two, because the correction is computed rather than searched: the same
/// law the controller steers by says how many quantiser steps a given
/// overshoot needs, so one re-code should land. The second exists because
/// the law is an approximation and the first correction can undershoot;
/// a third would be chasing a model that more attempts do not improve.
pub const MAX_ATTEMPTS: u32 = 3;

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

/// Which complexity estimate a picture draws on, and what share of the
/// budget it is given.
///
/// Three kinds rather than two because they cost genuinely different
/// amounts at the same quantiser — an intra picture predicts from nothing,
/// and a B picture predicts from both directions — and because nothing
/// references a B picture, so its share can be cut without harming
/// anything else.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PicKind {
    /// An IDR or other intra-coded picture.
    Intra,
    /// A P picture: predicted from the past, and referenced by others.
    Inter,
    /// A B picture: predicted from both directions, and referenced by
    /// nothing, so it is given the smallest share.
    B,
}

impl PicKind {
    /// The share of a plain inter picture's budget this kind is given.
    fn weight(self) -> f64 {
        match self {
            PicKind::Intra => INTRA_WEIGHT,
            PicKind::Inter => 1.0,
            PicKind::B => B_WEIGHT,
        }
    }
}

/// Per-kind complexity: `k` in `bits ≈ k * 2^(-qp/6)`.
#[derive(Clone, Copy)]
struct Complexity {
    k: f64,
    /// The last `(quantiser, bits)` actually observed for this kind, to
    /// tell a picture that got cheaper from one that never cared.
    last_obs: Option<(u8, u64)>,
    /// Set when the last two observations showed a real quantiser move and
    /// essentially no change in bits. See [`INSENSITIVE_BAND`].
    insensitive: bool,
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
    complexity: [Complexity; 3],
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
    last_informed: [Option<u8>; 3],
    /// The last quantiser chosen for each kind whether measured or guessed,
    /// so that even the first measured correction has something to be
    /// bounded against. See [`MAX_FIRST_STEP`].
    last_any: [Option<u8>; 3],
    /// **The ledger.** Total bits emitted, as measured from the access
    /// units themselves — start codes, NAL headers, parameter sets and
    /// all. The encoder asserts this against the bytes it has produced;
    /// see the module documentation for why this is the one thing here
    /// held to equality.
    pub bits_spent: u64,
    /// Pictures accounted for, for the same reason.
    pub pictures: u64,
    /// The declared coded picture buffer and how full it is, when the
    /// stream declares one. Bits, tracked in the same leaky-bucket terms
    /// `encode::hrd` uses to check the result — deliberately the same
    /// arithmetic, so that the controller aiming at a buffer and the
    /// checker measuring one cannot disagree about what the buffer is.
    cpb: Option<(f64, f64)>,
    /// What [`RateController::pick_qp`] chose for the picture currently
    /// being coded, so [`RateController::account`] can pin the model
    /// against the quantiser that actually produced the bits.
    pending: Option<(PicKind, u8)>,
}

impl RateController {
    /// A controller for `bps` bits per second at `fps` pictures per second
    /// over a `width` by `height` picture, with `gop` pictures between IDRs
    /// (0 meaning every picture is one) and `bframes` consecutive B
    /// pictures between references.
    pub fn new(bps: u32, fps: u32, width: u32, height: u32, gop: u32, bframes: u32) -> Self {
        Self::with_cpb(bps, fps, width, height, gop, bframes, None)
    }

    /// [`RateController::new`] against a declared coded picture buffer of
    /// `cpb_bits`, which each picture's target is capped to fit.
    #[allow(clippy::too_many_arguments)]
    pub fn with_cpb(bps: u32, fps: u32, width: u32, height: u32, gop: u32, bframes: u32, cpb_bits: Option<u64>) -> Self {
        let fps = fps.max(1) as f64;
        let per_picture = (bps as f64 / fps).max(1.0);
        // With `gop` pictures per keyframe, one carries INTRA_WEIGHT and
        // the rest split between P and B in the ratio the scheduler will
        // produce: `bframes` B pictures for every anchor. The average is
        // what keeps the weights a *redistribution* — they change which
        // picture gets the bits, never how many there are.
        let n = if gop == 0 { 1.0 } else { gop as f64 };
        let others = (n - 1.0).max(0.0);
        let b_share = bframes as f64 / (bframes as f64 + 1.0);
        let n_b = others * b_share;
        let n_p = others - n_b;
        let avg_weight = (INTRA_WEIGHT + n_p + n_b * B_WEIGHT) / n;

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
        let seed_for = |k: PicKind| per_picture * k.weight() / avg_weight;
        RateController {
            per_picture,
            avg_weight,
            budget: 0.0,
            complexity: [
                Complexity { k: seed_k(seed_for(PicKind::Intra)), observed: false, last_obs: None, insensitive: false },
                Complexity { k: seed_k(seed_for(PicKind::Inter)), observed: false, last_obs: None, insensitive: false },
                Complexity { k: seed_k(seed_for(PicKind::B)), observed: false, last_obs: None, insensitive: false },
            ],
            last_informed: [None; 3],
            last_any: [None; 3],
            // A buffering period begins with the buffer full: that is the
            // initial removal delay the stream declares, so it is the
            // fullness the controller must start from.
            cpb: cpb_bits.map(|c| (c as f64, c as f64)),
            bits_spent: 0,
            pictures: 0,
            pending: None,
        }
    }

    /// The bits this picture is aiming for: its share by kind, plus a
    /// slice of whatever the bucket has over- or under-spent so far.
    fn target_for(&self, kind: PicKind) -> f64 {
        let base = self.per_picture * kind.weight() / self.avg_weight;
        // Spread the correction, and never let it move the target by more
        // than half — a single expensive picture should bend the next few,
        // not flatten them.
        let correction = (self.budget / CORRECTION_PICTURES).clamp(-0.5 * base, 0.5 * base);
        (base + correction).max(16.0)
    }

    /// How many bits the buffer can hand over at the next picture's
    /// removal time, or `None` when no buffer was declared.
    ///
    /// **This is the same arithmetic `encode::hrd` checks with, and
    /// deliberately the same function rather than a second copy of a leaky
    /// bucket.** What makes that checker independent is its *inputs* — it
    /// reads the declared parameters out of the emitted stream and the
    /// sizes off the bytes, where this reads what the encoder believes —
    /// not having two implementations that can drift apart. Two copies of
    /// this formula would be two things to keep in step, and the one that
    /// went stale would be the one nobody ran.
    pub fn affordable_bits(&self) -> Option<u64> {
        self.cpb.map(|(size, fullness)| (fullness + self.per_picture).min(size).max(0.0) as u64)
    }

    /// The quantiser to try next after a picture came out at `actual` bits
    /// when only `affordable` were available.
    ///
    /// The step is computed, not searched: bits halve for every six added
    /// to the quantiser, so the overshoot names its own correction —
    /// `6 * log2(actual / affordable)`, rounded up, plus one step of
    /// margin because the law is an approximation and a re-code that still
    /// does not fit has cost a whole picture for nothing.
    pub fn escalate(qp: u8, actual: u64, affordable: u64) -> u8 {
        if affordable == 0 || actual <= affordable {
            return (qp as i32 + 1).clamp(QP_MIN, QP_MAX) as u8;
        }
        let steps = (6.0 * (actual as f64 / affordable as f64).log2()).ceil() as i32 + 1;
        (qp as i32 + steps.max(1)).clamp(QP_MIN, QP_MAX) as u8
    }

    /// Tell the controller the quantiser the picture was **actually** coded
    /// at, when a re-code moved it away from what [`RateController::pick_qp`]
    /// chose.
    ///
    /// Without this the model would learn from an attempt that was thrown
    /// away — the complexity estimate would be pinned against a quantiser
    /// no picture in the stream was ever coded at, and every later decision
    /// would inherit the error.
    pub fn note_recode(&mut self, qp: u8) {
        if let Some((kind, _)) = self.pending {
            self.pending = Some((kind, qp));
        }
    }

    /// Choose the quantiser for the next picture. Call once per picture,
    /// in coding order, before coding it.
    pub fn pick_qp(&mut self, kind: PicKind) -> u8 {
        let mut target = self.target_for(kind);
        // What the buffer can hand over at this picture's removal time.
        // The rate target says what the picture is *worth*; this says what
        // it can *have*, and the smaller of the two wins.
        if let Some((size, fullness)) = self.cpb {
            let available = (fullness + self.per_picture).min(size);
            // Before this kind has been measured the quantiser comes from
            // a seed, and a seed is routinely wrong by a factor of two —
            // the corpus measured eighteen quantiser steps of error. Being
            // wrong about a rate costs a soft picture; being wrong about a
            // buffer is a stream that does not conform. So an unmeasured
            // picture aims at half of what a measured one would, and the
            // very first picture of a stream — an intra picture against a
            // small buffer, which is the case that actually underflows —
            // is the one that benefits.
            let aim = if self.complexity[kind as usize].observed { CPB_AIM } else { CPB_AIM * 0.5 };
            target = target.min(available * aim).max(16.0);
        }
        let c = self.complexity[kind as usize];
        // Invert bits(qp) = k * 2^(-qp/6).
        let want = 6.0 * (c.k / target).log2();
        let mut qp = want.round().clamp(QP_MIN as f64, QP_MAX as f64) as i32;
        // The step limit stops the quantiser pulsing between *considered*
        // choices, so it applies only between two of them — see
        // `last_informed`. The first informed pick of each kind is allowed
        // to be as large a correction as it needs to be, because the thing
        // it is correcting is a guess.
        // Three regimes, narrowing as the controller learns: no bound at
        // all for the very first picture of a kind, a wide one for its
        // first measured correction, and the ordinary step limit forever
        // after.
        let informed = c.observed;
        match (informed, self.last_informed[kind as usize], self.last_any[kind as usize]) {
            (true, Some(last), _) => {
                let last = last as i32;
                qp = qp.clamp(last - MAX_QP_STEP, last + MAX_QP_STEP);
            }
            (_, None, Some(any)) => {
                let any = any as i32;
                qp = qp.clamp(any - MAX_FIRST_STEP, any + MAX_FIRST_STEP);
            }
            _ => {}
        }
        // Content that does not answer the quantiser cannot be steered by
        // it, and the model's advice — lower it further — is exactly wrong.
        // Hold the line instead of walking to zero.
        if c.insensitive {
            if let Some(last) = self.last_informed[kind as usize] {
                qp = qp.max(last as i32);
            }
        }
        let qp = qp.clamp(QP_MIN, QP_MAX) as u8;
        if informed {
            self.last_informed[kind as usize] = Some(qp);
        }
        self.last_any[kind as usize] = Some(qp);
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
        // The leaky bucket, in the same terms `encode::hrd` will check:
        // bits arrive at the declared rate between removals, the buffer
        // cannot hold more than its size, and this picture is removed
        // whole. A buffer driven below empty is recorded as empty — the
        // stream has already failed at that point and the controller's job
        // is to climb out, not to carry a negative.
        if let Some((size, fullness)) = self.cpb.as_mut() {
            *fullness = (*fullness + self.per_picture).min(*size) - bits as f64;
            if *fullness < 0.0 {
                *fullness = 0.0;
            }
        }
        // Pin the model: one observation determines k exactly, given the
        // quantiser that produced it. A picture that coded to nothing says
        // nothing about complexity, so it is not allowed to zero the
        // estimate.
        if bits > 0 {
            let k_obs = bits as f64 * 2f64.powf(qp as f64 / 6.0);
            let c = &mut self.complexity[kind as usize];
            if c.observed {
                // Bound the excursion before blending: see MAX_K_RATIO.
                let lo = c.k / MAX_K_RATIO;
                let hi = c.k * MAX_K_RATIO;
                c.k = 0.5 * c.k + 0.5 * k_obs.clamp(lo, hi);
            } else {
                // The first real observation replaces the seed outright. A
                // seed is a guess and deserves no weight once a fact
                // exists, and it is not a previous measurement to be
                // bounded against.
                c.k = k_obs;
            }
            // Did the quantiser move, and did the bits care? Recomputed
            // every time, so the verdict follows the content.
            // Only a picture coded at a *different* quantiser carries
            // information about whether the quantiser matters. When one is
            // held — which is exactly what the verdict below causes — the
            // absence of a move is not evidence against it, so the verdict
            // stands until a real move contradicts it. Recomputing it to
            // false on every held picture made the controller alternate
            // between holding and stepping down, walking to zero in pairs.
            match c.last_obs {
                Some((pqp, pbits)) if pbits > 0 && (qp as i32 - pqp as i32).abs() >= MAX_QP_STEP => {
                    let change = (bits as f64 / pbits as f64 - 1.0).abs();
                    c.insensitive = change < INSENSITIVE_BAND;
                    c.last_obs = Some((qp, bits));
                }
                None => c.last_obs = Some((qp, bits)),
                // A held quantiser: keep both the verdict and the reference
                // point it was reached from.
                _ => {}
            }
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
        let mut rc = RateController::new(500_000, 30, 64, 64, 8, 0);
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
            let mut rc = RateController::new(bps, 30, W, H, 8, 0);
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
            let mut rc = RateController::new(bps, 30, W, H, 8, 0);
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

    /// Content whose cost ignores the quantiser must not send the
    /// controller into a spiral.
    ///
    /// This is the H.264 failure reproduced in miniature: a picture that
    /// costs the same however it is quantised. The model has no `k` that
    /// explains it, so every observation implies a cheaper picture than the
    /// last, and without a bound the quantiser walks to zero chasing bits
    /// that were never going to arrive. What is asserted is not that the
    /// controller hits the target — it cannot, and should not pretend to —
    /// but that it fails *quietly*, staying inside a sane band instead of
    /// pinning itself at the extreme.
    #[test]
    fn a_picture_that_ignores_the_quantiser_does_not_send_it_running() {
        let mut rc = RateController::new(2_000_000, 30, W, H, 8, 0);
        let mut qps = Vec::new();
        for i in 0..40 {
            let kind = if i % 8 == 0 { PicKind::Intra } else { PicKind::Inter };
            qps.push(rc.pick_qp(kind));
            // The same size every time, whatever was asked for.
            rc.account(400);
        }
        let lowest = *qps.iter().min().expect("forty pictures");
        assert!(
            lowest > 4,
            "the quantiser ran away to {lowest} chasing bits that do not respond to it: {qps:?}"
        );
    }

    /// The quantiser may not lurch. A controller that jumps from 20 to 45
    /// to meet a budget produces a visible pulse, which is worse to watch
    /// than a steady small miss.
    #[test]
    fn the_quantiser_never_moves_more_than_the_step_limit() {
        let mut rc = RateController::new(600_000, 30, W, H, 8, 0);
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

    /// A B picture is given less than a P picture, and the three weights
    /// still only *redistribute* — a GOP's total is what it would have been
    /// with no weighting at all.
    ///
    /// The second half is the one worth pinning: weights that quietly
    /// changed the total would make every target wrong by a factor nobody
    /// could see, since the band is wide and the error would look like the
    /// transient.
    #[test]
    fn b_pictures_are_given_less_than_p_and_the_weights_only_redistribute() {
        // Two B pictures between anchors, eight pictures to a keyframe.
        let (gop, bframes) = (8u32, 2u32);
        let rc = RateController::new(500_000, 30, W, H, gop, bframes);
        let (i, p, b) = (rc.target_for(PicKind::Intra), rc.target_for(PicKind::Inter), rc.target_for(PicKind::B));
        assert!(b < p, "a B picture ({b:.0}) should be given less than a P ({p:.0})");
        assert!(p < i, "a P picture ({p:.0}) should be given less than an intra ({i:.0})");

        // The GOP as the scheduler will actually shape it.
        let others = (gop - 1) as f64;
        let n_b = others * bframes as f64 / (bframes as f64 + 1.0);
        let n_p = others - n_b;
        let total = i + p * n_p + b * n_b;
        let plain = rc.per_picture * gop as f64;
        assert!(
            (total - plain).abs() < plain * 0.01,
            "the weights changed the GOP's total: {total:.0} against {plain:.0}"
        );
    }

    /// An intra picture gets a larger share than an inter one at the same
    /// complexity — which shows up as a *lower* quantiser being affordable
    /// for it. Without the split, the picture after every keyframe is
    /// starved, and this is the assertion that would fail if
    /// `INTRA_WEIGHT` were quietly dropped to 1.
    #[test]
    fn an_intra_picture_is_given_more_bits_than_an_inter_one() {
        let rc = RateController::new(500_000, 30, W, H, 8, 0);
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
