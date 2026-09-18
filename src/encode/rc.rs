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
//! - **Lookahead only where the encoder offers it.** By itself every
//!   decision here is made from the past only. The H.265 encoder can hold
//!   pictures back and hand this controller a cost per picture — see
//!   [`RateController::pick_qp_ahead`] and the *lookahead* section below —
//!   and with that the controller places bits by what is coming rather
//!   than by what came. H.264 does not drive that path yet.
//! - **Per-CTB adaptation lives beside this, not in it.** The quantiser
//!   this module chooses is picture-wide; the H.265 encoder's adaptive
//!   quantisation (`encode::aq`) redistributes it across coding tree
//!   blocks zero-mean, so the two do not steer the same number.
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
//!
//! # Lookahead: the same law, with a cost in it
//!
//! An encoder that has measured the pictures it is holding can hand this
//! controller a **cost** per picture — the H.265 encoder's is a sum of
//! 8x8 SATDs, intra against each block's own mean and inter against the
//! previous source picture at zero motion (`encode::h265::PicCost`) — and
//! the law grows one factor:
//!
//! ```text
//!     bits(qp) ≈ k * cost * 2^(-qp / 6)
//! ```
//!
//! `k` is then **bits per unit of cost** rather than bits per picture,
//! which is a far more stable number: it is a property of how the codec
//! prices a unit of residual energy, not of the content, so one
//! observation of it transfers to the next picture whatever that picture
//! shows. Three things follow, each of them a mechanism the past-only
//! controller lacks:
//!
//! - **The share is measured, not assumed.** Without lookahead an intra
//!   picture is given [`INTRA_WEIGHT`] times an inter picture's bits, a
//!   round number. With it, each picture's share of the window's budget is
//!   its predicted bits at a common quantiser over the window's mean —
//!   `k(kind) * cost`, so a keyframe of flat content is not overpaid and a
//!   scene cut's first inter picture, whose cost is a keyframe's, is not
//!   starved. The ratio is bounded to `[1 / MAX_K_RATIO, MAX_K_RATIO]`
//!   so that a single wild picture cannot take the whole window.
//! - **The model sees the change before coding it.** `k * cost` predicts
//!   a picture's bits from its own samples, so the quantiser moves *with*
//!   the content rather than one picture after it — and the step limit,
//!   which exists to stop pulsing between similar pictures, is widened by
//!   exactly the measured cost ratio (`6 * log2(cost / last cost)`) so a
//!   measured change is not throttled as if it were noise.
//! - **A kind that has not been measured borrows from one that has.**
//!   The first P picture of a stream no longer starts from the bits-per-
//!   pixel seed: it takes the keyframe's observed `k` and its own cost.
//!   Before any observation at all the seed is [`SEED_BITS_PER_COST`],
//!   a calibration, clamped through the same [`SEED_QP_MIN`] and
//!   [`SEED_QP_MAX`] a past-only seed is. A picture planned from a seed
//!   alone that misses by far is coded again once (*The seeded first
//!   pictures* below).
//!
//! What lookahead does **not** change: the ledger, the bucket, the buffer
//! cap, and every check above. A cost of one and an empty window is the
//! past-only controller exactly, which is what [`RateController::pick_qp`]
//! passes — so a stream that does not ask for lookahead is coded by the
//! same arithmetic it always was.
//!
//! ## Fade pictures in lookahead ABR (the cap implemented, the rest measured)
//!
//! On a fade the lookahead controller ends over its target. The gate's
//! fade clip (`src_fade`, a luma gain fade) ended at 1.083x at 64 kbps,
//! 1.030x at 96 kbps and 1.109x with two B pictures at 64 kbps. The two
//! gain-and-offset fades, which no lookahead row of the gate visits, end
//! at 1.280x (`src_wpoff`, 8 bits) and 1.248x (`src_fdeep10`) at 96 kbps,
//! their P pictures at 1.15x and 1.18x of plan in total. Two mechanisms
//! were measured:
//!
//! - **The intra cap is fixed here.** It stays on a picture whose change
//!   is more than a brightness step, and it never applies under
//!   `--wpred` (`encode::h265::PicCost::inter_cost`).
//! - **The first mechanism is not.** A model fix for it was built and
//!   tried in 23 configurations, and none met the bar. This is the record
//!   of why.
//!
//! Measured 2026-09-14 on h26x 5dde3bb, whose encoder source is develop's
//! up to this change, with the model on the branch `agent/rcfix3`
//! (ed5ed4d); rechecked, and the cap measured and implemented, 2026-09-18.
//!
//! **The cost does not see what a fade's bits pay for.** The inter cost
//! is mostly the fade. An SATD over 4x4 Hadamard tiles prices a one-level
//! shift of an 8x8 block at 32 units: each tile's DC coefficient takes 16
//! levels and the tile sum is halved to 8. On the fade clip the mean moves
//! about 420 levels a picture summed over its blocks — 13.4k units, every
//! picture — against 3k to 6k for everything else the picture does. The
//! codec codes a brightness step cheaply. What costs bits is the rest, and
//! the three fades built on `testsrc2` (fade, wpoff, fdeep10) change a
//! small element on odd pictures. At a fixed quantiser those pictures cost
//! about twice the even ones, while the cost moves by an eighth:
//!
//! ```text
//!     src_fade at quantiser 20, no rate control   display 4   display 5
//!     bits                                            1408        3160   2.24x
//!     inter cost (zero-motion SATD)                  15252       17104   1.12x
//!       of which not the mean shift                   2991        5411   1.81x
//! ```
//!
//! `k` is blended half and half with the last observation, so every P
//! picture is planned from a picture of the other parity and is about
//! twice wrong, alternately each way. At 96 kbps with `--lookahead 8`
//! those two pictures code at 20 and 18 for 1480 and 4184 bits, 0.55 and
//! 1.43 of their plan, and the clip's P pictures range from 0.45 to 1.50
//! of plan. On the gain fade the misses nearly cancel (0.95x of plan over
//! all its P pictures at 96 kbps). On the gain-and-offset fades they do
//! not. The native 10-bit fade (`src_wsine10`) has no such element and
//! does not alternate.
//!
//! This was first read as skip-lag: a coarse quantiser rounding the step
//! away and the next picture coding two steps. The reconstruction rules
//! that out. No 8x8 block of any P picture of that run is identical to the
//! same block of the picture before it. The reconstructed luma mean tracks
//! the source to within 0.11 levels on every picture. And the alternation
//! is there at a fixed quantiser, with no controller to lag.
//!
//! **The intra cap changes units at the end of a fade.** An inter picture
//! was planned at `min(intra, max(inter, floor))`. The intra cost is taken
//! against each block's own mean, so it has no DC at all, while the inter
//! cost carries the whole brightness step. As a fade reaches black the
//! texture goes, the intra cost falls under the inter cost, and the plan
//! falls with it while the bits do not:
//!
//! ```text
//!     src_wpoff at 96 kbps, last GOP      display 9    10      11
//!     intra cost                              16000    11196    6000
//!     inter cost                              21665    19740   19508
//!     plan (bits)                              4206     3504    2415
//!     spent                                    5136     4176    7688   3.18x plan at display 11
//!     at quantiser 13, no rate control         4760     3488    6600
//! ```
//!
//! `src_fdeep10`'s display 11 is the same story: intra 7920 against inter
//! 21883, and 2.87x its plan. The cap matters less than the first mechanism.
//! Over every lookahead cell it binds on only 10 distinct pictures: the
//! last one to three of each fade, and the cut clip's cut, where it is
//! right. See *What is implemented* below.
//!
//! **What predicts a fade picture's bits.** For each candidate cost, the
//! spread of `log2(bits * 2^(qp / 6) / cost)` within a cell — how far one
//! `k` misses across a clip — pooled over 447 P pictures of the 28
//! lookahead cells without B pictures (the four fades against the rest).
//! "Fade bias" is the fades' mean `log2 k` less everyone else's. The
//! SATD is the encoder's own tiled kernel, reproduced offline to zero
//! mismatches against its costs. An earlier fit with a plain 8x8 Hadamard
//! gave 0.33 for the reference cost, and that number was wrong.
//!
//! ```text
//!     cost                                         fade    other   fade bias
//!     today: min(intra, max(inter, floor))        0.586    0.518     -1.902
//!     SATD against the previous source            0.447    1.078     -3.886
//!     SATD against the reconstructed reference    0.431    0.522     -1.146
//!       each block's mean difference removed      0.340    0.518     +0.121
//!       removed, then 8 per level added back      0.374    0.519     -0.361
//! ```
//!
//! The mean-removed difference against the reference is the best
//! predictor for fades and no worse for anything else. That is the first
//! mechanism in numbers: the bits follow the part of the difference that
//! is not the fade.
//!
//! **Telling a fade apart.** A picture is fade-like when its mean shift,
//! priced at the SATD's own 32 per level, is at least everything else:
//! `dc > 0 && 32 * dc >= inter_ac`. Here `inter_ac` is the inter SATD with
//! each block's mean difference removed and `dc` is the summed absolute
//! mean difference. Over the distinct inter pictures of every lookahead
//! cell:
//!
//! ```text
//!     32 * dc / inter_ac    pictures    min     p50     p90     max
//!     the four fades             40    1.341   2.630   4.493   6.548
//!     everything else           172    0.036   0.094   0.197   0.760   (20 with dc = 0)
//! ```
//!
//! A threshold of 1.0 sits between the cut clip's cut (0.760) and
//! wsine10's first P picture (1.341). Outside the corpus, at 1280x720:
//!
//! - A cinema trailer, 1525 pictures: 48 flagged in 8 runs. Each run is a
//!   real fade or grading pulse (a logo fading in, clouds fading up, a
//!   lighting pulse, a fade out).
//! - A UHD stock clip, 207 pictures: none flagged, maximum 0.366.
//! - A 1080p clip, 750 pictures: none flagged, maximum 0.137.
//!
//! The detector is the part that works.
//!
//! **What was tried.** The fix in `agent/rcfix3` used the detector:
//!
//! - A P picture it flagged, outside `--wpred`, had its quantiser inverted
//!   from a *model* cost, the SATD of its luma against the reconstructed
//!   reference it predicts from.
//! - That model cost had its own bits-per-cost estimate, while the window
//!   shares kept the source costs.
//! - A picture not flagged was untouched, and in every configuration no
//!   byte of any non-fade rate cell moved.
//!
//! The bar, for the five fade lookahead cells the gate then visited
//! (wsine10 by the 10-bit row; since a35a614 only `@wsine10` rows visit
//! it, and none of them is a lookahead row):
//!
//! - **(a)** Non-fade cells byte-identical.
//! - **(b)** Mean `|ratio - 1|` below 0.0474 over the five cells (mean5),
//!   or below 0.0585 over the four that are not wsine10 (mean4). None of
//!   those four more than 0.01 further from target.
//! - **(c)** No plane of a moved cell more than 0.30 dB under the quality
//!   floor.
//! - **(d)** No cell both larger and worse.
//!
//! Every configuration, by the names the round files use, scored against
//! that bar:
//!
//! ```text
//!     round  configuration            mean5   mean4  >0.01  la-ipb dCb   dCr   fails
//!            base (5dde3bb)          0.0474  0.0585
//!       1    model (ed5ed4d)         0.0586  0.0570     0      -0.60   -0.56   c d
//!       1    ac                      0.0524  0.0628     2      -0.36   -0.32   b c d
//!       1    damp=0.25               0.0646  0.0470     0      -0.46   -0.46   c
//!       1    ac,damp=0.25            0.0588  0.0595     1      -0.36   -0.32   b c d
//!       1    anchor=2                0.0578  0.0570     0      -0.60   -0.56   c d
//!       1    ac,anchor=2             0.0506  0.0628     2      -0.36   -0.32   b c d
//!       1    anchor=1                0.0622  0.0648     2      -0.46   -0.46   b c d
//!       1    ac,damp=0.25,anchor=1   0.0596  0.0610     1      -0.36   -0.32   b c d
//!       1    step=1                  0.1672  0.0467     1      -0.72   -0.70   b c
//!       1    ac,step=1,anchor=1      0.1710  0.0532     1      -0.57   -0.53   b c
//!       2    ac,rise=1               0.0504  0.0628     2      -0.36   -0.32   b c d
//!       2    ac,rise=2               0.0506  0.0628     2      -0.36   -0.32   b c d
//!       2    ac,rise=3               0.0524  0.0628     2      -0.36   -0.32   b c d
//!       2    ac,dcw=8,rise=2         0.0532  0.0585     1      -0.48   -0.40   b c
//!       2    ac,dcw=16,rise=2        0.0582  0.0648     2      -0.46   -0.46   b c
//!       2    rise=2                  0.0592  0.0588     0      -0.46   -0.46   b c d
//!       3    tail                    0.0528  0.0655     2      -0.40   -0.40   b c d
//!       3    ac,tail                 0.0556  0.0693     2      -0.40   -0.40   b c
//!       3    ac,rise=2,tail          0.0556  0.0693     2      -0.40   -0.40   b c
//!       3    damp=0.25,tail          0.0624  0.0613     1      -0.40   -0.40   b c d
//!       3    ac,damp=0.25,tail       0.0520  0.0645     2      -0.40   -0.40   b c
//!       3    ac,dcw=8,tail           0.0572  0.0665     2      -0.40   -0.40   b c
//! ```
//!
//! - `model`: the committed fix.
//! - `ac`: its cost with each block's mean difference removed.
//! - `dcw=w`: `ac` plus `w` per level of mean difference.
//! - `damp=a`: the model's `k` moves `a` of the way to each observation,
//!   not half.
//! - `anchor=d`: the quantiser at most `d` above the last reference
//!   picture's.
//! - `rise=d`: at most `d` above the last fade-planned P's.
//! - `step=s`: within `s` of it. The `step=1` rows put wsine10 at 0.351x
//!   and 0.358x, out of the band.
//! - `tail`: the window's last P planned by today's cost.
//!
//! Round 2 ran `ac` again as its control, and it scored the same.
//!
//! Before the model, four families were scored on rate alone:
//!
//! - **The inter cost as `inter_ac + w * dc` for every picture**
//!   (w 0 to 16). Moved 11 to 19 of the 32 non-fade lookahead cells, 7 to
//!   10 of them further from target. A cost change for everyone is not a
//!   fade fix.
//! - **The reference cost for every P picture, ungated.** The gate's mean
//!   `|ratio - 1|` went from 0.0782 to 0.0764, but 29 non-fade cells
//!   moved and 15 of them went further.
//! - **The cap raised to `intra + w * dc`.** Moved the four cut-clip
//!   cells, whose cut is capped. It did the most for the gain-and-offset
//!   fades (1.082x and 1.084x at w = 32) and pushed wsine10 to 0.888.
//! - **The cap dropped where `inter_ac + w * dc <= intra`.** This is
//!   what is implemented (below). As first measured it applied under
//!   `--wpred` too, and there it failed (b) on one cell, the weighted
//!   la-ipb row: 0.988 to 0.975, 0.013 further from target.
//!
//! **What is implemented.** The cap is dropped on a P or B picture where
//! `inter_ac + 16 * dc <= intra`: the change, its brightness step priced
//! at half the SATD's own 32 per level, still fits under the intra cost.
//! It is never dropped under `--wpred`, where the encoder takes the step
//! out itself with the weights. Scored on all 128 rate cells against the
//! same bar and base (`D:/rivet-rcfix4/capwp/capwp.txt`), it passes:
//!
//! ```text
//!     cell                                  before   after   bytes        dY     dCb    dCr
//!     src_fade hevc-abr-la-64k               1.083   1.070   3464 -> 3424  -0.02  -0.03  -0.02
//!     src_fade hevc-abr-la-96k               1.030   1.019   4942 -> 4889  -0.04  -0.02  -0.02
//!     src_fade hevc-abr-la-ipb-64k           1.109   1.102   3549 -> 3526  -0.04  -0.22  -0.11
//!     src_wsine10 hevc10-abr-la-96k          1.003   0.995   4815 -> 4776  +0.00  -0.01  -0.01
//!     every other rate cell, the weighted la-ipb row included: byte-identical
//!     mean5 0.0474 -> 0.0416, mean4 0.0585 -> 0.0508, none further, floor held
//! ```
//!
//! Every moved cell spends less, so every one loses a little PSNR, 0.02
//! to 0.22 dB. On the gain-and-offset fades, which no gate row visits,
//! fdeep10 goes from 1.248x to 1.229x and wpoff from 1.280x to 1.289x.
//! On wpoff the rule uncaps only display 9, which then spends 0.96 of
//! its plan instead of 1.22. Displays 10 and 11 are darker, their intra
//! cost has fallen under even `inter_ac + 16 * dc`, and they keep the
//! cap. Display 11 still spends 3.23x its plan (`latrace_b16`).
//!
//! The price was tried at 8 and 16. At 8 the same four gate cells moved
//! the same way except wsine10, which fell to 0.888: its capped end
//! pictures had been cancelling an under-spent keyframe, and a cheaper
//! price uncaps more of them.
//!
//! **Why la-ipb is the blocker.** The fade is luma only, so its chroma is
//! still. Chroma quality is set by the anchors — the IDR and P pictures —
//! and every B picture inherits it. The model moves spend inside the
//! first GOP: P 6 goes 2632 to 2376 bits, and B 5 goes 904 to 1496. The
//! IDR at display 8 then gets 3912 bits instead of 4152, and the second
//! GOP loses chroma in every variant: display 9 Cb -1.79 to -2.23. The
//! committed model also prices the last P (display 11, nearly black,
//! coded before displays 9 and 10) from its reference and raises it from
//! quantiser 24 to 29. It spends 3408 bits instead of 4608, as planned,
//! but its Cb falls from 41.19 to 36.72 dB (-4.47), display 10 loses 1.30
//! and display 6 loses 1.02. Exempting that picture (round 3) left -0.40
//! on both chroma planes, the same in all six variants. The IDR still
//! lost its bits, and the pictures after it still lost chroma (display 9
//! Cb -2.15, display 11 -2.03). A fade P picture is cheap in its own bits
//! and expensive in its dependants', and a per-picture model cannot see
//! the second part.
//!
//! **What a fix would need.**
//!
//! 1. **An anchor-aware quantiser.** An anchor's cost to the clip has to
//!    include the pictures predicted from it: a propagation of each
//!    picture's inter cost back to its references, as x264's MB-tree
//!    does. The lookahead already holds the window and the picture types
//!    to do it. Without that, anything that takes spend off a fade anchor
//!    fails la-ipb's chroma.
//! 2. **A `k` that is not fooled by alternation.** A half-and-half blend
//!    with the last picture is twice wrong on every picture of content that
//!    alternates. Planning fades from the mean-removed reference cost
//!    narrowed the swing: at la-96k, displays 5 and 6 coded at 3672 and
//!    1768 bits against the base's 4152 and 1152. It still did not beat the
//!    base ratio (1.044 against 1.030). Damping the model's `k` to 0.25 met
//!    the rate conditions and failed only the floor.
//! 3. **Gate rows where the overspend is.** The gain fade's P pictures
//!    are already near plan in total. The overspend is on wpoff and
//!    fdeep10 (1.280x, 1.248x), which no lookahead row visits. A fix
//!    should arrive with `@wpoff` and `@fdeep10` lookahead rows and a floor
//!    recorded for them.
//!
//! wsine10's 1.003 was two errors cancelling: keyframes at 0.25 and 0.24
//! of plan against capped end pictures overspending (display 11 at
//! 2.19x). Anything that fixes the end pictures moves it off 1.0 until the
//! keyframe seed is fixed. *The seeded first pictures* below fixes the
//! first keyframe and takes wsine10 to 1.101x.
//!
//! **Where the evidence is.** The measurements are on the machine they
//! were taken on, under `D:/rivet-rcfix3` and `D:/rivet-rcfix4`:
//!
//! - The alternation and the skip-lag refutation: `cqp/` (bits at a fixed
//!   quantiser) and `skiplag_fade96.out`. The zero copied blocks and the
//!   0.11-level mean came from the same reconstruction, `recon/`.
//! - The traces: `ftrace/*.log` and `latrace_ctl2/`.
//! - The fit: `refcost_recon.out`, from `refcost.py`.
//! - The detector: `detector_margin.out` and `detector_natural2.log`.
//! - The rounds: `round1.txt` to `round3.txt`, rescored in `rescore.txt`
//!   and `rescore_table.out`.
//! - The earlier families: `cmp_prerounds.out`.
//! - The cap: `capb/capb.txt` (under `--wpred`), `capwp/capwp.txt` (as
//!   implemented) and `latrace_b16/` (its per-picture trace).
//! - la-ipb: `ipb_psnr.out` and `ipb_variants_psnr.out`.
//!
//! ## The seeded first pictures, coded again when they miss
//!
//! Under lookahead the stream's first picture is planned from
//! [`SEED_BITS_PER_COST`] and nothing else. That calibration was taken
//! before the coding quadtree became the default, and it now misses:
//!
//! - **The gate's lookahead cells.** Over its 36 lookahead cells, first
//!   keyframes spent a median 0.82 of plan: 21 under 0.84, 15 under 0.71,
//!   and 2 over 1.19. The observed `k` had median 3.32 (quartiles 2.55 and
//!   4.02) against the seed's 4.2. Later keyframes, planned from a
//!   measurement, spent a median 0.97, so only the seeded pick misses.
//! - **The grad clip.** Its keyframes sat on [`SEED_QP_MIN`] at 0.11 to
//!   0.20 of plan.
//! - **A 720p natural clip** at 2 Mb/s. The seed asked for more than
//!   [`SEED_QP_MAX`], the keyframe was coded at 45 on 0.28 of plan (28.5 dB
//!   luma), and the first P spent 5.07 times its plan repairing it.
//!
//! **What is implemented** ([`RateController::seed_recode`]):
//!
//! - **The first picture.** A picture planned from a seed alone that lands
//!   more than [`SEED_RECODE_STEPS`] from its plan is coded again once, at
//!   the quantiser its own bits imply. That quantiser may be below the
//!   seed's floor, because it is a measurement, and it is bounded by
//!   [`MAX_FIRST_STEP`].
//! - **The first P after it.** When the keyframe was coded again, the
//!   first P borrows a measured `k` rather than the calibration, so the
//!   seed's clamp no longer holds it. It may be coded again once the same
//!   way.
//! - **Nothing else.** At most two extra codings per stream, and none
//!   without a lookahead.
//!
//! Scored on the 125 rate cells of develop f74b437 against its own
//! binary (`D:/rivet-rcfix4/kfclean/clean.txt`), the mean `|ratio - 1|`
//! over the 36 lookahead cells goes from 0.0795 to 0.0523:
//!
//! - **What moves.** 15 cells move, all with luma up 0.65 to 10.37 dB.
//!   13 end closer to target, and the two that end further are below. No
//!   plane falls more than 0.30 dB, nothing is larger and worse, and no
//!   cell outside the lookahead rows moves.
//! - **The fade clip at 96 kbps: 1.019 to 1.077**, with luma up 2.14 dB.
//!   Its P pictures alternate as the fade section above records, now
//!   0.53 to 1.57 of plan against 0.46 to 1.47. Over the clip the P
//!   pictures spend 1.06 of their plan where they spent 0.95, because
//!   the under-spent keyframe that absorbed the misses no longer does.
//! - **The odd clip at 96 kbps: 0.904 to 0.870**, but on 2089 bytes
//!   against 2169 and with luma up 10.37 dB. It is strictly better.
//! - **The native 10-bit fade** (no gate lookahead row) goes from 0.995 to
//!   1.101: the two errors above, with the keyframe's now gone.
//! - **The 720p natural clip** codes its keyframe again at 34 (35.8 dB,
//!   0.89 of plan) and ends at 1.009x with luma up 0.51 dB. Its plan error
//!   goes from 6.02 to 5.29 steps, and its worst picture from 28.5 to 35.5
//!   dB.
//!
//!   Its first P misses the other way. At 31 it spent 1.78 times its plan;
//!   coded again at 36 it spent 0.11. Near its skip threshold a P picture's
//!   bits fall far faster than six steps per doubling, and one re-code by
//!   the law cannot see that.
//!
//! **What it costs.** Two re-codes on a 2-frame 720p stream take 0.30 s,
//! about one extra keyframe. That is 14.5% of a 12-frame stream (2.05 s),
//! 1.3% of 120 frames (23.7 s) and 0.28% of 600 frames (108.6 s): best of
//! 3 to 9 on a loaded machine. A caller that builds an encoder per segment
//! pays it per segment.
//!
//! **What was tried and not kept** (36 cells, mean 0.0803 on the base
//! then; `D:/rivet-rcfix4/kf`, `kf2`):
//!
//! ```text
//!     variant                                            mean    >0.01 further   planes < -0.30
//!     re-code the keyframe at 3 steps, nothing else     0.0824        6                0
//!     the same at 1.5 steps                              0.0846        9                3
//!     recalibrate the seed to 3.3                        0.0971       19                0
//!     recalibrate and re-code at 3                       0.0973       22                0
//!     re-code, first P unclamped (not re-coded)          0.0712        4                0
//!     re-code the first picture of every kind            0.0748        9               12
//!     a second, secant re-code of the keyframe           0.0820        8                0
//!     first P seeded at 2.7 x the keyframe's k           0.0650        3                0
//!     implemented: re-code, first P unclamped and
//!       re-coded once                                    0.0531        2                0
//! ```
//!
//! - **The recalibrated seed** makes every first keyframe spend more. That
//!   uncovers the overspend the under-spent keyframes were hiding, all at
//!   once, and it moves detail10-444 the wrong way: that keyframe was
//!   already over its plan, and the cell goes from 1.098 to 1.117.
//! - **The unclamped first P without its own re-code** borrows an intra
//!   `k` that is 2 to 4 times too low for a P at a low quantiser. It spent
//!   3.35 to 4.45 times its plan on the odd and static clips.
//! - **The secant re-code** lands keyframes near plan: 0.84 to 1.04, and
//!   grad 0.42 to 0.73 at the step bound. A single re-code by the law
//!   lands short (0.29 to 0.82), because six steps per doubling is too
//!   steep below about 30. But landing on plan only makes the clips
//!   overshoot more.

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

/// How far the bits may *fall*, on a picture coded at a lower quantiser
/// than the walk began at, and still count as not having answered it.
///
/// Bounding the complexity step (above) slows a runaway; it cannot stop
/// one, because a picture whose cost never moves keeps implying a cheaper
/// picture forever and the quantiser keeps walking. The only way out is to
/// notice, and the noticing starts with a **walk**: the quantiser lowered,
/// picture after picture, while the bits stay where they were. Only a
/// lowered quantiser starts one, because the runaway only goes down and
/// the verdict only stops lowering. Bits are compared per unit of
/// lookahead cost, so a picture the lookahead measured as cheaper is not
/// read as one that ignored the quantiser.
///
/// Bits that fall while the quantiser falls are not a picture ignoring
/// the quantiser; they are a picture that got cheaper — references that
/// improved, a scene settling after a cut. So a fall beyond this band
/// restarts the walk instead of adding to it. The band itself is noise:
/// content that truly ignores the quantiser codes to the same size
/// within it.
///
/// This band was once the whole rule, applied to one move of three steps
/// in either direction, and it froze the quantiser for the rest of a clip
/// on a single observation. H.265 on the cut clip at 64 kbps with a
/// depth-0 quadtree: quantiser 41 to 38 while the bits fell 168 to 160 —
/// a picture right after the cut, cheaper because its reference had just
/// been coded — and all 37 inter pictures after it were coded at 38 on a
/// third of their plan, 0.86x for the clip. H.264 at 128 kbps: 28 to 32
/// with the bits up 3.6 %, a *raised* quantiser, and 62 inter pictures at
/// 32 on under half their plan. Ten of the gate's 18 cut-clip rate cells
/// ended between 0.81x and 0.95x that way. See [`INSENSITIVE_SPAN`] and
/// [`INSENSITIVE_RETRY`] for what replaced it.
const INSENSITIVE_BAND: f64 = 0.06;

/// How much of the law's predicted response counts as the bits answering
/// a quantiser move: a quarter of it, in the log, in either direction.
///
/// The law says `d` steps down multiplies the bits by `2^(d/6)`, and `d`
/// steps up divides them by it. Content that answers even weakly — a
/// quarter of that, nine percent for three steps — is being steered and
/// is left alone. The fraction is deliberately small, because what this
/// guards against is content that does not answer *at all*, and a
/// controller that wrongly concludes that stops spending.
const INSENSITIVE_RESPONSE: f64 = 0.25;

/// How far a walk must have lowered the quantiser before its silence is
/// worth asking about: six steps, a predicted doubling of the bits, across
/// at least two pictures — or twice that in one.
///
/// Two pictures because one is an anecdote: the ordinary step limit is
/// [`MAX_QP_STEP`], a predicted 41 %, and consecutive pictures of real
/// content routinely differ by that much on their own. A single move of
/// twelve steps or more — a first measured correction ([`MAX_FIRST_STEP`])
/// or a lookahead-widened one — predicts four times the bits, which no
/// picture-to-picture noise hides, and waiting a second picture there
/// would be twelve more steps of runaway.
///
/// **A silent walk is still not a verdict.** References that improve by
/// exactly the law's slope per picture — each lowered picture predicting
/// better from the one before — hold the bits flat all the way down, and
/// no amount of walking tells that apart from content that ignores the
/// quantiser. Asking from the other side does. So the next picture of the
/// kind is coded [`MAX_QP_STEP`] *above* the walk's bottom: content that
/// ignores the quantiser ignores a raised one too, and its bits stay put,
/// while improving references and a raised quantiser push the bits the
/// same way and they fall by more than the law alone. Only a walk that
/// stayed silent *and* a raised picture that did not answer
/// ([`INSENSITIVE_RESPONSE`]) is a verdict. The question costs one picture
/// three steps coarser, and only after a silent walk, which real content
/// rarely produces.
const INSENSITIVE_SPAN: i32 = 6;

/// How many picks a verdict holds the quantiser at its floor before one
/// is allowed [`MAX_QP_STEP`] below it, to ask the content again.
///
/// A verdict that nothing can overturn is a freeze, and the one this
/// replaced was exactly that: holding the quantiser means no move, no
/// move means no evidence, and the verdict stood for the rest of the clip
/// whatever the content did. Two things re-open it. The bits moving
/// further from the verdict's than [`MAX_QP_STEP`]'s worth of the law, at
/// any quantiser, which is the content changing. And this: every
/// `INSENSITIVE_RETRY + 1`th pick under a verdict probes one step limit
/// below the floor, and a probe that answers releases it. A probe that
/// does not leaves the floor where it was, not where the probe went —
/// content that truly ignores the quantiser costs one picture in nine
/// coded three steps finer, at no cost in bits, and never walks.
const INSENSITIVE_RETRY: u32 = 8;

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

/// The seed for `k` when a lookahead cost is available and nothing has
/// been observed yet: bits per unit of cost at quantiser 0, before the
/// `2^(-qp/6)` factor.
///
/// A calibration, taken on this project's corpus (2026-09-13, the four
/// `--bitrate` rows over every clip under `--lookahead 8`, 75 keyframes,
/// cost measured as `encode::h265::PicCost` measures it): the median of
/// `bits * 2^(qp/6) / cost` was 4.23 at the seed's own operating point
/// (quartiles 3.64 and 4.79, extremes 1.16 and 7.20). A first pass with a
/// placeholder of 40 — eight times too high — put every keyframe at the
/// clamp's ceiling and measured 4.76 there; the law is approximate
/// enough over twenty quantiser steps that the number at the operating
/// point is the one to keep. Inter pictures came out at a median of 3.55
/// over 417, the same order, which is what lets a keyframe's observed
/// value stand in for the first P. It replaces the bits-per-pixel anchor
/// only for the very first picture of a stream; every picture after that
/// is pinned by an observation, and the [`SEED_QP_MIN`] / [`SEED_QP_MAX`]
/// clamp bounds how wrong this constant is allowed to be — it is what
/// kept the placeholder from being worse than a seed.
const SEED_BITS_PER_COST: f64 = 4.2;

/// How far, in quantiser steps of the law, a picture planned from a seed
/// alone may land from its plan before it is coded again — once, at the
/// quantiser its own bits ask for ([`RateController::seed_recode`]).
///
/// Two pictures are planned that way under lookahead. The stream's first,
/// from [`SEED_BITS_PER_COST`], and the first P after it, from the
/// keyframe's bits per cost. The seed is a calibration taken before the
/// coding quadtree became the default. On the lookahead cells of the gate
/// it now misses the first keyframe by a median 0.82 of plan: 21 of 36 are
/// under 0.84 and the grad clip sits at [`SEED_QP_MIN`] on 0.11 of its
/// plan. On a 720p natural clip at 2 Mb/s it asks for more than
/// [`SEED_QP_MAX`] and codes the keyframe at 45 on 0.28 of plan. The first
/// P then spends five times its plan repairing it. See the module docs,
/// *The seeded first pictures*, for the measurement.
///
/// Three steps, a factor of 1.41 either way. At 1.5 the re-code moved 23
/// of the 36 cells, not 15, and three planes lost more than 0.30 dB.
const SEED_RECODE_STEPS: f64 = 3.0;

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
    /// The last observation of this kind: its quantiser and
    /// `log2(bits / cost)` — bits per unit of lookahead cost, so a
    /// picture that measured cheaper is not mistaken for one that ignored
    /// the quantiser. The cost is one without a lookahead.
    last_obs: Option<(u8, f64)>,
    /// Whether this kind's bits answer the quantiser, as far as the
    /// observations so far can tell. See [`INSENSITIVE_BAND`].
    response: Response,
    /// Whether `k` has been pinned by a real observation yet, or is still
    /// the seed. The first real observation replaces the seed outright
    /// rather than being blended into it — a seed is a guess and deserves
    /// no weight once a fact exists.
    observed: bool,
}

/// What the observations of one kind say about whether its bits answer the
/// quantiser. Quantisers are the ones pictures were coded at; `y` is
/// `log2(bits / cost)`, the cost being one without a lookahead.
#[derive(Clone, Copy, Debug)]
enum Response {
    /// Nothing observed yet.
    Unknown,
    /// Pictures since the anchor `(qp, y)` have been coded at quantisers no
    /// higher than the one before, and `moves` of them were lowered without
    /// the bits answering. See [`INSENSITIVE_BAND`].
    Walk { qp: u8, y: f64, moves: u32 },
    /// A walk went silent over [`INSENSITIVE_SPAN`], bottoming out at
    /// `(floor, y)`; the next pick is raised a step limit above the floor
    /// to ask whether a raised quantiser is ignored too.
    Confirm { floor: u8, y: f64 },
    /// The verdict: the bits ignore the quantiser. Picks are held at or
    /// above `floor` — fixed, so the verdict neither ratchets up behind a
    /// raised quantiser nor walks down behind a probe — and `held` counts
    /// them since the verdict or the last probe ([`INSENSITIVE_RETRY`]).
    /// `y` is the bits when it was reached, to notice the content change.
    Insensitive { floor: u8, y: f64, held: u32 },
}

impl Complexity {
    fn new(k: f64) -> Self {
        Complexity { k, last_obs: None, response: Response::Unknown, observed: false }
    }

    /// Fold one observation — the quantiser `qp` a picture was coded at and
    /// `y = log2(bits / cost)` — into [`Response`].
    fn observe_response(&mut self, qp: u8, y: f64) {
        let last = self.last_obs.replace((qp, y));
        let restart = Response::Walk { qp, y, moves: 0 };
        // The law's predicted change in `y` for `steps` quantiser steps
        // down, and the quarter of it that counts as an answer.
        let answer = |steps: i32| INSENSITIVE_RESPONSE * f64::from(steps) / 6.0;
        let band = (1.0 - INSENSITIVE_BAND).log2();
        self.response = match (self.response, last) {
            (Response::Insensitive { floor, y: vy, held }, Some((lqp, ly))) => {
                let answered = if qp < floor && qp < lqp {
                    // A probe below the floor, against the picture before it.
                    y - ly >= answer(i32::from(lqp - qp))
                } else {
                    // Anywhere else: has the content moved further than the
                    // law's response to one step limit?
                    (y - vy).abs() > f64::from(MAX_QP_STEP) / 6.0
                };
                if answered { restart } else { Response::Insensitive { floor, y: vy, held } }
            }
            (Response::Confirm { floor, y: fy }, _) => {
                // The raised picture: a verdict only if its bits neither fell
                // by an answer's worth nor rose beyond noise.
                let change = y - fy;
                if qp > floor && change > -answer(i32::from(qp - floor)) && change < -band {
                    Response::Insensitive { floor, y: fy, held: 0 }
                } else {
                    restart
                }
            }
            (Response::Walk { qp: wqp, y: wy, moves }, Some((lqp, ly))) => {
                if qp > lqp || (qp == lqp && (y - ly < band || y - ly > -band)) {
                    // A raised quantiser is not a walk; a held one whose bits
                    // moved means the content did, and the anchor no longer
                    // describes it.
                    restart
                } else if qp == lqp {
                    self.response
                } else {
                    let span = i32::from(wqp) - i32::from(qp);
                    let rise = y - wy;
                    if rise >= answer(span) || rise < band {
                        // Answered, or got cheaper: either way not a picture
                        // ignoring the quantiser.
                        restart
                    } else if span >= INSENSITIVE_SPAN && (moves >= 1 || span >= 2 * INSENSITIVE_SPAN) {
                        Response::Confirm { floor: qp, y }
                    } else {
                        Response::Walk { qp: wqp, y: wy, moves: moves + 1 }
                    }
                }
            }
            _ => restart,
        };
    }
}

/// What the insensitivity rule did over a stream, counted across every
/// picture kind: verdicts reached ([`INSENSITIVE_SPAN`]), probes coded
/// below a verdict's floor and verdicts released ([`INSENSITIVE_RETRY`]).
///
/// Reported, like the plan error, so a gate row can insist the rule was
/// exercised rather than assume it: on the corpus as it stood when the rule
/// was rewritten, no cell reached a verdict at all, so every property that
/// held said nothing about the verdict path.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Insensitivity {
    /// Walks confirmed from above into a verdict.
    pub verdicts: u64,
    /// Picks coded below a standing verdict's floor to ask again.
    pub probes: u64,
    /// Verdicts ended, by a probe that answered or by the content changing.
    pub releases: u64,
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
    /// being coded — kind, quantiser and lookahead cost (1 without one) —
    /// so [`RateController::account`] can pin the model against the
    /// quantiser and cost that actually produced the bits.
    pending: Option<(PicKind, u8, f64)>,
    /// Whether the pending pick was planned from a seed and nothing else —
    /// the stream's first under lookahead, or the first P after a keyframe
    /// that was coded again — so [`RateController::seed_recode`] may ask
    /// for it to be coded again too.
    pending_seeded: bool,
    /// Whether the stream's first picture was coded again by
    /// [`RateController::seed_recode`]. From then on its bits per cost is a
    /// measurement at a quantiser the picture shipped at, and a kind that
    /// borrows it is no longer held to the seed's clamp.
    keyframe_recoded: bool,
    /// The lookahead cost of the last picture *picked* for each kind, so
    /// the step limit can be widened by a measured change in content
    /// rather than throttle it. `None` until a kind has been picked with
    /// a cost.
    last_cost: [Option<f64>; 3],
    /// The kind most recently pinned by an observation, which an
    /// unobserved kind borrows its bits-per-cost from under lookahead.
    last_observed: Option<PicKind>,
    /// The bits the picture being coded was planned at, for the plan
    /// error below. 0 before the first pick.
    planned: f64,
    /// Accumulated `6 * |log2(actual / planned)|` over accounted
    /// pictures, and how many — the model check
    /// [`RateController::plan_error`] reports.
    plan_error: f64,
    plan_count: u64,
    /// See [`RateController::insensitivity`].
    insensitivity: Insensitivity,
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
                Complexity::new(seed_k(seed_for(PicKind::Intra))),
                Complexity::new(seed_k(seed_for(PicKind::Inter))),
                Complexity::new(seed_k(seed_for(PicKind::B))),
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
            pending_seeded: false,
            keyframe_recoded: false,
            last_cost: [None; 3],
            last_observed: None,
            planned: 0.0,
            plan_error: 0.0,
            plan_count: 0,
            insensitivity: Insensitivity::default(),
        }
    }

    /// How far pictures landed from their plan, on average, in quantiser
    /// steps of the law: `6 * |log2(actual bits / planned bits)|` per
    /// picture, meaned. The controller's model check — it chooses a
    /// quantiser to land a target, and this is how wrong that choice was,
    /// reported so a change to the model can be measured against the
    /// pictures it planned rather than only against the band. `None`
    /// before any picture has been accounted.
    pub fn plan_error(&self) -> Option<f64> {
        (self.plan_count > 0).then(|| self.plan_error / self.plan_count as f64)
    }

    /// What the insensitivity rule did so far: see [`Insensitivity`].
    pub fn insensitivity(&self) -> Insensitivity {
        self.insensitivity
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
        if let Some((kind, _, cost)) = self.pending {
            self.pending = Some((kind, qp, cost));
        }
    }

    /// Choose the quantiser for the next picture. Call once per picture,
    /// in coding order, before coding it.
    ///
    /// The past-only controller: [`RateController::pick_qp_ahead`] with a
    /// cost of one and nothing in the window, which is the same
    /// arithmetic it has always been.
    pub fn pick_qp(&mut self, kind: PicKind) -> u8 {
        self.pick_qp_ahead(kind, 1.0, &[])
    }

    /// The bits this picture is aiming for under lookahead: its share of
    /// the window's budget by predicted bits at a common quantiser —
    /// `k(kind) * cost` for it and for every picture in `window` (which
    /// includes it) — bounded so one picture cannot take the whole window,
    /// plus the same bucket correction [`RateController::target_for`]
    /// applies. See the module documentation's lookahead section.
    fn target_ahead(&self, kind: PicKind, cost: f64, window: &[(PicKind, f64)]) -> f64 {
        let mean = window.iter().map(|&(k, c)| self.k_for(k) * c).sum::<f64>() / window.len() as f64;
        let mine = self.k_for(kind) * cost;
        let share = if mean > 0.0 { (mine / mean).clamp(1.0 / MAX_K_RATIO, MAX_K_RATIO) } else { 1.0 };
        let base = self.per_picture * share;
        let correction = (self.budget / CORRECTION_PICTURES).clamp(-0.5 * base, 0.5 * base);
        (base + correction).max(16.0)
    }

    /// The bits-per-unit-cost estimate to plan a picture of `kind` with
    /// under lookahead: the kind's own once it has been observed, the most
    /// recently observed kind's before that — a keyframe's price per unit
    /// of residual energy is the best available guess for the first P —
    /// and the calibrated seed when nothing has been observed at all.
    fn k_for(&self, kind: PicKind) -> f64 {
        let c = self.complexity[kind as usize];
        if c.observed {
            return c.k;
        }
        match self.last_observed {
            Some(other) => self.complexity[other as usize].k,
            None => SEED_BITS_PER_COST,
        }
    }

    /// [`RateController::pick_qp`] informed by a lookahead: `cost` is this
    /// picture's complexity in the caller's units and `window` is every
    /// picture the caller holds — this one included — with the kind it
    /// expects to code each as. An empty window means no lookahead, in
    /// which case `cost` must be one and the kind weights allocate the
    /// budget as they always have.
    ///
    /// The calibration run behind [`SEED_BITS_PER_COST`]: `tools/verify_encode.sh`'s
    /// corpus under `--lookahead 8`, keyframes only, `bits * 2^(qp/6) /
    /// cost` — recorded on the constant.
    pub fn pick_qp_ahead(&mut self, kind: PicKind, cost: f64, window: &[(PicKind, f64)]) -> u8 {
        let ahead = !window.is_empty();
        debug_assert!(ahead || cost == 1.0, "a past-only pick has no cost to scale by");
        debug_assert!(cost > 0.0, "a lookahead cost must be positive");
        let mut target = if ahead { self.target_ahead(kind, cost, window) } else { self.target_for(kind) };
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
        // Invert bits(qp) = k * cost * 2^(-qp/6). Under lookahead `k` is
        // bits per unit of cost and may be borrowed from another kind or
        // seeded from the calibration; either way a pick for a kind that
        // has not been observed is a guess about that kind, and it is
        // bounded the way every other seed is. The past-only path seeds
        // through the same clamp inside `seed_k`. Without it a P picture
        // whose cost the lookahead put near zero — a held frame, whose
        // residual is really the reference's quantisation noise — was
        // planned at quantiser 0 and cost twelve times its keyframe.
        let (k_eff, guess) = if ahead { (self.k_for(kind) * cost, !c.observed) } else { (c.k, false) };
        // Planned from a seed and nothing else: the stream's first picture,
        // and the first P after a keyframe that was coded again. Either may
        // be coded again itself (`seed_recode`).
        let first_p = kind == PicKind::Inter && self.keyframe_recoded && self.last_observed == Some(PicKind::Intra);
        let seeded = ahead && guess && (self.last_observed.is_none() || first_p);
        // A kind borrowing the bits per cost of a keyframe that was coded
        // again borrows a measurement, not the calibration, so the seed's
        // clamp does not hold it. Held at `SEED_QP_MIN` it had planned the
        // grad clip's first P on 0.04 of its plan.
        let guess = guess && !(self.keyframe_recoded && self.last_observed.is_some());
        let want = 6.0 * (k_eff / target).log2();
        let want = if guess { want.clamp(SEED_QP_MIN, SEED_QP_MAX) } else { want };
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
        //
        // Under lookahead the limit is widened by the measured change in
        // cost since this kind was last picked, in the direction the cost
        // moved: the limit exists to stop the quantiser chasing noise
        // between similar pictures, and a picture four times the cost of
        // the last is not noise — holding it to three steps is what
        // spends a scene cut's whole budget on its first picture.
        let widen = match (ahead, self.last_cost[kind as usize]) {
            (true, Some(last)) if last > 0.0 => (6.0 * (cost / last).log2()).round() as i32,
            _ => 0,
        };
        let informed = c.observed;
        match (informed, self.last_informed[kind as usize], self.last_any[kind as usize]) {
            (true, Some(last), _) => {
                let last = last as i32;
                qp = qp.clamp(last - MAX_QP_STEP + widen.min(0), last + MAX_QP_STEP + widen.max(0));
            }
            (_, None, Some(any)) => {
                let any = any as i32;
                qp = qp.clamp(any - MAX_FIRST_STEP, any + MAX_FIRST_STEP);
            }
            _ => {}
        }
        // Content that does not answer the quantiser cannot be steered by
        // it, and the model's advice — lower it further — is exactly wrong.
        // Hold the line instead of walking to zero, except for the periodic
        // probe that asks again (INSENSITIVE_RETRY). A silent walk first
        // asks from above (INSENSITIVE_SPAN).
        match &mut self.complexity[kind as usize].response {
            Response::Confirm { floor, .. } => qp = qp.max(i32::from(*floor) + MAX_QP_STEP),
            Response::Insensitive { floor, held, .. } if *held >= INSENSITIVE_RETRY => {
                qp = qp.max(i32::from(*floor) - MAX_QP_STEP);
                *held = 0;
                if qp < i32::from(*floor) {
                    self.insensitivity.probes += 1;
                }
            }
            Response::Insensitive { floor, held, .. } => {
                qp = qp.max(i32::from(*floor));
                *held += 1;
            }
            _ => {}
        }
        let qp = qp.clamp(QP_MIN, QP_MAX) as u8;
        if informed {
            self.last_informed[kind as usize] = Some(qp);
        }
        self.last_any[kind as usize] = Some(qp);
        if ahead {
            self.last_cost[kind as usize] = Some(cost);
        }
        self.pending = Some((kind, qp, cost));
        self.pending_seeded = seeded;
        self.planned = target;
        qp
    }

    /// Whether the pending picture, coded once to `bits`, is to be coded
    /// again, and at what quantiser: `Some` when it was planned from a seed
    /// and nothing else and landed more than [`SEED_RECODE_STEPS`] from its
    /// plan. `None` when it landed near enough, and for every other
    /// picture.
    ///
    /// Two pictures of a stream can qualify: its first under lookahead,
    /// and, if that one was coded again, the first P after it.
    ///
    /// The answer is what one observation says: the quantiser moved by the
    /// miss in steps of the law, `6 * log2(bits / planned)`. It is bounded
    /// by [`MAX_FIRST_STEP`] like any first measured correction, and not by
    /// the seed's clamp, because it is no longer a guess. Under a declared
    /// buffer it may only rise: a picture that fitted must not be coded
    /// again into one that does not.
    ///
    /// A `Some` is also the controller taking that quantiser as the
    /// pending picture's, in one call so that no caller can re-code without
    /// telling it. The model is pinned to the coding that ships. The pick
    /// counts as a measured one, so the kind's next picture moves from it
    /// by the ordinary step limit rather than from a guess. And it is not
    /// asked about again.
    pub fn seed_recode(&mut self, bits: u64) -> Option<u8> {
        let (kind, qp, cost) = self.pending?;
        if !self.pending_seeded || bits == 0 || self.planned <= 0.0 {
            return None;
        }
        let miss = 6.0 * (bits as f64 / self.planned).log2();
        if miss.abs() <= SEED_RECODE_STEPS {
            return None;
        }
        let was = i32::from(qp);
        let again = ((was as f64 + miss).round() as i32).clamp(was - MAX_FIRST_STEP, was + MAX_FIRST_STEP).clamp(QP_MIN, QP_MAX);
        if again == was || (self.cpb.is_some() && again < was) {
            return None;
        }
        let again = again as u8;
        self.pending = Some((kind, again, cost));
        self.last_informed[kind as usize] = Some(again);
        self.last_any[kind as usize] = Some(again);
        self.pending_seeded = false;
        if kind == PicKind::Intra {
            self.keyframe_recoded = true;
        }
        Some(again)
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
        let Some((kind, qp, cost)) = self.pending.take() else {
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
        // The model check, reported never gated: how far the picture
        // landed from what it was planned at, in quantiser steps of the
        // law (six per doubling). Zero would mean the model was exact.
        if bits > 0 && self.planned > 0.0 {
            self.plan_error += (bits as f64 / self.planned).log2().abs() * 6.0;
            self.plan_count += 1;
        }
        if bits > 0 {
            // Per unit of lookahead cost, which is one without a lookahead.
            let k_obs = bits as f64 * 2f64.powf(qp as f64 / 6.0) / cost;
            self.last_observed = Some(kind);
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
            // Did the quantiser move down, and did the bits care? See
            // INSENSITIVE_BAND for the walk and INSENSITIVE_RETRY for how a
            // verdict is reopened.
            let was = c.response;
            c.observe_response(qp, (bits as f64 / cost).log2());
            c.observed = true;
            match (was, c.response) {
                (Response::Confirm { .. }, Response::Insensitive { .. }) => self.insensitivity.verdicts += 1,
                (Response::Insensitive { .. }, Response::Insensitive { .. }) => {}
                (Response::Insensitive { .. }, _) => self.insensitivity.releases += 1,
                _ => {}
            }
        }
    }

    /// A copy of the controller's whole state, for tests that branch one
    /// history two ways.
    #[cfg(test)]
    fn clone_for_test(&self) -> Self {
        RateController {
            per_picture: self.per_picture,
            avg_weight: self.avg_weight,
            budget: self.budget,
            complexity: self.complexity,
            last_informed: self.last_informed,
            last_any: self.last_any,
            bits_spent: self.bits_spent,
            pictures: self.pictures,
            cpb: self.cpb,
            pending: self.pending,
            pending_seeded: self.pending_seeded,
            keyframe_recoded: self.keyframe_recoded,
            last_cost: self.last_cost,
            last_observed: self.last_observed,
            planned: self.planned,
            plan_error: self.plan_error,
            plan_count: self.plan_count,
            insensitivity: self.insensitivity,
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

    /// Whether a kind's controller holds a standing verdict that its bits
    /// ignore the quantiser.
    fn verdict(rc: &RateController, kind: PicKind) -> bool {
        matches!(rc.complexity[kind as usize].response, Response::Insensitive { .. })
    }

    /// Fold a sequence of `(quantiser, bits)` observations into a fresh
    /// estimate, at a cost of one, and return the response after each —
    /// the rule alone, without a controller choosing the quantisers.
    fn responses(seq: &[(u8, f64)]) -> Vec<Response> {
        let mut c = Complexity::new(1.0);
        seq.iter()
            .map(|&(qp, bits)| {
                c.observe_response(qp, bits.log2());
                c.response
            })
            .collect()
    }

    /// Drive one observation at a quantiser of the test's choosing: pick
    /// (so the controller's bookkeeping runs), overrule the pick through
    /// the re-code hook, and account the bytes. What replays a real trace.
    fn observe_at(rc: &mut RateController, kind: PicKind, qp: u8, bytes: usize) {
        let _ = rc.pick_qp(kind);
        rc.note_recode(qp);
        rc.account(bytes);
    }

    /// **The misfires, replayed.** Two traces from the gate's cut clip that
    /// froze the quantiser for the rest of the clip under the one-move rule
    /// (h26x d88e24a), fed back verbatim — every picture from the first,
    /// kind, quantiser and bits, because the old rule's reference point was
    /// carried across every move smaller than three and an excerpt starts a
    /// different chain of references. Neither may reach a verdict.
    ///
    /// - H.265 at 64 kbps with a depth-0 quadtree, pictures 0 to 53: at 53,
    ///   41 to 38 while the bits fell 168 to 160, a picture right after the
    ///   cut, cheaper because its reference had just been coded. The old
    ///   rule held 38 for the 37 inter pictures left, on a third of their
    ///   plan.
    /// - H.264 at 128 kbps, pictures 0 to 25: at 25, 28 to 32, a *raised*
    ///   quantiser, with the bits up 3.6 %. The old rule held 32 for 62
    ///   inter pictures on under half their plan.
    #[test]
    fn a_single_move_after_a_cut_or_a_raised_quantiser_is_not_a_verdict() {
        const I: PicKind = PicKind::Intra;
        const P: PicKind = PicKind::Inter;
        #[rustfmt::skip]
        let h265: [(PicKind, u8, usize); 54] = [
            (I, 26, 12440), (P, 32, 1752), (P, 39, 760), (P, 39, 928), (P, 40, 856), (P, 40, 872), (P, 39, 1192), (P, 39, 1096),
            (I, 33, 8360), (P, 42, 528), (P, 40, 1032), (P, 42, 760), (P, 40, 1240), (P, 40, 576), (P, 37, 1528), (P, 37, 1584),
            (I, 34, 7632), (P, 40, 760), (P, 40, 904), (P, 39, 1280), (P, 39, 1208), (P, 39, 1288), (P, 38, 1520), (P, 38, 1344),
            (I, 35, 7432), (P, 41, 1152), (P, 42, 912), (P, 40, 1392), (P, 40, 1256), (P, 40, 1328), (P, 39, 1456), (P, 39, 1432),
            (I, 36, 6760), (P, 42, 528), (P, 39, 1208), (P, 38, 1240), (P, 37, 1560), (P, 37, 1632), (P, 37, 792), (P, 34, 2232),
            (I, 36, 6984), (P, 37, 1640), (P, 39, 1072), (P, 38, 1376), (P, 37, 1520), (P, 37, 1584), (P, 37, 1464), (P, 36, 1624),
            (I, 36, 6584), (P, 39, 1048), (P, 38, 1416), (P, 38, 3408), (P, 41, 168), (P, 38, 160),
        ];
        #[rustfmt::skip]
        let h264: [(PicKind, u8, usize); 26] = [
            (I, 26, 9552), (P, 28, 2672), (P, 28, 3224), (P, 29, 2560), (P, 28, 3120), (P, 28, 2896), (P, 27, 3416), (P, 27, 3376),
            (I, 23, 11768), (P, 29, 2672), (P, 29, 3024), (P, 29, 2976), (P, 28, 3448), (P, 28, 2104), (P, 26, 4040), (P, 27, 3560),
            (I, 23, 12232), (P, 29, 2680), (P, 29, 3000), (P, 29, 3224), (P, 29, 3304), (P, 29, 3224), (P, 29, 3352), (P, 29, 3464),
            (I, 23, 13144), (P, 32, 2768),
        ];
        let mut verdicts = Vec::new();
        for (name, bps, trace) in [("H.265 64k", 64_000, &h265[..]), ("H.264 128k", 128_000, &h264[..])] {
            let mut rc = RateController::new(bps, 30, 64, 64, 8, 0);
            for (i, &(kind, qp, bits)) in trace.iter().enumerate() {
                observe_at(&mut rc, kind, qp, bits / 8);
                if verdict(&rc, kind) {
                    verdicts.push(format!("{name}: picture {i}, {kind:?} quantiser {qp}, {bits} bits"));
                }
            }
        }
        assert!(verdicts.is_empty(), "the replayed traces reached verdicts: {verdicts:?}");
    }

    /// **Bits held flat by improving references are not a verdict.** A
    /// scene whose inter complexity falls by a three-step factor every
    /// picture — each picture predicting better from the finer one before
    /// it — while the quantiser walks down three a picture: the bits stay
    /// exactly flat, and the walk alone cannot tell that from content that
    /// ignores the quantiser. The raised picture can: the references are
    /// still improving and the raised quantiser pushes the same way, so its
    /// bits fall by twice the law's step and the walk restarts.
    #[test]
    fn bits_held_flat_by_improving_references_are_not_a_verdict() {
        let bits = |qp: u8, t: i32| 1e6 * 2f64.powf(-f64::from(qp) / 6.0 - f64::from(t) / 2.0);
        // The walk, then the raised picture the rule asks with, then the
        // scene settling at picture 5 while the controller walks on.
        let seq = [(40, 0), (37, 1), (34, 2), (37, 3), (34, 4), (31, 5), (28, 5), (25, 5)];
        let r = responses(&seq.map(|(qp, t)| (qp, bits(qp, t))));
        assert!(matches!(r[2], Response::Confirm { floor: 34, .. }), "the walk should have gone silent and asked: {r:?}");
        assert!(r.iter().all(|x| !matches!(x, Response::Insensitive { .. })), "improving references reached a verdict: {r:?}");
    }

    /// **Bits that truly stop answering still reach a verdict**, through
    /// each of its gates: a walk of two lowered pictures over six steps, or
    /// one move of twelve, then a raised picture whose bits also stay put.
    /// A single three- or six-step move is not enough, nor a walk whose
    /// raised picture answers, nor one whose bits fell along the way.
    #[test]
    fn bits_that_do_not_answer_the_quantiser_reach_a_verdict_and_nothing_less_does() {
        let f = 3000.0;
        let r = responses(&[(40, f), (37, f * 1.02), (34, f * 0.99), (37, f * 1.01)]);
        assert!(matches!(r[2], Response::Confirm { floor: 34, .. }) && matches!(r[3], Response::Insensitive { floor: 34, .. }), "{r:?}");
        let r = responses(&[(26, f), (10, f), (13, f)]);
        assert!(matches!(r[1], Response::Confirm { floor: 10, .. }) && matches!(r[2], Response::Insensitive { floor: 10, .. }), "{r:?}");

        for seq in [
            // One move of three, and one of six.
            &[(40, f), (37, f)][..],
            &[(40, f), (34, f), (34, f)][..],
            // The raised picture answers.
            &[(40, f), (37, f), (34, f), (37, f * 0.8)][..],
            // The bits fell along the walk.
            &[(40, f), (37, f * 0.97), (34, f * 0.9), (37, f * 0.9)][..],
            // The lowered pictures answered a quarter of the law.
            &[(40, f), (37, f * 1.1), (34, f * 1.2), (37, f * 1.2)][..],
        ] {
            let r = responses(seq);
            assert!(r.iter().all(|x| !matches!(x, Response::Insensitive { .. })), "{seq:?} reached a verdict: {r:?}");
        }
    }

    /// **A verdict re-opens.** Bits that move at the floor are content that
    /// changed, and release it at once; a probe below the floor whose bits
    /// answer releases it too; a probe that does not leaves it standing
    /// with its floor where it was.
    #[test]
    fn a_verdict_is_released_by_content_change_or_an_answering_probe() {
        let f = 3000.0;
        let base = [(40, f), (37, f), (34, f), (37, f)];
        let with = |tail: &[(u8, f64)]| responses(&[&base[..], tail].concat());
        let r = with(&[(34, f), (34, f * 1.5)]);
        assert!(matches!(r[4], Response::Insensitive { .. }) && matches!(r[5], Response::Walk { .. }), "content change: {r:?}");
        let r = with(&[(34, f), (31, f * 1.3)]);
        assert!(matches!(r[5], Response::Walk { .. }), "answering probe: {r:?}");
        let r = with(&[(34, f), (31, f * 1.02), (34, f)]);
        assert!(matches!(r[6], Response::Insensitive { floor: 34, .. }), "silent probe: {r:?}");
    }

    /// Closed loop, content that never answers: each kind reaches its
    /// verdict and then *holds* it — over the next three hundred and fifty
    /// pictures its floor never moves and no pick, probes included, goes
    /// more than one step limit below it.
    #[test]
    fn content_that_ignores_the_quantiser_is_held_without_walking() {
        let mut rc = RateController::new(2_000_000, 30, W, H, 8, 0);
        let floor_of = |rc: &RateController, kind: PicKind| match rc.complexity[kind as usize].response {
            Response::Insensitive { floor, .. } => Some(floor),
            _ => None,
        };
        let mut floors: [Option<u8>; 2] = [None; 2];
        let mut probes = 0;
        for i in 0..400 {
            let kind = if i % 8 == 0 { PicKind::Intra } else { PicKind::Inter };
            let qp = rc.pick_qp(kind);
            if i >= 50 {
                let fl = floors[kind as usize].expect("a verdict for both kinds within fifty pictures");
                assert!(i32::from(qp) >= i32::from(fl) - MAX_QP_STEP, "picture {i}: {kind:?} picked {qp} under a floor of {fl}");
                probes += usize::from(qp < fl);
            }
            rc.account(400);
            if i < 50 {
                floors[kind as usize] = floor_of(&rc, kind);
            } else {
                assert_eq!(floor_of(&rc, kind), floors[kind as usize], "picture {i}: {kind:?}'s verdict moved or was released");
            }
        }
        assert!(probes > 0, "no probe was ever made, so nothing above tested the hold against one");
        // The report says the same: a verdict per kind, at least the probes
        // counted above (which start at picture fifty), and no release.
        let events = rc.insensitivity();
        assert!(events.verdicts == 2 && events.probes as usize >= probes && events.releases == 0, "{events:?}, {probes} probes after picture 50");
    }

    /// Closed loop, content that ignores the quantiser at or above some
    /// level and answers below it — all-skip pictures that start coding
    /// residual once the quantiser is fine enough. The verdict forms at the
    /// level, the periodic probe finds the answer, and the controller then
    /// steers below the old floor instead of holding it for the rest of the
    /// stream.
    #[test]
    fn a_probe_below_the_floor_finds_content_that_answers_again() {
        let mut rc = RateController::new(2_000_000, 30, W, H, 8, 0);
        let mut floor = None;
        let mut below = u8::MAX;
        for i in 0..160 {
            let kind = if i % 8 == 0 { PicKind::Intra } else { PicKind::Inter };
            let qp = rc.pick_qp(kind);
            let bytes = match (kind, floor) {
                (PicKind::Inter, Some(fl)) if qp < fl => (400.0 * 2f64.powf(f64::from(fl - qp) / 6.0)) as usize,
                _ => 400,
            };
            if kind == PicKind::Inter && floor.is_some() {
                below = below.min(qp);
            }
            rc.account(bytes);
            if floor.is_none()
                && i >= 40
                && let Response::Insensitive { floor: fl, .. } = rc.complexity[PicKind::Inter as usize].response
            {
                floor = Some(fl);
            }
        }
        let fl = floor.expect("a verdict on the constant phase");
        assert!(i32::from(below) <= i32::from(fl) - 2 * MAX_QP_STEP, "the controller never went more than a probe below the floor {fl}: lowest {below}");
        let events = rc.insensitivity();
        assert!(events.verdicts >= 1 && events.probes >= 1 && events.releases >= 1, "the report missed a path: {events:?}");
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

    /// A cost of one and an empty window is the past-only controller,
    /// pick for pick: the lookahead path may not move a stream that did
    /// not ask for it.
    #[test]
    fn a_cost_of_one_and_no_window_is_the_past_only_controller() {
        let mut a = RateController::new(600_000, 30, W, H, 8, 2);
        let mut b = RateController::new(600_000, 30, W, H, 8, 2);
        for i in 0..24 {
            let kind = match i % 8 {
                0 => PicKind::Intra,
                1 | 4 | 7 => PicKind::Inter,
                _ => PicKind::B,
            };
            let qa = a.pick_qp(kind);
            let qb = b.pick_qp_ahead(kind, 1.0, &[]);
            assert_eq!(qa, qb, "picture {i}: the two paths chose different quantisers");
            let bytes = synth_bits(if kind == PicKind::Intra { K_INTRA } else { K_INTER }, qa);
            a.account(bytes);
            b.account(bytes);
        }
        assert_eq!(a.bits_spent, b.bits_spent);
    }

    /// Under lookahead the window's budget is split by predicted bits: the
    /// costlier picture is given more, and the shares only redistribute —
    /// two pictures' targets sum to two pictures' worth.
    #[test]
    fn lookahead_gives_the_costlier_picture_the_larger_share_and_redistributes() {
        let rc = RateController::new(600_000, 30, W, H, 8, 0);
        let window = [(PicKind::Inter, 1000.0), (PicKind::Inter, 3000.0)];
        let cheap = rc.target_ahead(PicKind::Inter, 1000.0, &window);
        let dear = rc.target_ahead(PicKind::Inter, 3000.0, &window);
        assert!(dear > cheap * 2.5, "the picture at three times the cost was given {dear:.0} against {cheap:.0}");
        let plain = rc.per_picture * 2.0;
        assert!(((cheap + dear) - plain).abs() < plain * 0.01, "the shares changed the window's total: {:.0} against {plain:.0}", cheap + dear);
        // Bounded: a picture a hundred times the cost of the rest cannot
        // take more than MAX_K_RATIO pictures' worth.
        let wild = rc.target_ahead(PicKind::Inter, 100_000.0, &[(PicKind::Inter, 100_000.0), (PicKind::Inter, 1000.0), (PicKind::Inter, 1000.0)]);
        assert!(wild <= rc.per_picture * MAX_K_RATIO * 1.01, "{wild:.0}");
    }

    /// The first P picture of a stream under lookahead is planned from
    /// the keyframe's measured bits per unit of cost, not from the
    /// bits-per-pixel seed: four times the cost asks for twelve more
    /// quantiser steps, the law's own slope — inside the seed clamp,
    /// which a guess about an unobserved kind never escapes, whether the
    /// guess is the calibration or a borrowed measurement.
    #[test]
    fn an_unmeasured_kind_borrows_the_measured_bits_per_cost() {
        // Nothing observed: the calibration alone, clamped like a seed at
        // both ends — an absurd cost either way cannot run away.
        let mut fresh = RateController::new(600_000, 30, W, H, 8, 0);
        let huge = fresh.pick_qp_ahead(PicKind::Intra, 1e12, &[(PicKind::Intra, 1e12)]);
        assert_eq!(f64::from(huge), SEED_QP_MAX, "an absurd cost against the calibration must hit the seed clamp, not run away");
        let mut fresh = RateController::new(600_000, 30, W, H, 8, 0);
        let tiny = fresh.pick_qp_ahead(PicKind::Intra, 1e-3, &[(PicKind::Intra, 1e-3)]);
        assert_eq!(f64::from(tiny), SEED_QP_MIN, "a negligible cost must be held at the seed floor");

        // One keyframe observed at a plausible cost; its k per unit cost
        // is then what plans the first P, so a P at cost c and a P at
        // cost 4c on two copies of the same state differ by the law's own
        // slope, twelve steps, when both land inside the clamp.
        let mut rc = RateController::new(600_000, 30, W, H, 8, 0);
        let cost_i = 1e6;
        let first = rc.pick_qp_ahead(PicKind::Intra, cost_i, &[(PicKind::Intra, cost_i)]);
        rc.account(synth_bits(K_INTRA, first));
        let mut lo = rc.clone_for_test();
        let mut hi = rc.clone_for_test();
        let q_lo = lo.pick_qp_ahead(PicKind::Inter, 4e5, &[(PicKind::Inter, 4e5)]);
        let q_hi = hi.pick_qp_ahead(PicKind::Inter, 1.6e6, &[(PicKind::Inter, 1.6e6)]);
        assert!(f64::from(q_lo) > SEED_QP_MIN && f64::from(q_hi) < SEED_QP_MAX, "the picks must sit inside the clamp for the slope to show: {q_lo}, {q_hi}");
        assert!((i32::from(q_hi) - i32::from(q_lo) - 12).abs() <= 1, "cost x4 moved the quantiser from {q_lo} to {q_hi}, not by twelve");
        // A borrowed measurement is still a guess about this kind: a
        // negligible cost is held at the seed floor rather than planned
        // at quantiser 0 — the held-frame case, whose true residual is
        // the reference's quantisation noise.
        let mut held = rc.clone_for_test();
        let q_held = held.pick_qp_ahead(PicKind::Inter, 1.0, &[(PicKind::Inter, 1.0)]);
        assert_eq!(f64::from(q_held), SEED_QP_MIN, "a borrowed pick at a negligible cost escaped the seed clamp: {q_held}");
        // And it was the keyframe's measurement that planned it, not the
        // calibration: the same P on a controller that observed nothing
        // lands elsewhere.
        let mut blind = RateController::new(600_000, 30, W, H, 8, 0);
        let q_blind = blind.pick_qp_ahead(PicKind::Inter, 4e5, &[(PicKind::Inter, 4e5)]);
        assert_ne!(q_blind, q_lo, "the borrowed k made no difference to the first P");
    }

    /// **A seeded first picture that misses by far is coded again.** The
    /// stream's first picture under lookahead is planned from the
    /// calibration alone. Landing within [`SEED_RECODE_STEPS`] of its plan
    /// it is left alone. Further out, the controller asks for it again at
    /// the quantiser its own bits imply, below the seed's floor if need be
    /// and never more than [`MAX_FIRST_STEP`] away. Once coded again it is
    /// a measured pick:
    ///
    /// - the model is pinned to the coding that shipped;
    /// - the next keyframe moves from it by the ordinary step limit;
    /// - nothing else is asked about.
    #[test]
    fn a_seeded_first_picture_that_misses_by_far_is_coded_again_at_its_own_quantiser() {
        let cost = 1e-3;
        let window = |c: f64| vec![(PicKind::Intra, c), (PicKind::Inter, c / 4.0), (PicKind::Inter, c / 4.0)];
        // A negligible cost puts the seeded pick on the seed's floor.
        let mut rc = RateController::new(600_000, 30, W, H, 8, 0);
        let first = rc.pick_qp_ahead(PicKind::Intra, cost, &window(cost));
        assert_eq!(f64::from(first), SEED_QP_MIN, "the seeded pick of a negligible cost sits on the floor");
        let plan = rc.planned;
        let at = |steps: f64| (plan * 2f64.powf(steps / 6.0)) as u64;
        let ask = |steps: f64| rc.clone_for_test().seed_recode(at(steps));
        assert_eq!(ask(-2.9), None, "2.9 steps under plan is near enough");
        assert_eq!(ask(2.9), None, "2.9 steps over plan is near enough");
        assert_eq!(ask(-4.0), Some(first - 4), "4 steps under plan asks for 4 steps lower");
        assert_eq!(ask(4.0), Some(first + 4), "4 steps over plan asks for 4 steps higher");
        assert_eq!(ask(30.0), Some(first + 16), "30 steps over plan is bounded to the first-correction limit");

        let again = rc.seed_recode(at(-12.0)).expect("12 steps under plan is coded again");
        assert_eq!(again, first - 12, "12 steps under plan asks for 12 steps lower, under the seed's floor");
        assert_eq!(rc.seed_recode(at(-12.0)), None, "a picture is coded again once, not twice");
        // Coded again it still lands ten steps under. The model is pinned
        // to that coding, and the next keyframe, which wants ten lower,
        // moves by the ordinary limit and not the first correction's.
        let bytes = (at(-10.0) / 8) as usize;
        rc.account(bytes);
        let k = (bytes * 8) as f64 * 2f64.powf(f64::from(again) / 6.0) / cost;
        let got = rc.complexity[PicKind::Intra as usize].k;
        assert!((got / k - 1.0).abs() < 1e-9, "the model was pinned to {got}, not to the coding that shipped ({k})");
        let next = rc.pick_qp_ahead(PicKind::Intra, cost, &window(cost));
        assert_eq!(i32::from(next), i32::from(again) - MAX_QP_STEP, "the next keyframe did not move from a measured pick");
        assert_eq!(rc.seed_recode(1), None, "a keyframe planned from a measurement is never asked again");

        // Without a lookahead nothing is seeded from the calibration, and
        // nothing is ever asked again.
        let mut past = RateController::new(600_000, 30, W, H, 8, 0);
        let _ = past.pick_qp(PicKind::Intra);
        assert_eq!(past.seed_recode(1), None, "the past-only controller asked for a re-code");

        // Under a declared buffer only a rise is asked for.
        let mut cpb = RateController::with_cpb(600_000, 30, W, H, 8, 0, Some(400_000));
        let q = cpb.pick_qp_ahead(PicKind::Intra, 1e6, &window(1e6));
        let plan = cpb.planned;
        assert_eq!(cpb.clone_for_test().seed_recode((plan * 0.1) as u64), None, "a buffered keyframe under plan was asked to spend more");
        assert!(cpb.seed_recode((plan * 10.0) as u64).is_some_and(|again| again > q), "a buffered keyframe over plan was not raised");
    }

    /// **The first P after a re-coded keyframe plans from a measurement.**
    /// It borrows the keyframe's bits per cost, as every first P does, but
    /// that is now a measurement at a quantiser a picture shipped at, so
    /// the seed's floor does not hold it: held there, the grad clip's first
    /// P was planned on 0.04 of its plan. It may be coded again once
    /// itself; the second P, and the first B, never are. Without a
    /// re-coded keyframe the first P is held at the floor as before.
    #[test]
    fn the_first_p_after_a_recoded_keyframe_is_not_held_at_the_seed_floor() {
        let (cost_i, cost_p) = (1e6, 3e4);
        let window = |kind: PicKind, c: f64| vec![(kind, c), (PicKind::Inter, cost_p), (PicKind::B, cost_p)];
        // The keyframe lands on plan either way: at the seed's pick, or
        // coded again 12 steps lower after missing by 12.
        let keyframe = |recode: bool| {
            let mut rc = RateController::new(600_000, 30, W, H, 8, 2);
            let first = rc.pick_qp_ahead(PicKind::Intra, cost_i, &window(PicKind::Intra, cost_i));
            if recode {
                let again = rc.seed_recode((rc.planned * 2f64.powf(-12.0 / 6.0)) as u64).expect("the keyframe is coded again");
                assert_eq!(again, first - 12);
            } else {
                assert_eq!(rc.seed_recode(rc.planned as u64), None);
            }
            rc.account((rc.planned / 8.0) as usize);
            rc
        };
        let mut rc = keyframe(true);
        let p1 = rc.pick_qp_ahead(PicKind::Inter, cost_p, &window(PicKind::Inter, cost_p));
        assert!(f64::from(p1) < SEED_QP_MIN, "the first P after a re-coded keyframe was held at the seed floor: {p1}");
        let plan = rc.planned;
        assert_eq!(rc.clone_for_test().seed_recode((plan * 2f64.powf(2.9 / 6.0)) as u64), None, "near enough is left alone");
        let again = rc.seed_recode((plan * 2f64.powf(9.0 / 6.0)) as u64).expect("the first P over its plan by 9 steps is coded again");
        assert_eq!(again, p1 + 9);
        rc.account((plan / 8.0) as usize);
        let _ = rc.pick_qp_ahead(PicKind::B, cost_p, &window(PicKind::B, cost_p));
        assert_eq!(rc.seed_recode(1), None, "the first B was asked again");
        rc.account((rc.planned / 8.0) as usize);
        let _ = rc.pick_qp_ahead(PicKind::Inter, cost_p, &window(PicKind::Inter, cost_p));
        assert_eq!(rc.seed_recode(1), None, "the second P was asked again");

        // A first P whose keyframe landed near its plan borrows as before,
        // held at the floor, and is not asked again.
        let mut plain = keyframe(false);
        let p = plain.pick_qp_ahead(PicKind::Inter, cost_p, &window(PicKind::Inter, cost_p));
        assert_eq!(f64::from(p), SEED_QP_MIN, "without a re-coded keyframe the first P must stay on the seed floor");
        assert_eq!(plain.seed_recode(1), None, "without a re-coded keyframe the first P was asked again");
    }

    /// A measured jump in cost widens the step limit in the direction of
    /// the jump; the same jump unmeasured is held to the limit. This is
    /// what lets a scene cut's first picture be coded at the quantiser it
    /// needs rather than three steps from the one before it.
    #[test]
    fn a_measured_cost_jump_widens_the_step_limit() {
        let mut ahead = RateController::new(600_000, 30, W, H, 8, 0);
        let mut past = RateController::new(600_000, 30, W, H, 8, 0);
        let window = |c: f64| [(PicKind::Inter, c)];
        // Two settled inter pictures at a steady cost, then one at eight
        // times the cost.
        let mut last_a = 0u8;
        let mut last_p = 0u8;
        for _ in 0..3 {
            last_a = ahead.pick_qp_ahead(PicKind::Inter, 1000.0, &window(1000.0));
            ahead.account(synth_bits(K_INTER, last_a));
            last_p = past.pick_qp(PicKind::Inter);
            past.account(synth_bits(K_INTER, last_p));
        }
        let jump_a = ahead.pick_qp_ahead(PicKind::Inter, 8000.0, &window(8000.0));
        let jump_p = past.pick_qp(PicKind::Inter);
        assert!(i32::from(jump_a) - i32::from(last_a) > MAX_QP_STEP, "lookahead held the cut to {last_a} -> {jump_a}");
        assert!(i32::from(jump_p) - i32::from(last_p) <= MAX_QP_STEP, "the past-only path exceeded its limit: {last_p} -> {jump_p}");
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
