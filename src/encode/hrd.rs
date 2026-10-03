//! The coded picture buffer, simulated — the only instrument that can see
//! whether a stream conforms to the buffer it declares.
//!
//! # Why this is not SELF, not CROSS, and not quality
//!
//! The gate's three properties are conformance (SELF and CROSS: the stream
//! means what the encoder thinks, exactly, checked against a decoder),
//! quality (PSNR: a measurement, reported), and control (`encode::rc`: did
//! the encoder achieve what it was handed, with no ground truth at all).
//!
//! Buffer conformance is a fourth position and belongs to none of them. It
//! **has a right answer** — given the declared parameters and the coded
//! access unit sizes, whether the buffer underflows is determined integer
//! arithmetic, not a judgement — which makes it conformance by nature
//! rather than control. But **neither of our conformance instruments can
//! see it**:
//!
//! - Our own decoder cannot. Until this change both parsers read the HRD
//!   fields purely to stay bit-aligned and discarded them, and that was not
//!   an oversight: **a decoder is not required to check the HRD at all.**
//!   It is a constraint on the *encoder*, verified by a separate
//!   conformance checker.
//! - libavcodec cannot either. It decodes a stream that overflows its
//!   declared buffer exactly as happily as one that does not, so CROSS is
//!   structurally blind.
//!
//! So the check has to be written, and the one rule that keeps it honest is
//! that it is **driven by the stream, never by the encoder**: every number
//! below comes out of the emitted bytes through the production parsers. An
//! encoder that told this module what buffer it had intended would be
//! marking its own homework, and would agree with itself no matter what it
//! wrote. Same discipline as the SAO check comparing against the filter's
//! actual output rather than its own prediction.
//!
//! # The model
//!
//! The leaky bucket of Annex C, in the one configuration this encoder
//! writes — NAL HRD, a single coded picture buffer, constant bit rate,
//! fixed picture rate, no sub-picture parameters:
//!
//! - Bits arrive continuously at `BitRate`.
//! - Access unit `n` is **removed whole** at `t_r(n)`, which for a fixed
//!   picture rate is `t_r(0) + n / fps`, with `t_r(0)` the
//!   `initial_cpb_removal_delay` the buffering period SEI carries.
//! - **Underflow** is the failure: at `t_r(n)` the buffer holds fewer bits
//!   than the access unit needs. The stream promised a decoder it could
//!   start decoding after `t_r(0)` and keep up, and it cannot.
//! - **Overflow** matters only under `cbr_flag`, where the arrival never
//!   pauses: bits that would push the buffer past `CpbSize` have nowhere to
//!   go. With `cbr_flag` clear the arrival simply stops and a full buffer
//!   is not an error, which is why [`Report::overflow`] is only populated
//!   for the constant-rate case.
//!
//! Everything is integer: bits and 90 kHz ticks, no floating point
//! anywhere, so the verdict is reproducible rather than nearly so. A
//! frame rate that is not a whole number of 90 kHz ticks — 29.97 in
//! H.264's field clock is 1501.5 of them — is kept as the exact fraction
//! (`Schedule::tick_den`), and the removal times are counted in the
//! fraction's own units, so a long stream does not drift by the rounding.

use crate::hevc::sps::Sps;
use crate::{Error, Result};

/// What the buffer did over one stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Report {
    /// Access units examined.
    pub units: usize,
    /// Declared `BitRate`, bits per second.
    pub bit_rate: u64,
    /// Declared `CpbSize`, bits.
    pub cpb_size: u64,
    /// The largest shortfall at any removal time, in bits, and which access
    /// unit it happened at. `None` when the stream conforms.
    ///
    /// The *largest* rather than the first, because how badly a stream
    /// misses is the number that says whether it was close.
    pub underflow: Option<(usize, u64)>,
    /// The largest excess over `CpbSize`, under `cbr_flag` only.
    pub overflow: Option<(usize, u64)>,
    /// Buffer occupancy immediately after each removal, in bits — the
    /// trace, for a caller that wants to see the shape rather than the
    /// verdict.
    pub occupancy: Vec<u64>,
}

impl Report {
    /// Whether the stream conformed to the buffer it declared.
    pub fn conforms(&self) -> bool {
        self.underflow.is_none() && self.overflow.is_none()
    }
}

/// The declared schedule, in the units the arithmetic wants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Schedule {
    /// `BitRate`, bits per second.
    pub bit_rate: u64,
    /// `CpbSize`, bits.
    pub cpb_size: u64,
    /// `cbr_flag`.
    pub cbr: bool,
    /// Removal interval, in 90 kHz ticks — `90_000 * num_units_in_tick /
    /// time_scale` for a fixed picture rate — over [`Schedule::tick_den`].
    pub tick_90k: u64,
    /// Denominator of the removal interval: `tick_90k / tick_den` 90 kHz
    /// ticks, in lowest terms. 1 for a clock that divides 90 kHz — every
    /// whole-number frame rate this encoder writes — and 2 for 29.97 in
    /// H.264's field clock (3003/2 ticks).
    pub tick_den: u64,
    /// `initial_cpb_removal_delay`, in 90 kHz ticks.
    pub initial_delay_90k: u64,
}

/// Walk the buffer over `sizes` (access unit sizes in **bits**, in coding
/// order) against `s`, at a fixed picture rate: unit `n` is removed at
/// `initial_delay + n * tick`.
///
/// Separated from any parsing so it can be tested against sequences chosen
/// to underflow and to overflow — which is the only way to know the check
/// can fail at all. A conformance checker that has never rejected anything
/// is indistinguishable from one that cannot.
pub fn simulate(sizes: &[u64], s: &Schedule) -> Report {
    let removal: Vec<u64> =
        (0..sizes.len()).map(|n| s.initial_delay_90k * s.tick_den + (n as u64) * s.tick_90k).collect();
    simulate_at(sizes, &removal, s)
}

/// [`simulate`] with each unit's removal time given outright, in units of
/// `1 / (90 000 * tick_den)` seconds — 90 kHz ticks when `tick_den` is 1
/// — how an H.264 stream is walked, whose removal times come from the
/// `cpb_removal_delay` of each picture's own timing SEI rather than from a
/// fixed-rate inference.
pub fn simulate_at(sizes: &[u64], removal_90k: &[u64], s: &Schedule) -> Report {
    debug_assert_eq!(sizes.len(), removal_90k.len());
    let mut occupancy = Vec::with_capacity(sizes.len());
    let mut underflow: Option<(usize, u64)> = None;
    let mut overflow: Option<(usize, u64)> = None;

    // Everything in bits and fractions of a 90 kHz tick. Bits arriving in
    // `d` of them is `bit_rate * d / (90_000 * tick_den)`, done as one
    // product before the divide so the truncation happens once rather than
    // compounding per picture — and in 128 bits, which the product of a
    // rate and a time counted that finely needs.
    let arrived = |ticks: u64| arrived_by(s, ticks);

    for (n, &size) in sizes.iter().enumerate() {
        // Bits that have arrived by this removal time, and bits already
        // removed by the units before it.
        let t = removal_90k[n];
        let total_in = arrived(t);
        let removed: u64 = sizes[..n].iter().sum();

        // Under a constant rate the buffer cannot hold more than CpbSize:
        // arrival never pauses, so anything above it is lost, which is the
        // overflow the standard forbids.
        let raw = total_in.saturating_sub(removed);
        if s.cbr && raw > s.cpb_size {
            let excess = raw - s.cpb_size;
            if overflow.is_none_or(|(_, e)| excess > e) {
                overflow = Some((n, excess));
            }
        }
        let present = raw.min(s.cpb_size);
        if present < size {
            let short = size - present;
            if underflow.is_none_or(|(_, d)| short > d) {
                underflow = Some((n, short));
            }
        }
        occupancy.push(present.saturating_sub(size));
    }

    Report {
        units: sizes.len(),
        bit_rate: s.bit_rate,
        cpb_size: s.cpb_size,
        underflow,
        overflow,
        occupancy,
    }
}

/// Bits that have arrived `ticks` units of `1 / (90 000 * tick_den)`
/// seconds after the first bit: `bit_rate * ticks / (90 000 * tick_den)`,
/// done as one product before the divide so the truncation happens once
/// rather than compounding per picture — and in 128 bits, which the
/// product of a rate and a time counted that finely needs.
fn arrived_by(s: &Schedule, ticks: u64) -> u64 {
    let per_second = 90_000u128 * u128::from(s.tick_den.max(1));
    (u128::from(s.bit_rate) * u128::from(ticks) / per_second).min(u128::from(u64::MAX)) as u64
}

/// The buffer of a constant-rate stream, walked by the **encoder** as it
/// writes: the one place filler data is sized and a later buffering
/// period's delay is read off.
///
/// The same arithmetic as [`simulate`] — `arrived_by`, the removal of
/// unit `n` at `initial_delay + n` pictures — rather than a second copy of
/// a leaky bucket, for the reason `encode::rc` gives: what keeps the
/// checker independent is its inputs, read off the emitted bytes, not a
/// second implementation that could drift from the first. The encoder
/// sizes filler here from what it wrote and [`verify`] judges the result
/// from what the stream says; if the two disagreed about the buffer the
/// stream would overflow and `verify` would say so.
///
/// Under `cbr_flag` the arrival never pauses (H.264 C.1.1, H.265 C.2.2:
/// each unit's first bit arrives when the last one's last bit did), so
/// the buffer at unit `n`'s removal holds everything that has arrived
/// less everything already removed, and that is at most `CpbSize` only if
/// the stream spent at least what arrived beyond it.
#[derive(Debug, Clone)]
pub struct ConstantRate {
    schedule: Schedule,
    /// Bits of every access unit removed so far, filler included.
    removed: u64,
    /// Access units removed so far: the next one is removed at
    /// `initial_delay + units` pictures.
    units: u64,
}

impl ConstantRate {
    /// The buffer `cpb` declares, full at the first removal — the initial
    /// delay [`crate::encode::h265_syntax::Cpb::initial_removal_delay_90k`]
    /// declares — at `fps_num / fps_den` pictures a second.
    pub fn new(cpb: &crate::encode::h265_syntax::Cpb, (fps_num, fps_den): (u32, u32)) -> ConstantRate {
        let (tick_90k, tick_den) = removal_interval(fps_den, fps_num);
        ConstantRate {
            schedule: Schedule {
                bit_rate: cpb.bit_rate,
                cpb_size: cpb.size,
                cbr: true,
                tick_90k,
                tick_den,
                initial_delay_90k: u64::from(cpb.initial_removal_delay_90k()),
            },
            removed: 0,
            units: 0,
        }
    }

    /// Removal time of unit `n`, in `1 / (90 000 * tick_den)` seconds.
    fn removal(&self, n: u64) -> u64 {
        self.schedule.initial_delay_90k * self.schedule.tick_den + n * self.schedule.tick_90k
    }

    /// What the buffer holds at the next unit's removal: every bit arrived
    /// by then, less every bit already removed. The most that unit may
    /// spend.
    pub fn available(&self) -> u64 {
        arrived_by(&self.schedule, self.removal(self.units)).saturating_sub(self.removed)
    }

    /// The bits of filler the next unit must carry beyond its own `bits`
    /// for the buffer not to overflow at the removal after it: what will
    /// have arrived by then, less what will have been removed, over
    /// `CpbSize`. Zero when the unit spent enough.
    pub fn filler_bits(&self, bits: u64) -> u64 {
        let arrived = arrived_by(&self.schedule, self.removal(self.units + 1));
        arrived.saturating_sub(self.removed + bits).saturating_sub(self.schedule.cpb_size)
    }

    /// The `initial_cpb_removal_delay` a buffering period beginning at the
    /// next unit must carry: `90000 * (t_r(n) - t_af(n - 1))`, the time the
    /// buffer has been filling for that unit, floored. The standard asks
    /// for the value between its floor and its ceiling under a constant
    /// rate (H.264 C.3, H.265 C.4); for the first unit it is exactly the
    /// declared initial delay.
    pub fn initial_delay_90k(&self) -> u32 {
        // t_r(n) is `removal / (90000 * den)` s and t_af(n - 1) is
        // `removed / bit_rate` s, so the difference in 90 kHz ticks is
        // `(bit_rate * removal - 90000 * den * removed) / (bit_rate * den)`.
        let s = &self.schedule;
        let den = u128::from(s.tick_den.max(1));
        let rate = u128::from(s.bit_rate.max(1));
        let num = (rate * u128::from(self.removal(self.units))).saturating_sub(90_000 * den * u128::from(self.removed));
        (num / (rate * den)).min(u128::from(u32::MAX)) as u32
    }

    /// Remove the next unit, `bits` long with its filler.
    pub fn remove(&mut self, bits: u64) {
        self.removed += bits;
        self.units += 1;
    }
}

/// Split an Annex B byte stream into access units and return each one's
/// size **in bits**, in coding order.
///
/// An access unit starts at a parameter set or an SEI, or at the first
/// slice after another slice — the shape this encoder writes, where an IRAP
/// carries VPS, SPS, PPS and optionally a buffering period SEI ahead of its
/// slice and every other picture is one slice alone. The size counted is
/// the whole unit including start codes and NAL headers, because that is
/// what arrives at a decoder and therefore what the buffer holds.
fn access_unit_bits(annexb: &[u8]) -> Vec<u64> {
    let mut starts: Vec<usize> = Vec::new();
    let mut types: Vec<u8> = Vec::new();
    let mut i = 0usize;
    while i + 4 <= annexb.len() {
        if annexb[i] == 0 && annexb[i + 1] == 0 && annexb[i + 2] == 0 && annexb[i + 3] == 1 {
            if i + 4 < annexb.len() {
                starts.push(i);
                types.push((annexb[i + 4] >> 1) & 0x3f);
            }
            i += 4;
        } else {
            i += 1;
        }
    }
    let mut out: Vec<u64> = Vec::new();
    let mut cur = 0u64;
    for k in 0..starts.len() {
        let end = starts.get(k + 1).copied().unwrap_or(annexb.len());
        let n = (end - starts[k]) as u64;
        // An access unit is a run of leading non-slice NALs — parameter
        // sets, SEI — followed by its slices, so the next one begins at
        // the first NAL *after* a slice, whatever kind that NAL is.
        //
        // The first version began a unit at every parameter set, which
        // split each keyframe's VPS, SPS, PPS and SEI into four units of
        // their own: 132 units for a 96-picture clip. The removal schedule
        // then ran 4.4 seconds over a 3.2-second stream, and the extra
        // arrival time overflowed the buffer by more than a megabit. Two
        // wrong numbers that looked like one buffer problem.
        //
        // Filler data, a suffix SEI and the end-of-sequence and
        // end-of-bitstream units follow a picture's slices inside its
        // access unit (7.4.2.4.4) — a constant-rate stream's filler above
        // all, which arrives, and is counted, as part of the unit it
        // stuffs — so they neither begin a unit nor stop the next NAL from
        // beginning one.
        let trails = |t: u8| matches!(t, 36 | 37 | 38 | 40);
        let begins = k == 0 || ((types[k - 1] < 32 || trails(types[k - 1])) && !trails(types[k]));
        if begins && cur != 0 {
            out.push(cur * 8);
            cur = 0;
        }
        cur += n;
    }
    if cur != 0 {
        out.push(cur * 8);
    }
    out
}

/// One clock tick, `num_units_in_tick / time_scale` seconds, in 90 kHz
/// ticks as a fraction in lowest terms: `(90 000 * num_units_in_tick,
/// time_scale)` over their common divisor.
fn removal_interval(num_units: u32, time_scale: u32) -> (u64, u64) {
    let (num, den) = (90_000u64 * u64::from(num_units), u64::from(time_scale));
    let (mut a, mut b) = (num, den);
    while b != 0 {
        (a, b) = (b, a % b);
    }
    (num / a.max(1), den / a.max(1))
}

/// Read the schedule out of the stream itself: the HRD from the SPS's VUI,
/// and the initial removal delay from the first buffering period SEI.
fn schedule_from_stream(annexb: &[u8]) -> Result<Schedule> {
    let mut sps: Option<Sps> = None;
    let mut initial_delay: Option<u64> = None;
    for nal in crate::nal::annexb_nals(annexb) {
        if nal.len() < 3 {
            continue;
        }
        let t = (nal[0] >> 1) & 0x3f;
        let rbsp = crate::nal::unescape_rbsp(&nal[2..]);
        if t == 33 && sps.is_none() {
            sps = Sps::parse(&rbsp).ok();
        } else if t == 39 && initial_delay.is_none() {
            if let Some(s) = sps.as_ref() {
                initial_delay = buffering_period_delay(&rbsp, s);
            }
        }
    }
    let sps = sps.ok_or_else(|| Error::bitstream("HRD: the stream carries no sequence parameter set"))?;
    let vui = sps
        .vui
        .as_ref()
        .ok_or_else(|| Error::bitstream("HRD: the sequence parameter set declares no VUI, so no buffer"))?;
    let hrd = vui
        .hrd
        .ok_or_else(|| Error::bitstream("HRD: the VUI declares no hypothetical reference decoder"))?;
    let (num_units, time_scale) = vui
        .timing
        .ok_or_else(|| Error::bitstream("HRD: the VUI declares no frame rate, so removal times are undefined"))?;
    if time_scale == 0 {
        return Err(Error::bitstream("HRD: time_scale is zero"));
    }
    let initial_delay_90k =
        initial_delay.ok_or_else(|| Error::bitstream("HRD: no buffering period SEI, so the initial removal delay is unknown"))?;
    let (tick_90k, tick_den) = removal_interval(num_units, time_scale);
    Ok(Schedule { bit_rate: hrd.bit_rate, cpb_size: hrd.cpb_size, cbr: hrd.cbr, tick_90k, tick_den, initial_delay_90k })
}

/// `initial_cpb_removal_delay[0]` out of a `buffering_period` SEI payload.
///
/// The field widths come from the SPS's own HRD, which is why this cannot
/// be parsed without it — and why the parser retaining those lengths was a
/// precondition for any of this, not a nicety.
fn buffering_period_delay(rbsp: &[u8], sps: &Sps) -> Option<u64> {
    let hrd = sps.vui.as_ref()?.hrd?;
    let mut r = crate::bitreader::BitReader::new(rbsp);
    // SEI header: payload type and size, each a run of 0xff terminated by a
    // byte below 255.
    let mut ty = 0u32;
    loop {
        let b = r.bits(8);
        ty += b;
        if b != 255 {
            break;
        }
    }
    if ty != 0 {
        return None; // not a buffering period
    }
    let mut size = 0u32;
    loop {
        let b = r.bits(8);
        size += b;
        if b != 255 {
            break;
        }
    }
    let _ = size;
    r.ue(); // bp_seq_parameter_set_id
    // sub_pic_hrd_params_present_flag is 0 in everything this crate writes,
    // so irap_cpb_params_present_flag is present.
    let irap = r.flag();
    if irap {
        // cpb_delay_offset / dpb_delay_offset, widths from the SPS.
        r.bits(hrd.removal_delay_length);
        r.bits(hrd.output_delay_length);
    }
    r.flag(); // concatenation_flag
    r.bits(hrd.removal_delay_length); // au_cpb_removal_delay_delta_minus1
    Some(r.bits(hrd.initial_delay_length) as u64)
}

/// Whether an Annex B stream is H.264 rather than H.265, read off its
/// first NAL unit: an H.265 stream opens with a parameter set whose
/// two-byte header has type 32..=34; an H.264 header is one byte whose
/// low five bits name the type, and no H.264 type read as an H.265 one
/// lands in that range (an SPS's `0x67` reads as 51).
fn is_h264(annexb: &[u8]) -> bool {
    match crate::nal::annexb_nals(annexb).next() {
        Some(nal) if !nal.is_empty() => !matches!((nal[0] >> 1) & 0x3f, 32..=34),
        _ => false,
    }
}

/// The H.264 messages of one SEI NAL that the buffer model reads:
/// `initial_cpb_removal_delay` from a buffering period, and
/// `(cpb_removal_delay, dpb_output_delay)` from a picture timing.
#[derive(Default)]
struct H264Sei {
    buffering_period: Option<u64>,
    pic_timing: Option<(u64, u64)>,
}

/// Parse the SEI messages of one H.264 SEI RBSP — `payloadType` and
/// `payloadSize` each a run of 0xff bytes plus one below 255 — at the
/// field widths the SPS's HRD declared, which is why the SPS must have
/// been seen first.
fn h264_sei(rbsp: &[u8], hrd: &crate::h264::sps::Hrd) -> H264Sei {
    let mut out = H264Sei::default();
    let mut i = 0usize;
    while i < rbsp.len() && rbsp[i] != 0x80 {
        let mut ty = 0usize;
        while i < rbsp.len() && rbsp[i] == 0xff {
            ty += 255;
            i += 1;
        }
        if i >= rbsp.len() {
            break;
        }
        ty += rbsp[i] as usize;
        i += 1;
        let mut size = 0usize;
        while i < rbsp.len() && rbsp[i] == 0xff {
            size += 255;
            i += 1;
        }
        if i >= rbsp.len() {
            break;
        }
        size += rbsp[i] as usize;
        i += 1;
        let end = (i + size).min(rbsp.len());
        let mut r = crate::bitreader::BitReader::new(&rbsp[i..end]);
        match ty {
            0 => {
                r.ue(); // seq_parameter_set_id
                // NalHrdBpPresentFlag: the first (only) SchedSelIdx.
                out.buffering_period = Some(r.bits(hrd.initial_delay_length) as u64);
            }
            1 => {
                // CpbDpbDelaysPresentFlag, set by the NAL HRD.
                let removal = r.bits(hrd.removal_delay_length) as u64;
                let output = r.bits(hrd.output_delay_length) as u64;
                out.pic_timing = Some((removal, output));
            }
            _ => {}
        }
        i = end;
    }
    out
}

/// Walk an H.264 stream: access unit sizes and removal times, with the
/// schedule, all read off the bytes.
///
/// An H.264 access unit is a run of non-VCL NAL units — SEI, parameter
/// sets — followed by its slices (types 1..=5); the next begins at the
/// first NAL after a slice. Each unit's removal time is
/// `t_r(n_b) + t_c * cpb_removal_delay(n)` (C.1.2): its own timing SEI's
/// delay in clock ticks after the removal of the last access unit that
/// carried a buffering period — and the first unit's is the initial
/// delay the buffering period itself carries. A stream with a NAL HRD
/// that leaves a picture without a timing SEI has no removal time for
/// it, and is refused rather than guessed at.
fn h264_units(annexb: &[u8]) -> Result<(Vec<u64>, Vec<u64>, Schedule)> {
    use crate::h264::sps::Sps;
    let mut sps: Option<Sps> = None;
    let mut schedule: Option<Schedule> = None;
    let mut sizes: Vec<u64> = Vec::new();
    let mut removal: Vec<u64> = Vec::new();
    // The unit being accumulated: its bytes so far, and what its SEI said.
    let mut cur_bytes = 0u64;
    let mut cur_sei = H264Sei::default();
    let mut in_slices = false;
    // Whether the unit has reached its trailing NALs — filler data, end of
    // sequence or of stream — after which the next other NAL, a slice
    // included, begins the next unit.
    let mut in_tail = false;
    // Removal time of the last buffering-period unit, the base every
    // `cpb_removal_delay` counts from — like every removal time here, in
    // `1 / (90 000 * tick_den)` seconds.
    let mut base_90k: Option<u64> = None;
    let mut close = |bytes: u64, sei: &H264Sei, schedule: &Schedule, sizes: &mut Vec<u64>, removal: &mut Vec<u64>| -> Result<()> {
        let n = sizes.len();
        let t = match (base_90k, sei.buffering_period, sei.pic_timing) {
            (None, Some(initial), Some((delay, _))) => initial * schedule.tick_den + delay * schedule.tick_90k,
            (None, Some(initial), None) if n == 0 => initial * schedule.tick_den,
            (None, None, _) => {
                return Err(Error::bitstream(
                    "HRD: the first access unit carries no buffering period SEI, so the initial removal delay is unknown",
                ));
            }
            (Some(base), _, Some((delay, _))) => base + delay * schedule.tick_90k,
            (_, _, None) => {
                return Err(Error::bitstream(format!(
                    "HRD: access unit {n} carries no picture timing SEI, so its removal time is undefined"
                )));
            }
        };
        if sei.buffering_period.is_some() {
            base_90k = Some(t);
        }
        sizes.push(bytes * 8);
        removal.push(t);
        Ok(())
    };
    // Start codes are four bytes throughout what this encoder writes; the
    // size counted is the whole unit including them and the NAL headers,
    // because that is what arrives at a decoder.
    let mut i = 0usize;
    let mut starts: Vec<usize> = Vec::new();
    while i + 4 <= annexb.len() {
        if annexb[i] == 0 && annexb[i + 1] == 0 && annexb[i + 2] == 0 && annexb[i + 3] == 1 {
            if i + 4 < annexb.len() {
                starts.push(i);
            }
            i += 4;
        } else {
            i += 1;
        }
    }
    for k in 0..starts.len() {
        let end = starts.get(k + 1).copied().unwrap_or(annexb.len());
        let nal = &annexb[starts[k] + 4..end];
        let t = nal[0] & 0x1f;
        let vcl = (1..=5).contains(&t);
        // Filler data and the end-of-sequence and end-of-stream units
        // follow the primary coded picture inside its access unit
        // (7.4.1.2.3); a constant-rate stream's filler is counted as part
        // of the unit it stuffs, because that is where it arrives.
        let trails = matches!(t, 10..=12);
        if (in_slices && !vcl && !trails) || (in_tail && !trails) {
            // A new unit begins.
            let Some(s) = schedule.as_ref() else {
                return Err(Error::bitstream("HRD: slices before any sequence parameter set"));
            };
            close(cur_bytes, &cur_sei, s, &mut sizes, &mut removal)?;
            cur_bytes = 0;
            cur_sei = H264Sei::default();
            in_slices = false;
            in_tail = false;
        }
        cur_bytes += (end - starts[k]) as u64;
        in_slices |= vcl;
        in_tail |= in_slices && trails;
        match t {
            7 if sps.is_none() => {
                let parsed = Sps::parse(&crate::nal::unescape_rbsp(&nal[1..]))?;
                let vui = parsed.vui.as_ref().ok_or_else(|| {
                    Error::bitstream("HRD: the sequence parameter set declares no VUI, so no buffer")
                })?;
                let hrd = vui.nal_hrd.ok_or_else(|| {
                    Error::bitstream("HRD: the VUI declares no hypothetical reference decoder")
                })?;
                let (num_units, time_scale) = vui.timing.ok_or_else(|| {
                    Error::bitstream("HRD: the VUI declares no clock, so removal times are undefined")
                })?;
                if time_scale == 0 {
                    return Err(Error::bitstream("HRD: time_scale is zero"));
                }
                let (tick_90k, tick_den) = removal_interval(num_units, time_scale);
                schedule = Some(Schedule {
                    bit_rate: hrd.bit_rate,
                    cpb_size: hrd.cpb_size,
                    cbr: hrd.cbr,
                    tick_90k,
                    tick_den,
                    initial_delay_90k: 0,
                });
                sps = Some(parsed);
            }
            6 => {
                if let Some(hrd) = sps.as_ref().and_then(|s| s.vui.as_ref()).and_then(|v| v.nal_hrd) {
                    let sei = h264_sei(&crate::nal::unescape_rbsp(&nal[1..]), &hrd);
                    if sei.buffering_period.is_some() {
                        cur_sei.buffering_period = sei.buffering_period;
                    }
                    if sei.pic_timing.is_some() {
                        cur_sei.pic_timing = sei.pic_timing;
                    }
                }
            }
            _ => {}
        }
    }
    let Some(mut s) = schedule else {
        return Err(Error::bitstream("HRD: the stream carries no sequence parameter set"));
    };
    if cur_bytes != 0 {
        close(cur_bytes, &cur_sei, &s, &mut sizes, &mut removal)?;
    }
    // The report quotes the initial delay through the schedule; it is
    // the first unit's removal time.
    s.initial_delay_90k = removal.first().copied().unwrap_or(0);
    Ok((sizes, removal, s))
}

/// Verify an Annex B stream — H.264 or H.265, told apart by its first
/// NAL — against the buffer **it declares**.
///
/// Every number comes from the bytes: the rate and buffer size from the
/// SPS's VUI, the clock beside it, the initial delay from the buffering
/// period SEI — and, for H.264, each picture's removal time from its own
/// timing SEI. Nothing is passed in, so nothing can be assumed.
pub fn verify(annexb: &[u8]) -> Result<Report> {
    if is_h264(annexb) {
        let (sizes, removal, schedule) = h264_units(annexb)?;
        if sizes.is_empty() {
            return Err(Error::bitstream("HRD: the stream carries no access units"));
        }
        return Ok(simulate_at(&sizes, &removal, &schedule));
    }
    let schedule = schedule_from_stream(annexb)?;
    let sizes = access_unit_bits(annexb);
    if sizes.is_empty() {
        return Err(Error::bitstream("HRD: the stream carries no access units"));
    }
    Ok(simulate(&sizes, &schedule))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A schedule whose arithmetic is easy to do by hand: 90 000 bits a
    /// second at 90 000 ticks a second is one bit per tick, and 30 pictures
    /// a second is 3000 ticks and therefore 3000 bits between removals.
    fn sched(cpb: u64, cbr: bool) -> Schedule {
        Schedule { bit_rate: 90_000, cpb_size: cpb, cbr, tick_90k: 3_000, tick_den: 1, initial_delay_90k: cpb }
    }

    /// A stream that spends exactly what arrives never moves the buffer.
    #[test]
    fn spending_the_arrival_rate_exactly_conforms() {
        let r = simulate(&[3_000; 40], &sched(30_000, true));
        assert!(r.conforms(), "{r:?}");
        // Occupancy is recorded *after* each removal, so a buffer that
        // starts full and gives up one picture's worth sits one picture
        // below full for ever after — steady, which is the point.
        assert!(r.occupancy.iter().all(|&f| f == 27_000), "{:?}", &r.occupancy[..5]);
    }

    /// **The check must be able to fail.** One access unit larger than the
    /// whole buffer cannot be removed however long a decoder waits, and it
    /// is the case a small buffer makes common: a keyframe that does not
    /// fit.
    #[test]
    fn an_access_unit_larger_than_the_buffer_underflows() {
        let mut sizes = vec![3_000u64; 20];
        sizes[7] = 40_000;
        let r = simulate(&sizes, &sched(30_000, true));
        assert!(!r.conforms(), "a unit above the buffer size must not conform");
        let (n, short) = r.underflow.expect("underflow");
        assert_eq!(n, 7, "the failure should be at the oversized unit");
        assert_eq!(short, 10_000, "40000 bits removed from a 30000-bit buffer is 10000 short");
    }

    /// Underflow by accumulation rather than by one big picture: spending
    /// consistently above the arrival rate drains the buffer, and the
    /// failure arrives some pictures later.
    #[test]
    fn spending_above_the_rate_drains_the_buffer_and_then_fails() {
        // 4000 bits a picture against 3000 arriving: 1000 lost each time,
        // so a 30000-bit buffer is empty after thirty.
        let r = simulate(&[4_000; 40], &sched(30_000, true));
        assert!(!r.conforms(), "spending above the rate forever must fail");
        let (n, _) = r.underflow.expect("underflow");
        assert!((28..=34).contains(&n), "the drain should fail around picture 30, not {n}");
        // And it is a drain, not a cliff: occupancy falls monotonically
        // until it hits the floor.
        assert!(r.occupancy[0] > r.occupancy[10] && r.occupancy[10] > r.occupancy[20], "{:?}", &r.occupancy[..21]);
    }

    /// Under a constant rate, spending consistently *below* it overflows —
    /// the arrival cannot pause, so the bits have nowhere to go. Under a
    /// variable rate the same stream is fine, which is the whole difference
    /// the flag makes.
    #[test]
    fn underspending_overflows_only_under_a_constant_rate() {
        let cbr = simulate(&[1_000; 60], &sched(30_000, true));
        assert!(cbr.overflow.is_some(), "constant rate: underspending must overflow");
        assert!(cbr.underflow.is_none(), "constant rate: underspending must not underflow");

        let vbr = simulate(&[1_000; 60], &sched(30_000, false));
        assert!(vbr.conforms(), "variable rate: a full buffer is not an error — {vbr:?}");
    }

    /// The initial delay is what buys the first picture its room: the same
    /// stream conforms when the decoder waits for the buffer to fill and
    /// fails when it does not wait at all.
    #[test]
    fn the_initial_delay_is_what_the_first_picture_spends() {
        let sizes = [20_000u64, 3_000, 3_000, 3_000];
        let waited = Schedule { initial_delay_90k: 30_000, ..sched(30_000, false) };
        assert!(simulate(&sizes, &waited).conforms(), "with a full buffer the first unit fits");

        let eager = Schedule { initial_delay_90k: 3_000, ..sched(30_000, false) };
        let r = simulate(&sizes, &eager);
        assert_eq!(r.underflow, Some((0, 17_000)), "starting after one tick only 3000 bits have arrived");
    }

    /// A stream with no VUI, or a VUI with no HRD, is not a stream that
    /// failed the buffer — it is one that declared no buffer, and saying so
    /// is different from saying it conformed. Both codecs. Every stream
    /// these encoders write carries a VUI with its clock, so one that
    /// declares no buffer is the second kind: a clock and no HRD.
    #[test]
    fn a_stream_that_declares_no_buffer_is_refused_rather_than_passed() {
        use crate::encode::h265_syntax::{Geometry, write_sps};
        use crate::encode::Config;
        let cfg = Config { width: 64, height: 64, ..Config::default() };
        let g = Geometry::new(&cfg);
        let sps = crate::encode::h265_syntax::annexb(33, &write_sps(&cfg, &g, 8, None));
        let err = verify(&sps).expect_err("no HRD means no verdict");
        let s = format!("{err}");
        assert!(s.contains("no hypothetical reference decoder"), "{s}");

        let g = crate::encode::h264_syntax::Geometry::new(&cfg);
        let sps = crate::encode::h264_syntax::annexb(
            crate::encode::h264_syntax::NAL_SPS,
            3,
            &crate::encode::h264_syntax::write_sps(&cfg, &g, 16, 16, None),
        );
        assert!(is_h264(&sps));
        let err = verify(&sps).expect_err("H.264: no HRD means no verdict");
        let s = format!("{err}");
        assert!(s.contains("no hypothetical reference decoder"), "{s}");
    }

    /// Every frame rate a caller can name round-trips through both
    /// parameter sets' VUI exactly, in lowest terms: H.264's field clock
    /// (`num_units_in_tick` the rate's denominator, `time_scale` twice its
    /// numerator), H.265's picture clock (denominator over numerator). A
    /// whole-number rate writes what it always did — 30 is 1 over 60 and 1
    /// over 30 — and 60/2 is written as 30.
    #[test]
    fn fractional_frame_rates_round_trip_through_the_vui() {
        use crate::encode::{Config, RateControl};
        for (fps, fps_den, num, den) in [
            (30000, 1001, 30000, 1001),
            (24000, 1001, 24000, 1001),
            (60000, 1001, 60000, 1001),
            (25, 1, 25, 1),
            (30, 1, 30, 1),
            (25, 2, 25, 2),
            (60, 2, 30, 1),
        ] {
            let tag = format!("{fps}/{fps_den}");
            let cfg = Config {
                width: 64,
                height: 64,
                fps,
                fps_den,
                rate: RateControl::Bitrate { bps: 1_000_000 },
                cpb_ms: 500,
                ..Config::default()
            };
            assert_eq!(cfg.frame_rate(), (num, den), "{tag}");
            let cpb = crate::encode::h265_syntax::Cpb::new(1_000_000, 500).unwrap();
            let g4 = crate::encode::h264_syntax::Geometry::new(&cfg);
            let sps4 = crate::encode::h264_syntax::write_sps(&cfg, &g4, 16, 16, Some(&cpb));
            let sps4 = crate::h264::Sps::parse(&crate::nal::unescape_rbsp(&sps4)).unwrap();
            assert_eq!(sps4.vui.as_ref().and_then(|v| v.timing), Some((den, 2 * num)), "{tag}: H.264 (num_units_in_tick, time_scale)");
            let g5 = crate::encode::h265_syntax::Geometry::new(&cfg);
            let sps5 = crate::encode::h265_syntax::write_sps(&cfg, &g5, 8, Some(&cpb));
            let sps5 = Sps::parse(&crate::nal::unescape_rbsp(&sps5)).unwrap();
            assert_eq!(sps5.vui.as_ref().and_then(|v| v.timing), Some((den, num)), "{tag}: H.265 (num_units_in_tick, time_scale)");
        }
        let zero = Config { width: 64, height: 64, fps_den: 0, ..Config::default() };
        assert!(zero.validate().unwrap_err().to_string().contains("fps_den"));
    }

    /// Every stream carries its clock, buffer or none — a raw stream has
    /// its rate nowhere else — and H.264's `fixed_frame_rate_flag` says so
    /// exactly where it is true: a progressive stream, and not an
    /// interlaced one, whose field pictures claim nothing.
    #[test]
    fn every_stream_carries_its_clock() {
        use crate::encode::{Config, FieldOrder};
        for (fps, fps_den) in [(30, 1), (25, 1), (30000, 1001), (60000, 1001)] {
            let cfg = Config { width: 64, height: 64, fps, fps_den, ..Config::default() };
            let (num, den) = cfg.frame_rate();
            let g4 = crate::encode::h264_syntax::Geometry::new(&cfg);
            let sps4 = crate::encode::h264_syntax::write_sps(&cfg, &g4, 16, 16, None);
            let vui4 = crate::h264::Sps::parse(&crate::nal::unescape_rbsp(&sps4)).unwrap().vui.expect("an H.264 VUI");
            assert_eq!((vui4.timing, vui4.fixed_frame_rate), (Some((den, 2 * num)), true), "H.264 {fps}/{fps_den}");
            assert!(vui4.nal_hrd.is_none(), "H.264 {fps}/{fps_den}: no buffer, no HRD");
            let g5 = crate::encode::h265_syntax::Geometry::new(&cfg);
            let sps5 = crate::encode::h265_syntax::write_sps(&cfg, &g5, 8, None);
            let vui5 = Sps::parse(&crate::nal::unescape_rbsp(&sps5)).unwrap().vui.expect("an H.265 VUI");
            assert_eq!(vui5.timing, Some((den, num)), "H.265 {fps}/{fps_den}");
            assert!(vui5.hrd.is_none(), "H.265 {fps}/{fps_den}: no buffer, no HRD");
        }
        let cfg = Config { width: 64, height: 64, interlace: Some(FieldOrder::TopFirst), ..Config::default() };
        let g = crate::encode::h264_syntax::Geometry::new(&cfg);
        let sps = crate::encode::h264_syntax::write_sps(&cfg, &g, 16, 16, None);
        let vui = crate::h264::Sps::parse(&crate::nal::unescape_rbsp(&sps)).unwrap().vui.expect("a VUI");
        assert_eq!((vui.timing, vui.fixed_frame_rate), (Some((1, 60)), false), "interlaced: the clock, no fixed-rate claim");
    }

    /// At 29.97 H.264's clock tick is 1501.5 of the 90 kHz ones — 3003/2,
    /// kept exactly — so the thousandth picture of a stream is removed at
    /// exactly 1000 * 1001 / 30000 seconds after the first, not 250
    /// ticks early as the rounded tick would have it. H.265's picture
    /// tick at 29.97 is a whole 3003.
    #[test]
    fn a_fractional_clock_is_kept_exact() {
        assert_eq!(removal_interval(1001, 60_000), (3003, 2));
        assert_eq!(removal_interval(1001, 30_000), (3003, 1));
        assert_eq!(removal_interval(1, 60), (1500, 1));
        assert_eq!(removal_interval(2, 50), (3600, 1), "12.5 pictures a second: 7200 a tick pair");
        let s = Schedule { bit_rate: 90_000, cpb_size: 1 << 30, cbr: false, tick_90k: 3003, tick_den: 2, initial_delay_90k: 9000 };
        // A frame is two ticks: unit n of a fixed-rate walk is removed
        // `n` ticks after the first, which for H.264 is half a frame; the
        // H.264 walk counts two per frame through cpb_removal_delay.
        let sizes = vec![1u64; 2001];
        let r = simulate(&sizes, &s);
        assert!(r.conforms());
        // After 2000 ticks (1000 frames) at 90 000 bits a second, exactly
        // 90 000 * (0.1 + 1000 * 1001 / 30000) bits have arrived.
        let arrived = 9_000 + 1000 * 3003;
        assert_eq!(r.occupancy[2000], arrived - 2001, "the buffer after the last removal, to the bit");
    }

    /// Streams coded at 29.97 by both encoders, against a declared
    /// buffer, conform to it — read back, walked and judged at the exact
    /// clock their VUI declares.
    #[test]
    fn streams_at_29_97_conform_to_their_buffer() {
        use crate::encode::{Config, RateControl};
        let frames: Vec<Vec<u8>> = (0..24)
            .map(|i| {
                let mut f = vec![128u8; 64 * 64 * 3 / 2];
                for y in 0..64 {
                    for x in 0..64 {
                        f[y * 64 + x] = (((x + 2 * i) * 5) ^ ((y + i) * 3)) as u8;
                    }
                }
                f
            })
            .collect();
        let cfg = Config {
            width: 64,
            height: 64,
            fps: 30000,
            fps_den: 1001,
            gop: 12,
            rate: RateControl::Bitrate { bps: 96_000 },
            cpb_ms: 500,
            ..Config::default()
        };
        let mut s264 = Vec::new();
        let mut e = crate::encode::h264::H264Encoder::new(cfg.clone()).unwrap();
        for f in &frames {
            s264.extend(e.push(f).unwrap().into_iter().flat_map(|a| a.data));
        }
        s264.extend(e.flush().unwrap().into_iter().flat_map(|a| a.data));
        let mut s265 = Vec::new();
        let mut e = crate::encode::h265::H265Encoder::new(Config { max_cu_depth: Some(0), ..cfg }).unwrap();
        for f in &frames {
            s265.extend(e.push(f).unwrap().into_iter().flat_map(|a| a.data));
        }
        s265.extend(e.flush().unwrap().into_iter().flat_map(|a| a.data));
        let (_, _, sched) = h264_units(&s264).unwrap();
        assert_eq!((sched.tick_90k, sched.tick_den), (3003, 2), "H.264 at 29.97: a 1501.5-tick field clock");
        assert_eq!(schedule_from_stream(&s265).map(|s| (s.tick_90k, s.tick_den)).unwrap(), (3003, 1));
        for (codec, stream) in [("H.264", &s264), ("H.265", &s265)] {
            let r = verify(stream).unwrap_or_else(|err| panic!("{codec}: {err}"));
            assert_eq!(r.units, frames.len(), "{codec}");
            assert!(r.conforms(), "{codec} at 29.97 breaks its own buffer: {r:?}");
        }
    }

    /// An H.264 stream's schedule is read off its own SEI: the initial
    /// delay from the buffering period, each unit's removal time from its
    /// timing SEI's `cpb_removal_delay` — with a second buffering period
    /// rebasing the count — and the units split at the slices. Built from
    /// the encoder's own writers, then walked, and checked against the
    /// same walk done by hand.
    #[test]
    fn an_h264_schedule_is_read_off_the_stream() {
        use crate::encode::h264_syntax::{
            Cpb, Geometry, NAL_IDR, NAL_PPS, NAL_SEI, NAL_SLICE, NAL_SPS, annexb, write_pps,
            write_buffering_period_sei, write_pic_timing_sei, write_sps,
        };
        use crate::encode::{Config, RateControl};
        let cfg = Config {
            width: 64,
            height: 64,
            fps: 30,
            rate: RateControl::Bitrate { bps: 90_000 },
            cpb_ms: 500,
            ..Config::default()
        };
        let g = Geometry::new(&cfg);
        let cpb = Cpb::new(90_000, 500).unwrap();
        // Five pictures: an IDR, two P, an IDR (a new buffering period,
        // its delay counted from the first), one P — two clock ticks per
        // frame, slices padded to known sizes.
        let slice = |bytes: usize| -> Vec<u8> { vec![0x55u8; bytes] };
        let mut stream = Vec::new();
        let unit = |idr: bool, delay: u32, bytes: usize, out: &mut Vec<u8>| {
            if idr {
                out.extend_from_slice(&annexb(NAL_SPS, 3, &write_sps(&cfg, &g, 16, 16, Some(&cpb))));
                out.extend_from_slice(&annexb(NAL_PPS, 3, &write_pps(&cfg, 26)));
                out.extend_from_slice(&annexb(NAL_SEI, 0, &write_buffering_period_sei(&cpb)));
            }
            out.extend_from_slice(&annexb(NAL_SEI, 0, &write_pic_timing_sei(&cpb, delay, 0)));
            out.extend_from_slice(&annexb(if idr { NAL_IDR } else { NAL_SLICE }, 3, &slice(bytes)));
        };
        unit(true, 0, 3000, &mut stream);
        unit(false, 2, 200, &mut stream);
        unit(false, 4, 200, &mut stream);
        unit(true, 6, 2500, &mut stream);
        unit(false, 2, 200, &mut stream);

        let (sizes, removal, s) = h264_units(&stream).expect("a readable schedule");
        assert_eq!(sizes.len(), 5, "five access units");
        assert_eq!(s.bit_rate, cpb.bit_rate);
        assert_eq!(s.cpb_size, cpb.size);
        // 60 ticks a second: one clock tick is 1500 of the 90 kHz.
        assert_eq!(s.tick_90k, 1500);
        let t0 = cpb.initial_removal_delay_90k() as u64;
        assert_eq!(removal, vec![t0, t0 + 3000, t0 + 6000, t0 + 9000, t0 + 9000 + 3000]);
        // The unit sizes are the whole units: for an IDR, SPS + PPS + two
        // SEI + slice, start codes and headers included.
        assert!(sizes[0] > 3000 * 8 && sizes[1] > 200 * 8 && sizes[1] < 300 * 8, "{sizes:?}");
        let r = verify(&stream).unwrap();
        assert_eq!(r, simulate_at(&sizes, &removal, &s));
        assert!(r.conforms(), "{r:?}");

        // Drop a picture's timing SEI and the stream has no removal time
        // for it: refused, not guessed. (The PPS is what makes the second
        // slice a unit of its own: a unit begins at the first non-slice
        // NAL after a slice.)
        let mut broken = Vec::new();
        unit(true, 0, 300, &mut broken);
        broken.extend_from_slice(&annexb(NAL_PPS, 3, &write_pps(&cfg, 26)));
        broken.extend_from_slice(&annexb(NAL_SLICE, 3, &slice(300)));
        let err = verify(&broken).expect_err("no timing SEI, no removal time");
        assert!(format!("{err}").contains("timing"), "{err}");
    }

    /// The encoder's walk sizes filler to exactly what the checker would
    /// call overflow: over a run of access units that spend far below the
    /// rate, the stuffed sizes walked by [`simulate`] never overflow and
    /// leave the buffer full to within the one filler unit's rounding at
    /// every removal — and a unit that spent its share needs none.
    #[test]
    fn a_constant_rate_walk_stuffs_exactly_the_overflow() {
        use crate::encode::h265_syntax::Cpb;
        let cpb = Cpb::new(90_000, 500).unwrap().with_cbr(true);
        let mut buffer = ConstantRate::new(&cpb, (30, 1));
        assert_eq!(buffer.initial_delay_90k(), cpb.initial_removal_delay_90k(), "the first period's delay is the declared one");
        let spent = [20_000u64, 100, 100, 3_000, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100];
        let mut stuffed = Vec::new();
        for &bits in &spent {
            assert!(bits <= buffer.available(), "{bits} bits against {} available", buffer.available());
            let filler = buffer.filler_bits(bits);
            // Whole bytes, never below the smallest filler unit: what the
            // encoder's NAL would add.
            let filler = if filler > 0 { filler.div_ceil(8).max(6) * 8 } else { 0 };
            buffer.remove(bits + filler);
            stuffed.push(bits + filler);
        }
        let schedule = Schedule { bit_rate: cpb.bit_rate, cpb_size: cpb.size, cbr: true, tick_90k: 3_000, tick_den: 1, initial_delay_90k: u64::from(cpb.initial_removal_delay_90k()) };
        let r = simulate(&stuffed, &schedule);
        assert!(r.conforms(), "{r:?}");
        let unstuffed = simulate(&spent, &schedule);
        assert!(unstuffed.overflow.is_some(), "the same sizes without filler must overflow, or this proved nothing");
        // After the keyframe drains it, a unit at the arrival rate (3000
        // bits a picture) needs no filler until the buffer is full again.
        let mut fresh = ConstantRate::new(&cpb, (30, 1));
        fresh.remove(20_000);
        assert_eq!(fresh.filler_bits(3_000), 0);
    }

    /// Streams coded at a constant rate by both encoders, of content that
    /// spends far below the rate (a still picture) and of content that
    /// needs all of it (noise that moves), conform to the buffer they
    /// declare — `cbr_flag` set, read back through the production parsers,
    /// walked from the emitted bytes with the filler counted — neither
    /// overflowing nor underflowing, at an average rate within a percent of
    /// the declared one over the whole seconds coded.
    ///
    /// Beside the buffer: every filler unit is a well-formed filler NAL
    /// placed after its access unit's slice; each later buffering period
    /// carries the delay the buffer's actual fullness gives it; the stream
    /// decodes to the encoder's reconstructions exactly, with the filler
    /// and with it stripped; and the same configuration without `cbr`
    /// writes no filler and no `cbr_flag`.
    #[test]
    fn constant_rate_streams_stuff_and_conform_to_their_buffer() {
        use crate::encode::{Access, Config, RateControl};
        const W: usize = 64;
        const H: usize = 64;
        const FPS: u32 = 30;
        const SECONDS: usize = 10;
        let n = FPS as usize * SECONDS;
        let mut seed = 0x2545_f491u32;
        let mut noise = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        let still: Vec<Vec<u8>> = {
            let mut f = vec![128u8; W * H * 3 / 2];
            for y in 0..H {
                for x in 0..W {
                    f[y * W + x] = ((x * 3) ^ (y * 5)) as u8;
                }
            }
            vec![f; n]
        };
        let moving: Vec<Vec<u8>> = (0..n)
            .map(|i| {
                let mut f = vec![128u8; W * H * 3 / 2];
                for y in 0..H {
                    for x in 0..W {
                        let pattern = ((x + 3 * i) * 7) ^ ((y + 2 * i) * 5);
                        f[y * W + x] = (pattern as u32 ^ (noise() & 0x1f)) as u8;
                    }
                }
                for c in &mut f[W * H..] {
                    *c = 120 + (noise() & 0xf) as u8;
                }
                f
            })
            .collect();
        // Motion that stops for a second in the middle and starts again:
        // the controller has to meet the stop with filler and the restart
        // without underflowing.
        //
        // The still pictures walk the quantiser down, and the first moving
        // picture after them is planned far too fine. Until the last
        // attempt went to quantiser 51 (`rc::MAX_ATTEMPTS`) a stop of two
        // seconds was refused by name, the law's three codings all missing
        // the 300 ms buffer; a three-second stop now conforms too.
        let mixed: Vec<Vec<u8>> = (0..n).map(|i| if (4 * FPS as usize..5 * FPS as usize).contains(&i) { still[0].clone() } else { moving[i].clone() }).collect();
        let encode = |h264: bool, cfg: &Config, frames: &[Vec<u8>]| -> (Vec<Access>, Vec<Vec<u8>>) {
            let mut units = Vec::new();
            if h264 {
                let mut e = crate::encode::h264::H264Encoder::new(cfg.clone()).unwrap();
                for f in frames {
                    units.extend(e.push(f).unwrap());
                }
                units.extend(e.flush().unwrap());
                (units, e.reconstructions().to_vec())
            } else {
                let mut e = crate::encode::h265::H265Encoder::new(Config { max_cu_depth: Some(0), ..cfg.clone() }).unwrap();
                for f in frames {
                    units.extend(e.push(f).unwrap());
                }
                units.extend(e.flush().unwrap());
                (units, e.reconstructions().to_vec())
            }
        };
        let decode = |h264: bool, stream: &[u8]| -> Vec<Vec<u8>> {
            let mut out = Vec::new();
            if h264 {
                let mut d = crate::h264::H264Decoder::new();
                d.push_annexb(stream).unwrap();
                d.flush().unwrap();
                assert_eq!(d.warnings(), 0);
                while let Some(p) = d.next_picture() {
                    out.push(p.into_packed());
                }
            } else {
                let mut d = crate::hevc::HevcDecoder::new();
                d.push_annexb(stream).unwrap();
                d.flush().unwrap();
                assert_eq!(d.warnings(), 0);
                while let Some(p) = d.next_picture() {
                    out.push(p.into_packed());
                }
            }
            out
        };
        for (h264, codec) in [(true, "H.264"), (false, "H.265")] {
            for (content, frames, bframes) in [("still", &still, 0u32), ("moving", &moving, 0), ("moving IPB", &moving, 2), ("mixed", &mixed, 0)] {
                let tag = format!("{codec} {content}");
                let cfg = Config {
                    width: W as u32,
                    height: H as u32,
                    fps: FPS,
                    gop: 60,
                    bframes,
                    rate: RateControl::Bitrate { bps: 160_000 },
                    cpb_ms: 300,
                    cbr: true,
                    ..Config::default()
                };
                let (units, recon) = encode(h264, &cfg, frames);
                assert_eq!(units.len(), n, "{tag}");
                let stream: Vec<u8> = units.iter().flat_map(|u| u.data.iter().copied()).collect();

                // The buffer, read off the stream.
                let r = verify(&stream).unwrap_or_else(|err| panic!("{tag}: {err}"));
                assert_eq!(r.units, n, "{tag}: the filler must not split an access unit");
                assert!(r.overflow.is_none(), "{tag}: overflows its constant-rate buffer: {:?}", r.overflow);
                assert!(r.underflow.is_none(), "{tag}: underflows its buffer: {:?}", r.underflow);

                // cbr_flag, and the declared rate the average is held to.
                let first = crate::nal::annexb_nals(&stream).find(|nal| if h264 { nal[0] & 0x1f == 7 } else { (nal[0] >> 1) & 0x3f == 33 }).unwrap();
                let (cbr, rate, size) = if h264 {
                    let sps = crate::h264::Sps::parse(&crate::nal::unescape_rbsp(&first[1..])).unwrap();
                    let hrd = sps.vui.and_then(|v| v.nal_hrd).unwrap();
                    (hrd.cbr, hrd.bit_rate, hrd.cpb_size)
                } else {
                    let sps = Sps::parse(&crate::nal::unescape_rbsp(&first[2..])).unwrap();
                    let hrd = sps.vui.and_then(|v| v.hrd).unwrap();
                    (hrd.cbr, hrd.bit_rate, hrd.cpb_size)
                };
                assert!(cbr, "{tag}: cbr_flag clear");
                let total: u64 = units.iter().map(|u| u.data.len() as u64 * 8).sum();
                let achieved = total as f64 / SECONDS as f64;
                assert!((achieved / rate as f64 - 1.0).abs() < 0.01, "{tag}: {achieved:.0} bits a second against a declared {rate}");

                // Every filler unit: after a slice of its access unit,
                // nal_ref_idc 0, a run of 0xFF and the trailing bits.
                let mut filler = 0u64;
                for u in &units {
                    let nals: Vec<&[u8]> = crate::nal::annexb_nals(&u.data).collect();
                    let is_filler = |nal: &[u8]| if h264 { nal[0] & 0x1f == 12 } else { (nal[0] >> 1) & 0x3f == 38 };
                    let is_slice = |nal: &[u8]| if h264 { (1..=5).contains(&(nal[0] & 0x1f)) } else { (nal[0] >> 1) & 0x3f < 32 };
                    for (k, nal) in nals.iter().enumerate() {
                        if !is_filler(nal) {
                            continue;
                        }
                        assert!(k > 0 && is_slice(nals[k - 1]), "{tag}: filler before its picture's slice");
                        assert_eq!(k, nals.len() - 1, "{tag}: filler is the last NAL of its unit");
                        let body = if h264 {
                            assert_eq!(nal[0], 12, "{tag}: filler nal_ref_idc must be 0");
                            &nal[1..]
                        } else {
                            assert_eq!(&nal[..2], &[38 << 1, 1], "{tag}: filler header");
                            &nal[2..]
                        };
                        let (last, run) = body.split_last().unwrap();
                        assert!(*last == 0x80 && run.iter().all(|&b| b == 0xff), "{tag}: filler payload");
                        filler += (nal.len() + 4) as u64 * 8;
                    }
                }
                match content {
                    "still" => assert!(filler > total / 2, "{tag}: a still picture is mostly filler at this rate ({filler} of {total})"),
                    "mixed" => assert!(filler > 0 && filler * 5 < total, "{tag}: the stop is stuffed, the motion is not ({filler} of {total})"),
                    _ => assert!(filler * 50 < total, "{tag}: filler is the exception, not the norm ({filler} of {total} bits)"),
                }

                // Each buffering period's delay: the time the buffer has
                // filled for its unit, from the sizes as they arrived.
                let (tick, den) = removal_interval(cfg.fps_den, cfg.fps);
                let mut t0: Option<u64> = None;
                let mut removed = 0u64;
                let mut periods = 0;
                for u in &units {
                    let delay = crate::nal::annexb_nals(&u.data).find_map(|nal| {
                        if h264 {
                            if nal[0] & 0x1f != 6 {
                                return None;
                            }
                            let sps = crate::h264::Sps::parse(&crate::nal::unescape_rbsp(&first[1..])).unwrap();
                            h264_sei(&crate::nal::unescape_rbsp(&nal[1..]), &sps.vui.unwrap().nal_hrd.unwrap()).buffering_period
                        } else {
                            if (nal[0] >> 1) & 0x3f != 39 {
                                return None;
                            }
                            let sps = Sps::parse(&crate::nal::unescape_rbsp(&first[2..])).unwrap();
                            buffering_period_delay(&crate::nal::unescape_rbsp(&nal[2..]), &sps)
                        }
                    });
                    if let Some(delay) = delay {
                        let t0 = *t0.get_or_insert(delay);
                        let removal = u128::from(t0 * den + u.encode_index * tick);
                        let want = (u128::from(rate) * removal - 90_000 * u128::from(den) * u128::from(removed)) / (u128::from(rate) * u128::from(den));
                        assert_eq!(u128::from(delay), want, "{tag}: the buffering period at unit {}", u.encode_index);
                        assert!(delay > 0 && delay <= size * 90_000 / rate, "{tag}: a delay the buffer cannot hold");
                        periods += 1;
                    }
                    removed += u.data.len() as u64 * 8;
                }
                assert_eq!(periods, n / 60, "{tag}: a buffering period at every IDR");

                // Bit-exact against the reconstructions, with the filler and
                // without it.
                let decoded = decode(h264, &stream);
                assert_eq!(decoded.len(), n, "{tag}");
                for u in &units {
                    assert!(decoded[u.display as usize] == recon[u.encode_index as usize], "{tag}: picture {} decoded differently", u.display);
                }
                let mut stripped = Vec::new();
                for nal in crate::nal::annexb_nals(&stream) {
                    if (h264 && nal[0] & 0x1f == 12) || (!h264 && (nal[0] >> 1) & 0x3f == 38) {
                        continue;
                    }
                    stripped.extend_from_slice(&[0, 0, 0, 1]);
                    stripped.extend_from_slice(nal);
                }
                assert_eq!(stripped.len() as u64 + filler / 8, stream.len() as u64, "{tag}: the filler stripped is the filler counted");
                if filler > 0 {
                    assert!(decode(h264, &stripped) == decoded, "{tag}: the filler changed what decodes");
                }

                // And without `cbr`: no flag, no filler.
                let (plain, _) = encode(h264, &Config { cbr: false, ..cfg.clone() }, &frames[..60]);
                let plain: Vec<u8> = plain.iter().flat_map(|u| u.data.iter().copied()).collect();
                let r = verify(&plain).unwrap();
                assert!(r.conforms(), "{tag} without cbr: {r:?}");
                for nal in crate::nal::annexb_nals(&plain) {
                    let t = if h264 { nal[0] & 0x1f } else { (nal[0] >> 1) & 0x3f };
                    assert_ne!(t, if h264 { 12 } else { 38 }, "{tag}: filler without cbr");
                }
                eprintln!("{tag}: {achieved:.0} b/s against {rate}, filler {:.2}% of the stream", 100.0 * filler as f64 / total as f64);
            }
        }
    }

    /// A constant rate is a property of a declared buffer: asked for
    /// without one, or without a rate for it to be, it is refused by name.
    #[test]
    fn a_constant_rate_without_a_buffer_is_refused() {
        use crate::encode::{Config, RateControl};
        let base = Config { width: 64, height: 64, cbr: true, ..Config::default() };
        for cfg in [
            Config { rate: RateControl::Bitrate { bps: 100_000 }, cpb_ms: 0, ..base.clone() },
            Config { rate: RateControl::ConstantQp(26), cpb_ms: 0, ..base.clone() },
        ] {
            let err = cfg.validate().unwrap_err().to_string();
            assert!(err.contains("constant bit rate"), "{err}");
            assert!(crate::encode::h264::H264Encoder::new(cfg.clone()).is_err());
            assert!(crate::encode::h265::H265Encoder::new(cfg).is_err());
        }
        let ok = Config { rate: RateControl::Bitrate { bps: 100_000 }, cpb_ms: 500, ..base };
        assert!(ok.validate().is_ok());
    }

    /// A constant-rate stream keeps the model it declares whatever it
    /// carries: pure noise (more than any quantiser can bring down to the
    /// rate at a fine one), hard scene cuts that do not fall on a keyframe,
    /// fast motion and a held picture, each in H.264, H.265, and H.265
    /// planning through its lookahead, at a one-second buffer and a short
    /// one.
    ///
    /// What the standards require of it (H.264 C.1 / C.3, H.265 C.1 /
    /// C.4): the coded picture buffer neither underflows nor, with
    /// `cbr_flag` set, overflows — checked by [`verify`] off the bytes.
    /// That model bounds every interval directly: the bits removed over
    /// one second cannot exceed what arrived in it (the rate) plus what
    /// the buffer held at its start (at most its size), so no one-second
    /// window of pictures spends more than the rate plus the buffer. That
    /// consequence is asserted separately, from the access unit sizes
    /// alone, because it is what a player's network sees. And the average
    /// is the declared rate within ten percent.
    #[test]
    fn constant_rate_holds_its_model_across_content() {
        use crate::encode::{Config, RateControl};
        const W: usize = 64;
        const H: usize = 64;
        const FPS: u32 = 30;
        const SECONDS: usize = 6;
        const BPS: u32 = 160_000;
        let n = FPS as usize * SECONDS;
        let mut seed = 0x9e37_79b9u32;
        let mut rnd = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        let frame = |luma: &dyn Fn(usize, usize) -> u8, chroma: u8| {
            let mut f = vec![chroma; W * H * 3 / 2];
            for y in 0..H {
                for x in 0..W {
                    f[y * W + x] = luma(x, y);
                }
            }
            f
        };
        let noise: Vec<Vec<u8>> = (0..n).map(|_| (0..W * H * 3 / 2).map(|_| rnd() as u8).collect()).collect();
        let motion: Vec<Vec<u8>> = (0..n).map(|i| frame(&|x, y| (((x + 5 * i) * 9) ^ ((y + 3 * i) * 7)) as u8, 128)).collect();
        let held = vec![frame(&|x, y| ((x * 3) ^ (y * 5)) as u8, 120); n];
        // Four unrelated scenes, cut every 41 pictures — never on the
        // 30-picture GOP — between smooth, detailed, noisy and moving.
        let cuts: Vec<Vec<u8>> = (0..n)
            .map(|i| match (i / 41) % 4 {
                0 => frame(&|x, y| (x + y + i) as u8, 128),
                1 => held[0].clone(),
                2 => noise[i].clone(),
                _ => motion[i].clone(),
            })
            .collect();
        for (codec, lookahead) in [("H.264", 0u32), ("H.265", 0), ("H.265 lookahead", 8)] {
            let h264 = codec == "H.264";
            for (content, frames) in [("noise", &noise), ("cuts", &cuts), ("motion", &motion), ("held", &held)] {
                for cpb_ms in [1000u32, 300] {
                    let tag = format!("{codec} {content} {cpb_ms} ms");
                    let cfg = Config {
                        width: W as u32,
                        height: H as u32,
                        fps: FPS,
                        gop: 30,
                        rate: RateControl::Bitrate { bps: BPS },
                        cpb_ms,
                        cbr: true,
                        lookahead,
                        max_cu_depth: if h264 { None } else { Some(1) },
                        ..Config::default()
                    };
                    let mut sizes = Vec::new();
                    let mut stream = Vec::new();
                    let mut take = |units: Vec<crate::encode::Access>| {
                        for u in units {
                            sizes.push(u.data.len() as u64 * 8);
                            stream.extend_from_slice(&u.data);
                        }
                    };
                    if h264 {
                        let mut e = crate::encode::h264::H264Encoder::new(cfg.clone()).unwrap();
                        for f in frames.iter() {
                            take(e.push(f).unwrap_or_else(|err| panic!("{tag}: {err}")));
                        }
                        take(e.flush().unwrap_or_else(|err| panic!("{tag}: {err}")));
                    } else {
                        let mut e = crate::encode::h265::H265Encoder::new(cfg.clone()).unwrap();
                        for f in frames.iter() {
                            take(e.push(f).unwrap_or_else(|err| panic!("{tag}: {err}")));
                        }
                        take(e.flush().unwrap_or_else(|err| panic!("{tag}: {err}")));
                    }
                    assert_eq!(sizes.len(), n, "{tag}");
                    let r = verify(&stream).unwrap_or_else(|err| panic!("{tag}: {err}"));
                    assert!(r.conforms(), "{tag}: {:?} {:?}", r.underflow, r.overflow);
                    let (rate, size) = (r.bit_rate as f64, r.cpb_size as f64);
                    let total: u64 = sizes.iter().sum();
                    let average = total as f64 / SECONDS as f64;
                    assert!((average / f64::from(BPS) - 1.0).abs() <= 0.10, "{tag}: {average:.0} b/s against {BPS}");
                    let peak = sizes.windows(FPS as usize).map(|w| w.iter().sum::<u64>()).max().unwrap() as f64;
                    assert!(peak <= rate + size, "{tag}: a one-second window spent {peak:.0} bits, over the rate plus the buffer ({:.0})", rate + size);
                    eprintln!("{tag}: average {:.3}x, peak one-second window {:.3}x of rate + buffer", average / rate, peak / (rate + size));
                }
            }
        }
    }
}
