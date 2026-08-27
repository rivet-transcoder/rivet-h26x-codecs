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
//! anywhere, so the verdict is reproducible rather than nearly so.

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
    /// time_scale` for a fixed picture rate.
    pub tick_90k: u64,
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
        (0..sizes.len()).map(|n| s.initial_delay_90k + (n as u64) * s.tick_90k).collect();
    simulate_at(sizes, &removal, s)
}

/// [`simulate`] with each unit's removal time given outright, in 90 kHz
/// ticks — how an H.264 stream is walked, whose removal times come from
/// the `cpb_removal_delay` of each picture's own timing SEI rather than
/// from a fixed-rate inference.
pub fn simulate_at(sizes: &[u64], removal_90k: &[u64], s: &Schedule) -> Report {
    debug_assert_eq!(sizes.len(), removal_90k.len());
    let mut occupancy = Vec::with_capacity(sizes.len());
    let mut underflow: Option<(usize, u64)> = None;
    let mut overflow: Option<(usize, u64)> = None;

    // Everything in 90 kHz ticks and bits. Bits arriving in `d` ticks is
    // `bit_rate * d / 90_000`, done as one product before the divide so the
    // truncation happens once rather than compounding per picture.
    let arrived = |ticks: u64| -> u64 { s.bit_rate.saturating_mul(ticks) / 90_000 };

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
        let begins = k == 0 || types[k - 1] < 32;
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
    Ok(Schedule {
        bit_rate: hrd.bit_rate,
        cpb_size: hrd.cpb_size,
        cbr: hrd.cbr,
        tick_90k: 90_000u64 * num_units as u64 / time_scale as u64,
        initial_delay_90k,
    })
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
    // Removal time of the last buffering-period unit, the base every
    // `cpb_removal_delay` counts from.
    let mut base_90k: Option<u64> = None;
    let mut close = |bytes: u64, sei: &H264Sei, schedule: &Schedule, sizes: &mut Vec<u64>, removal: &mut Vec<u64>| -> Result<()> {
        let n = sizes.len();
        let t = match (base_90k, sei.buffering_period, sei.pic_timing) {
            (None, Some(initial), Some((delay, _))) => initial + delay * schedule.tick_90k,
            (None, Some(initial), None) if n == 0 => initial,
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
        if !vcl && in_slices {
            // A new unit begins.
            let Some(s) = schedule.as_ref() else {
                return Err(Error::bitstream("HRD: slices before any sequence parameter set"));
            };
            close(cur_bytes, &cur_sei, s, &mut sizes, &mut removal)?;
            cur_bytes = 0;
            cur_sei = H264Sei::default();
            in_slices = false;
        }
        cur_bytes += (end - starts[k]) as u64;
        in_slices |= vcl;
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
                schedule = Some(Schedule {
                    bit_rate: hrd.bit_rate,
                    cpb_size: hrd.cpb_size,
                    cbr: hrd.cbr,
                    tick_90k: 90_000u64 * num_units as u64 / time_scale as u64,
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
        Schedule { bit_rate: 90_000, cpb_size: cpb, cbr, tick_90k: 3_000, initial_delay_90k: cpb }
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
    /// is different from saying it conformed. Both codecs.
    #[test]
    fn a_stream_that_declares_no_buffer_is_refused_rather_than_passed() {
        use crate::encode::h265_syntax::{Geometry, write_sps};
        use crate::encode::Config;
        let cfg = Config { width: 64, height: 64, ..Config::default() };
        let g = Geometry::new(&cfg);
        let sps = crate::encode::h265_syntax::annexb(33, &write_sps(&cfg, &g, 8, None));
        let err = verify(&sps).expect_err("no VUI means no verdict");
        let s = format!("{err}");
        assert!(s.contains("no VUI") || s.contains("no buffer"), "{s}");

        let g = crate::encode::h264_syntax::Geometry::new(&cfg);
        let sps = crate::encode::h264_syntax::annexb(
            crate::encode::h264_syntax::NAL_SPS,
            3,
            &crate::encode::h264_syntax::write_sps(&cfg, &g, 16, 16, None),
        );
        assert!(is_h264(&sps));
        let err = verify(&sps).expect_err("H.264: no VUI means no verdict");
        let s = format!("{err}");
        assert!(s.contains("no VUI") || s.contains("no buffer"), "{s}");
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
}
