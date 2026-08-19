//! A wasm entry point for `tools/wasm.sh`: decode a stream and hand back the
//! frame count and a hash of every output picture.
//!
//! This exists because nothing else can check that the decoder *works* on
//! wasm. `cargo test` does not run there — `wasm32-unknown-unknown` has no
//! test harness, no filesystem and no clock — so the only thing CI can prove
//! about that target is that it compiles, and a decoder that compiles and
//! aborts on its first picture is the failure shape that passes every check
//! and fails in front of a user. That is not hypothetical: it is what this
//! crate did until the profiling clock reads were guarded.
//!
//! The hash is FNV-1a over the packed planes with each frame's dimensions
//! mixed in — deliberately the same function and the same constants as
//! `tests/decode.rs`, so a wasm run is held to the expectations that were
//! anchored against libavcodec's `framemd5`, rather than merely to whatever
//! the native build of the moment happens to produce. `tools/wasm.sh` reads
//! those constants out of that file so the two cannot drift apart.
//!
//! Exported deliberately small: wasm has no argv and no files, so the caller
//! allocates, copies the stream in, and reads the answer out.

use std::alloc::{Layout, alloc};

/// FNV-1a, as in `tests/decode.rs`.
struct Hasher(u64);

impl Hasher {
    fn new() -> Self {
        Hasher(0xcbf2_9ce4_8422_2325)
    }

    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= b as u64;
            self.0 = self.0.wrapping_mul(0x100_0000_01b3);
        }
    }

    fn frame(&mut self, pic: h26x::Picture) {
        let (w, h) = (pic.width, pic.height);
        self.write(&w.to_le_bytes());
        self.write(&h.to_le_bytes());
        self.write(&pic.into_packed());
    }
}

/// Which kernels this module was built with, written as text to `out` (at
/// most 16 bytes), returning the length.
///
/// A decode that produces the right bytes proves nothing about *which* code
/// produced them — a tier that installed nothing would pass every comparison
/// in `tools/wasm.sh` without running one vector instruction. This is how the
/// script tells the two apart.
#[unsafe(no_mangle)]
pub extern "C" fn h26x_rung(out: *mut u8) -> u32 {
    let s = h26x::dsp::Cpu::detect().rung().as_bytes();
    let n = s.len().min(16);
    unsafe { std::ptr::copy_nonoverlapping(s.as_ptr(), out, n) };
    n as u32
}

/// `len` bytes of scratch inside the module's memory, for the caller to write
/// the stream into and to read the hash out of.
///
/// Never freed: the module decodes one stream and is thrown away.
#[unsafe(no_mangle)]
pub extern "C" fn h26x_scratch(len: usize) -> *mut u8 {
    unsafe { alloc(Layout::from_size_align(len.max(1), 8).unwrap()) }
}

/// Decode `len` bytes of Annex-B at `ptr` — HEVC when `hevc` is nonzero, H.264
/// otherwise — writing the hash as eight little-endian bytes to `out_hash`.
///
/// Returns the number of output pictures, or `u32::MAX` if the decoder
/// refused the stream.
#[unsafe(no_mangle)]
pub extern "C" fn h26x_decode(ptr: *const u8, len: usize, hevc: u32, out_hash: *mut u8) -> u32 {
    let data = unsafe { std::slice::from_raw_parts(ptr, len) };
    let mut h = Hasher::new();
    let mut frames = 0u32;
    let ok = if hevc != 0 {
        let mut d = h26x::hevc::HevcDecoder::new();
        run(data, &mut frames, &mut h, &mut d, |d, n| d.push_nal(n).is_ok(), |d| d.try_next_picture(), |d| d.flush().is_ok(), |d| d.next_picture())
    } else {
        let mut d = h26x::h264::H264Decoder::new();
        run(data, &mut frames, &mut h, &mut d, |d, n| d.push_nal(n).is_ok(), |d| d.try_next_picture(), |d| d.flush().is_ok(), |d| d.next_picture())
    };
    if !ok {
        return u32::MAX;
    }
    unsafe { std::ptr::copy_nonoverlapping(h.0.to_le_bytes().as_ptr(), out_hash, 8) };
    frames
}

/// The push / drain / flush / drain sequence, shared by the two decoders
/// because it is the same sequence and only the types differ.
fn run<D>(
    data: &[u8],
    frames: &mut u32,
    h: &mut Hasher,
    d: &mut D,
    push: impl Fn(&mut D, &[u8]) -> bool,
    try_next: impl Fn(&mut D) -> Option<h26x::Picture>,
    flush: impl Fn(&mut D) -> bool,
    next: impl Fn(&mut D) -> Option<h26x::Picture>,
) -> bool {
    for nal in h26x::nal::annexb_nals(data) {
        if !push(d, nal) {
            return false;
        }
        while let Some(p) = try_next(d) {
            h.frame(p);
            *frames += 1;
        }
    }
    if !flush(d) {
        return false;
    }
    while let Some(p) = next(d) {
        h.frame(p);
        *frames += 1;
    }
    true
}

// ---------------------------------------------------------------------------
// The HEVC kernel sweep
// ---------------------------------------------------------------------------

/// The LCG the dsp tests use, so the sweep here draws the same kind of data.
fn lcg(seed: &mut u64) -> u32 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*seed >> 33) as u32
}

/// Compare every entry of the installed 8-bit HEVC kernel table against the
/// scalar reference over randomized inputs — all the block shapes the
/// dispatch serves, every fractional position, and the clipping corners —
/// returning the number of comparisons that disagreed (0 = bit-exact).
///
/// This is the `#[cfg(test)]` module the wasm kernel files cannot have:
/// `wasm32-unknown-unknown` has no test harness, so the sweep is exported
/// from the probe instead and driven by `tools/wasm_dsp_check.mjs`. On a
/// native build it checks whatever ladder the host installs, which is the
/// same sweep the dsp files' own tests run — harmless and still true.
#[unsafe(no_mangle)]
pub extern "C" fn h26x_hevc_dsp_check() -> u32 {
    use h26x::dsp::hevc::HevcDsp;
    let s = HevcDsp::<u8>::SCALAR;
    let d = HevcDsp::<u8>::new(h26x::dsp::Cpu::detect());
    let mut fails = 0u32;
    // Every PU width the dispatch serves, including the AMP widths (12, 24,
    // 48) and the odd chroma one (6).
    const SIZES: [(usize, usize); 12] = [(2, 4), (4, 4), (4, 8), (6, 8), (8, 4), (8, 8), (12, 16), (16, 16), (24, 32), (32, 8), (48, 64), (64, 64)];

    // Interpolation: first stage over bytes, second stage over 14-bit
    // intermediates — the latter in both the contiguous (stride == w, the
    // shape the two-pass path feeds it) and the strided form.
    let mut seed = 31u64;
    let stride = 96;
    let src: Vec<u8> = (0..stride * 96).map(|_| lcg(&mut seed) as u8).collect();
    for &(w, h) in &SIZES {
        let mut a = vec![0i16; w * h];
        let mut b = vec![0i16; w * h];
        for frac in 0..4 {
            (s.qpel_h)(&mut a, &src, stride, w, h, frac, 0);
            (d.qpel_h)(&mut b, &src, stride, w, h, frac, 0);
            fails += (a != b) as u32;
            (s.qpel_v)(&mut a, &src, stride, w, h, frac, 0);
            (d.qpel_v)(&mut b, &src, stride, w, h, frac, 0);
            fails += (a != b) as u32;
            let mid: Vec<i16> = (0..stride * 96).map(|_| (lcg(&mut seed) % 30000) as i16 - 15000).collect();
            (s.qpel_v2)(&mut a, &mid, stride, w, h, frac);
            (d.qpel_v2)(&mut b, &mid, stride, w, h, frac);
            fails += (a != b) as u32;
            let flat: Vec<i16> = (0..w * (h + 7)).map(|_| (lcg(&mut seed) % 30000) as i16 - 15000).collect();
            (s.qpel_v2)(&mut a, &flat, w, w, h, frac);
            (d.qpel_v2)(&mut b, &flat, w, w, h, frac);
            fails += (a != b) as u32;
        }
        for frac in 0..8 {
            (s.epel_h)(&mut a, &src, stride, w, h, frac, 0);
            (d.epel_h)(&mut b, &src, stride, w, h, frac, 0);
            fails += (a != b) as u32;
            (s.epel_v)(&mut a, &src, stride, w, h, frac, 0);
            (d.epel_v)(&mut b, &src, stride, w, h, frac, 0);
            fails += (a != b) as u32;
            let mid: Vec<i16> = (0..stride * 96).map(|_| (lcg(&mut seed) % 30000) as i16 - 15000).collect();
            (s.epel_v2)(&mut a, &mid, stride, w, h, frac);
            (d.epel_v2)(&mut b, &mid, stride, w, h, frac);
            fails += (a != b) as u32;
            let flat: Vec<i16> = (0..w * (h + 3)).map(|_| (lcg(&mut seed) % 30000) as i16 - 15000).collect();
            (s.epel_v2)(&mut a, &flat, w, w, h, frac);
            (d.epel_v2)(&mut b, &flat, w, w, h, frac);
            fails += (a != b) as u32;
        }
        (s.qpel_copy)(&mut a, &src, stride, w, h, 6);
        (d.qpel_copy)(&mut b, &src, stride, w, h, 6);
        fails += (a != b) as u32;
    }

    // Combination and weighting, over the 14-bit domain and over the
    // saturation corners (the extreme palette drives the saturating-add
    // shortcut through every branch of its exactness argument).
    let mut seed = 41u64;
    const CORNERS: [i16; 12] = [-32768, -32767, -16384, -64, -1, 0, 1, 63, 16383, 16384, 32766, 32767];
    for &(w, h) in &SIZES {
        for corner in [false, true] {
            let draw = |seed: &mut u64| -> i16 {
                if corner { CORNERS[(lcg(seed) as usize) % CORNERS.len()] } else { (lcg(seed) % 32768) as i16 - 16384 }
            };
            let pa: Vec<i16> = (0..w * h).map(|_| draw(&mut seed)).collect();
            let pb: Vec<i16> = (0..w * h).map(|_| draw(&mut seed)).collect();
            let stride = w + 5;
            let mut a = vec![0u8; stride * h];
            let mut b = vec![0u8; stride * h];
            (s.uni)(&mut a, stride, &pa, w, h, 6, 255);
            (d.uni)(&mut b, stride, &pa, w, h, 6, 255);
            fails += (a != b) as u32;
            (s.bi)(&mut a, stride, &pa, &pb, w, h, 7, 255);
            (d.bi)(&mut b, stride, &pa, &pb, w, h, 7, 255);
            fails += (a != b) as u32;
            for &(lwd, wt, o) in &[(6i32, 64i32, 0i32), (0, 1, 3), (5, -20, -7), (7, 127, 100), (7, -128, -128)] {
                (s.weighted_uni)(&mut a, stride, &pa, w, h, lwd, wt, o, 255);
                (d.weighted_uni)(&mut b, stride, &pa, w, h, lwd, wt, o, 255);
                fails += (a != b) as u32;
                (s.weighted_bi)(&mut a, stride, &pa, &pb, w, h, lwd, wt, 64 - wt, o, -o, 255);
                (d.weighted_bi)(&mut b, stride, &pa, &pb, w, h, lwd, wt, 64 - wt, o, -o, 255);
                fails += (a != b) as u32;
            }
        }
    }

    // Residual add, with full-range residuals so the saturating add's
    // exactness argument is exercised at both ends.
    let mut seed = 43u64;
    for &n in &[4usize, 8, 16, 32] {
        for corner in [false, true] {
            let stride = n + 7;
            let base: Vec<u8> = (0..stride * n).map(|_| lcg(&mut seed) as u8).collect();
            let res: Vec<i16> = (0..n * n)
                .map(|_| if corner { CORNERS[(lcg(&mut seed) as usize) % CORNERS.len()] } else { (lcg(&mut seed) % 512) as i16 - 256 })
                .collect();
            let mut a = base.clone();
            let mut b = base.clone();
            (s.add_residual)(&mut a, stride, &res, n, 255);
            (d.add_residual)(&mut b, stride, &res, n, 255);
            fails += (a != b) as u32;
        }
    }

    // Inverse transforms: sparse blocks with a bounding box, plus dense ones.
    let mut seed = 9u64;
    for &(n, log2) in &[(4usize, 2u32), (8, 3), (16, 4), (32, 5)] {
        for trial in 0..120 {
            let mut c = vec![0i16; n * n];
            let (mx, my) = if trial % 4 == 0 { (n - 1, n - 1) } else { ((lcg(&mut seed) as usize) % n, (lcg(&mut seed) as usize) % n) };
            for y in 0..=my {
                for x in 0..=mx {
                    if lcg(&mut seed) % 2 == 0 {
                        c[y * n + x] = (lcg(&mut seed) as i32 % 65536 - 32768) as i16;
                    }
                }
            }
            let bd_shift = 12 - (trial % 3) as i32 * 2;
            let mut a = c.clone();
            let mut b = c;
            (s.idct[(log2 - 2) as usize])(&mut a, bd_shift, mx, my);
            (d.idct[(log2 - 2) as usize])(&mut b, bd_shift, mx, my);
            fails += (a != b) as u32;
        }
    }

    // SAO.
    let mut seed = 47u64;
    let stride = 72;
    let src: Vec<u8> = (0..stride * 80).map(|_| lcg(&mut seed) as u8).collect();
    for &(w, h) in &SIZES {
        let mut table = [0i16; 32];
        let start = (lcg(&mut seed) % 28) as usize;
        for k in 0..4 {
            table[start + k] = (lcg(&mut seed) % 15) as i16 - 7;
        }
        let mut a = vec![0u8; src.len()];
        let mut b = vec![0u8; src.len()];
        (s.sao_band)(&mut a, stride, &src, stride, w, h, &table, 3, 255);
        (d.sao_band)(&mut b, stride, &src, stride, w, h, &table, 3, 255);
        fails += (a != b) as u32;
        let mut off = [0i16; 5];
        for k in [0usize, 1, 3, 4] {
            off[k] = (lcg(&mut seed) % 15) as i16 - 7;
        }
        for &(na, nb) in &[(-1isize, 1isize), (-(stride as isize), stride as isize), (-(stride as isize) - 1, stride as isize + 1), (-(stride as isize) + 1, stride as isize - 1)] {
            let origin = 4 * stride + 4;
            let mut a = src.clone();
            let mut b = src.clone();
            (s.sao_edge)(&mut a, &src, origin, stride, w, h, na, nb, &off, 255);
            (d.sao_edge)(&mut b, &src, origin, stride, w, h, na, nb, &off, 255);
            fails += (a != b) as u32;
        }
    }

    fails
}
