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
