//! Decode three small vendored streams and check every output byte.
//!
//! The rest of the crate's tests compare one kernel against its scalar
//! reference. None of them decodes anything, and the conformance suites are
//! fetched rather than vendored — they are large and not redistributable — so
//! until this file existed, CI could prove that a change *compiled* on a given
//! architecture and never that it *decoded correctly* there. That gap is not
//! theoretical: the kernel ladder installs each rung over the one below, so a
//! kernel that exists at some rungs and not others is normal, and whether the
//! rungs that inherit it still produce the right bytes is exactly the question
//! a compile cannot answer.
//!
//! These three streams are 80x64 and twelve frames, a few kilobytes each, and
//! between them they cover CABAC with B-pyramid and the 8x8 transform, CAVLC,
//! and HEVC — enough that a broken interpolation filter, transform,
//! deblocking edge or reference list shows up. They are not a substitute for
//! `tools/verify.sh`, which runs 412 real conformance streams; they are the
//! part of it that fits in a repository and runs everywhere.
//!
//! The expected hashes were taken from this decoder's output *after* checking
//! it frame-by-frame against libavcodec's `framemd5` for all three streams, so
//! they are anchored to an independent decoder rather than to ourselves.
//! Regenerating them to make a red test go green is therefore never the fix.

use h26x::Picture;

/// FNV-1a over the packed planes of every frame, in output order, mixed with
/// each frame's dimensions so a picture emitted at the wrong size cannot
/// collide with the right one.
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

    fn frame(&mut self, pic: Picture) {
        let (w, h) = (pic.width, pic.height);
        self.write(&w.to_le_bytes());
        self.write(&h.to_le_bytes());
        self.write(&pic.into_packed());
    }
}

fn decode_h264(data: &[u8]) -> (usize, u64) {
    let mut dec = h26x::h264::H264Decoder::new();
    let mut hasher = Hasher::new();
    let mut frames = 0;
    for nal in h26x::nal::annexb_nals(data) {
        dec.push_nal(nal).expect("h264 nal");
        while let Some(pic) = dec.try_next_picture() {
            hasher.frame(pic);
            frames += 1;
        }
    }
    dec.flush().expect("h264 flush");
    while let Some(pic) = dec.next_picture() {
        hasher.frame(pic);
        frames += 1;
    }
    (frames, hasher.0)
}

fn decode_hevc(data: &[u8]) -> (usize, u64) {
    let mut dec = h26x::hevc::HevcDecoder::new();
    let mut hasher = Hasher::new();
    let mut frames = 0;
    for nal in h26x::nal::annexb_nals(data) {
        dec.push_nal(nal).expect("hevc nal");
        while let Some(pic) = dec.try_next_picture() {
            hasher.frame(pic);
            frames += 1;
        }
    }
    dec.flush().expect("hevc flush");
    while let Some(pic) = dec.next_picture() {
        hasher.frame(pic);
        frames += 1;
    }
    (frames, hasher.0)
}

#[test]
fn h264_cabac_decodes_to_the_expected_bytes() {
    let (frames, hash) = decode_h264(include_bytes!("data/tiny_cabac.264"));
    assert_eq!(frames, 12, "frame count");
    assert_eq!(hash, 0xf9c88492eba65cad, "output bytes");
}

#[test]
fn h264_cavlc_decodes_to_the_expected_bytes() {
    let (frames, hash) = decode_h264(include_bytes!("data/tiny_cavlc.264"));
    assert_eq!(frames, 12, "frame count");
    assert_eq!(hash, 0xd3f30ea4806b6fa1, "output bytes");
}

#[test]
fn hevc_decodes_to_the_expected_bytes() {
    let (frames, hash) = decode_hevc(include_bytes!("data/tiny.265"));
    assert_eq!(frames, 12, "frame count");
    assert_eq!(hash, 0x55a0af35d7a74a68, "output bytes");
}
