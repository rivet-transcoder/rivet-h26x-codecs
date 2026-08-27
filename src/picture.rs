//! The decoded picture both decoders hand back.

/// Chroma sampling of a decoded picture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChromaFormat {
    /// Luma only (`chroma_format_idc` 0).
    Monochrome,
    /// 4:2:0.
    Yuv420,
    /// 4:2:2.
    Yuv422,
    /// 4:4:4.
    Yuv444,
}

impl ChromaFormat {
    /// `(SubWidthC, SubHeightC)` — how many luma samples one chroma sample
    /// covers in each direction (1 for monochrome, which has no chroma).
    pub fn subsampling(self) -> (u32, u32) {
        match self {
            ChromaFormat::Monochrome => (1, 1),
            ChromaFormat::Yuv420 => (2, 2),
            ChromaFormat::Yuv422 => (2, 1),
            ChromaFormat::Yuv444 => (1, 1),
        }
    }
    /// From `chroma_format_idc`.
    pub fn from_idc(idc: u32) -> Option<Self> {
        Some(match idc {
            0 => ChromaFormat::Monochrome,
            1 => ChromaFormat::Yuv420,
            2 => ChromaFormat::Yuv422,
            3 => ChromaFormat::Yuv444,
            _ => return None,
        })
    }
}

/// One plane of a [`Picture`]: where it sits in the picture's data. Samples
/// are tightly packed (stride == width), one byte each when every component
/// of the picture is 8-bit, else little-endian `u16` for every plane (values
/// in the low bits) — see [`Picture::bytes_per_sample`].
#[derive(Debug, Clone, Copy)]
pub struct Plane {
    /// Byte offset of the plane's first sample in [`Picture::data`].
    pub offset: usize,
    /// Width in samples.
    pub width: u32,
    /// Height in samples.
    pub height: u32,
}

impl Plane {
    /// The plane's size in bytes for a picture whose widest component has
    /// `bit_depth` bits (see [`Picture::bytes_per_sample`]).
    pub fn len(&self, bit_depth: u32) -> usize {
        self.width as usize * self.height as usize * if bit_depth > 8 { 2 } else { 1 }
    }
}

/// Output buffers a decoder hands out and takes back: a dropped [`Picture`]
/// returns its buffer here and the next picture of the same size reuses it,
/// so steady-state decoding allocates nothing per frame (a fresh
/// picture-sized allocation is page faults and zeroing on every platform,
/// and on Windows a system call each way).
#[derive(Debug, Clone, Default)]
pub struct OutputPool(std::sync::Arc<std::sync::Mutex<Vec<Vec<u8>>>>);

impl OutputPool {
    /// A pooled buffer of exactly `len` bytes (contents stale — every byte
    /// is overwritten by the caller), or a fresh zeroed one.
    pub fn take(&self, len: usize) -> Vec<u8> {
        let mut g = self.0.lock().unwrap();
        // Buffers of another size (a resolution change) are let go.
        g.retain(|b| b.len() == len);
        g.pop().unwrap_or_else(|| vec![0u8; len])
    }

    fn give(&self, buf: Vec<u8>) {
        if buf.is_empty() {
            return;
        }
        let mut g = self.0.lock().unwrap();
        // Bounded: more than a pipeline's worth is a consumer that keeps
        // pictures around, not a reason to hoard.
        if g.len() < 32 {
            g.push(buf);
        }
    }
}

/// A decoded, cropped picture in output order: one buffer holding the
/// planes one after the other (Y, then Cb, then Cr — the layout of a packed
/// planar frame, and what libavcodec's `framemd5` hashes), so a consumer
/// that wants exactly that takes it without a copy ([`Self::into_packed`]).
/// Dropped, it gives its buffer back to the decoder's [`OutputPool`].
#[derive(Debug, Clone)]
pub struct Picture {
    /// Luma width after cropping.
    pub width: u32,
    /// Luma height after cropping.
    pub height: u32,
    /// Bits per luma sample (8–16).
    pub bit_depth: u32,
    /// Bits per chroma sample (8–16). Equal to `bit_depth` in every profile
    /// but the range extensions' unequal-depth case (`TSUNEQBD`,
    /// `Bitdepth_A/B`); a monochrome picture reports the SPS value, which
    /// then decides nothing but the sample size (below).
    pub bit_depth_chroma: u32,
    /// Chroma sampling.
    pub chroma: ChromaFormat,
    /// The samples of every plane, packed.
    pub data: Vec<u8>,
    /// Y, then Cb, then Cr (the last two absent for monochrome).
    pub planes: Vec<Plane>,
    /// Picture order count — the standard's own display-order key. Frames
    /// come out sorted by it; a caller can use it to detect gaps.
    pub poc: i32,
    /// Decode-order index of the frame (0 for the first decoded picture).
    pub decode_index: u64,
    /// Where the buffer goes when the picture is dropped.
    pub(crate) pool: Option<OutputPool>,
}

impl Drop for Picture {
    fn drop(&mut self) {
        if let Some(p) = &self.pool {
            p.give(std::mem::take(&mut self.data));
        }
    }
}

impl Picture {
    /// Bytes per sample, the same for every plane: 1 when both bit depths
    /// are 8, else 2 (little-endian, values in the low bits) — the layout
    /// the HM reference decoder writes for unequal depths too, so a
    /// `Bitdepth_B` picture (8-bit luma, 12-bit chroma) carries its luma as
    /// 16-bit words holding 8-bit values.
    pub fn bytes_per_sample(&self) -> usize {
        if self.bit_depth > 8 || self.bit_depth_chroma > 8 {
            2
        } else {
            1
        }
    }

    /// The samples of plane `i` (0 Y, 1 Cb, 2 Cr).
    pub fn plane(&self, i: usize) -> &[u8] {
        let p = &self.planes[i];
        &self.data[p.offset..p.offset + p.len(self.bit_depth.max(self.bit_depth_chroma))]
    }

    /// The planes concatenated: Y then Cb then Cr — the layout of a packed
    /// planar frame (and of libavcodec's `framemd5`).
    pub fn packed(&self) -> &[u8] {
        &self.data
    }

    /// The packed planes, taking the buffer (it then belongs to the caller
    /// and does not return to the decoder's pool).
    pub fn into_packed(mut self) -> Vec<u8> {
        std::mem::take(&mut self.data)
    }
}
