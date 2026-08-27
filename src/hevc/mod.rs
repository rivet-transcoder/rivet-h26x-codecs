//! H.265 / HEVC decoding: Main / Main 10 / Main 12 and the format range
//! extensions (4:0:0 – 4:4:4 at 8–16 bits, unequal luma / chroma depths,
//! extended precision, CABAC bypass alignment — everything but separate
//! colour planes), written from ITU-T H.265 (V11, 01/2026) with the same architecture as the
//! H.264 half of the crate: parameter sets, slice segments, CTU parsing and
//! reconstruction, deblocking + SAO, and an output-order DPB.

pub(crate) mod ctu;
pub(crate) mod ctx;
pub(crate) mod deblock;
pub(crate) mod decoder;
pub(crate) mod dpb;
pub(crate) mod frame;
pub(crate) mod hash;
pub(crate) mod inter;
pub(crate) mod intra;
pub(crate) mod mvpred;
pub(crate) mod pic;
pub(crate) mod pps;
pub(crate) mod residual;
pub(crate) mod sao;
pub(crate) mod slice;
pub(crate) mod sps;
pub(crate) mod tables;
// Generated: the standard's tables in full, whether or not the decoder
// reaches for every entry.
#[allow(dead_code, missing_docs, clippy::all, rustdoc::broken_intra_doc_links)]
pub(crate) mod tables_gen;

pub use decoder::HevcDecoder;
