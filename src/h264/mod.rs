//! The H.264 / AVC decoder.
//!
//! What is implemented: the Baseline, Main, High, High 10, High 4:2:2 and
//! High 4:4:4 (Predictive and Intra, CAVLC 4:4:4 Intra) profiles — frame
//! pictures, field pictures (PAFF) and MBAFF, 4:0:0 / 4:2:0 / 4:2:2 / 4:4:4
//! at 8–14 bits, separate colour planes, lossless transform bypass; CAVLC
//! and CABAC entropy coding, I/P/B slices, every macroblock and
//! sub-macroblock type including P_Skip, B_Skip and the spatial and temporal
//! direct modes, multiple reference frames and fields with reordering,
//! long-term references and adaptive marking (MMCO 1–6), weighted prediction
//! (explicit and implicit), the 8x8 transform and Intra_8x8, scaling
//! matrices, I_PCM, constrained intra prediction, multiple slices per
//! picture in any order (ASO), slice groups (FMO, all seven map types),
//! SP and SI slices (the Extended profile's switching pictures), frame_num
//! gaps, POC types 0/1/2, cropping, and the deblocking filter.
//!
//! What is refused with [`Error::Unsupported`](crate::Error::Unsupported)
//! (a caller with another decoder should hand the stream over): data
//! partitioning, SP/SI slices outside the Extended profile's shape (CABAC,
//! 4:2:2 / 4:4:4, more than 8 bits, the 8x8 transform), unequal luma /
//! chroma bit depths and bit depths above 14.
//!
//! Module map: `sps` / `pps` / `slice` parse the parameter sets and slice
//! header; `fmo` maps macroblocks to slice groups; `cavlc` and `cabac_mb`
//! parse a macroblock into an `mb::MbLayer`; `recon` turns it into samples
//! (through `intra`, `inter`, `transform`, and `sp` for SP / SI slices);
//! `deblock` filters the finished picture; `dpb` owns POC, reference marking,
//! list construction and output order; `decoder` drives it all.

pub(crate) mod cabac_mb;
pub(crate) mod cavlc;
pub(crate) mod deblock;
pub(crate) mod decoder;
pub(crate) mod dpb;
pub(crate) mod fmo;
pub(crate) mod frame;
pub(crate) mod inter;
pub(crate) mod intra;
pub(crate) mod mb;
pub(crate) mod pps;
pub(crate) mod recon;
pub(crate) mod slice;
pub(crate) mod sp;
pub(crate) mod sps;
pub(crate) mod tables;
// Generated: the standard's tables in full, whether or not the decoder
// reaches for every entry.
#[allow(dead_code, missing_docs, clippy::all, rustdoc::broken_intra_doc_links)]
pub(crate) mod tables_gen;
pub(crate) mod transform;

pub use decoder::H264Decoder;
pub use pps::Pps;
pub use slice::{SliceHeader, SliceType};
pub use sps::Sps;
