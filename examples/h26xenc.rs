//! Encode raw planar YUV to an Annex B stream. The counterpart of
//! `h26xdec`, and what `tools/verify_encode.sh` drives.
//!
//! Deliberately dumb about I/O: raw frames in, one file out, and — the part
//! the gate needs — the encoder's own reconstruction written separately, so
//! that decoding the bitstream can be compared against what the encoder
//! believed it was writing. That comparison is the SELF property, and it is
//! the check that finds encoder/decoder desync without any reference data.
//!
//!   h26xenc --input src.yuv --size 64x64 --format 420 --output out.264 \
//!           --recon out.rec.yuv [--codec h264|h265] [--qp N | --lossless]
//!           [--gop N] [--bframes N] [--cavlc] [--threads N]

use std::io::Write;

use h26x::ChromaFormat;
use h26x::encode::{Config, Entropy, RateControl};

fn die(msg: &str) -> ! {
    eprintln!("h26xenc: {msg}");
    eprintln!(
        "usage: h26xenc --input F --size WxH [--format 400|420|422|444] --output F\n\
         \x20      [--recon F] [--codec h264|h265] [--qp N | --lossless]\n\
         \x20      [--gop N] [--bframes N] [--cavlc] [--depth N] [--threads N]"
    );
    std::process::exit(2);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut input = None;
    let mut output = None;
    let mut recon = None;
    let mut codec = "h264".to_string();
    let mut cfg = Config::default();
    let mut fmt = "420".to_string();

    let mut i = 1;
    let val = |i: &mut usize, args: &Vec<String>, what: &str| -> String {
        *i += 1;
        args.get(*i).cloned().unwrap_or_else(|| die(&format!("{what} needs a value")))
    };
    while i < args.len() {
        match args[i].as_str() {
            "--input" => input = Some(val(&mut i, &args, "--input")),
            "--output" => output = Some(val(&mut i, &args, "--output")),
            "--recon" => recon = Some(val(&mut i, &args, "--recon")),
            "--codec" => codec = val(&mut i, &args, "--codec"),
            "--format" => fmt = val(&mut i, &args, "--format"),
            "--size" => {
                let s = val(&mut i, &args, "--size");
                let (w, h) = s.split_once('x').unwrap_or_else(|| die("--size wants WxH"));
                cfg.width = w.parse().unwrap_or_else(|_| die("--size width"));
                cfg.height = h.parse().unwrap_or_else(|_| die("--size height"));
            }
            "--qp" => {
                let q: u8 = val(&mut i, &args, "--qp").parse().unwrap_or_else(|_| die("--qp"));
                cfg.rate = RateControl::ConstantQp(q);
            }
            "--lossless" => cfg.rate = RateControl::Lossless,
            "--gop" => cfg.gop = val(&mut i, &args, "--gop").parse().unwrap_or_else(|_| die("--gop")),
            "--bframes" => {
                cfg.bframes = val(&mut i, &args, "--bframes").parse().unwrap_or_else(|_| die("--bframes"))
            }
            "--depth" => {
                cfg.bit_depth = val(&mut i, &args, "--depth").parse().unwrap_or_else(|_| die("--depth"))
            }
            "--threads" => {
                cfg.threads = val(&mut i, &args, "--threads").parse().unwrap_or_else(|_| die("--threads"))
            }
            "--cavlc" => cfg.entropy = Entropy::Cavlc,
            other => die(&format!("unknown argument {other}")),
        }
        i += 1;
    }

    let input = input.unwrap_or_else(|| die("--input is required"));
    let output = output.unwrap_or_else(|| die("--output is required"));
    cfg.chroma = match fmt.as_str() {
        "400" | "gray" => ChromaFormat::Monochrome,
        "420" => ChromaFormat::Yuv420,
        "422" => ChromaFormat::Yuv422,
        "444" => ChromaFormat::Yuv444,
        other => die(&format!("unknown --format {other}")),
    };
    if codec != "h264" && codec != "h265" {
        die("--codec must be h264 or h265");
    }

    let raw = std::fs::read(&input).unwrap_or_else(|e| die(&format!("read {input}: {e}")));

    // H.265 has an encoder skeleton whose refusal names the missing piece
    // (the CABAC coding tree), so drive it for real rather than refusing at
    // the argument parser — the gate then reports the precise hole.
    if codec == "h265" {
        let mut enc = match h26x::encode::h265::H265Encoder::new(cfg) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("h26xenc: {e}");
                std::process::exit(1);
            }
        };
        let fb = enc.frame_bytes();
        if fb == 0 || raw.len() < fb {
            die(&format!("input is {} bytes, less than one {fb}-byte picture", raw.len()));
        }
        let mut stream: Vec<u8> = Vec::new();
        let mut frames = 0usize;
        let mut fail = |e: h26x::Error| -> ! {
            eprintln!("h26xenc: {e}");
            std::process::exit(1);
        };
        for chunk in raw.chunks_exact(fb) {
            match enc.push(chunk) {
                Ok(units) => {
                    for a in units {
                        stream.extend_from_slice(&a.data);
                        frames += 1;
                    }
                }
                Err(e) => fail(e),
            }
        }
        match enc.flush() {
            Ok(units) => {
                for a in units {
                    stream.extend_from_slice(&a.data);
                    frames += 1;
                }
            }
            Err(e) => fail(e),
        }
        std::fs::write(&output, &stream).unwrap_or_else(|e| die(&format!("write {output}: {e}")));
        if let Some(path) = recon {
            let mut f = std::fs::File::create(&path)
                .unwrap_or_else(|e| die(&format!("create {path}: {e}")));
            for r in enc.reconstructions() {
                f.write_all(r).unwrap_or_else(|e| die(&format!("write {path}: {e}")));
            }
        }
        eprintln!("{frames} pictures, {} bytes", stream.len());
        return;
    }

    let mut enc = match h26x::encode::h264::H264Encoder::new(cfg) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("h26xenc: {e}");
            std::process::exit(1);
        }
    };

    let fb = enc.frame_bytes();
    if fb == 0 || raw.len() < fb {
        die(&format!(
            "input is {} bytes, which is less than one {fb}-byte picture",
            raw.len()
        ));
    }
    if raw.len() % fb != 0 {
        eprintln!(
            "h26xenc: warning: input is {} bytes, not a whole number of {fb}-byte pictures; \
             the tail is ignored",
            raw.len()
        );
    }

    let mut stream: Vec<u8> = Vec::new();
    let mut frames = 0usize;
    for chunk in raw.chunks_exact(fb) {
        match enc.push(chunk) {
            Ok(units) => {
                for a in units {
                    stream.extend_from_slice(&a.data);
                    frames += 1;
                }
            }
            Err(e) => {
                eprintln!("h26xenc: {e}");
                std::process::exit(1);
            }
        }
    }
    match enc.flush() {
        Ok(units) => {
            for a in units {
                stream.extend_from_slice(&a.data);
                frames += 1;
            }
        }
        Err(e) => {
            eprintln!("h26xenc: {e}");
            std::process::exit(1);
        }
    }

    std::fs::write(&output, &stream).unwrap_or_else(|e| die(&format!("write {output}: {e}")));
    if let Some(path) = recon {
        let mut f = std::fs::File::create(&path)
            .unwrap_or_else(|e| die(&format!("create {path}: {e}")));
        for r in enc.reconstructions() {
            f.write_all(r).unwrap_or_else(|e| die(&format!("write {path}: {e}")));
        }
    }
    eprintln!("{frames} pictures, {} bytes", stream.len());
}
