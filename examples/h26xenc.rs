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


use h26x::ChromaFormat;
use h26x::encode::{Config, Entropy, RateControl};

fn die(msg: &str) -> ! {
    eprintln!("h26xenc: {msg}");
    eprintln!(
        "usage: h26xenc --input F --size WxH [--format 400|420|422|444] --output F\n\
         \x20      [--recon F] [--codec h264|h265] [--qp N | --lossless | --bitrate BPS]\n\
         \x20      [--fps N] [--cpb-ms N]\n\
         \x20      [--gop N] [--bframes N] [--cavlc] [--t8x8] [--subparts] [--sao]\n\
         \x20      [--depth N] [--threads N]"
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
            "--bitrate" => {
                let b: u32 = val(&mut i, &args, "--bitrate").parse().unwrap_or_else(|_| die("--bitrate"));
                cfg.rate = RateControl::Bitrate { bps: b };
            }
            "--fps" => cfg.fps = val(&mut i, &args, "--fps").parse().unwrap_or_else(|_| die("--fps")),
            "--cpb-ms" => cfg.cpb_ms = val(&mut i, &args, "--cpb-ms").parse().unwrap_or_else(|_| die("--cpb-ms")),
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
            // H.264 only: offer the 8x8 transform in the PPS and let the
            // decisions use it. Ignored by H.265.
            "--t8x8" => cfg.transform_8x8 = true,
            // H.265 only: offer sample adaptive offset. Refused on H.264,
            // which has no such filter.
            "--sao" => cfg.sao = true,
            // H.264 only: offer inter partitions below 16x16.
            "--subparts" => cfg.subparts = true,
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
        let mut pocs: Vec<(bool, i32)> = Vec::new();
        let fail = |e: h26x::Error| -> ! {
            eprintln!("h26xenc: {e}");
            std::process::exit(1);
        };
        for chunk in raw.chunks_exact(fb) {
            match enc.push(chunk) {
                Ok(units) => {
                    for a in units {
                        stream.extend_from_slice(&a.data);
                        pocs.push((a.keyframe, a.poc));
                    }
                }
                Err(e) => fail(e),
            }
        }
        match enc.flush() {
            Ok(units) => {
                for a in units {
                    stream.extend_from_slice(&a.data);
                    pocs.push((a.keyframe, a.poc));
                }
            }
            Err(e) => fail(e),
        }
        std::fs::write(&output, &stream).unwrap_or_else(|e| die(&format!("write {output}: {e}")));
        if let Some(path) = recon {
            write_recon_display_order(&path, enc.reconstructions(), &pocs);
        }
        eprintln!("{} pictures, {} bytes", pocs.len(), stream.len());
        if let Some((achieved, target)) = enc.rate_report() {
            // The gate parses this line. Ratio included so a human reading
            // a log sees the shape of the error without dividing.
            eprintln!("rate: achieved {achieved:.0} bps, target {target:.0} bps, ratio {:.3}", achieved / target);
        }
        if enc.recodes() != 0 {
            eprintln!("rate: {} extra codings to fit the declared buffer", enc.recodes());
        }
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
    let mut pocs: Vec<(bool, i32)> = Vec::new();
    for chunk in raw.chunks_exact(fb) {
        match enc.push(chunk) {
            Ok(units) => {
                for a in units {
                    stream.extend_from_slice(&a.data);
                    pocs.push((a.keyframe, a.poc));
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
                pocs.push((a.keyframe, a.poc));
            }
        }
        Err(e) => {
            eprintln!("h26xenc: {e}");
            std::process::exit(1);
        }
    }

    std::fs::write(&output, &stream).unwrap_or_else(|e| die(&format!("write {output}: {e}")));
    if let Some(path) = recon {
        write_recon_display_order(&path, enc.reconstructions(), &pocs);
    }
    eprintln!("{} pictures, {} bytes", pocs.len(), stream.len());
    if let Some((achieved, target)) = enc.rate_report() {
        // Parsed by the gate. Same line, same shape, as the H.265 path.
        eprintln!("rate: achieved {achieved:.0} bps, target {target:.0} bps, ratio {:.3}", achieved / target);
    }
    // The shape census: which macroblock kinds each picture type took.
    // A row turns a shape on; only this line says whether the clip took
    // it, which is the difference between a cell that proves a feature
    // and one that proves its syntax.
    for (pic, name) in ["I", "P", "B"].iter().enumerate() {
        let taken = enc.shape_census().taken(pic);
        if taken.is_empty() {
            continue;
        }
        let list: Vec<String> = taken.iter().map(|(k, n)| format!("{k} {n}")).collect();
        eprintln!("shapes {name}: {}", list.join(", "));
    }
}

/// Write the reconstructions in *display* order — sorted by each coded
/// picture's POC — because that is the order a decoder emits pictures and
/// therefore the order the SELF comparison reads. The encoder hands them
/// back in coding order, which differs the moment B pictures exist; an
/// all-skip GOP hid that (every picture in it decoded identical), and the
/// first real B pictures surfaced it as a phantom SELF failure whose
/// per-frame diffs were exactly the reorder distance.
fn write_recon_display_order(path: &str, recons: &[Vec<u8>], pocs: &[(bool, i32)]) {
    use std::io::Write;
    assert_eq!(recons.len(), pocs.len(), "one reconstruction per coded picture");
    // POC restarts at every IDR, so display order is per coded video
    // sequence: sort by (sequence, poc), the sequence counted up at each
    // keyframe. A global poc sort interleaves GOPs — the first two-GOP
    // clip with real B pictures found that the hard way.
    let mut seq = 0u32;
    let keys: Vec<(u32, i32)> = pocs
        .iter()
        .map(|&(key, poc)| {
            if key {
                seq += 1;
            }
            (seq, poc)
        })
        .collect();
    let mut order: Vec<usize> = (0..recons.len()).collect();
    order.sort_by_key(|&i| keys[i]);
    let mut f = std::fs::File::create(path)
        .unwrap_or_else(|e| die(&format!("create {path}: {e}")));
    for &i in &order {
        f.write_all(&recons[i]).unwrap_or_else(|e| die(&format!("write {path}: {e}")));
    }
}
