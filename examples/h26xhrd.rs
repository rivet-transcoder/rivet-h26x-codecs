//! Verify a coded stream against the coded picture buffer **it declares**.
//!
//! Deliberately a separate tool rather than a flag on the decoder: a
//! decoder is not required to check the hypothetical reference decoder, and
//! ours does not. This reads the declaration out of the stream — rate and
//! buffer size from the sequence parameter set's VUI, the removal interval
//! from the frame rate beside it, the initial delay from the buffering
//! period SEI — and walks the buffer. Nothing is passed in, so nothing is
//! assumed.
//!
//! Usage: h26xhrd FILE

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let Some(path) = args.get(1) else {
        eprintln!("usage: h26xhrd FILE");
        std::process::exit(2);
    };
    let data = match std::fs::read(path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("h26xhrd: {path}: {e}");
            std::process::exit(2);
        }
    };
    match h26x::encode::hrd::verify(&data) {
        Ok(r) => {
            let verdict = if r.conforms() { "conforms" } else { "VIOLATES" };
            println!(
                "hrd: {verdict} ({} access units, {} bps, {} bit buffer)",
                r.units, r.bit_rate, r.cpb_size
            );
            if let Some((n, bits)) = r.underflow {
                println!("hrd: underflow at access unit {n}, short by {bits} bits");
            }
            if let Some((n, bits)) = r.overflow {
                println!("hrd: overflow at access unit {n}, over by {bits} bits");
            }
            std::process::exit(i32::from(!r.conforms()));
        }
        Err(e) => {
            eprintln!("h26xhrd: {e}");
            std::process::exit(2);
        }
    }
}
