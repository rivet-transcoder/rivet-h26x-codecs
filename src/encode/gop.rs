//! Deciding what each picture is, and what order to code them in.
//!
//! Shared by both encoders, because H.264 and H.265 disagree about almost
//! everything below the slice header and agree completely here: pictures
//! arrive in display order, references must be coded before the pictures that
//! use them, and B pictures reference in both directions, so coding order is
//! not display order.
//!
//! The scheduling is deliberately the simple one — a fixed mini-GOP, anchor
//! first, then the B pictures between it and the previous anchor. Adaptive
//! decisions (scene cuts, adaptive B placement, pyramids) are a quality
//! feature and belong on top of a correct reorderer rather than inside the
//! first one.
//!
//! The invariant everything rests on is checked in the tests below rather than
//! argued for: **no picture is emitted before a picture it references**. A
//! reorderer that breaks it produces a bitstream that is either illegal or
//! silently wrong depending on the decoder, which is the worst pair of
//! outcomes available.

/// What a picture is coded as.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// Instantaneous decoder refresh: intra, and every earlier picture is
    /// removed from the reference lists. Where a decoder may start.
    Idr,
    /// Intra, but not a refresh point.
    I,
    /// One reference list, backwards only.
    P,
    /// Two reference lists.
    B,
}

impl Kind {
    /// Whether a decoder may begin the stream here.
    pub fn is_keyframe(self) -> bool {
        matches!(self, Kind::Idr)
    }

    /// Whether this picture may be predicted from at all.
    pub fn is_intra(self) -> bool {
        matches!(self, Kind::Idr | Kind::I)
    }
}

/// One picture, placed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Coded {
    /// Position in display order, counted from the first picture ever pushed.
    pub display: u64,
    /// Position in coding order, counted the same way.
    pub encode: u64,
    /// Picture order count, reset at each IDR. Doubled because H.264 counts
    /// POC in fields even when coding frames, and H.265 tolerates the same
    /// spacing; leaving room costs nothing and adding it later is a bitstream
    /// change.
    pub poc: i32,
    /// See [`Kind`].
    pub kind: Kind,
    /// Whether later pictures may reference this one. False for the B
    /// pictures of a non-pyramid GOP, which nothing refers to and which a
    /// decoder may therefore discard.
    pub reference: bool,
}

/// Turns a display-order stream of pictures into a coding-order one.
#[derive(Debug)]
pub struct Scheduler {
    gop: u32,
    bframes: u32,
    display: u64,
    encode: u64,
    /// Display index of the most recent IDR, for POC.
    idr_at: u64,
    /// B pictures held back, waiting for the anchor that follows them.
    pending: Vec<u64>,
    first: bool,
}

impl Scheduler {
    /// `gop` is pictures between IDRs, 0 meaning every picture is an IDR.
    /// `bframes` is consecutive B pictures between anchors.
    pub fn new(gop: u32, bframes: u32) -> Self {
        Self {
            gop,
            bframes,
            display: 0,
            encode: 0,
            idr_at: 0,
            pending: Vec::new(),
            first: true,
        }
    }

    /// Whether the picture at this display index starts a new GOP.
    fn is_idr(&self, display: u64) -> bool {
        if self.first && display == 0 {
            return true;
        }
        match self.gop {
            0 => true,
            g => (display - self.idr_at) >= g as u64,
        }
    }

    fn emit(&mut self, display: u64, kind: Kind, reference: bool) -> Coded {
        if kind == Kind::Idr {
            self.idr_at = display;
        }
        let c = Coded {
            display,
            encode: self.encode,
            poc: ((display - self.idr_at) as i32) * 2,
            kind,
            reference,
        };
        self.encode += 1;
        c
    }

    /// Offer the next picture in display order. Returns whatever is now ready
    /// to code, which may be nothing (the picture is a B held for its anchor),
    /// or several (an anchor releases the Bs waiting behind it).
    pub fn push(&mut self) -> Vec<Coded> {
        let display = self.display;
        self.display += 1;
        let mut out = Vec::new();

        if self.is_idr(display) {
            // An IDR ends the GOP, so anything still held cannot reference
            // forwards past it: those pictures become P, in display order.
            let held = std::mem::take(&mut self.pending);
            for d in held {
                out.push(self.emit(d, Kind::P, true));
            }
            self.first = false;
            out.push(self.emit(display, Kind::Idr, true));
            return out;
        }

        if self.bframes == 0 {
            out.push(self.emit(display, Kind::P, true));
            return out;
        }

        if self.pending.len() < self.bframes as usize {
            // Hold it: the anchor that follows has not arrived.
            self.pending.push(display);
            return out;
        }

        // This picture is the anchor. It codes first, then the Bs behind it.
        out.push(self.emit(display, Kind::P, true));
        let held = std::mem::take(&mut self.pending);
        for d in held {
            out.push(self.emit(d, Kind::B, false));
        }
        out
    }

    /// Release everything still held. A B picture with no following anchor
    /// cannot be a B, so it is coded as a P — which is why this exists rather
    /// than the caller simply dropping the tail.
    pub fn flush(&mut self) -> Vec<Coded> {
        let held = std::mem::take(&mut self.pending);
        held.into_iter().map(|d| self.emit(d, Kind::P, true)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(gop: u32, bframes: u32, n: u64) -> Vec<Coded> {
        let mut s = Scheduler::new(gop, bframes);
        let mut out = Vec::new();
        for _ in 0..n {
            out.extend(s.push());
        }
        out.extend(s.flush());
        out
    }

    #[test]
    fn every_picture_is_coded_exactly_once() {
        for &(gop, b) in &[(0, 0), (1, 0), (8, 0), (8, 1), (8, 2), (8, 3), (250, 2)] {
            for n in [1u64, 2, 7, 8, 9, 33] {
                let coded = run(gop, b, n);
                let mut seen: Vec<u64> = coded.iter().map(|c| c.display).collect();
                seen.sort_unstable();
                let want: Vec<u64> = (0..n).collect();
                assert_eq!(seen, want, "gop={gop} b={b} n={n}");
                let mut order: Vec<u64> = coded.iter().map(|c| c.encode).collect();
                order.sort_unstable();
                assert_eq!(order, want, "encode order gop={gop} b={b} n={n}");
            }
        }
    }

    /// The invariant the whole module exists for. A B picture references the
    /// nearest anchor on each side; a P references the previous anchor. Both
    /// must already have been coded.
    #[test]
    fn nothing_is_coded_before_what_it_references() {
        for &(gop, b) in &[(0, 0), (8, 0), (8, 1), (8, 2), (8, 3), (250, 2)] {
            for n in [1u64, 5, 8, 9, 20, 33] {
                let coded = run(gop, b, n);
                let mut done: Vec<u64> = Vec::new();
                for c in &coded {
                    if c.kind.is_intra() {
                        done.push(c.display);
                        continue;
                    }
                    let prev = done.iter().copied().filter(|&d| d < c.display).max();
                    assert!(
                        prev.is_some(),
                        "gop={gop} b={b} n={n}: {:?} coded with no earlier reference",
                        c
                    );
                    if c.kind == Kind::B {
                        let next = done.iter().copied().find(|&d| d > c.display);
                        assert!(
                            next.is_some(),
                            "gop={gop} b={b} n={n}: B at display {} coded before any later \
                             reference existed",
                            c.display
                        );
                    }
                    done.push(c.display);
                }
            }
        }
    }

    #[test]
    fn poc_is_zero_at_every_idr_and_orders_display_within_a_gop() {
        let coded = run(8, 2, 24);
        for c in &coded {
            if c.kind == Kind::Idr {
                assert_eq!(c.poc, 0, "IDR at display {} has poc {}", c.display, c.poc);
            }
        }
        // Within one GOP, POC must sort the same way display order does.
        let mut gops: Vec<Vec<&Coded>> = Vec::new();
        for c in &coded {
            if c.kind == Kind::Idr {
                gops.push(Vec::new());
            }
            gops.last_mut().unwrap().push(c);
        }
        for g in gops {
            let mut by_poc = g.clone();
            by_poc.sort_by_key(|c| c.poc);
            let mut by_display = g.clone();
            by_display.sort_by_key(|c| c.display);
            let a: Vec<u64> = by_poc.iter().map(|c| c.display).collect();
            let b: Vec<u64> = by_display.iter().map(|c| c.display).collect();
            assert_eq!(a, b);
        }
    }

    #[test]
    fn gop_zero_makes_every_picture_a_keyframe() {
        for c in run(0, 0, 5) {
            assert!(c.kind.is_keyframe(), "{c:?}");
            assert_eq!(c.poc, 0);
        }
    }

    /// A held B with no anchor after it cannot stay a B.
    #[test]
    fn a_trailing_b_becomes_a_p() {
        let coded = run(250, 3, 3);
        assert_eq!(coded.len(), 3);
        assert_eq!(coded[0].kind, Kind::Idr);
        for c in &coded[1..] {
            assert_ne!(c.kind, Kind::B, "{c:?} is a B with nothing after it");
        }
    }
}
