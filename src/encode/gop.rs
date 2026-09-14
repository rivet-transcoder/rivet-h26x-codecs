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
#[derive(Debug, Clone)]
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
    /// The next picture offered starts a new GOP whatever the cadence says.
    /// Set by [`Scheduler::force_idr`], cleared when that IDR is emitted.
    force_idr: bool,
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
            force_idr: false,
        }
    }

    /// Make the **next** picture offered an IDR, wherever the GOP cadence
    /// would have put one.
    ///
    /// A caller that splits a stream into independently decodable pieces
    /// needs this: it feeds a run-in of pictures to warm the encoder up and
    /// discards their output, so the first picture it *keeps* is not the
    /// encoder's picture zero and has to be promoted to a random access
    /// point by name. The GOP restarts there — the cadence counts from the
    /// forced IDR, not from where the previous one would have fallen — so
    /// what follows is an ordinary closed GOP and the piece stands alone.
    ///
    /// Any B pictures held back at that point are released ahead of it as
    /// P pictures, the same way a scheduled IDR releases them: nothing may
    /// predict forwards across a random access point.
    pub fn force_idr(&mut self) {
        self.force_idr = true;
    }

    /// Whether the picture at this display index starts a new GOP.
    fn is_idr(&self, display: u64) -> bool {
        if self.force_idr {
            return true;
        }
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
            self.force_idr = false;
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

    /// What the pictures not yet released would be coded as if the next
    /// `ahead` were offered now: the B pictures held back and the `ahead`
    /// displays after them, each with the kind [`Scheduler::push`] would
    /// give it. For a rate controller that has those pictures in hand and
    /// wants to plan across them.
    ///
    /// Not a second copy of the cadence: it *runs* `push` on a copy of
    /// this scheduler. The copy is pushed `bframes` pictures past the
    /// horizon so that a B picture just inside it is released by the
    /// anchor that would follow rather than left held; those phantom
    /// pictures are then dropped from the answer. The one thing this
    /// cannot know is whether the stream ends inside the horizon — a B
    /// picture the caller will flush is previewed as the B it would have
    /// been, not the P it becomes.
    pub fn preview(&self, ahead: u64) -> Vec<Coded> {
        let mut sim = self.clone();
        let horizon = self.display + ahead;
        let mut out = Vec::new();
        for _ in 0..ahead + u64::from(self.bframes) {
            out.extend(sim.push().into_iter().filter(|c| c.display < horizon));
        }
        out
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

    /// A forced IDR lands on the next picture offered, restarts the GOP
    /// cadence from there, and releases any held B pictures ahead of it as
    /// P pictures — the same shape a scheduled IDR has, so a decoder that
    /// starts at the forced one sees an ordinary closed GOP.
    #[test]
    fn a_forced_idr_is_the_next_picture_and_restarts_the_gop() {
        for &b in &[0u32, 2] {
            let mut s = Scheduler::new(8, b);
            let mut coded = Vec::new();
            for _ in 0..5 {
                coded.extend(s.push());
            }
            s.force_idr();
            coded.extend(s.push()); // display 5
            for _ in 6..20 {
                coded.extend(s.push());
            }
            coded.extend(s.flush());

            let idrs: Vec<u64> =
                coded.iter().filter(|c| c.kind == Kind::Idr).map(|c| c.display).collect();
            // Picture 0 by rule, 5 by request, then 13 because the cadence
            // counts from 5 — and NOT 8, where it would have fallen.
            assert_eq!(idrs, vec![0, 5, 13], "bframes={b}: {coded:?}");
            let forced = coded.iter().find(|c| c.display == 5).unwrap();
            assert_eq!(forced.poc, 0, "bframes={b}: a forced IDR restarts POC");
            // Nothing predicts forwards across it: whatever was held back
            // before display 5 is coded before it, and none of it as a B.
            let at = coded.iter().position(|c| c.display == 5).unwrap();
            assert!(coded[..at].iter().all(|c| c.display < 5), "bframes={b}: {coded:?}");
            assert!(coded[at..].iter().all(|c| c.display >= 5), "bframes={b}: {coded:?}");
            // With bframes=2, display 4 was held back waiting for an anchor
            // at 6 when the request came; the forced IDR releases it, and it
            // cannot be a B because nothing may predict across the IDR.
            let held = coded.iter().find(|c| c.display == 4).unwrap();
            assert_eq!(held.kind, Kind::P, "bframes={b}: {coded:?}");
            // Every picture is still coded exactly once.
            let mut seen: Vec<u64> = coded.iter().map(|c| c.display).collect();
            seen.sort_unstable();
            assert_eq!(seen, (0..20).collect::<Vec<u64>>(), "bframes={b}");
            // The request is consumed by the IDR it produced: it does not
            // linger and turn the picture after it into a second IDR.
            let after = coded.iter().find(|c| c.display == 6).unwrap();
            assert_ne!(after.kind, Kind::Idr, "bframes={b}: {after:?}");
        }
    }

    /// A preview taken at any point of a stream names every picture inside
    /// its horizon exactly as the real run later codes it — the IDR
    /// positions, the B pictures, the ones a forced IDR turns into P — up
    /// to the stream's end, where a flushed B is the one thing it cannot
    /// foresee.
    #[test]
    fn preview_matches_push() {
        for &(gop, b) in &[(0u32, 0u32), (8, 0), (8, 1), (8, 2), (8, 3), (250, 2)] {
            for n in [1u64, 5, 8, 9, 20, 33] {
                for ahead in [1u64, 2, 4, 8] {
                    let mut s = Scheduler::new(gop, b);
                    let mut real = Vec::new();
                    // (time, what was previewed then, how many pictures
                    // the real run had released by then).
                    let mut previews = Vec::new();
                    for t in 0..n {
                        if t == 7 {
                            s.force_idr();
                        }
                        previews.push((t, s.preview(ahead.min(n - t)), real.len() as u64));
                        real.extend(s.push());
                    }
                    real.extend(s.flush());
                    for (t, preview, released) in previews {
                        let horizon = (t + ahead).min(n);
                        // Every unreleased picture inside the horizon
                        // appears exactly once, and none from outside it.
                        let mut seen: Vec<u64> = preview.iter().map(|c| c.display).collect();
                        seen.sort_unstable();
                        assert!(seen.iter().all(|d| *d < horizon), "gop={gop} b={b} n={n} t={t}: {seen:?} past the horizon {horizon}");
                        assert!(seen.windows(2).all(|w| w[0] != w[1]), "gop={gop} b={b} n={n} t={t}: a display previewed twice: {seen:?}");
                        for r in real.iter().filter(|r| r.display < horizon && r.encode >= released) {
                            assert!(seen.contains(&r.display), "gop={gop} b={b} n={n} t={t}: display {} was unreleased inside the horizon and not previewed: {seen:?}", r.display);
                        }
                        for p in &preview {
                            let r = real.iter().find(|c| c.display == p.display).expect("previewed a picture the run never coded");
                            // The kinds agree except at the stream's end,
                            // where flush makes a P of a held B, and past
                            // an IDR forced *after* the preview was taken,
                            // which nothing could have foreseen.
                            let flushed_b = p.kind == Kind::B && r.kind == Kind::P && p.display + u64::from(b) >= n;
                            let forced_later = t < 7 && p.display >= 7;
                            assert!(p.kind == r.kind || flushed_b || forced_later, "gop={gop} b={b} n={n} t={t}: display {} previewed as {:?}, coded as {:?}", p.display, p.kind, r.kind);
                        }
                    }
                }
            }
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
