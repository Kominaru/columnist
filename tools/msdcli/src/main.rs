// Reads charts from stdin: a header line "<id> <keys> <n>" then n lines "<row_time_s> <mask>".
// Prints "<id> overall stream jumpstream handstream stamina jackspeed chordjack technical" (MSD at 1.0x).
use minacalc_rs::{Calc, CalcMode, Note};
use std::io::{self, BufRead, Write};

fn main() {
    let calc = Calc::new().expect("calc");
    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();
    let out = io::stdout();
    let mut out = out.lock();
    while let Some(Ok(h)) = lines.next() {
        let p: Vec<&str> = h.split_whitespace().collect();
        if p.len() < 3 { continue; }
        let (id, keys, n): (&str, u32, usize) = (p[0], p[1].parse().unwrap(), p[2].parse().unwrap());
        let mut notes = Vec::with_capacity(n);
        for _ in 0..n {
            let l = lines.next().unwrap().unwrap();
            let mut it = l.split_whitespace();
            let t: f32 = it.next().unwrap().parse().unwrap();
            let m: u32 = it.next().unwrap().parse().unwrap();
            notes.push(Note { notes: m, row_time: t });
        }
        match calc.calc_at_rate(&notes, 1.0, 0.93, keys, CalcMode::Msd) {
            Ok(s) => writeln!(out, "{} {} {} {} {} {} {} {} {}", id, s.overall, s.stream, s.jumpstream, s.handstream,
                              s.stamina, s.jackspeed, s.chordjack, s.technical).unwrap(),
            Err(e) => writeln!(out, "{} ERR {:?}", id, e).unwrap(),
        }
        out.flush().unwrap();
    }
}
