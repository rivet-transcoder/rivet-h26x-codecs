// wasm_enc.mjs <module.wasm> <command> — the encode-side half of what
// tools/wasm.sh checks inside the module. Driven by that script; see there.
//
//   --installed              which encode-side kernels the build's tables
//                            took (bitmask, see `h26x_enc_installed`)
//   --selftest               the randomised encode-kernel sweep against the
//                            scalar reference (`h26x_enc_dsp_check`)
//   --bench GROUP SHAPE ITERS  time `h26x_enc_bench` from outside, best of
//                            three, printing ns per call group
//   --encode WxH CODEC QP GOP BFRAMES [file.yuv]
//                            encode raw 8-bit 4:2:0 frames (from the file,
//                            or eight synthesised ones) and decode them back
//                            inside the module; prints "<frames> <stream
//                            bytes> <stream hash> <decoded hash> <recon
//                            hash> <best-of-3 ms>"
//
// The module has no clock, so timing is performance.now() around the export
// call. Best of three, because on a shared machine the minimum is the
// number with the least noise in it.
import { readFileSync } from "node:fs";
import { performance } from "node:perf_hooks";

const [, , wasmPath, cmd, ...rest] = process.argv;
if (!wasmPath || !cmd) {
  console.error("usage: wasm_enc.mjs <module.wasm> --installed | --selftest | --bench G S N | --encode WxH codec qp gop bframes [file]");
  process.exit(2);
}

let instance;
try {
  ({ instance } = await WebAssembly.instantiate(readFileSync(wasmPath), {}));
} catch (e) {
  console.error(`instantiate failed: ${e.message}`);
  process.exit(1);
}
const { memory, h26x_scratch, h26x_enc_installed, h26x_enc_dsp_check, h26x_enc_bench, h26x_encode } = instance.exports;

const hex = (bytes, off) => {
  let v = 0n;
  for (let i = 7; i >= 0; i--) v = (v << 8n) | BigInt(bytes[off + i]);
  return "0x" + v.toString(16).padStart(16, "0");
};

const bestOf3 = (f) => {
  let best = Infinity;
  for (let i = 0; i < 3; i++) {
    const t = performance.now();
    f();
    best = Math.min(best, performance.now() - t);
  }
  return best;
};

try {
  if (cmd === "--installed") {
    console.log(String(h26x_enc_installed()));
  } else if (cmd === "--selftest") {
    const mask = h26x_enc_dsp_check();
    console.log(mask === 0 ? "OK" : `FAILED (group mask ${mask})`);
    process.exit(mask === 0 ? 0 : 1);
  } else if (cmd === "--bench") {
    const [g, s, n] = rest.map(Number);
    // One warm-up call, then the timed ones.
    h26x_enc_bench(g, s, Math.max(1, n >> 4));
    const ms = bestOf3(() => h26x_enc_bench(g, s, n));
    console.log((ms * 1e6 / n).toFixed(1));
  } else if (cmd === "--encode") {
    const [size, codec, qp, gop, bframes, file] = rest;
    const [w, h] = size.split("x").map(Number);
    const fb = w * h + 2 * Math.ceil(w / 2) * Math.ceil(h / 2);
    let raw;
    if (file) {
      raw = readFileSync(file);
    } else {
      // Eight frames of a drifting gradient with noise: enough motion for
      // the inter paths to have something to search for, deterministic so
      // the two builds see the same bytes.
      const frames = 8;
      raw = Buffer.alloc(fb * frames);
      let seed = 12345;
      const lcg = () => {
        seed = (Math.imul(seed, 1103515245) + 12345) >>> 0;
        return seed >>> 16;
      };
      for (let f = 0; f < frames; f++) {
        const base = f * fb;
        for (let y = 0; y < h; y++)
          for (let x = 0; x < w; x++) raw[base + y * w + x] = ((x * 3 + y * 2 + f * 5) + (lcg() % 24)) & 255;
        const cw = Math.ceil(w / 2), ch = Math.ceil(h / 2);
        for (let p = 0; p < 2; p++)
          for (let y = 0; y < ch; y++)
            for (let x = 0; x < cw; x++) raw[base + w * h + p * cw * ch + y * cw + x] = (128 + (p ? -1 : 1) * (x + y + f) * 2 + (lcg() % 8)) & 255;
      }
    }
    const inPtr = h26x_scratch(raw.length);
    new Uint8Array(memory.buffer, inPtr, raw.length).set(raw);
    const outPtr = h26x_scratch(32);
    const hevc = codec === "h265" || codec === "hevc" ? 1 : 0;
    let frames;
    const ms = bestOf3(() => {
      frames = h26x_encode(inPtr, raw.length, w, h, hevc, Number(qp), Number(gop), Number(bframes), outPtr);
    });
    // u32::MAX comes back through the i32 ABI as -1.
    if (frames === -1 || frames === 0xffffffff) {
      console.error("the encoder or decoder refused");
      process.exit(1);
    }
    const out = new Uint8Array(memory.buffer, outPtr, 32);
    let len = 0n;
    for (let i = 7; i >= 0; i--) len = (len << 8n) | BigInt(out[24 + i]);
    console.log(`${frames} ${len} ${hex(out, 0)} ${hex(out, 8)} ${hex(out, 16)} ${ms.toFixed(1)}`);
  } else {
    console.error(`unknown command ${cmd}`);
    process.exit(2);
  }
} catch (e) {
  // A panic inside the module traps, which is how "it compiled but cannot
  // run here" shows up. Say so rather than reporting a wrong number.
  console.error(`trapped: ${e.message}`);
  process.exit(1);
}
