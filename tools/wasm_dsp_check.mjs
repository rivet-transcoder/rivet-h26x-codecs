// wasm_dsp_check.mjs <module.wasm> — run the randomized HEVC kernel sweep
// inside the module (`h26x_hevc_dsp_check` in examples/wasm_probe.rs) and
// print "<rung>: OK" or the number of comparisons that disagreed.
//
// This is the wasm stand-in for the `#[cfg(test)]` modules the x86 kernel
// files carry: wasm32-unknown-unknown has no test harness, so the sweep is
// compiled into the probe and driven from here. Run it on the +simd128 build
// to check the simd128 tier; on the scalar build it compares the scalar
// table with itself, which only proves the sweep does not trap.
import { readFileSync } from "node:fs";

const [, , wasmPath] = process.argv;
if (!wasmPath) {
  console.error("usage: wasm_dsp_check.mjs <module.wasm>");
  process.exit(2);
}

let instance;
try {
  ({ instance } = await WebAssembly.instantiate(readFileSync(wasmPath), {}));
} catch (e) {
  console.error(`instantiate failed: ${e.message}`);
  process.exit(1);
}

const { memory, h26x_scratch, h26x_rung, h26x_hevc_dsp_check } = instance.exports;
const p = h26x_scratch(16);
const n = h26x_rung(p);
const rung = new TextDecoder().decode(new Uint8Array(memory.buffer, p, n));

let fails;
try {
  fails = h26x_hevc_dsp_check();
} catch (e) {
  console.error(`${rung}: trapped during the sweep: ${e.message}`);
  process.exit(1);
}
console.log(fails === 0 ? `${rung}: OK` : `${rung}: ${fails} kernel comparisons FAILED`);
process.exit(fails === 0 ? 0 : 1);
