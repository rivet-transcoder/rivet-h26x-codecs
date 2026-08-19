// wasm_run.mjs <module.wasm> <stream> — decode the stream inside the module
// and print "<frames> <hash>", the hash as the 0x… literal `tests/decode.rs`
// asserts on. Driven by tools/wasm.sh; see there for why this exists.
//
// The module imports nothing: wasm32-unknown-unknown std reaches for no host
// functions on this path, which is the point — the decoder runs in a bare
// sandbox with no clock, no files and no threads.
import { readFileSync } from "node:fs";

const [, , wasmPath, streamPath] = process.argv;
if (!wasmPath || !streamPath) {
  console.error("usage: wasm_run.mjs <module.wasm> <stream>");
  process.exit(2);
}

const stream = streamPath === "--rung" ? Buffer.alloc(0) : readFileSync(streamPath);
// The codec is the extension's, matching how the decoder's own example picks.
const hevc = /\.(265|hevc|h265)$/i.test(streamPath) ? 1 : 0;

let instance;
try {
  ({ instance } = await WebAssembly.instantiate(readFileSync(wasmPath), {}));
} catch (e) {
  console.error(`instantiate failed: ${e.message}`);
  process.exit(1);
}

const { memory, h26x_scratch, h26x_decode, h26x_rung } = instance.exports;

// --rung: which kernels the module was built with. A tier that installed
// nothing would pass every decode comparison without running one vector
// instruction, so the script checks this separately.
if (streamPath === "--rung") {
  const p = h26x_scratch(16);
  const n = h26x_rung(p);
  console.log(new TextDecoder().decode(new Uint8Array(memory.buffer, p, n)));
  process.exit(0);
}
const inPtr = h26x_scratch(stream.length);
new Uint8Array(memory.buffer, inPtr, stream.length).set(stream);
const hashPtr = h26x_scratch(8);

let frames;
try {
  frames = h26x_decode(inPtr, stream.length, hevc, hashPtr);
} catch (e) {
  // A panic inside the module traps, which is how "it compiled but cannot run
  // here" shows up. Say so rather than reporting a wrong hash.
  console.error(`trapped during decode: ${e.message}`);
  process.exit(1);
}
if (frames === 0xffffffff) {
  console.error("the decoder refused the stream");
  process.exit(1);
}

// The buffer may have been replaced if the module grew its memory.
const bytes = new Uint8Array(memory.buffer, hashPtr, 8);
let hash = 0n;
for (let i = 7; i >= 0; i--) hash = (hash << 8n) | BigInt(bytes[i]);
console.log(`${frames} 0x${hash.toString(16).padStart(16, "0")}`);
