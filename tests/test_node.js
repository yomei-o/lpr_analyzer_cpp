// Headless end-to-end test of docs/analyzer.wasm: the same module the page runs, on the sample video.
// node has no WebCodecs, so mediakit falls back to FFmpeg's software H.264 decoder.
//
//   node tests/test_node.js [video.mp4]
const fs = require('fs');
const path = require('path');

const docs = path.join(__dirname, '..', 'docs');
const video = process.argv[2] || path.join(docs, 'samples', 'IMG_7121.mp4');
// plates a person can read in the sample; the analyzer must report every one of them
const EXPECTED = ['12-09', '・・・9', '・・・6', '22-56', '13-08'];

(async () => {
  const createAnalyzer = require(path.join(docs, 'analyzer.js'));
  const M = await createAnalyzer();
  const load = (kind, file) => {
    const b = fs.readFileSync(path.join(docs, 'models', file));
    const p = M._malloc(b.length);
    M.HEAPU8.set(b, p);
    const r = M._az_load_model(kind, p, b.length);
    M._free(p);
    if (r <= 0) throw new Error('model failed: ' + file);
  };
  load(0, 'plate_det_v8n_640.onnx');
  load(1, 'plate_det_v8n_320.onnx');
  load(2, path.join('..', 'models_bsd', 'plate_ocr_syn1.onnx'));
  load(3, 'plate_corner.onnx');
  const spec = fs.readFileSync(path.join(docs, 'models', 'labels.txt'), 'utf8');
  const n = M.lengthBytesUTF8(spec) + 1, sp = M._malloc(n);
  M.stringToUTF8(spec, sp, n);
  M._az_load_spec(sp);

  M.FS.writeFile('/in.mp4', fs.readFileSync(video));
  M._az_config(0.5, 0.2, 0, 1, 0.5);   // min_legible 0.5: what the page uses for the BSD model set
  const pp = M._malloc(16);
  M.stringToUTF8('/in.mp4', pp, 16);
  const state = () => JSON.parse(M.UTF8ToString(M._az_state()));
  if (!M._az_open(pp)) throw new Error(state().error);

  const t0 = Date.now();
  for (;;) {
    const r = M._az_step();
    if (r === 1) {
      const s = state();
      const dets = s.dets.map((d) => `#${d.track} ${d.number || '?'} (${d.num_conf})`).join(' | ');
      console.log(`t=${s.t.toFixed(2).padStart(6)} ${s.mode.padEnd(4)} ${s.ms.toFixed(0).padStart(5)} ms  ${dets}`);
    } else if (r === 0) {
      await new Promise((res) => setTimeout(res, 1));
    } else if (r === 3) {
      break;
    } else if (r < 0) {
      throw new Error(state().error);
    }
  }
  const s = state();
  const results = JSON.parse(M.UTF8ToString(M._az_results()));
  console.log(`\n${s.analyzed} frames analysed, ${(s.total_ms / 1000).toFixed(1)} s compute, ${((Date.now() - t0) / 1000).toFixed(1)} s wall, decoder ${s.decoder}`);
  for (const r of results) console.log(`${r.time}  ${r.number.padEnd(6)}  ${r.text}  (num ${r.num_conf}, ${r.frames} frames)`);
  const got = new Set(results.map((r) => r.number));
  const missing = EXPECTED.filter((x) => !got.has(x));
  if (missing.length) {
    console.log('FAIL: missing ' + missing.join(', '));
    process.exit(1);
  }
  console.log('PASS: all expected plates found');
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
