// Analysis worker: the whole pipeline (MP4 read, H.264 decode, plate detection / reading / tracking,
// drawing) is C++ in analyzer.wasm. This file only moves bytes between the page and the module and
// yields to the event loop so the browser's decoder (WebCodecs) can hand frames back.
//
// in : {type:'init'}
//      {type:'analyze', file: File|Blob, name, cfg: {coarse, dense, full640}}
//      {type:'stop'}
// out: {type:'log', text} | {type:'ready', ...} | {type:'frame', w, h, buf, state}
//      {type:'state', state} | {type:'results', list} | {type:'done', state} | {type:'error', text}
let M = null;
let running = false, stopRequested = false;
let seenResults = new Set();

const post = (m, transfer) => self.postMessage(m, transfer || []);
const log = (text) => post({ type: 'log', text });

// setTimeout(0) is clamped to 4 ms after a few nested calls; a MessageChannel yield is not.
const ch = new MessageChannel();
const pending = [];
ch.port1.onmessage = () => { const f = pending.shift(); if (f) f(); };
const yieldNow = () => new Promise((r) => { pending.push(r); ch.port2.postMessage(0); });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function fetchBytes(url, label) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(url + ': HTTP ' + r.status);
  // content-length is only a hint for the progress display: a gzip-serving host (GitHub Pages)
  // reports the compressed size while the stream delivers the decompressed bytes.
  const total = +r.headers.get('content-length') || 0;
  if (!r.body) return new Uint8Array(await r.arrayBuffer());
  const reader = r.body.getReader();
  const chunks = [];
  let off = 0, lastPct = -1;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    off += value.length;
    const pct = total ? Math.min(99, Math.floor(off * 100 / total)) : -1;
    if (pct >= 0 && pct !== lastPct && pct % 10 === 0) { lastPct = pct; log(label + ' ' + pct + '%'); }
  }
  const buf = new Uint8Array(off);
  let p = 0;
  for (const c of chunks) { buf.set(c, p); p += c.length; }
  return buf;
}

function loadModel(kind, bytes) {
  const p = M._malloc(bytes.length);
  M.HEAPU8.set(bytes, p);
  const r = M._az_load_model(kind, p, bytes.length);
  M._free(p);
  return r;
}

function readImage(id, which, size) {
  const p = M._az_result_image(id, which);
  if (!p || !size[0]) return null;
  const n = size[0] * size[1] * 4;
  return { w: size[0], h: size[1], buf: M.HEAPU8.slice(p, p + n).buffer };
}

function postResults() {
  const list = JSON.parse(M.UTF8ToString(M._az_results()));
  const transfer = [];
  for (const r of list) {
    if (seenResults.has(r.id + ':' + r.frames)) continue;  // pictures only for new / changed entries
    seenResults.add(r.id + ':' + r.frames);
    r.thumbImg = readImage(r.id, 0, r.thumb);
    r.snapImg = readImage(r.id, 1, r.snap);
    for (const im of [r.thumbImg, r.snapImg]) if (im) transfer.push(im.buf);
  }
  post({ type: 'results', list }, transfer);
}

async function init() {
  self.importScripts('analyzer.js');
  M = await createAnalyzer({ locateFile: (f) => f });
  const models = [
    [0, 'models/plate_det_v8n_640.onnx', '検出モデル 640'],
    [1, 'models/plate_det_v8n_320.onnx', '検出モデル 320'],
    [2, 'models/plate_ocr_v7_bal.onnx', '認識モデル'],
    [3, 'models/plate_corner.onnx', '4隅補正モデル'],
  ];
  const bytes = await Promise.all(models.map((m) => fetchBytes(m[1], m[2])));
  const nodes = models.map((m, i) => loadModel(m[0], bytes[i]));
  const spec = await (await fetch('models/labels.txt')).text();
  const n = M.lengthBytesUTF8(spec) + 1;
  const sp = M._malloc(n);
  M.stringToUTF8(spec, sp, n);
  const groups = M._az_load_spec(sp);
  M._free(sp);
  post({ type: 'ready', nodes, groups, webcodecs: typeof VideoDecoder !== 'undefined' });
}

async function analyze(m) {
  running = true;
  stopRequested = false;
  seenResults = new Set();
  try {
    try { M.FS.unmount('/input'); } catch (e) {}
    try { M.FS.mkdir('/input'); } catch (e) {}
    const name = 'video.mp4';
    M.FS.mount(M.WORKERFS, { blobs: [{ name, data: m.file }] }, '/input');  // no copy: read straight from the File
    const c = m.cfg || {};
    M._az_config(c.coarse || 0.5, c.dense || 0.2, 0, c.full640 === false ? 0 : 1);
    const pp = M._malloc(64);
    M.stringToUTF8('/input/' + name, pp, 64);
    const ok = M._az_open(pp);
    M._free(pp);
    const state = () => JSON.parse(M.UTF8ToString(M._az_state()));
    if (!ok) throw new Error(state().error || 'open failed');
    post({ type: 'state', state: state() });
    let lastResults = 0, lastStatePost = 0;
    for (;;) {
      if (stopRequested) { post({ type: 'stopped', state: state() }); break; }
      const r = M._az_step();
      if (r === 1) {
        const s = state();
        const p = M._az_frame();
        const buf = M.HEAPU8.slice(p, p + s.dw * s.dh * 4).buffer;
        post({ type: 'frame', w: s.dw, h: s.dh, buf, state: s }, [buf]);
        if (s.results !== lastResults) { lastResults = s.results; postResults(); }
        await yieldNow();
      } else if (r === 2) {
        const now = performance.now();
        if (now - lastStatePost > 100) { lastStatePost = now; post({ type: 'state', state: state() }); }
        await yieldNow();
      } else if (r === 0) {
        await sleep(2);  // the browser's decoder is working
      } else if (r === 3) {
        postResults();
        post({ type: 'done', state: state() });
        break;
      } else {
        throw new Error(state().error || 'analysis failed');
      }
    }
  } catch (e) {
    post({ type: 'error', text: String((e && e.message) || e) });
  } finally {
    M._az_close();
    running = false;
  }
}

self.onmessage = (e) => {
  const m = e.data;
  if (m.type === 'init') init().catch((err) => post({ type: 'error', text: String((err && err.message) || err) }));
  else if (m.type === 'analyze' && !running) analyze(m);
  else if (m.type === 'stop') stopRequested = true;
};
