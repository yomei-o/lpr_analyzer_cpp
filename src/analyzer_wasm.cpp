// Browser (wasm) driver: MP4 -> H.264 decode (mediakit: FFmpeg demux + the browser's WebCodecs, or
// FFmpeg's software decoder where there is no WebCodecs, e.g. node) -> RGB -> lpr_analyzer.
//
// It runs in a Web Worker. The worker calls az_step() in a loop and yields to the event loop between
// calls, because WebCodecs hands decoded frames back asynchronously:
//
//   az_step() ->  0  waiting for the decoder: yield a moment and call again
//                 1  a frame was analysed: the drawn picture (az_frame) and az_state() are new
//                 2  progress only (frames skipped / packets fed)
//                 3  finished: every plate has been closed into the result list
//                -1  error (az_state() has "error")
#include <emscripten/emscripten.h>

#include <algorithm>
#include <deque>
#include <string>
#include <vector>

#include "lpr_analyzer.hpp"
#include "mk_codec.h"

namespace {

onx::Graph g_det640, g_det320, g_ocr, g_corner;
bool g_ok[4] = {false, false, false, false};
spec::Spec g_spec;
bool g_spec_ok = false;

lpa::Analyzer g_az;
lpa::Image g_disp;
std::string g_state, g_results;

struct Session {
  mk::Mp4Reader reader;
  mk::VideoDecoder dec;
  mk::FrameConverter conv;
  mk::VideoFrame rgb;
  std::deque<mk::VideoFrame> queue;
  bool open = false, eof = false, done = false;
  int64_t fed = 0, got = 0, first_pts = INT64_MIN;
  int waits = 0;
  double duration = 0, t = 0, last_ms = 0, total_ms = 0;
  int width = 0, height = 0;
  std::string error;
} * g_s = nullptr;

int g_use_full640 = 1;  // 0: the light 320 detector for full frames too (fast, misses small plates)

double now_ms() { return emscripten_get_now(); }

std::string esc(const std::string& s) { return jl::json_escape(s); }

void build_state(bool with_dets) {
  Session& s = *g_s;
  char b[512];
  std::snprintf(b, sizeof b,
                "{\"t\":%.3f,\"duration\":%.3f,\"w\":%d,\"h\":%d,\"dw\":%d,\"dh\":%d,\"analyzed\":%d,\"mode\":\"%s\",\"rois\":%zu,"
                "\"ms\":%.1f,\"total_ms\":%.1f,\"tracks\":%zu,\"done\":%s,\"decoder\":\"%s\",\"results\":%zu,\"error\":\"%s\"",
                s.t, s.duration, s.width, s.height, g_disp.w, g_disp.h, g_az.frames_analyzed(), g_az.last_was_full() ? "full" : "roi",
                g_az.roi_count(), s.last_ms, s.total_ms, g_az.tracks().size(), s.done ? "true" : "false", esc(s.dec.name()).c_str(),
                g_az.results().size(), esc(s.error).c_str());
  g_state = b;
  g_state += ",\"dets\":[";
  if (with_dets) {
    bool first = true;
    for (const lpa::Sighting& si : g_az.last()) {
      std::string num;
      for (const lpa::Track& tr : g_az.tracks())
        if (tr.id == si.track && tr.frames > 0) num = lpa::number_display(tr.plate) + "\",\"text\":\"" + esc(tr.plate.text);
      std::snprintf(b, sizeof b, "%s{\"box\":[%.0f,%.0f,%.0f,%.0f],\"det\":%.2f,\"num_conf\":%.2f,\"track\":%d,\"number\":\"", first ? "" : ",",
                    si.box.x1, si.box.y1, si.box.x2, si.box.y2, si.box.score, si.num_conf, si.track);
      g_state += b;
      g_state += num.empty() ? "\",\"text\":\"" : num;
      g_state += "\"}";
      first = false;
    }
  }
  g_state += "]}";
}

}  // namespace

extern "C" {

// kind: 0 full-frame detector (640), 1 window detector (320), 2 recognizer, 3 corner net
EMSCRIPTEN_KEEPALIVE int az_load_model(int kind, const unsigned char* buf, int len) {
  onx::Graph* g[4] = {&g_det640, &g_det320, &g_ocr, &g_corner};
  if (kind < 0 || kind > 3) return -1;
  *g[kind] = onx::parse_onnx(buf, (size_t)len);
  g_ok[kind] = !g[kind]->nodes.empty();
  return g_ok[kind] ? (int)g[kind]->nodes.size() : -1;
}

EMSCRIPTEN_KEEPALIVE int az_load_spec(const char* text) {
  g_spec = spec::parse(std::string(text));
  g_spec_ok = !g_spec.groups.empty();
  return g_spec_ok ? (int)g_spec.groups.size() : -1;
}

// coarse / dense intervals in seconds, full640: 1 = yolov8n 640 on full frames, 0 = 320
EMSCRIPTEN_KEEPALIVE void az_config(double coarse, double dense, float min_num_conf, int full640, float min_legible) {
  g_az.cfg.min_legible = min_legible;   // 0 = off; the BSD recognizer marks non-plates with legible = no
  if (coarse > 0) g_az.cfg.coarse_interval = coarse;
  if (dense > 0) g_az.cfg.dense_interval = dense;
  if (min_num_conf > 0) g_az.cfg.min_num_conf = min_num_conf;
  g_use_full640 = full640;
}

// Open an MP4 (a path in the module's file system). Returns 1, or 0 with the reason in az_state().
EMSCRIPTEN_KEEPALIVE int az_open(const char* path) {
  delete g_s;
  g_s = new Session();
  g_az.reset();
  g_disp = {};
  Session& s = *g_s;
  if (!g_ok[2] || !g_spec_ok || (!g_ok[0] && !g_ok[1])) {
    s.error = "models not loaded";
  } else if (!s.reader.open(path)) {
    s.error = "cannot open the video: " + s.reader.error();
  } else if (!s.reader.info().has_video) {
    s.error = "no video track";
  } else if (!s.dec.open(true)) {
    s.error = "no H.264 decoder: " + s.dec.error();
  } else {
    s.open = true;
    s.duration = s.reader.info().duration_us / 1e6;
    s.width = s.reader.info().width;
    s.height = s.reader.info().height;
  }
  build_state(false);
  return s.open ? 1 : 0;
}

EMSCRIPTEN_KEEPALIVE int az_step() {
  if (!g_s || !g_s->open) return -1;
  Session& s = *g_s;
  if (s.done) return 3;
  lpa::Models m;
  m.det_full = g_use_full640 && g_ok[0] ? &g_det640 : &g_det320;
  m.det_roi = g_ok[1] ? &g_det320 : &g_det640;
  m.ocr = &g_ocr;
  m.corner = g_ok[3] ? &g_corner : nullptr;
  m.spec = &g_spec;

  const double t0 = now_ms();
  std::vector<mk::VideoFrame> out;
  for (;;) {
    // 1. decoded frames, in order: analyse the ones the analyzer wants, drop the rest
    while (!s.queue.empty()) {
      mk::VideoFrame f = std::move(s.queue.front());
      s.queue.pop_front();
      if (s.first_pts == INT64_MIN) s.first_pts = f.pts_us;
      s.t = (f.pts_us - s.first_pts) / 1e6;
      if (!g_az.wants(s.t)) continue;
      if (!s.conv.convert(f, s.rgb, mk::PixelFormat::RGB24)) {
        s.error = "pixel conversion failed";
        build_state(false);
        return -1;
      }
      s.width = s.rgb.width, s.height = s.rgb.height;
      const double a0 = now_ms();
      g_az.process(m, s.rgb.data[0], s.rgb.width, s.rgb.height, s.t);
      g_az.render(s.rgb.data[0], s.rgb.width, s.rgb.height, g_disp);
      s.last_ms = now_ms() - a0;
      s.total_ms += s.last_ms;
      build_state(true);
      return 1;
    }
    // 2. feed the decoder, a few frames ahead at most (decoded 1080p frames are 3 MB each)
    out.clear();
    s.dec.poll(out);  // WebCodecs frames that arrived while we were away
    s.got += (int64_t)out.size();
    for (mk::VideoFrame& f : out) s.queue.push_back(std::move(f));
    if (!s.queue.empty()) continue;
    if (!s.eof) {
      const int64_t inflight = s.fed - s.got;
      if (inflight >= 12 && s.waits < 200) {
        ++s.waits;
        return 0;  // the browser has not handed frames back yet
      }
      s.waits = 0;
      mk::Packet p;
      if (!s.reader.read(p)) {
        s.eof = true;
        continue;
      }
      if (p.type != mk::MediaType::Video) continue;
      out.clear();
      if (!s.dec.decode(p, out)) {
        s.error = "decode: " + s.dec.error();
        build_state(false);
        return -1;
      }
      ++s.fed;
      s.got += (int64_t)out.size();
      for (mk::VideoFrame& f : out) s.queue.push_back(std::move(f));
    } else {
      out.clear();
      const bool drained = s.dec.drain(out);
      s.got += (int64_t)out.size();
      for (mk::VideoFrame& f : out) s.queue.push_back(std::move(f));
      if (s.queue.empty()) {
        if (!drained) return 0;
        g_az.finish();
        s.done = true;
        build_state(false);
        return 3;
      }
    }
    if (now_ms() - t0 > 30) {  // keep the worker responsive to "stop"
      build_state(false);
      return 2;
    }
  }
}

EMSCRIPTEN_KEEPALIVE const char* az_state() { return g_state.c_str(); }

// The analysed frame with boxes and numbers drawn (RGBA, size in az_state() "dw"/"dh").
EMSCRIPTEN_KEEPALIVE const uint8_t* az_frame() { return g_disp.px.empty() ? nullptr : g_disp.px.data(); }

// The result list, sorted by the time the plate first appeared.
EMSCRIPTEN_KEEPALIVE const char* az_results() {
  std::vector<const lpa::Result*> r;
  for (const lpa::Result& x : g_az.results()) r.push_back(&x);
  std::stable_sort(r.begin(), r.end(), [](const lpa::Result* a, const lpa::Result* b) { return a->t_first < b->t_first; });
  g_results = "[";
  for (size_t i = 0; i < r.size(); ++i) g_results += (i ? "," : "") + lpa::result_json(*r[i]);
  g_results += "]";
  return g_results.c_str();
}

// Pictures of result `id`: which 0 = the plate crop, 1 = the scene. RGBA; size in the result JSON.
EMSCRIPTEN_KEEPALIVE const uint8_t* az_result_image(int id, int which) {
  for (const lpa::Result& x : g_az.results())
    if (x.id == id) {
      const lpa::Image& im = which ? x.snap : x.thumb;
      return im.px.empty() ? nullptr : im.px.data();
    }
  return nullptr;
}

EMSCRIPTEN_KEEPALIVE void az_close() {
  delete g_s;
  g_s = nullptr;
}

}  // extern "C"
