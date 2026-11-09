// lpr_analyzer: find the number plates that pass through a video and when they pass.
//
// The video is fed in as RGB frames with timestamps (any source: the browser's decoder via
// mediakit, ffmpeg piping raw frames, ...). The analyzer decides which frames it wants:
//
//   coarse pass  every `coarse_interval` s (0.5 s = 2 fps by default) the whole frame goes through the
//                full detector (yolov8n 640) -> plates that are new to us.
//   dense pass   while a plate is being followed, every `dense_interval` s (0.2 s = 5 fps) only a
//                square window around where it is expected goes through the light detector
//                (yolov8n 320). Small crop + small net = cheap, and the plate is seen by many
//                slightly different frames.
//
// Each sighting is read (4-corner rectification + per-position classifier, from yolo_lpr_cpp) and
// its per-head probabilities are summed into the plate's track, so the reported number is the one
// the evidence over all frames supports, not the guess of a single frame. A track that has not been
// seen for `track_timeout` s is closed and becomes one entry of the result list: time of passing,
// the plate, a crop of the plate and a snapshot of the scene.
//
// Drawing (boxes, the 4-digit number, search windows) is done here too, into an RGBA picture the
// page only has to blit.
#pragma once
#include "pipeline.hpp"
#include "spec.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace lpa {

struct Config {
  double coarse_interval = 0.5;   // s between full-frame scans (2 fps)
  double dense_interval = 0.2;    // s between re-reads of a followed plate (5 fps)
  double track_timeout = 1.0;     // s unseen -> the plate has passed
  float det_conf = 0.30f;         // full-frame detector threshold
  float roi_det_conf = 0.25f;     // window detector threshold
  float min_num_conf = 0.55f;     // mean confidence of the 4 number positions for a plate to count
  // P(legible = yes) a sighting needs to count as a plate. 0 = off. The recognizers trained with
  // negatives (train/train_ocr_real.py --neg) use this head to say "this is not a plate".
  float min_legible = 0.f;
  bool tta = false;               // read small plates from 6 margins instead of 1 (6x the reading cost)
  int display_max_w = 1280;       // width of the picture handed to the page
  int thumb_h = 56;               // plate crop height in the result list
  int snap_w = 480;               // scene snapshot width in the result list
};

struct Image {  // RGBA, tightly packed
  int w = 0, h = 0;
  std::vector<uint8_t> px;
};

// ---------------------------------------------------------------- the 4-digit number
// Japanese plates show 4 digits as "12-34" and fewer with leading dots: "・123", "・・12", "・・・1".
inline std::string number_display(const spec::Plate& p) {
  if (p.num.size() == 4) return p.num.substr(0, 2) + "-" + p.num.substr(2, 2);
  return p.disp;
}

// Mean confidence of the four number positions (heads 5..8): what decides "this is a plate we read".
inline float number_conf(const std::vector<float>& conf) {
  if (conf.size() < 9) return 0.f;
  return (conf[5] + conf[6] + conf[7] + conf[8]) / 4.f;
}

// P(legible = yes) from the 11th head (spec: no / yes); 1 when the model has no such head.
inline float legible_prob(const std::vector<std::vector<float>>& probs) {
  if (probs.size() < 11 || probs[10].size() < 2) return 1.f;
  return probs[10][1];
}

// ---------------------------------------------------------------- drawing
namespace draw {

// 5x7 glyphs, one byte per row (low 5 bits, MSB = left).
struct Glyph { char c; uint8_t rows[7]; };
inline const Glyph* glyph(char c) {
  static const Glyph g[] = {
      {'0', {0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E}}, {'1', {0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E}},
      {'2', {0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F}}, {'3', {0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E}},
      {'4', {0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02}}, {'5', {0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E}},
      {'6', {0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E}}, {'7', {0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08}},
      {'8', {0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E}}, {'9', {0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C}},
      {'-', {0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00}}, {'.', {0x00, 0x00, 0x00, 0x0C, 0x0C, 0x00, 0x00}},
      {'?', {0x0E, 0x11, 0x01, 0x02, 0x04, 0x00, 0x04}}, {':', {0x00, 0x0C, 0x0C, 0x00, 0x0C, 0x0C, 0x00}},
      {' ', {0, 0, 0, 0, 0, 0, 0}},
  };
  for (const Glyph& x : g)
    if (x.c == c) return &x;
  return nullptr;
}

// UTF-8 '・' (U+30FB) is drawn as the centred dot.
inline std::string to_ascii(const std::string& s) {
  std::string o;
  for (size_t i = 0; i < s.size();) {
    if ((unsigned char)s[i] == 0xE3 && i + 2 < s.size() && (unsigned char)s[i + 1] == 0x83 &&
        (unsigned char)s[i + 2] == 0xBB) {
      o += '.';
      i += 3;
    } else if ((unsigned char)s[i] < 0x80) {
      o += s[i++];
    } else {
      o += '?';
      ++i;
      while (i < s.size() && ((unsigned char)s[i] & 0xC0) == 0x80) ++i;
    }
  }
  return o;
}

inline void fill(Image& im, int x0, int y0, int x1, int y1, uint32_t rgba, int alpha = 255) {
  x0 = std::max(0, x0), y0 = std::max(0, y0), x1 = std::min(im.w, x1), y1 = std::min(im.h, y1);
  const uint8_t c[3] = {(uint8_t)(rgba >> 24), (uint8_t)(rgba >> 16), (uint8_t)(rgba >> 8)};
  for (int y = y0; y < y1; ++y) {
    uint8_t* p = im.px.data() + ((size_t)y * im.w + x0) * 4;
    for (int x = x0; x < x1; ++x, p += 4)
      for (int k = 0; k < 3; ++k) p[k] = (uint8_t)((p[k] * (255 - alpha) + c[k] * alpha) / 255);
  }
}

inline void rect(Image& im, int x0, int y0, int x1, int y1, uint32_t rgba, int t) {
  fill(im, x0, y0, x1, y0 + t, rgba);
  fill(im, x0, y1 - t, x1, y1, rgba);
  fill(im, x0, y0, x0 + t, y1, rgba);
  fill(im, x1 - t, y0, x1, y1, rgba);
}

inline int text_w(const std::string& s, int scale) { return (int)s.size() * 6 * scale - scale; }

inline void text(Image& im, int x, int y, const std::string& s, uint32_t rgba, int scale) {
  for (char c : s) {
    if (const Glyph* g = glyph(c))
      for (int r = 0; r < 7; ++r)
        for (int b = 0; b < 5; ++b)
          if (g->rows[r] & (0x10 >> b)) fill(im, x + b * scale, y + r * scale, x + (b + 1) * scale, y + (r + 1) * scale, rgba);
    x += 6 * scale;
  }
}

// A label on a dark plate, above the box when there is room, else below it.
inline void label(Image& im, int bx0, int by0, int by1, const std::string& s, uint32_t fg, int scale) {
  const int pad = scale * 2, w = text_w(s, scale) + pad * 2, h = 7 * scale + pad * 2;
  int y = by0 - h - scale;
  if (y < 0) y = by1 + scale;
  int x = std::clamp(bx0, 0, std::max(0, im.w - w));
  fill(im, x, y, x + w, y + h, 0x000000ffu, 190);
  text(im, x + pad, y + pad, s, fg, scale);
}

// Area-average downscale of an RGB frame into RGBA.
inline void downscale_rgb(const uint8_t* rgb, int W, int H, int x0, int y0, int cw, int ch, Image& out, int ow, int oh) {
  out.w = ow, out.h = oh;
  out.px.assign((size_t)ow * oh * 4, 255);
  for (int y = 0; y < oh; ++y) {
    const int sy0 = y0 + (int)((int64_t)y * ch / oh), sy1 = std::max(sy0 + 1, y0 + (int)((int64_t)(y + 1) * ch / oh));
    for (int x = 0; x < ow; ++x) {
      const int sx0 = x0 + (int)((int64_t)x * cw / ow), sx1 = std::max(sx0 + 1, x0 + (int)((int64_t)(x + 1) * cw / ow));
      int s[3] = {0, 0, 0}, n = 0;
      for (int sy = sy0; sy < sy1 && sy < H; ++sy)
        for (int sx = sx0; sx < sx1 && sx < W; ++sx) {
          const uint8_t* p = rgb + ((size_t)sy * W + sx) * 3;
          s[0] += p[0], s[1] += p[1], s[2] += p[2], ++n;
        }
      uint8_t* o = out.px.data() + ((size_t)y * ow + x) * 4;
      if (n)
        for (int k = 0; k < 3; ++k) o[k] = (uint8_t)(s[k] / n);
    }
  }
}

// Bilinear crop + resize (for upscaling small plate crops).
inline void crop_rgb(const uint8_t* rgb, int W, int H, float x0, float y0, float x1, float y1, Image& out, int ow, int oh) {
  out.w = ow, out.h = oh;
  out.px.assign((size_t)ow * oh * 4, 255);
  for (int y = 0; y < oh; ++y)
    for (int x = 0; x < ow; ++x) {
      const float sx = std::clamp(x0 + (x + 0.5f) * (x1 - x0) / ow - 0.5f, 0.f, (float)W - 1);
      const float sy = std::clamp(y0 + (y + 0.5f) * (y1 - y0) / oh - 0.5f, 0.f, (float)H - 1);
      const int ix = std::min((int)sx, W - 2 < 0 ? 0 : W - 2), iy = std::min((int)sy, H - 2 < 0 ? 0 : H - 2);
      const float fx = sx - ix, fy = sy - iy;
      uint8_t* o = out.px.data() + ((size_t)y * ow + x) * 4;
      for (int k = 0; k < 3; ++k) {
        auto at = [&](int xx, int yy) { return (float)rgb[((size_t)yy * W + xx) * 3 + k]; };
        const float v = at(ix, iy) * (1 - fx) * (1 - fy) + at(ix + 1, iy) * fx * (1 - fy) + at(ix, iy + 1) * (1 - fx) * fy +
                        at(ix + 1, iy + 1) * fx * fy;
        o[k] = (uint8_t)std::clamp(v + 0.5f, 0.f, 255.f);
      }
    }
}

}  // namespace draw

// ---------------------------------------------------------------- tracking + results

struct Sighting {
  jl::Box box;
  jl::Read read;
  float num_conf = 0;
  float legible = 1;
  int track = 0;  // id of the track it went to (0: not tracked)
  bool cornered = false;  // read through the 4-corner rectification; quad = the corners used
  float quad[8] = {0, 0, 0, 0, 0, 0, 0, 0};
};

struct Track {
  int id = 0;
  double t_first = 0, t_last = 0;
  jl::Box box{};
  jl::Box first_box{};   // where the track started
  float vx = 0, vy = 0;  // px/s
  int frames = 0;
  std::vector<std::vector<double>> sum;  // [head][class], summed over the frames
  std::vector<int> arg;
  std::vector<float> conf;
  spec::Plate plate;
  float best_score = -1;
  float best_legible = 0;  // highest P(legible = yes) among the sightings that voted
  double t_best = 0;
  Image thumb, snap;
};

struct Result {
  int id = 0;
  double t_first = 0, t_last = 0, t_best = 0;
  int frames = 0;
  spec::Plate plate;
  std::string number;  // "12-34"
  std::vector<float> conf;
  Image thumb, snap;
  jl::Box first_box{}, last_box{};  // where the plate was first / last seen (frame pixels), for merging
};

struct Models {
  const onx::Graph* det_full = nullptr;  // yolov8n 640 (cxcywh)
  const onx::Graph* det_roi = nullptr;   // yolov8n 320 (cxcywh); null -> det_full
  const onx::Graph* ocr = nullptr;
  const onx::Graph* corner = nullptr;    // optional
  const spec::Spec* spec = nullptr;
};

class Analyzer {
 public:
  Config cfg;

  void reset() {
    tracks_.clear();
    results_.clear();
    last_.clear();
    rois_.clear();
    next_id_ = 1;
    next_coarse_ = 0;
    last_dense_ = -1e9;
    frames_analyzed_ = 0;
    last_full_ = false;
  }

  // Does the analyzer want the frame at time t (seconds)? Frames it does not want are skipped
  // without being converted.
  bool wants(double t) const {
    if (t >= next_coarse_ - 1e-6) return true;
    return !tracks_.empty() && t >= last_dense_ + cfg.dense_interval - 1e-3;
  }

  // Analyse one RGB frame. Returns the number of plates seen in it.
  int process(const Models& m, const uint8_t* rgb, int W, int H, double t) {
    W_ = W, H_ = H;
    spec_of_ = m.spec;
    const bool full = t >= next_coarse_ - 1e-6;
    if (full) next_coarse_ = std::max(next_coarse_ + cfg.coarse_interval, t + cfg.coarse_interval * 0.5);
    last_dense_ = t;
    last_full_ = full;
    ++frames_analyzed_;
    t_ = t;

    // 1. candidate boxes: the whole frame on a coarse step, windows around the followed plates always
    std::vector<jl::Box> boxes;
    if (full) boxes = detect(*m.det_full, rgb, W, H, 0, 0, W, H, cfg.det_conf);
    rois_.clear();
    for (const Track& tr : tracks_) {
      const double dt = t - tr.t_last;
      const float cx = (tr.box.x1 + tr.box.x2) / 2 + tr.vx * (float)dt, cy = (tr.box.y1 + tr.box.y2) / 2 + tr.vy * (float)dt;
      const float bw = tr.box.x2 - tr.box.x1, bh = tr.box.y2 - tr.box.y1;
      int side = (int)std::max({bw * 5.f, bh * 10.f, 256.f});
      side = std::min({side, W, H});
      const int x0 = std::clamp((int)(cx - side / 2), 0, W - side), y0 = std::clamp((int)(cy - side / 2), 0, H - side);
      bool covered = false;  // a window inside one we already have adds nothing
      for (const jl::Box& r : rois_)
        if (x0 >= r.x1 && y0 >= r.y1 && x0 + side <= r.x2 && y0 + side <= r.y2) covered = true;
      if (covered) continue;
      rois_.push_back({(float)x0, (float)y0, (float)(x0 + side), (float)(y0 + side), 0});
      for (const jl::Box& b : detect(m.det_roi ? *m.det_roi : *m.det_full, rgb, W, H, x0, y0, side, side, cfg.roi_det_conf))
        boxes.push_back(b);
    }
    boxes = nms(boxes, 0.4f);

    // 2. read every box
    last_.clear();
    jl::CornerCfg ccfg;
    for (const jl::Box& b : boxes) {
      Sighting s;
      s.box = b;
      float corners[8];
      if (m.corner && jl::predict_corners(*m.corner, rgb, W, H, b, ccfg, corners)) {
        s.read = jl::read_plate_warped(*m.ocr, *m.spec, rgb, W, H, corners, ccfg);
        s.cornered = true;
        std::copy(corners, corners + 8, s.quad);
      } else
        // one crop: the margin TTA of a still photo is replaced by voting over the frames of the track
        s.read = cfg.tta ? jl::read_plate_tta(*m.ocr, *m.spec, rgb, W, H, b.x1, b.y1, b.x2, b.y2)
                         : jl::read_plate_single(*m.ocr, *m.spec, rgb, W, H, b.x1, b.y1, b.x2, b.y2);
      s.num_conf = number_conf(s.read.conf);
      s.legible = legible_prob(s.read.probs);
      if (s.legible < cfg.min_legible) s.num_conf = 0.f;   // "not a plate": neither starts nor feeds a track
      last_.push_back(std::move(s));
    }

    // 3. match to tracks (nearest predicted centre, greedy), fold the probabilities in
    std::vector<char> used(tracks_.size(), 0);
    std::vector<int> order(last_.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = (int)i;
    std::sort(order.begin(), order.end(), [&](int a, int b) { return last_[a].num_conf > last_[b].num_conf; });
    for (int i : order) {
      Sighting& s = last_[(size_t)i];
      const float cx = (s.box.x1 + s.box.x2) / 2, cy = (s.box.y1 + s.box.y2) / 2;
      int best = -1;
      float best_d = 1e9f;
      for (size_t k = 0; k < tracks_.size(); ++k) {
        if (used[k]) continue;
        const Track& tr = tracks_[k];
        const double dt = t - tr.t_last;
        const float px = (tr.box.x1 + tr.box.x2) / 2 + tr.vx * (float)dt, py = (tr.box.y1 + tr.box.y2) / 2 + tr.vy * (float)dt;
        const float gate = 1.5f * std::max({tr.box.x2 - tr.box.x1, s.box.x2 - s.box.x1, 40.f});
        const float d = std::hypot(cx - px, cy - py);
        if (d < gate && d < best_d) best_d = d, best = (int)k;
      }
      const bool readable = s.num_conf >= cfg.min_num_conf;
      if (best < 0) {
        if (!readable) continue;  // only a plate we can read starts a track (keeps signs out)
        Track nt;
        nt.id = next_id_++;
        nt.t_first = t;
        nt.box = s.box;
        nt.first_box = s.box;
        nt.t_last = t;
        tracks_.push_back(std::move(nt));
        used.push_back(1);
        best = (int)tracks_.size() - 1;
      } else {
        used[(size_t)best] = 1;
      }
      Track& tr = tracks_[(size_t)best];
      const double dt = t - tr.t_last;
      if (dt > 1e-3) {
        const float nvx = ((s.box.x1 + s.box.x2) / 2 - (tr.box.x1 + tr.box.x2) / 2) / (float)dt;
        const float nvy = ((s.box.y1 + s.box.y2) / 2 - (tr.box.y1 + tr.box.y2) / 2) / (float)dt;
        tr.vx = tr.frames > 1 ? 0.5f * tr.vx + 0.5f * nvx : nvx;
        tr.vy = tr.frames > 1 ? 0.5f * tr.vy + 0.5f * nvy : nvy;
      }
      tr.box = s.box;
      tr.t_last = t;
      s.track = tr.id;
      if (s.num_conf < 0.35f) continue;  // followed, but too blurred to vote
      tr.frames++;
      tr.best_legible = std::max(tr.best_legible, s.legible);
      const auto& p = s.read.probs;
      if (tr.sum.size() != p.size()) tr.sum.assign(p.size(), {});
      const double w = 0.5 + s.num_conf;  // sharper sightings count a bit more
      for (size_t h = 0; h < p.size(); ++h) {
        if (tr.sum[h].size() != p[h].size()) tr.sum[h].assign(p[h].size(), 0.0);
        for (size_t c = 0; c < p[h].size(); ++c) tr.sum[h][c] += w * p[h][c];
      }
      settle(tr);
      const float score = s.box.score * s.num_conf * std::min(1.f, (s.box.x2 - s.box.x1) / 120.f);
      if (score > tr.best_score) {
        tr.best_score = score;
        tr.t_best = t;
        keep_pictures(tr, s.box, rgb, W, H);
      }
    }

    // 4. plates not seen for a while have passed
    for (size_t k = 0; k < tracks_.size();) {
      if (t - tracks_[k].t_last > cfg.track_timeout) {
        close(tracks_[k]);
        tracks_.erase(tracks_.begin() + (long)k);
      } else {
        ++k;
      }
    }
    return (int)last_.size();
  }

  // End of the video: everything still followed is closed.
  void finish() {
    for (Track& tr : tracks_) close(tr);
    tracks_.clear();
    drop_misreads();
  }

  // The frame scaled to display size with the search windows, boxes and numbers drawn on it.
  void render(const uint8_t* rgb, int W, int H, Image& out) const {
    const int ow = std::min(W, cfg.display_max_w), oh = (int)((int64_t)H * ow / W);
    draw::downscale_rgb(rgb, W, H, 0, 0, W, H, out, ow, oh);
    const float s = (float)ow / W;
    const int th = std::max(2, ow / 400), scale = std::max(2, ow / 320);
    for (const jl::Box& r : rois_) draw::rect(out, (int)(r.x1 * s), (int)(r.y1 * s), (int)(r.x2 * s), (int)(r.y2 * s), 0x40c0ff00u, 1);
    for (const Sighting& si : last_) {
      const Track* tr = nullptr;
      for (const Track& t : tracks_)
        if (t.id == si.track) tr = &t;
      const bool ok = tr && tr->frames > 0;
      const uint32_t col = ok ? 0x30e060ffu : 0xffc000ffu;
      const int x0 = (int)(si.box.x1 * s), y0 = (int)(si.box.y1 * s), x1 = (int)(si.box.x2 * s) + 1, y1 = (int)(si.box.y2 * s) + 1;
      draw::rect(out, x0 - th, y0 - th, x1 + th, y1 + th, col, th);
      if (ok) draw::label(out, x0 - th, y0 - th, y1 + th, draw::to_ascii(number_display(tr->plate)), 0xffffffffu, scale);
    }
  }

  const std::vector<Result>& results() const { return results_; }
  const std::vector<Track>& tracks() const { return tracks_; }
  const std::vector<Sighting>& last() const { return last_; }
  int frames_analyzed() const { return frames_analyzed_; }
  bool last_was_full() const { return last_full_; }
  size_t roi_count() const { return rois_.size(); }

 private:
  static std::vector<jl::Box> nms(std::vector<jl::Box> b, float thr) {
    std::sort(b.begin(), b.end(), [](const jl::Box& x, const jl::Box& y) { return x.score > y.score; });
    std::vector<jl::Box> out;
    for (const jl::Box& c : b) {
      bool keep = true;
      for (const jl::Box& o : out) {
        const float iw = std::min(c.x2, o.x2) - std::max(c.x1, o.x1), ih = std::min(c.y2, o.y2) - std::max(c.y1, o.y1);
        if (iw <= 0 || ih <= 0) continue;
        const float inter = iw * ih, a = (c.x2 - c.x1) * (c.y2 - c.y1), bb = (o.x2 - o.x1) * (o.y2 - o.y1);
        if (inter / std::min(a, bb) > thr) keep = false;
      }
      if (keep) out.push_back(c);
    }
    return out;
  }

  // Detector on a window of the frame (the whole frame when the window is the frame).
  std::vector<jl::Box> detect(const onx::Graph& g, const uint8_t* rgb, int W, int H, int x0, int y0, int cw, int ch, float conf) {
    jl::DetCfg dc;
    dc.kind = jl::DetKind::V8;
    dc.nc = 1;
    dc.plate_class = 0;
    dc.imgsz = 0;
    dc.v8_fmt = BoxFmt::CXCYWH;
    dc.conf = conf;
    if (x0 == 0 && y0 == 0 && cw == W && ch == H) return jl::detect_plates(g, rgb, W, H, dc);
    crop_.resize((size_t)cw * ch * 3);
    for (int y = 0; y < ch; ++y) std::copy_n(rgb + ((size_t)(y0 + y) * W + x0) * 3, (size_t)cw * 3, crop_.data() + (size_t)y * cw * 3);
    std::vector<jl::Box> b = jl::detect_plates(g, crop_.data(), cw, ch, dc);
    for (jl::Box& x : b) x.x1 += x0, x.x2 += x0, x.y1 += y0, x.y2 += y0;
    return b;
  }

  void settle(Track& t) const {
    t.arg.clear();
    t.conf.clear();
    for (const std::vector<double>& s : t.sum) {
      int best = 0;
      double bv = -1, tot = 0;
      for (size_t c = 0; c < s.size(); ++c) {
        tot += s[c];
        if (s[c] > bv) bv = s[c], best = (int)c;
      }
      t.arg.push_back(best);
      t.conf.push_back(tot > 0 ? (float)(bv / tot) : 0.f);
    }
    t.plate = spec::decode(*spec_of_, t.arg);
  }

  void keep_pictures(Track& tr, const jl::Box& b, const uint8_t* rgb, int W, int H) {
    const float bw = b.x2 - b.x1, bh = b.y2 - b.y1, mx = bw * 0.12f, my = bh * 0.15f;
    const float x0 = std::max(0.f, b.x1 - mx), y0 = std::max(0.f, b.y1 - my), x1 = std::min((float)W, b.x2 + mx), y1 = std::min((float)H, b.y2 + my);
    const int th = cfg.thumb_h, tw = std::max(1, (int)(th * (x1 - x0) / std::max(1.f, y1 - y0)));
    draw::crop_rgb(rgb, W, H, x0, y0, x1, y1, tr.thumb, tw, th);
    const int sw = std::min(W, cfg.snap_w), sh = (int)((int64_t)H * sw / W);
    draw::downscale_rgb(rgb, W, H, 0, 0, W, H, tr.snap, sw, sh);
    const float s = (float)sw / W;
    draw::rect(tr.snap, (int)(b.x1 * s) - 2, (int)(b.y1 * s) - 2, (int)(b.x2 * s) + 3, (int)(b.y2 * s) + 3, 0x30e060ffu, 2);
  }

  void close(Track& tr) {
    if (tr.frames == 0 || tr.conf.size() < 9) return;
    const float nc = number_conf(tr.conf);
    // one sharp sighting is enough; a vague plate needs company
    if (nc < cfg.min_num_conf || (tr.frames < 2 && nc < 0.85f)) return;
    // seen once: also needs the recognizer to be fairly sure it is a plate at all (IMG_7121: a bus-stop
    // sign at the frame edge, read once as 01-07 with legible 0.55)
    if (tr.frames < 2 && cfg.min_legible > 0 && tr.best_legible < 0.8f) return;
    Result r;
    r.id = tr.id;
    r.t_first = tr.t_first;
    r.t_last = tr.t_last;
    r.t_best = tr.t_best;
    r.frames = tr.frames;
    r.plate = tr.plate;
    r.number = number_display(tr.plate);
    r.conf = tr.conf;
    r.last_box = tr.box;
    r.first_box = tr.first_box;
    r.thumb = std::move(tr.thumb);
    r.snap = std::move(tr.snap);
    // the same plate lost for a moment and found again: one entry
    for (Result& o : results_)
      if (o.number == r.number && r.t_first - o.t_last < 4.0) {
        o.t_last = std::max(o.t_last, r.t_last);
        if (r.frames > o.frames) {
          const double tf = o.t_first;
          const int fr = o.frames + r.frames;
          o = std::move(r);
          o.t_first = tf;
          o.frames = fr;
        } else {
          o.frames += r.frames;
        }
        return;
      }
    results_.push_back(std::move(r));
  }

  // A plate misread for a frame or two gets a short track of its own next to the real one
  // (IMG_7121 in Edge: 13-08 also appeared once as 11-08). An entry built from <= 2 sightings whose
  // number differs from a much better-supported entry of about the same time in one digit only is
  // that misreading: drop it.
  void drop_misreads() {
    auto one_off = [](const spec::Plate& a, const spec::Plate& b) {
      if (a.disp.size() != b.disp.size()) return false;
      int diff = 0;
      for (size_t k = 0; k < a.disp.size(); ++k) diff += a.disp[k] != b.disp[k];
      return diff > 0 && diff <= 1;   // UTF-8 '・' differs in more than one byte, so it never matches here
    };
    for (size_t i = 0; i < results_.size();) {
      const Result& f = results_[i];
      bool drop = false;
      if (f.frames <= 2)
        for (size_t j = 0; j < results_.size() && !drop; ++j) {
          const Result& o = results_[j];
          drop = j != i && o.frames >= 4 * f.frames && std::fabs(o.t_first - f.t_first) <= 3.0 && one_off(f.plate, o.plate);
        }
      if (drop) results_.erase(results_.begin() + (long)i);
      else ++i;
    }
  }

 public:
  const spec::Spec* spec_of_ = nullptr;  // for closing tracks

 private:
  std::vector<Track> tracks_;
  std::vector<Result> results_;
  std::vector<Sighting> last_;
  std::vector<jl::Box> rois_;
  std::vector<uint8_t> crop_;
  int next_id_ = 1;
  double next_coarse_ = 0, last_dense_ = -1e9, t_ = 0;
  int frames_analyzed_ = 0;
  bool last_full_ = false;
  int W_ = 0, H_ = 0;
};

// ---------------------------------------------------------------- JSON for the page
inline std::string fmt_time(double t) {
  char b[32];
  const int m = (int)(t / 60);
  std::snprintf(b, sizeof b, "%02d:%04.1f", m, t - m * 60);
  return b;
}

inline std::string result_json(const Result& r) {
  using jl::json_escape;
  char b[256];
  std::snprintf(b, sizeof b, "{\"id\":%d,\"t\":%.2f,\"t_last\":%.2f,\"t_best\":%.2f,\"frames\":%d,\"time\":\"%s\",", r.id, r.t_first,
                r.t_last, r.t_best, r.frames, fmt_time(r.t_first).c_str());
  std::string o = b;
  o += "\"number\":\"" + json_escape(r.number) + "\",";
  o += "\"text\":\"" + json_escape(r.plate.text) + "\",";
  o += "\"region\":\"" + json_escape(r.plate.region) + "\",";
  o += "\"cls\":\"" + json_escape(r.plate.cls) + "\",";
  o += "\"hira\":\"" + json_escape(r.plate.hira) + "\",";
  std::snprintf(b, sizeof b, "\"num_conf\":%.3f,\"region_conf\":%.3f,\"thumb\":[%d,%d],\"snap\":[%d,%d],\"first_box\":[%.0f,%.0f,%.0f,%.0f]}",
                number_conf(r.conf), r.conf.empty() ? 0.f : r.conf[0], r.thumb.w, r.thumb.h, r.snap.w, r.snap.h, r.first_box.x1,
                r.first_box.y1, r.first_box.x2, r.first_box.y2);
  o += b;
  return o;
}

}  // namespace lpa
