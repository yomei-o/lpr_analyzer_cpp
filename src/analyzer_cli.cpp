// Native test driver for lpr_analyzer.hpp: raw RGB frames on stdin (from ffmpeg), the result list on
// stdout. The same analyzer the browser runs, without the browser -- for tuning and for regression.
//
//   ffmpeg -i IMG_7121.mp4 -f rawvideo -pix_fmt rgb24 - | lpr_cli --size 1920x1080 --fps 30 [--out dir]
//
// --out writes every analysed frame as drawn for the page (dir/fNNNN.png) and each result's crops.
#include "lpr_analyzer.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

static std::string arg_of(int argc, char** argv, const char* k, const char* def) {
  for (int i = 1; i + 1 < argc; ++i)
    if (!std::strcmp(argv[i], k)) return argv[i + 1];
  return def;
}

static void save(const std::string& path, const lpa::Image& im) {
  if (im.w > 0) stbi_write_png(path.c_str(), im.w, im.h, 4, im.px.data(), im.w * 4);
}

int main(int argc, char** argv) {
#ifdef _WIN32
  _setmode(_fileno(stdin), _O_BINARY);
#endif
  int W = 0, H = 0;
  std::sscanf(arg_of(argc, argv, "--size", "1920x1080").c_str(), "%dx%d", &W, &H);
  const double fps = std::atof(arg_of(argc, argv, "--fps", "30").c_str());
  const std::string models = arg_of(argc, argv, "--models", "docs/models");
  const std::string out = arg_of(argc, argv, "--out", "");

  onx::Graph det640 = onx::load_onnx(models + "/plate_det_v8n_640.onnx");
  onx::Graph det320 = onx::load_onnx(models + "/plate_det_v8n_320.onnx");
  onx::Graph ocr = onx::load_onnx(models + "/plate_ocr_v7_bal.onnx");
  onx::Graph corner = onx::load_onnx(models + "/plate_corner.onnx");
  FILE* sf = std::fopen((models + "/labels.txt").c_str(), "rb");
  if (!sf) {
    std::fprintf(stderr, "no labels.txt in %s\n", models.c_str());
    return 1;
  }
  std::string st;
  char buf[4096];
  for (size_t n; (n = std::fread(buf, 1, sizeof buf, sf)) > 0;) st.append(buf, n);
  std::fclose(sf);
  spec::Spec sp = spec::parse(st);

  lpa::Models m;
  m.det_full = &det640;
  m.det_roi = &det320;
  m.ocr = &ocr;
  m.corner = &corner;
  m.spec = &sp;

  lpa::Analyzer az;
  az.cfg.coarse_interval = std::atof(arg_of(argc, argv, "--coarse", "0.5").c_str());
  az.cfg.dense_interval = std::atof(arg_of(argc, argv, "--dense", "0.2").c_str());
  az.reset();

  std::vector<uint8_t> rgb((size_t)W * H * 3);
  lpa::Image disp;
  double total_ms = 0;
  for (int64_t i = 0;; ++i) {
    if (std::fread(rgb.data(), 1, rgb.size(), stdin) != rgb.size()) break;
    const double t = i / fps;
    if (!az.wants(t)) continue;
    const auto t0 = std::chrono::steady_clock::now();
    const int n = az.process(m, rgb.data(), W, H, t);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    total_ms += ms;
    std::fprintf(stderr, "t=%6.2f %s rois=%zu plates=%d tracks=%zu %.0f ms", t, az.last_was_full() ? "FULL" : "roi ", az.roi_count(), n,
                 az.tracks().size(), ms);
    for (const lpa::Sighting& s : az.last())
      std::fprintf(stderr, " | #%d %.2f %s (%.2f)", s.track, s.box.score, s.read.plate.disp.c_str(), s.num_conf);
    std::fprintf(stderr, "\n");
    if (!out.empty()) {
      az.render(rgb.data(), W, H, disp);
      char p[512];
      std::snprintf(p, sizeof p, "%s/f%05d.png", out.c_str(), (int)i);
      save(p, disp);
    }
  }
  az.finish();
  std::fprintf(stderr, "analysed %d frames, %.1f s of compute\n", az.frames_analyzed(), total_ms / 1000);
  std::printf("[\n");
  for (size_t i = 0; i < az.results().size(); ++i) {
    const lpa::Result& r = az.results()[i];
    std::printf("%s%s\n", lpa::result_json(r).c_str(), i + 1 < az.results().size() ? "," : "");
    if (!out.empty()) {
      save(out + "/result" + std::to_string(r.id) + "_plate.png", r.thumb);
      save(out + "/result" + std::to_string(r.id) + "_scene.png", r.snap);
    }
  }
  std::printf("]\n");
  return 0;
}
