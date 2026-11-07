cmake_minimum_required(VERSION 3.16)
project(lpr_analyzer C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(MK_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/mediakit)
set(LPR_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/yolo_lpr)

# ---------------------------------------------------------------- FFmpeg (from mediakit_cpp)
include(${MK_DIR}/codec/ffmpeg/ffmpeg.cmake)
# Only what reading an MP4 needs: the mov / h264 / aac / ts demuxers and the H.264 / AAC decoders.
# These lists replace FFmpeg's full ones (found first on the include path), which keeps hundreds of
# codecs out of the wasm.
foreach(t mk_avcodec mk_avformat)
  target_include_directories(${t} BEFORE PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cmake/ffmpeg_min)
endforeach()
if(EMSCRIPTEN)
  foreach(t mk_avutil mk_avcodec mk_avformat mk_swscale mk_swresample)
    target_compile_options(${t} PRIVATE -w -fno-strict-aliasing)
  endforeach()
endif()

add_library(mk_codec STATIC ${MK_DIR}/codec/mk_codec.cpp)
target_include_directories(mk_codec PUBLIC ${MK_DIR}/common ${MK_DIR}/codec)
target_link_libraries(mk_codec PUBLIC mk_ffmpeg)

# ---------------------------------------------------------------- the analyzer
set(LPR_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/src ${LPR_DIR}/pure ${LPR_DIR}/pure/third_party)

if(EMSCRIPTEN)
  # docs/analyzer.js + .wasm: loaded by docs/worker.js (the page runs it in a Web Worker)
  add_executable(analyzer src/analyzer_wasm.cpp)
  target_include_directories(analyzer PRIVATE ${LPR_INCLUDES})
  target_link_libraries(analyzer PRIVATE mk_codec)
  target_compile_options(analyzer PRIVATE -O3 -msimd128)
  # ASYNCIFY: mediakit's WebCodecs bridge has awaiting calls (flush). The analyzer never makes them,
  # and IGNORE_INDIRECT keeps the instrumentation off the inference code (it calls through pointers).
  target_link_options(analyzer PRIVATE -O3 -msimd128 -sASYNCIFY -sASYNCIFY_IGNORE_INDIRECT=1
      -sMODULARIZE=1 -sEXPORT_NAME=createAnalyzer -sENVIRONMENT=web,worker,node
      -sALLOW_MEMORY_GROWTH=1 -sINITIAL_MEMORY=268435456 -sMAXIMUM_MEMORY=4294967296 -sSTACK_SIZE=4194304
      -lworkerfs.js -sFORCE_FILESYSTEM=1
      "-sEXPORTED_RUNTIME_METHODS=['FS','WORKERFS','UTF8ToString','stringToUTF8','lengthBytesUTF8','HEAPU8']"
      "-sEXPORTED_FUNCTIONS=['_malloc','_free']")
  set_target_properties(analyzer PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/docs)
else()
  # native test driver: raw RGB frames from ffmpeg on stdin (see src/analyzer_cli.cpp)
  add_executable(lpr_cli src/analyzer_cli.cpp)
  target_include_directories(lpr_cli PRIVATE ${LPR_INCLUDES})
endif()
