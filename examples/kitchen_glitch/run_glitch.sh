#!/usr/bin/env bash
# Macroblock datamosh glitch on the moving person, ffmpeg-only pixel pipeline.
# usage (from kitchen_glitch): bash run_glitch.sh [LMAX]
set -e
LMAX=${1:-12}
mkdir -p work
# PYTHON=/path/to/python bash run_glitch.sh  (defaults to the repository .venv, then python3, then python)
PY=${PYTHON:-}
if [ -z "$PY" ]; then for c in ../../.venv/Scripts/python.exe ../../.venv/bin/python python3 python; do if command -v "$c" >/dev/null 2>&1 || [ -x "$c" ]; then PY=$c; break; fi; done; fi
"$PY" make_graph.py "$LMAX" 391 > work/glitch.ffgraph
ffmpeg -v error -y -i source/Pexels_7525471_720x1280.mp4 -/filter_complex work/glitch.ffgraph \
  -map "[v]" -c:v libx264 -crf 16 -preset slow -pix_fmt yuv420p \
  -color_primaries bt709 -color_trc bt709 -colorspace bt709 -movflags +faststart Kitchen_Datamosh_Glitch.mp4 \
  -map "[maskdbg]" -c:v libx264 -crf 18 work/motion_mask.mp4
