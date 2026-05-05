CXX        ?= g++
NVCXX      ?= nvc++

CXXSTD     := -std=c++17
WARN       := -Wall -Wextra
OPT        := -O2
INC        := -Iinclude

CXXFLAGS_BASE := $(OPT) $(CXXSTD) $(WARN) $(INC)
# Pragmas intentionally inert in builds without -fopenmp / -acc are not warnings.
NOPRAGMA      := -Wno-unknown-pragmas
LDFLAGS_BASE  := -lm

# OpenCV (lane_detect only) ---------------------------------------------------
OPENCV_CFLAGS := $(shell pkg-config --cflags opencv4 2>/dev/null)
OPENCV_LIBS   := $(shell pkg-config --libs   opencv4 2>/dev/null)

# OpenGL / GLUT (viewer only) -------------------------------------------------
GL_LIBS := -lGL -lGLU -lglut

# Filesystem lib for older GCCs (gcc < 9 needed -lstdc++fs).  GCC 13 doesn't.
FS_LIBS :=

BIN := bin
SRC := src

LANE_SRC := $(SRC)/lane_detect.cpp
IPM_SRC  := $(SRC)/ipm_3d.cpp
VIEW_SRC := $(SRC)/3d_viewer.cpp
TIMING_H := $(SRC)/timing.h
STB_H    := include/stb_image.h

TARGETS := \
    $(BIN)/lane_detect_serial \
    $(BIN)/lane_detect_omp    \
    $(BIN)/ipm_3d_serial      \
    $(BIN)/ipm_3d_omp         \
    $(BIN)/viewer

.PHONY: all clean bench run ipm_3d_acc help

all: $(TARGETS)
	@echo "==> built: $(TARGETS)"

$(BIN):
	@mkdir -p $@

# lane_detect
$(BIN)/lane_detect_serial: $(LANE_SRC) $(TIMING_H) | $(BIN)
	$(CXX) $(CXXFLAGS_BASE) $(NOPRAGMA) $(OPENCV_CFLAGS) -o $@ $(LANE_SRC) \
	    $(OPENCV_LIBS) $(FS_LIBS) $(LDFLAGS_BASE)

$(BIN)/lane_detect_omp: $(LANE_SRC) $(TIMING_H) | $(BIN)
	$(CXX) $(CXXFLAGS_BASE) -fopenmp $(OPENCV_CFLAGS) -o $@ $(LANE_SRC) \
	    $(OPENCV_LIBS) $(FS_LIBS) $(LDFLAGS_BASE)

# ipm_3d
$(BIN)/ipm_3d_serial: $(IPM_SRC) $(TIMING_H) $(STB_H) | $(BIN)
	$(CXX) $(CXXFLAGS_BASE) $(NOPRAGMA) -o $@ $(IPM_SRC) $(FS_LIBS) $(LDFLAGS_BASE)

$(BIN)/ipm_3d_omp: $(IPM_SRC) $(TIMING_H) $(STB_H) | $(BIN)
	$(CXX) $(CXXFLAGS_BASE) $(NOPRAGMA) -fopenmp -o $@ $(IPM_SRC) $(FS_LIBS) $(LDFLAGS_BASE)

# Optional: GPU build with NVHPC.  Not part of `all`.
ipm_3d_acc: $(BIN)/ipm_3d_acc
$(BIN)/ipm_3d_acc: $(IPM_SRC) $(TIMING_H) $(STB_H) | $(BIN)
	@command -v $(NVCXX) >/dev/null 2>&1 || \
	    { echo "ERROR: $(NVCXX) not found in PATH"; exit 1; }
	$(NVCXX) -O2 -std=c++17 -acc -mp $(INC) -o $@ $(IPM_SRC) -lm

# viewer
$(BIN)/viewer: $(VIEW_SRC) | $(BIN)
	$(CXX) $(CXXFLAGS_BASE) -o $@ $(VIEW_SRC) $(GL_LIBS) $(FS_LIBS) $(LDFLAGS_BASE)

# convenience targets
bench: all
	@bash scripts/bench.sh

run: all
	@echo "==> stage 1: lane_detect"
	./$(BIN)/lane_detect_omp data/raw_data data/lane_masks --time
	@echo "==> stage 2: ipm_3d"
	./$(BIN)/ipm_3d_omp --input-dir data/lane_masks --output-dir data/points_3d --time
	@echo "==> stage 3: viewer"
	./$(BIN)/viewer data/points_3d

clean:
	rm -rf $(BIN)

help:
	@echo "make             build all CPU binaries"
	@echo "make ipm_3d_acc  build the GPU binary (needs nvc++)"
	@echo "make bench       run the benchmark script"
	@echo "make run         end-to-end demo (lane_detect -> ipm_3d -> viewer)"
	@echo "make clean       remove build outputs"
