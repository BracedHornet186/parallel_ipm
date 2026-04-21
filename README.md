# Pixel to 3D

## Build Instructions
# 1. Get the single header image loader
curl -sSL https://raw.githubusercontent.com/nothings/stb/master/stb_image.h -o stb_image.h

# GPU build (NVHPC / PGI)
nvc++ -O2 -acc -Minfo=accel -o ipm_to_3d ipm_to_3d.cpp -lm

# CPU-only fallback (any g++)
g++ -O2 -o ipm_to_3d ipm_to_3d.cpp -lm

# Generate a test image (needs cv2)
python3 gen_dummy_lanes.py

# Run
./ipm_to_3d dummy_lanes.png output_3d.txt

# ID5130: Parallel Scientific Computing