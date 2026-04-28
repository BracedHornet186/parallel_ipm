import cv2
import numpy as np
import re
import os
import glob

def parse_line(line):
    # This pattern looks for: [NUMBER, NUMBER]
    # \d+ matches one or more digits
    pattern = r"\[(\d+),\s*(\d+)\]"
    
    matches = re.findall(pattern, line)
    
    # Converts the string matches into a list of integer tuples
    return [(int(x), int(y)) for x, y in matches]

def read_output_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Safety check: ensure the file has enough lines to avoid IndexError
    if len(lines) <= 15:
        print(f"Warning: {file_path} does not have enough lines.")
        return [], [], [], []

    wl = parse_line(lines[0])  # White left
    yl = parse_line(lines[5])  # Yellow left
    yr = parse_line(lines[10]) # Yellow right    
    wr = parse_line(lines[15]) # White right

    return wl, yl, yr, wr

def draw_points(image_shape, wl, yl, yr, wr, base_image=None):
    if base_image is None:
        canvas = np.zeros(image_shape, dtype=np.uint8)
    else:
        canvas = base_image.copy()

    # Colors (BGR)
    color_wl = (0, 255, 0)     # green
    color_wr = (255, 0, 0)     # blue
    color_yl = (0, 255, 255)   # yellow
    color_yr = (0, 0, 255)     # red

    for x, y in wl:
        cv2.circle(canvas, (x, y), 3, color_wl, -1)

    for x, y in wr:
        cv2.circle(canvas, (x, y), 3, color_wr, -1)

    for x, y in yl:
        cv2.circle(canvas, (x, y), 3, color_yl, -1)

    for x, y in yr:
        cv2.circle(canvas, (x, y), 3, color_yr, -1)

    return canvas

def generate_masked_images(raw_data_dir, points_dir, output_dir):
    """
    Iterates through points_dir, finds corresponding images in raw_data_dir, 
    and saves the drawn output in output_dir.
    """
    # 1. Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # 2. Find all text files in the points directory
    txt_files = glob.glob(os.path.join(points_dir, "*.txt"))
    
    if not txt_files:
        print(f"No point data (.txt) files found in {points_dir}")
        return

    # Valid image extensions to look for
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}

    for txt_path in txt_files:
        # Extract the base name (e.g., "image_01" from "image_01.txt")
        base_name = os.path.splitext(os.path.basename(txt_path))[0]
        
        # Read the points
        wl, yl, yr, wr = read_output_file(txt_path)
        
        # Find the corresponding image in the raw_data directory
        possible_images = glob.glob(os.path.join(raw_data_dir, f"{base_name}.*"))
        ref_image_path = None
        
        for img_path in possible_images:
            if os.path.splitext(img_path)[1].lower() in valid_extensions:
                ref_image_path = img_path
                break
        
        # Define where the output image will be saved
        output_image_path = os.path.join(output_dir, f"{base_name}_masked.png")
        
        if ref_image_path and os.path.exists(ref_image_path):
            base_img = cv2.imread(ref_image_path)
            if base_img is not None:
                h, w, _ = base_img.shape
                result = draw_points((h, w, 3), wl, yl, yr, wr, base_img)
            else:
                print(f"Error: Could not read image at {ref_image_path}. Skipping...")
                continue
        else:
            print(f"Warning: No matching image found for {base_name}. Generating blank canvas.")
            h, w = 1080, 1920 # Default fallback dimensions
            result = draw_points((h, w, 3), wl, yl, yr, wr)

        # Save the image
        cv2.imwrite(output_image_path, result)
        print(f"Saved: {output_image_path}")

if __name__ == "__main__":
    # Define your specific folder paths
    RAW_DATA_DIR = "data/raw_data"
    POINTS_DIR = "data/image_points"
    OUTPUT_DIR = "data/masked_images"
    
    print("Starting generation process...")
    generate_masked_images(RAW_DATA_DIR, POINTS_DIR, OUTPUT_DIR)
    print("Process complete!")