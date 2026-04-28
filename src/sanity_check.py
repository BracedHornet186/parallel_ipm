import cv2
import numpy as np
import re


'''
def parse_line(line):
    # This pattern specifically looks for the digits inside np.int64(...)
    # It matches: (np.int64(NUMBER), np.int64(NUMBER))
    pattern = r"\(np\.int64\((\d+)\),\s*np\.int64\((\d+)\)\)"
    matches = re.findall(pattern, line)
    return [(int(x), int(y)) for x, y in matches]
'''

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

    wl = parse_line(lines[0])  # White left
    yl = parse_line(lines[5])  # Yellow left
    yr = parse_line(lines[10])  # Yellow right    
    wr = parse_line(lines[15])  # White right

    print("WL:", len(wl), "YL:", len(yl), "YR:", len(yr), "WR:", len(wr))
    print(wl[1])
    return wl, yl, yr, wr

def draw_points(image_shape, wl, yl, yr, wr, base_image=None):
    if base_image is None:
        canvas = np.zeros(image_shape, dtype=np.uint8)
    else:
        canvas = base_image.copy()

    # Colors (BGR)
    color_wl = (0,255,0)   # green
    color_wr = (255, 0, 0)   # blue
    color_yl = (0, 255, 255)     # yellow
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

def visualize_lanes(txt_path, output_image_path, reference_image_path=None):
    wl, yl, yr, wr = read_output_file(txt_path)

    if reference_image_path:
        base_img = cv2.imread(reference_image_path)
        h, w, _ = base_img.shape
        result = draw_points((h, w, 3), wl, yl, yr, wr, base_img)
    else:
        # Default size (adjust if needed)
        h, w = 1080, 1920
        result = draw_points((h, w, 3), wl, yl, yr, wr)

    cv2.imwrite(output_image_path, result)

# Example usage

#visualize_lanes("trial_v1.txt", "check_v1.png", None)
visualize_lanes("trial_v2.txt", "check_v2.png", None)