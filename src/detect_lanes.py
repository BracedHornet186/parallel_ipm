import cv2
import numpy as np
import os 

INPUT_DIR = "data/raw_data/"
OUTPUT_DIR = "data/masked_images/"

os.makedirs(OUTPUT_DIR, exist_ok=True)

for filename in os.listdir(INPUT_DIR):
    if not filename.endswith((".jpg", ".png")):
        continue

    img_path = os.path.join(INPUT_DIR, filename)
    img = cv2.imread(img_path)

    # 2. Convert to HLS and mask yellow/white
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    lower_white = np.array([50, 200, 0], dtype=np.uint8)
    upper_white = np.array([100, 255, 180], dtype=np.uint8)
    mask_white = cv2.inRange(hls, lower_white, upper_white)

    lower_yellow = np.array([30, 150, 30], dtype=np.uint8)
    upper_yellow = np.array([90, 200, 50], dtype=np.uint8)
    mask_yellow = cv2.inRange(hls, lower_yellow, upper_yellow)

    combined_mask = cv2.bitwise_or(mask_white, mask_yellow)
    color_isolated = cv2.bitwise_and(img, img, mask=combined_mask)

    # 3. Grayscale & Blur
    gray = cv2.cvtColor(color_isolated, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    # 4. Canny Edge
    edges = cv2.Canny(blur, 50, 150)

    # 5. Region of Interest
    height, width = edges.shape
    polygons = np.array([[
        (0, 980),    # bottom left
        (0, 600),    # bottom left
        (650, 120),    # top left
        (1400, 120),   # lower left
        (1920, 300),   # mid right 
        (1920, 980)    # bottom right
    ]])
    roi_mask = np.zeros_like(edges)
    cv2.fillPoly(roi_mask, polygons, 255)
    cropped_edges = cv2.bitwise_and(edges, roi_mask)

    # 6. Hough Transform
    lines = cv2.HoughLinesP(cropped_edges, rho=2, theta=np.pi/180, threshold=50, 
                            minLineLength=40, maxLineGap=20)

    # 7. Draw Lines (Simplistic drawing, without averaging logic)
    line_image = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)

    # Create red ROI overlay
    roi_overlay = np.zeros_like(img)
    roi_overlay[:, :, 2] = roi_mask  # Red channel

    # Blend ROI with original image
    img_with_roi = cv2.addWeighted(img, 1.0, roi_overlay, 0.4, 0)

    # Final overlay with detected lines
    final_result = cv2.addWeighted(img_with_roi, 0.8, line_image, 1, 0)

    # Save output
    output_path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(output_path, final_result)

print("Processing complete. Images saved to:", OUTPUT_DIR)