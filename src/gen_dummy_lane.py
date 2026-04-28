import cv2
import numpy as np

def generate_dummy_lanes(width=1920, height=1080, save_path='dummy_lanes.png'):
    image = np.zeros((height, width), dtype=np.uint8)
    vp_x = int(width / 2)
    vp_y = int(height / 3)

    left_bottom_x = 100
    right_bottom_x = width - 100
    bottom_y = height

    horizon_offset_y = 50
    left_top_x = vp_x - 20
    right_top_x = vp_x + 20
    top_y = vp_y + horizon_offset_y

    thickness = 5
    cv2.line(image, (left_bottom_x, bottom_y), (left_top_x, top_y), 255, thickness)
    cv2.line(image, (right_bottom_x, bottom_y), (right_top_x, top_y), 255, thickness)

    center_bottom_x = int((left_bottom_x + right_bottom_x) / 2)
    center_top_x = vp_x

    num_dashes = 6
    for i in range(num_dashes):
        t1 = i / num_dashes
        t2 = (i + 0.6) / num_dashes 

        y1 = int(bottom_y * (1 - t1) + top_y * t1)
        y2 = int(bottom_y * (1 - t2) + top_y * t2)

        x1 = int(center_bottom_x * (1 - t1) + center_top_x * t1)
        x2 = int(center_bottom_x * (1 - t2) + center_top_x * t2)

        current_thickness = max(1, thickness - int(3 * t1))
        
        cv2.line(image, (x1, y1), (x2, y2), 255, current_thickness)

    cv2.imwrite(save_path, image)
    print(f"Successfully generated dummy lane image at: {save_path}")

    # Display the image (Optional)
    cv2.imshow('Dummy Perspective Lanes', image)
    cv2.waitKey(1)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    generate_dummy_lanes()