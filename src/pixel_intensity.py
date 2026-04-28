import cv2
import numpy as np

# Global variables to store the images
img = None
hls_img = None

def mouse_click_event(event, x, y, flags, param):
    """
    Mouse callback function. 
    Triggers when a mouse event happens on the image window.
    """
    # Listen for Left Mouse Button Clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        # OpenCV images are indexed as [Row(y), Column(x)]
        # Extract the Hue, Lightness, and Saturation values
        h, l, s = hls_img[y, x]
        
        # Print to terminal
        print(f"Location: (X: {x}, Y: {y}) | HLS Value: [H: {h}, L: {l}, S: {s}]")
        
        # Create a fresh copy so text doesn't permanently overwrite the original
        display_img = img.copy()
        
        # Formatting the text
        text = f"X:{x} Y:{y} | HLS:[{h}, {l}, {s}]"
        
        # Add a black background rectangle for text readability
        cv2.rectangle(display_img, (10, 10), (500, 50), (0, 0, 0), -1)
        
        # Put the text on the image (White text)
        cv2.putText(display_img, text, (15, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Show the updated image
        cv2.imshow("Pixel Color Inspector", display_img)

def main(image_path):
    global img, hls_img
    
    # 1. Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return

    # 2. Convert the image to HLS color space immediately
    hls_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    
    # 3. Create a named window and set the mouse callback function
    cv2.namedWindow("Pixel Color Inspector")
    cv2.setMouseCallback("Pixel Color Inspector", mouse_click_event)
    
    # 4. Show the initial image
    cv2.imshow("Pixel Color Inspector", img)
    
    print("Click anywhere on the image to get the HLS values.")
    print("Press 'q' or 'ESC' to exit.")
    
    # 5. Keep the window open until the user presses 'q' or the Esc key
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    IMAGE_PATH = "data/raw_data/frame000212.jpg" 
    main(IMAGE_PATH)