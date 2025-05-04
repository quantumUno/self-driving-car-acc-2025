import cv2
import numpy as np
import time
import os
from pal.products.qcar import QCarCameras, IS_PHYSICAL_QCAR

# Configuration
RUN_TIME = 30.0  # seconds
OUTPUT_DIR = "lane_detection_output"
FRAME_RATE = 30  # Hz
IMAGE_WIDTH = 820
IMAGE_HEIGHT = 410

# Ensure output directory exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def detect_lane_offset(image):
    """
    Detect lane lines and compute offset from lane center.
    Returns offset in pixels (positive: QCar is right of center, negative: left).
    """
    global prev_offset
    prev_offset = 0 if 'prev_offset' not in globals() else prev_offset
    
    # Downsample image to reduce memory usage
    image = cv2.resize(image, (410, 205))  # Half resolution
    height, width = 205, 410  # Update dimensions
    
    # Convert to HSV for color filtering
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # White lines (relaxed range)
    lower_white = np.array([0, 0, 150])
    upper_white = np.array([180, 50, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    # Yellow lines (relaxed range)
    lower_yellow = np.array([15, 80, 80])
    upper_yellow = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    color_mask = cv2.bitwise_or(white_mask, yellow_mask)
    # Visualize color mask
    color_mask_display = cv2.cvtColor(color_mask, cv2.COLOR_GRAY2BGR)
    cv2.imshow('Color Mask', color_mask_display)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply histogram equalization to improve contrast
    gray = cv2.equalizeHist(gray)
    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Canny edge detection
    edges = cv2.Canny(blur, 50, 150)
    # Visualize edges
    edges_display = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cv2.imshow('Edges', edges_display)
    
    # Apply color mask to edges
    masked_edges = cv2.bitwise_and(edges, color_mask)
    
    # Define ROI (wider trapezoid)
    mask = np.zeros_like(masked_edges)
    polygon = np.array([[
        (20, height), (width-20, height),
        (width//2 + 80, height//2 - 20), (width//2 - 80, height//2 - 20)
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(masked_edges, mask)
    # Visualize masked edges
    masked_edges_display = cv2.cvtColor(masked_edges, cv2.COLOR_GRAY2BGR)
    cv2.imshow('Masked Edges', masked_edges_display)
    
    # Hough transform to detect lines (relaxed parameters)
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, threshold=15, minLineLength=20, maxLineGap=150)
    # Separate left and right lanes based on slope and position
    left_lines, right_lines = [], []
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            avg_x = (x1 + x2) / 2
            slope = (y2 - y1) / (x2 - x1 + 1e-5)
            if abs(y2 - y1) > 10 and abs(slope) > 0.3:  # Ensure vertical extent and slope
                if slope < 0 and avg_x < width / 2:  # Left lane: negative slope, left side
                    left_lines.append([x1, y1, x2, y2])
                    cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                elif slope > 0 and avg_x > width / 2:  # Right lane: positive slope, right side
                    right_lines.append([x1, y1, x2, y2])
                    cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    # Compute lane center
    offset_pixels = 0
    lane_center = width / 2  # Default to image center if detection fails
    image_center = width / 2
    if left_lines and right_lines:
        left_x = min([min(line[0], line[2]) for line in left_lines])  # Use leftmost x
        right_x = max([max(line[0], line[2]) for line in right_lines])  # Use rightmost x
        lane_center = (left_x + right_x) / 2
        offset_pixels = image_center - lane_center  # Positive: QCar is right of center
        # Reduce smoothing to prevent history from dominating
        offset_pixels = 0.9 * offset_pixels + 0.1 * prev_offset
        prev_offset = offset_pixels
        # Scale back to original resolution for reporting
        left_x_orig = left_x * 2
        right_x_orig = right_x * 2
        offset_pixels_orig = offset_pixels * 2
        print(f"Left_x: {left_x_orig:.1f}, Right_x: {right_x_orig:.1f}, Width: {right_x_orig - left_x_orig:.1f} pixels")
    else:
        offset_pixels_orig = 0
    # Visualize lane center and offset
    cv2.line(line_image, (int(lane_center), height), (int(lane_center), height-25), (255, 255, 0), 2)
    cv2.line(line_image, (int(image_center), height), (int(image_center), height-25), (255, 0, 0), 2)
    combined_image = cv2.addWeighted(image, 0.8, line_image, 0.2, 0)
    cv2.putText(combined_image, f"Offset: {offset_pixels:.1f}px", (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return offset_pixels_orig, combined_image

def main():
    # Initialize front CSI camera
    cameras = QCarCameras(
        enableFront=True,
        frameWidth=IMAGE_WIDTH,
        frameHeight=IMAGE_HEIGHT,
        frameRate=FRAME_RATE
    )
    
    frame_count = 0
    try:
        with cameras:
            t0 = time.time()
            while time.time() - t0 < RUN_TIME:
                # Read front camera
                cameras.readAll()
                front_image = cameras.csiFront.imageData
                if front_image is None:
                    print("Failed to read front camera image")
                    continue
                
                # Detect lane offset
                offset_pixels, combined_image = detect_lane_offset(front_image)
                if combined_image is None:
                    print("Combined image is None, skipping display")
                    continue
                
                # Display images
                cv2.imshow('Front Camera', front_image)
                cv2.imshow('Lane Lines', combined_image)
                
                # Save images for debugging
                cv2.imwrite(os.path.join(OUTPUT_DIR, f"frame_{frame_count:04d}.jpg"), front_image)
                cv2.imwrite(os.path.join(OUTPUT_DIR, f"lines_{frame_count:04d}.jpg"), combined_image)
                
                print(f"Frame {frame_count}: Offset = {offset_pixels:.1f} pixels")
                
                # Wait to maintain frame rate
                cv2.waitKey(int(1000 / FRAME_RATE))
                frame_count += 1
                
    except Exception as e:
        print(f"Exception occurred: {e}")
    finally:
        cv2.destroyAllWindows()
        cameras.terminate()  # Explicitly terminate camera
        print(f"Test complete. {frame_count} frames processed. Images saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
