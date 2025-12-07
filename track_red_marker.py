import cv2
import numpy as np
from google.colab import files
import time

# -------------------- Paths --------------------
video_path = "production_speed_test.mp4"
output_path = "production_speed_test_visual_marker.mp4"

# -------------------- Polygon Setup (VIA format) --------------------
polygon_x = [1, 60, 144, 287, 407, 527, 645, 671, 1, 4]
polygon_y = [123, 120, 116, 101, 78, 55, 15, 3, 3, 119]
polygon_pts = np.array(list(zip(polygon_x, polygon_y)), dtype=np.int32).reshape((-1, 1, 2))

# -------------------- HSV Red Color Thresholds --------------------
lower_red1 = np.array([0, 70, 50], dtype=np.uint8)
upper_red1 = np.array([10, 255, 255], dtype=np.uint8)
lower_red2 = np.array([160, 70, 50], dtype=np.uint8)
upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

# -------------------- Drawing Settings --------------------
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 2.0
font_thickness = 4
text_color = (0, 255, 0)

# Vertical reference line for crossing detection
line_length = 120
line_color = (255, 0, 0)
line_thickness = 4
start_x = 130
start_y = 0
end_x = start_x
end_y = start_y + line_length

# Morphological kernel
kernel = np.ones((3, 3), np.uint8)

# -------------------- OpenCV Optimizations --------------------
cv2.setUseOptimized(True)
cv2.setNumThreads(0)  # auto thread selection

# -------------------- Video Setup --------------------
cap = cv2.VideoCapture(video_path)
ret, sample_frame = cap.read()
if not ret or sample_frame is None:
    cap.release()
    raise RuntimeError("Could not read first frame. Check video path.")

frame_height, frame_width = sample_frame.shape[:2]

# -------------------- Polygon Masks --------------------
# Full-frame polygon mask
mask_full = np.zeros((frame_height, frame_width), dtype=np.uint8)
cv2.fillPoly(mask_full, [polygon_pts], 255)
total_polygon_pixels = int(np.count_nonzero(mask_full))

# Bounding box ROI around polygon
x, y, w, h = cv2.boundingRect(polygon_pts)
x = max(0, x)
y = max(0, y)
w = min(w, frame_width - x)
h = min(h, frame_height - y)

# ROI-specific polygon mask (boolean version for logical operations)
mask_roi = mask_full[y:y+h, x:x+w]
mask_roi_bool = mask_roi.astype(bool)

# -------------------- Video Writer --------------------
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
fps = 30.0 if not fps or fps <= 1e-3 else fps
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# -------------------- State Variables --------------------
frame_index = 0
frame_crossed = 0
rotation_time = 0.0

# Precompute text placement (centered)
percent_dummy = f"{0.00:.2f}% red pixels"
text_size, _ = cv2.getTextSize(percent_dummy, font, font_scale, font_thickness)
text_x_center = int((frame_width - text_size[0]) / 2)
text_y_center = int((frame_height + text_size[1]) / 2)

# -------------------- Frame Processing Loop --------------------
while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    frame_out = frame
    roi_bgr = frame[y:y+h, x:x+w]
    roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

    # Red color thresholding
    mask1 = cv2.inRange(roi_hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(roi_hsv, lower_red2, upper_red2)
    red_mask_roi = cv2.bitwise_or(mask1, mask2)

    # Basic noise reduction
    red_mask_roi = cv2.GaussianBlur(red_mask_roi, (5, 5), 0)
    red_mask_roi = cv2.dilate(red_mask_roi, kernel, iterations=1)

    # Restrict red detection to polygon region
    red_in_poly_roi = np.zeros_like(red_mask_roi, dtype=np.uint8)
    red_in_poly_roi[mask_roi_bool] = red_mask_roi[mask_roi_bool]

    # Compute red percentage
    red_pixels = int(np.count_nonzero(red_in_poly_roi))
    percent_red = (red_pixels / total_polygon_pixels) * 100 if total_polygon_pixels > 0 else 0.0

    # Draw vertical reference line
    cv2.line(frame_out, (start_x, start_y), (end_x, end_y), line_color, line_thickness)

    # Marker detection and crossing check
    if percent_red >= 0.4 and red_pixels > 0:
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(red_in_poly_roi, connectivity=8)

        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            max_idx = 1 + np.argmax(areas)
            cx_roi, cy_roi = centroids[max_idx]

            # Convert to full-frame coordinates
            marker_x = int(x + cx_roi)
            marker_y = int(y + cy_roi)

            # Draw marker center
            cv2.circle(frame_out, (marker_x, marker_y), 5, (255, 255, 255), -1)

            # Trigger on crossing with cooldown of 2.5 seconds
            if (marker_x >= start_x) and ((frame_index - frame_crossed) / fps >= 2.5):
                rotation_time = (frame_index - frame_crossed) / fps
                time_sec = frame_index / fps
                print(f" Red marker crossed line at {time_sec:.2f} seconds (frame {frame_index}), Rotation Time: {rotation_time:.3f}s")
                frame_crossed = frame_index

    # Color detected red regions in ROI
    roi_bgr[red_in_poly_roi > 0] = (0, 0, 255)

    end_time = time.time()
    print("time elapsed : ", end_time - start_time)

    # Uncomment if saving output
    # out.write(frame_out)

    frame_index += 1

# -------------------- Cleanup --------------------
cap.release()
out.release()
print(f" Final video saved to: {output_path}")
files.download(output_path)
