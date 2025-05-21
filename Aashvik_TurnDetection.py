import cv2
import numpy as np


left = False
right = False
def saturation_mask(frame, saturation_threshold=0.2, min_percent=10, min_region_size=600):




   hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
   saturation = hsv[:, :, 1]
   total_pixels = frame.shape[0] * frame.shape[1]


   threshold_value = int(saturation_threshold * 255)
   binary_mask = (saturation > threshold_value).astype(np.uint8) * 255


   # Connected components analysis to remove small noisy blobs
   num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)


   # Filter out small regions
   filtered_mask = np.zeros_like(binary_mask)
   for i in range(1, num_labels):  # skip background
       if stats[i, cv2.CC_STAT_AREA] >= min_region_size:
           filtered_mask[labels == i] = 255


   return filtered_mask
previous_lines = []
previous_mid_x = None
alpha = 0.9
Vertical_Detected = False
Horizontal_Detected = False
direction = None
def ROI(image, frame_width, frame_height):
   # Create a blank ROI mask the same size as the image
   roi_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
  
   # Define ROI regions (left and middle)
   left_width = int(frame_width * 0.33) 
   right_start = int(frame_width * .66) 
   right_width = int(frame_width) 
   height_start = int(frame_height * 0.8)
   height_end = int(frame_height * 0.85) 
   roi_mask[height_start:height_end, 0:left_width] = 255
   roi_mask[height_start:height_end, right_start:right_width] = 255




  
   # Apply the ROI mask to the image
   masked_image = cv2.bitwise_and(image, image, mask=roi_mask)
   cv2.imshow('ROI Mask', masked_image)


  
   return masked_image, roi_mask
def determine_roi(x, frame_width):
   """
   Determines which ROI the x-coordinate belongs to.
   """
   left = False
   right = False
   left_width = int(frame_width * 0.33)
   right_start = int(frame_width * .66) 
   if 0 < x < left_width:
       left = True
   if right_start < x < frame_width:
       right = True
   return left, right
  
def calculate_distance(x1, y1, x2, y2):
   return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def smooth_value(new_value, previous_value, alpha=0.9):
   if previous_value is None:
       return new_value
   return int(alpha * previous_value + (1 - alpha) * new_value)


def filter_lines(lines, angle_threshold=20):
   global previous_lines
  
   if lines is None:
       if previous_lines:
           return previous_lines
       return []
      
   filtered_lines = []
   for line in lines:
       x1, y1, x2, y2 = line[0]
       angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
  
       if 90 - angle_threshold <= angle <= 90 + angle_threshold:
           if y2 > y1: 
               x1, y1, x2, y2 = x2, y2, x1, y1
           filtered_lines.append([x1, y1, x2, y2])
          
      
       if angle <= angle_threshold or angle >= 180 - angle_threshold:
           if x2 < x1: 
               x1, y1, x2, y2 = x2, y2, x1, y1
           filtered_lines.append([x1, y1, x2, y2])
  
   if filtered_lines:
       previous_lines = filtered_lines
   elif previous_lines:
       return previous_lines
          
   return filtered_lines


def get_two_main_lines(lines, frame_width):
   if len(lines) < 2:
       return []
  
   lines.sort(key=lambda line: line[0])
  
   left_lines = []
   right_lines = []
   mid_x = frame_width // 2
  
   for line in lines:
       x1 = line[0]
       if x1 < mid_x:
           left_lines.append(line)
       else:
           right_lines.append(line)
    
   if left_lines and right_lines:
       left_line = max(left_lines, key=lambda line: line[0])
       right_line = min(right_lines, key=lambda line: line[0])
       return [left_line, right_line]
   return []


def draw_lines_and_center(frame, lines):
   global previous_mid_x, Vertical_Detected, Horizontal_Detected, left, right
   line_image = np.zeros_like(frame)
   line_detected = False
   section = None
  
   for x1, y1, x2, y2 in lines:
       cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 3)


   vertical_lines = [line for line in lines if abs(np.degrees(np.arctan2(line[3] - line[1], line[2] - line[0]))) > 70]


  
   main_vertical_lines = get_two_main_lines(vertical_lines, frame.shape[1])
   if len(main_vertical_lines) == 1:
       left_x = main_vertical_lines[0][0]
       right_x = main_vertical_lines[1][0]
       mid_x = int((left_x + right_x) / 2)
       mid_x = smooth_value(mid_x, previous_mid_x)
       previous_mid_x = mid_x
       Vertical_Detected = True
       height = frame.shape[0]
       cv2.line(line_image, (mid_x, 0), (mid_x, height), (0, 0, 255), 3)
       left, right = determine_roi(mid_x, frame.shape[1])
  
   return line_detected, section
def determine_direction(left, right):
   if left == True and right == True:
       return "middle"
   if left:
       return "right"
   if right:
       return "left"
  
def main():
    global direction, left_detected, right_detected
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply ROI
        roi_frame, roi_mask = ROI(frame, frame_width, frame_height)

        # Visualize ROI
        roi_visualization = frame.copy()
        colored_mask = np.zeros_like(frame)
        colored_mask[roi_mask == 255] = [0, 255, 0]
        roi_visualization = cv2.addWeighted(roi_visualization, 1, colored_mask, 0.3, 0)
        cv2.imshow('ROI Visualization', roi_visualization)

        # Apply saturation mask
        mask = saturation_mask(roi_frame, saturation_threshold=0.4, min_percent=10, min_region_size=600)
        cv2.imshow('Saturation Mask', mask)

        # Detect presence in left and right ROI
        height_start = int(frame_height * 0.8)
        height_end = int(frame_height * 0.85)
        left_width = int(frame_width * 0.33)
        right_start = int(frame_width * 0.66)

        left_roi_mask = mask[height_start:height_end, 0:left_width]
        right_roi_mask = mask[height_start:height_end, right_start:frame_width]

        left_detected = np.count_nonzero(left_roi_mask) > 100
        right_detected = np.count_nonzero(right_roi_mask) > 100
    
        print(determine_direction(left_detected, right_detected))
        # Edge detection
        edges = cv2.Canny(mask, 100, 200)

        # Line detection
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=80, maxLineGap=10)

        filtered_lines = filter_lines(lines)
        if filtered_lines:
            line_image, direction = draw_lines_and_center(frame, filtered_lines)
            frame = cv2.addWeighted(frame, 0.8, line_image, 1, 0)

        # Show output
        cv2.imshow('Line Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
   main()

