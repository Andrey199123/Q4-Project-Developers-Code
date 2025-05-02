import cv2
import numpy as np
import time
previous_lines = []
previous_mid_x = None
alpha = 0.9
Vertical_Detected = False
Horizontal_Detected = False
printed_turn_msg = False
def ROI(image, frame_width, frame_height):
    # Create a blank ROI mask the same size as the image
    roi_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    
    # Define ROI regions (left and middle)
    left_width = int(frame_width * 0.33)  # Left region: 33% of frame width
    middle_start = int(frame_width * 0.33)
    middle_end = int(frame_width * .66)  # Middle region: 33%-66% of frame width
    right_width = int(frame_width)  # Right region: 33% of frame width
    # Fill the ROI mask for left region
    roi_mask[:, 0:left_width] = 255
    roi_mask[:, middle_end:right_width] = 255
    # Fill the ROI mask for middle region
    roi_mask[:, middle_start:middle_end] = 255
    
    # Apply the ROI mask to the image
    masked_image = cv2.bitwise_and(image, image, mask=roi_mask)
    cv2.imshow('ROI Mask', masked_image)

    
    return masked_image, roi_mask
def determine_roi(x, frame_width):
    """
    Determines which ROI the x-coordinate belongs to.
    """
    left_width = int(frame_width * 0.33)
    middle_start = int(frame_width * 0.33)
    middle_end = int(frame_width * 0.66)

    if x < left_width:
        return "Left"
    elif middle_start <= x < middle_end:
        return "Middle"
    else:
        return "Right"

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
    global previous_mid_x, Vertical_Detected, Horizontal_Detected
    line_image = np.zeros_like(frame)
    line_detected = False
    
    for x1, y1, x2, y2 in lines:
        cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 3)

    # Filter lines by angle
    vertical_lines = [line for line in lines if abs(np.degrees(np.arctan2(line[3] - line[1], line[2] - line[0]))) > 70]
    horizontal_lines = [line for line in lines if abs(np.degrees(np.arctan2(line[3] - line[1], line[2] - line[0]))) < 20]
    
    # Handle vertical lines
    main_vertical_lines = get_two_main_lines(vertical_lines, frame.shape[1])
    if len(main_vertical_lines) == 2:
        left_x = main_vertical_lines[0][0]
        right_x = main_vertical_lines[1][0]
        mid_x = int((left_x + right_x) / 2)
        mid_x = smooth_value(mid_x, previous_mid_x)
        previous_mid_x = mid_x
        section = determine_roi(mid_x, frame.shape[1])
        Vertical_Detected = True
        print(f"Vertical Detected: {Vertical_Detected}, Mid X: {mid_x}")
        height = frame.shape[0]
        cv2.line(line_image, (mid_x, 0), (mid_x, height), (0, 0, 255), 3)
    else:
        Vertical_Detected = False
    
    # Handle horizontal lines
    if len(horizontal_lines) >= 2:
        horizontal_lines.sort(key=lambda line: line[1])
        top_line = horizontal_lines[0]
        bottom_line = horizontal_lines[-1]
        line_detected = True
        section = determine_roi(top_line[0], frame.shape[1])
        print(f"Horizontal Detected: {line_detected}")
        mid_y = int((top_line[1] + bottom_line[1]) / 2)
        width = frame.shape[1]
        cv2.line(line_image, (0, mid_y), (width, mid_y), (0, 0, 255), 3)
    else:
        line_detected = False
    
    return line_detected, section

def main():
    cap = cv2.VideoCapture(0)
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

        # Apply ROI to the frame
        roi_frame, roi_mask = ROI(frame, frame_width, frame_height)
        
        # For visualization, you might want to show the original frame with ROI overlay
        roi_visualization = frame.copy()
        # Create a colored mask for visualization (green with 50% transparency)
        colored_mask = np.zeros_like(frame)
        colored_mask[roi_mask == 255] = [0, 255, 0]  # Green color
        roi_visualization = cv2.addWeighted(roi_visualization, 1, colored_mask, 0.3, 0)
        cv2.imshow('ROI Visualization', roi_visualization)
        
        # Continue with processing only the ROI areas
        hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
        lower_color = np.array([90, 100, 50])  # Broadened lower bound (H, S, V)
        upper_color = np.array([150, 255, 255])  # Broadened upper bound (H, S, V)
        mask = cv2.inRange(hsv, lower_color, upper_color)
        
        # Apply the mask to the original frame (only show tape-colored areas)
        masked_frame = cv2.bitwise_and(roi_frame, roi_frame, mask=mask)
        
        # Convert to grayscale and detect edges
        gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 100, 200)
        
        # Detect lines
        lines = cv2.HoughLinesP(edges,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=80,
            maxLineGap=10
        )

        # Filter and draw lines
        filtered_lines = filter_lines(lines)
        if filtered_lines:
            line_image, direction = draw_lines_and_center(frame, filtered_lines)
            frame = cv2.addWeighted(frame, 0.8, line_image, 1, 0)
        # Show the result
        cv2.imshow('Line Detection', frame)
        if direction == "Left":
            if printed_turn_msg == False:
                print("Turning Left")
                printed_turn_msg = True
                time.sleep(3)
                printed_turn_msg = False
        elif direction == "Right":
            if printed_turn_msg == False:
                print("Turning Right")
                printed_turn_msg = True
                time.sleep(3)
                printed_turn_msg = False
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows() 

if __name__ == "__main__":
    main()