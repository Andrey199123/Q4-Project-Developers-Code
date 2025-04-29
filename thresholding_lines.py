import cv2
import numpy as np
import math

def is_5_percent_blue(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    total_pixels = frame.shape[0] * frame.shape[1]
    blue_pixels = np.sum(mask > 0)
    blue_percentage = (blue_pixels / total_pixels) * 100
    return blue_percentage >= 5

def lanes(frame):
    '''
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur =
    ret, thresh = cv2.threshold(img,120,255,cv2.THRESH_TOZERO)
    cv2.imshow("thresh",thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    frame = cv2.blur(frame, (13,13))

    _, binary = cv2.threshold(frame, 50, 255, cv2.THRESH_BINARY)
    gray_result = cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY)
    equalized_result = cv2.equalizeHist(gray_result)

    return equalized_result

cap = cv2.VideoCapture(0)

if not cap.isOpened():
   print("Error: Could not open video file.")
else:
   while True:
      ret, frame = cap.read()
      if not ret:
         break  # Break the loop if no more frames are read

      cv2.imshow('Martian Detection PWP', lanes(frame))

      if cv2.waitKey(25) & 0xFF == ord('q'):
         break  # Break the loop if 'q' is pressed

   cap.release()
   cv2.destroyAllWindows()
