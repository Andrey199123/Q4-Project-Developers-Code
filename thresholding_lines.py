import cv2
import numpy as np
import math

def is_5_percent_blue(frame):
    mask = lanes(frame)
    total_pixels = frame.shape[0] * frame.shape[1]
    black_pixels = np.sum(mask < 1)
    black_percentage = (black_pixels / total_pixels) * 100
    return black_percentage

def lanes(frame):
    '''
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur =
    ret, thresh = cv2.threshold(img,120,255,cv2.THRESH_TOZERO)
    cv2.imshow("thresh",thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''

    #frame = cv2.blur(frame, (31, 31))
    #_, binary = cv2.threshold(frame, 50, 255, cv2.THRESH_BINARY)
    gray_result = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    equalized_result = cv2.equalizeHist(gray_result)
    frame = cv2.blur(equalized_result, (31, 31))
    _, binary = cv2.threshold(frame, 50, 255, cv2.THRESH_BINARY)

    return binary

cap = cv2.VideoCapture(0)

if not cap.isOpened():
   print("Error: Could not open video file.")
else:
   while True:
      ret, frame = cap.read()
      if not ret:
         break  # Break the loop if no more frames are read

      cv2.imshow('Martian Detection PWP', lanes(frame))
      if is_5_percent_blue(frame) >= 5:
          print("Over 5%")
      else:
          print("Not Over 5%")

      if cv2.waitKey(25) & 0xFF == ord('q'):
         break  # Break the loop if 'q' is pressed

   cap.release()
   cv2.destroyAllWindows()
