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
    hsv= cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([110, 85, 85])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    lower_green = np.array([40,100,20])
    upper_green = np.array([90,255,255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    lower_yellow = np.array([20, 200, 20])
    upper_yellow = np.array([40, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    lower_magenta = np.array([140, 100, 20])
    upper_magenta = np.array([170, 255, 255])
    magenta_mask = cv2.inRange(hsv, lower_magenta, upper_magenta)
    lower_redorange = np.array([170,150,20])
    upper_redorange = np.array([10, 255,255])
    green_redorange = cv2.inRange(hsv, lower_redorange, upper_redorange)
    lower_orange = np.array([10, 150,20])
    upper_orange = np.array([30, 255, 255])
    orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)
    #equalized_result = cv2.equalizeHist(gray_result)
    #frame = cv2.blur(equalized_result, (31, 31))
    #_, binary = cv2.threshold(frame, 70, 255, cv2.THRESH_BINARY)

    return green_mask

img = cv2.imshow("IMG_5133.jpeg", 0)
cv2.imshow("result", lanes(img))
'''
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
'''
