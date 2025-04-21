import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os


def ORB(img2):
   img = cv.imread('templates/martian.png', 0)
   img2 = cv.resize(img2, (312,380))
   img2 = cv.blur(img2, (15,15))
   #gray_img2 = cv.imread(img2, cv.IMREAD_GRAYSCALE)
   #gray_img = cv.imread(img, cv.IMREAD_GRAYSCALE)


   orb = cv.ORB_create()

   keypoints1, descriptors1 = orb.detectAndCompute(img, None)
   keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

   bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
   try:
      matches = bf.match(descriptors1,descriptors2) # descriptors2 is None
      matches = sorted(matches, key=lambda x: x.distance)
      num_matches = len(matches)  # 104 for other matches, 400+ for same image w/o blur
      # Draw first 50 matches.*
      img2 = cv.drawMatches(img, keypoints1, img2, keypoints2, matches[:50], None, flags=2)

      #plt.imshow(img3), plt.show()
      #print(num_matches)

      if num_matches > 5:
         print("we are not alone")
         not_alone('NOT ALONE', img2)
      else:
         print("no matches")
   except:
      print("no matches")
   return img2

def not_alone(text, img):
   font = cv.FONT_HERSHEY_SIMPLEX
   font_scale = 1
   color = (255, 255, 255)  # White color
   thickness = 2
   position = (10, 50)  # Coordinates of the bottom-left corner of the text string in the image

   # Get text size to create a background rectangle
   text_size = cv.getTextSize(text, font, font_scale, thickness)[0]
   text_width, text_height = text_size

   # Background rectangle settings
   rectangle_color = (0, 0, 255)  # Red color
   rectangle_position1 = position  # Top-left corner of the rectangle is the same as the text position
   rectangle_position2 = (position[0] + text_width, position[1] - text_height - 10)  # Bottom-right corner

   # Draw the background rectangle
   cv.rectangle(img, rectangle_position1, rectangle_position2, rectangle_color, -1)

   # Put the text on the image
   cv.putText(img, text, position, font, font_scale, color, thickness, cv.LINE_AA)


video_path = '/Users/pl1002215/PycharmProjects/object_detection/templates/IMG_5017.mov'  # Replace with your video file path
cap = cv.VideoCapture(video_path)

if not cap.isOpened():
   print("Error: Could not open video file.")
else:
   while True:
      ret, frame = cap.read()
      if not ret:
         break  # Break the loop if no more frames are read

      cv.imshow('Video Player', ORB(frame))

      if cv.waitKey(25) & 0xFF == ord('q'):
         break  # Break the loop if 'q' is pressed

   cap.release()
   cv.destroyAllWindows()
