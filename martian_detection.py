"""Version 1.0.0
Author: Emma Chetan
Functions:
   ORB(frame) --> Returns an image with either no change or a text box displaying
               'WE ARE NOT ALONE!!'

   show_not_alone(frame) --> Puts a textbox with red background and white lettering
               on the image; displays 'WE ARE NOT ALONE!!'

Sources:
https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html
https://youtu.be/lr1Sr0HJOoM?feature=shared
https://www.geeksforgeeks.org/python-opencv-cv2-puttext-method/

"""

import cv2

def ORB(frame):
   """Detect and display the Martian on each frame.

   Keyword arguments:
      frame -- An image captured from a live video stream,
               Mat | ndarray[Any, dtype] | ndarray = (cap. read())[1]
   Returns:
      frame -- An image with either no change or a text box displaying
               'WE ARE NOT ALONE!!'
   """

   martian = cv2.imread('templates/martian.png', 0)
   frame_resized = cv2.resize(frame, (312,380)) # Resize to dimensions of the martian template for ORB
   frame_blur = cv2.blur(frame_resized, (15,15)) # Reduce noise for less inaccurate matches

   orb = cv2.ORB_create()

   keypoints1, descriptors1 = orb.detectAndCompute(martian, None)
   keypoints2, descriptors2 = orb.detectAndCompute(frame_blur, None)

   bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
   try:
      matches = bf.match(descriptors1,descriptors2)
      matches = sorted(matches, key=lambda x: x.distance)
      num_matches = len(matches)

      if num_matches > 5:
         show_not_alone(frame)
      else:
         pass
   except:
      pass
   return frame

def show_not_alone(frame):
   """Put a text box on the frame.

   Keyword arguments:
       frame -- An image captured from a live video stream,
               Mat | ndarray[Any, dtype] | ndarray = (cap. read())[1]

   Returns:
      frame -- An image with a red text box and white lettering displaying
               'WE ARE NOT ALONE!!' in the top left corner.
   """

   font = cv2.FONT_HERSHEY_SIMPLEX
   font_scale = 1
   color = (255, 255, 255)  # White color
   thickness = 2
   position = (10, 50)  # Coordinates of the bottom-left corner of the text string in the image

   text = "WE ARE NOT ALONE!!"

   text_size = cv2.getTextSize(text, font, font_scale, thickness)[0] # Get text size to create a background rectangle
   text_width, text_height = text_size

   rectangle_color = (0, 0, 255)  # Red color
   rectangle_position1 = position  # Top-left corner of the rectangle is the same as the text position
   rectangle_position2 = (position[0] + text_width, position[1] - text_height - 10)  # Bottom-right corner

   cv2.rectangle(frame, rectangle_position1, rectangle_position2, rectangle_color, -1)

   cv2.putText(frame, text, position, font, font_scale, color, thickness, cv2.LINE_AA)


video_path = '/Users/pl1002215/PycharmProjects/object_detection/templates/IMG_5017.mov'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
   print("Error: Could not open video file.")
else:
   while True:
      ret, frame = cap.read()
      if not ret:
         break  # Break the loop if no more frames are read

      cv2.imshow('Martian Detection PWP', ORB(frame))

      if cv2.waitKey(25) & 0xFF == ord('q'):
         break  # Break the loop if 'q' is pressed

   cap.release()
   cv2.destroyAllWindows()
