"""Version 1.0.0
Author: Emma Chetan
Functions:
    isolate_high_saturation(image, saturation_threshold) --> Returns a mask of an input image isolating high saturation
                                                            values above a certain saturation threshold

Sources:
    https://stackoverflow.com/questions/17185151/how-to-obtain-a-single-channel-value-image-from-hsv-image-in-opencv-2-1
"""


import cv2
import numpy as np

def isolate_high_saturation(image, saturation_threshold):
    """Isolates areas with high saturation in an image.

    Keyword arguments:
        image -- Input image.
        saturation_threshold -- Decimal saturation value above which to isolate.

    Returns:
        mask -- Binary mask highlighting high saturation areas.
    """

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Extract the saturation channel from the hsv
    saturation_channel = hsv_image[:, :, 1]

    # Create a mask based on the saturated the values are
    mask = saturation_channel > (saturation_threshold * 255) # Assuming saturation values are between 0 and 255
    mask = mask.astype(np.uint8) * 255
    mask = cv2.bitwise_not(mask)
    return mask

cap = cv2.VideoCapture(0)

if not cap.isOpened():
   print("Error: Could not open video file.")
else:
   while True:
      ret, frame = cap.read()
      if not ret:
         break
      saturation_threshold = 0.65  # Threshold for how saturated values should be, between 0-1

      mask = isolate_high_saturation(frame, saturation_threshold)
      cv2.imshow("Original Image", frame)
      cv2.imshow("Saturation Mask", mask)  # Convert mask to 0-255 for display

      if cv2.waitKey(25) & 0xFF == ord('q'):
         break  # Break the loop if 'q' is pressed

   cap.release()
   cv2.destroyAllWindows()
