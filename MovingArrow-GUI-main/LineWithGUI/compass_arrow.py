# BETA 1.0.0
# ALL FUNCTIONS IN THIS FILE WERE WRITTEN BY ANTHONY FURCAL FOR THE INDIVIDUAL LANE LINE DETECTION PROJECT
# image - The image you would like to compass to be overlayed on
# angle - angle that you want the arrow should be point in (intended to be calculated with another function)
# This function returns the original image with an arrow in the corner pointing in the direction of the specified angle
def compass_overlay(image, angle):

   angle = -angle

   if angle > 0:
       overlay_angle = angle - 90
   elif angle < 0:
       overlay_angle = angle + 90
   else:
       overlay_angle = angle


   overlay = cv2.imread('ExtraResources/CompassArrow.png', cv2.IMREAD_UNCHANGED)
   overlay_height, overlay_width = overlay.shape[:2]
   overlay_center = (overlay_width / 2, overlay_height / 2)
   overlay_rotation = cv2.getRotationMatrix2D(overlay_center, overlay_angle, 0.5)
   rotated_overlay = cv2.warpAffine(overlay, overlay_rotation, (overlay_width, overlay_height))
   background = image


   if rotated_overlay.shape[2] == 4:
       b, g, r, alpha = cv2.split(rotated_overlay)
       alpha = alpha / 255.0
       overlay_rgb = cv2.merge([b, g, r])
   elif len(overlay.shape) == 2:
       alpha = overlay / 255.0
       overlay_rgb = overlay


   new_width = 100
   new_height = 100
   resized_overlay_rgb = cv2.resize(overlay_rgb, (new_width, new_height))
   resized_alpha = cv2.resize(alpha, (new_width, new_height))


   x_offset = 50
   y_offset = 50
   roi = background[y_offset:y_offset + new_height, x_offset:x_offset + new_width]


   roi_rgb = roi.astype(np.float32)
   overlay_rgb = resized_overlay_rgb.astype(np.float32)


   blended_roi = (
           roi_rgb * (1 - resized_alpha[:, :, np.newaxis]) + overlay_rgb * resized_alpha[:, :, np.newaxis]).astype(
       np.uint8)
   background[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = blended_roi


   return background

