# BETA 1.0.0
# ALL CODE IN THIS FILE WAS WRITTEN BY DESERTRON MEMBER ANTHONY FURCAL FOR THE INDEPENDENT LANE LINE DETECTION PROJECT


import cv2
import numpy as np
from numpy.ma.core import arctan
from compass_arrow import compass_overlay


# Image - the image file you would like to warp
# Reverse - boolean value, keep to false to transform an image normally, set to true if you are reversing another transform
# This function returns the warped image
def warp_frame(image, reverse):
   roi_points = np.array([
       (322, 174),  # Top-left corner
       (202, 237),  # Bottom-left corner
       (500, 237),  # Bottom-right corner
       (400, 174)  # Top-right corner
   ])


   dest_points = np.array([
       (0, 0),
       (0, 360),
       (640, 360),
       (640, 0)
   ])


   pts = np.float32(roi_points)
   pts2 = np.float32(dest_points)


   if not reverse:


       perspectiveMatrix = cv2.getPerspectiveTransform(pts, pts2)


   else:


       perspectiveMatrix = cv2.getPerspectiveTransform(pts2, pts)


   result = cv2.warpPerspective(image, perspectiveMatrix, (640, 360))


   return result

# binary_image - A prefiltered image that will be used to calculate the location of lines
# image - original image that filtered image was made from, this is where the lines will be overlayed.
# This function returns the original image with the overlayed lines.
def hist_detection(binary_image, image):
   histogram = np.sum(binary_image[binary_image.shape[0] // 2:, :], axis=0)


   midpoint = np.int32(histogram.shape[0] // 2)
   leftx_base = np.argmax(histogram[:midpoint])
   rightx_base = np.argmax(histogram[midpoint:]) + midpoint


   nwindows = 9


   margin = 100


   minpix = 50


   window_height = np.int32(binary_image.shape[0] // nwindows)
   nonzero = binary_image.nonzero()
   nonzeroy = np.array(nonzero[0])
   nonzerox = np.array(nonzero[1])
   leftx_current = leftx_base
   rightx_current = rightx_base


   left_lane_inds = []
   right_lane_inds = []


   for window in range(nwindows):
       win_y_low = binary_image.shape[0] - (window + 1) * window_height
       win_y_high = binary_image.shape[0] - window * window_height
       win_xleft_low = leftx_current - margin
       win_xleft_high = leftx_current + margin
       win_xright_low = rightx_current - margin
       win_xright_high = rightx_current + margin


       good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                         (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
       good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]


       left_lane_inds.append(good_left_inds)
       right_lane_inds.append(good_right_inds)


       if len(good_left_inds) > minpix:
           leftx_current = np.int32(np.mean(nonzerox[good_left_inds]))
       if len(good_right_inds) > minpix:
           rightx_current = np.int32(np.mean(nonzerox[good_right_inds]))


   try:
       left_lane_inds = np.concatenate(left_lane_inds)
       right_lane_inds = np.concatenate(right_lane_inds)
   except ValueError:
       pass


   leftx = nonzerox[left_lane_inds]
   lefty = nonzeroy[left_lane_inds]
   rightx = nonzerox[right_lane_inds]
   righty = nonzeroy[right_lane_inds]


   try:
       left_fit = np.polyfit(lefty, leftx, 2)
       right_fit = np.polyfit(righty, rightx, 2)


       ploty = np.linspace(0, binary_image.shape[0] - 1, binary_image.shape[0])
       try:
           left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
           right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
       except TypeError:
           print('The function failed to fit a line!')
           left_fitx = 1 * ploty ** 2 + 1 * ploty
           right_fitx = 1 * ploty ** 2 + 1 * ploty


       window_img = image
       left_line_window1 = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
       left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx,
                                                                       ploty])))])
       left_line_pts = np.hstack((left_line_window1, left_line_window2))
       right_line_window1 = np.array([np.transpose(np.vstack([right_fitx, ploty]))])
       right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx,
                                                                        ploty])))])
       right_line_pts = np.hstack((right_line_window1, right_line_window2))



       center_pts = (right_line_pts + left_line_pts) / 2

       flat_pts = np.concatenate(center_pts)

       center_coeffecients = np.polyfit(flat_pts[:,0], flat_pts[:,1], 1)
       slope = center_coeffecients[0]



       angle = np.degrees(arctan(slope))

       print(angle)

       cv2.polylines(window_img, np.int64([left_line_pts]), False, (255, 0, 0), 15)
       cv2.polylines(window_img, np.int64([right_line_pts]), False, (255, 0, 0), 15)
       cv2.polylines(window_img, np.int64([center_pts]), False, (0, 255, 0), 15)
       result = window_img


       return result, angle


   except TypeError:
       print("No line found")
       return image, 0


# frame - the frame you need to be processed
# This function returns the original image after being pipelined through all the other functions in this file.
def stream_processing(frame):
    resized_frame = cv2.resize(frame, (640, 360))
    warped = warp_frame(resized_frame, False)

    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    sx_binary = np.zeros_like(scaled_sobel)
    sx_binary[(scaled_sobel >= 90) & (scaled_sobel <= 255)] = 1


    white_binary = np.zeros_like(gray)
    white_binary[(gray > 230) & (gray <= 255)] = 1


    hls = cv2.cvtColor(warped, cv2.COLOR_BGR2HLS)
    H = hls[:, :, 0]
    S = hls[:, :, 2]


    sat_binary = np.zeros_like(S)
    sat_binary[(S > 90) & (S <= 100)] = 1


    hue_binary = np.zeros_like(H)
    hue_binary[(H > 0) & (H <= 10)] = 1


    binary_1 = cv2.bitwise_or(sx_binary, white_binary)
    binary_2 = cv2.bitwise_or(sat_binary, sat_binary)
    binary = cv2.bitwise_or(binary_1, binary_2)


    cv2.imshow("image", binary_2 * 255)


    with_lines, rotation_angle = hist_detection(binary, warped)


    rev_warp = warp_frame(with_lines, True)


    final_result = resized_frame.copy()


    mask = np.zeros_like(final_result)
    mask[rev_warp > 0] = 255


    final_result[mask == 255] = rev_warp[mask == 255]

    overlay = compass_overlay(final_result, rotation_angle)

    return final_result


