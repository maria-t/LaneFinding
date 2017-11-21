import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle
from moviepy.editor import VideoFileClip

# Load camera matrix and distortion coefficients
cam_info = pickle.load(open("cam.pickle", "rb"))
mtx = cam_info["mtx"]
dist = cam_info["dist"]

# Read in and make a list of all test images located in the test_images folder
images = glob.glob('../test_images/*.jpg')

######################################################################################################

# Function that performs distortion correction on images
def undistorted_image(img, mtx, dist):
	# Use cv2.undistort()
	undistorted = cv2.undistort(img, mtx, dist, None, mtx)
	
	return undistorted	

######################################################################################################

# Functions that convert image to HSV color space and apply yellow and white color masks

def select_yellow(img):
	hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
	lower = np.array([20,60,60])
	upper = np.array([38,174, 250])
	mask = cv2.inRange(hsv, lower, upper)

	return mask

def select_white(img):
	hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
	lower = np.array([202,202,202])
	upper = np.array([255,255,255])
	mask = cv2.inRange(img, lower, upper)

	return mask

def hsv_select(img):
	yellow = select_yellow(img)
	white = select_white(img)

	hsv_binary = np.zeros_like(yellow)
	hsv_binary[(yellow >= 1) | (white >= 1)] = 1

	return hsv_binary

######################################################################################################


def warper(img):
	# Function that performs perspective transform on a given image
	
	image = undistorted_image(img, mtx, dist)
	img_size = (image.shape[1], image.shape[0])	
	
	hsv_binary = hsv_select(image)
	
	src = np.float32([[230, 700],[1035, 700],[690, 460],[570, 460]])
	dst = np.float32([[280, 720],[980, 720],[980, 0],[280, 0]])
	
	M = cv2.getPerspectiveTransform(src, dst)
	Minv = cv2.getPerspectiveTransform(dst, src)
	warped = cv2.warpPerspective(hsv_binary, M, img_size, flags=cv2.INTER_LINEAR)

	return warped, M, Minv

######################################################################################################


def pipeline_final(img):
	#test_image = mpimg.imread(img)
	transformed_image = warper(img)[0]
		
	# Take a histogram of the bottom half of the transformed images to identify which pixels are part of the lines
	histogram = np.sum(transformed_image[transformed_image.shape[0]//2:,:], axis = 0)
	# Create an output image to draw on and  visualize the result
	out_img = (np.dstack((transformed_image, transformed_image, transformed_image))*255).astype('uint8')
	# Find the peak of the left and right halves of the histogram
	# These will be the starting point for the left and right lines
	midpoint = np.int(histogram.shape[0]/2)
	leftx_base = np.argmax(histogram[:midpoint])
	rightx_base = np.argmax(histogram[midpoint:]) + midpoint

	# Choose the number of sliding windows
	nwindows = 9
	# Set height of windows
	window_height = np.int(transformed_image.shape[0]/nwindows)
	# Identify the x and y positions of all nonzero pixels in the image
	nonzero = transformed_image.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	# Current positions to be updated for each window
	leftx_current = leftx_base
	rightx_current = rightx_base
	# Set the width of the windows +/- margin
	margin = 80
	# Set minimum number of pixels found to recenter window
	minpix = 50
	# Create empty lists to receive left and right lane pixel indices
	left_lane_inds = []
	right_lane_inds = []

	# Step through the windows one by one
	for window in range(nwindows):
		# Identify window boundaries in x and y (and right and left)
		win_y_low = transformed_image.shape[0] - (window+1)*window_height
		win_y_high = transformed_image.shape[0] - window*window_height
		win_xleft_low = leftx_current - margin
		win_xleft_high = leftx_current + margin
		win_xright_low = rightx_current - margin
		win_xright_high = rightx_current + margin
		# Draw the windows on the visualization image
		cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
		(0,255,0), 2) 
		cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
		(0,255,0), 2) 
		# Identify the nonzero pixels in x and y within the window
		good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
		(nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
		good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
		(nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
		# Append these indices to the lists
		left_lane_inds.append(good_left_inds)
		right_lane_inds.append(good_right_inds)
		# If you found > minpix pixels, recenter next window on their mean position
		if len(good_left_inds) > minpix:
		    leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
		if len(good_right_inds) > minpix:        
		    rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

	# Concatenate the arrays of indices
	left_lane_inds = np.concatenate(left_lane_inds)
	right_lane_inds = np.concatenate(right_lane_inds)

	# Extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds] 
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds] 

	# Fit a second order polynomial to each
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)


	nonzero = transformed_image.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	margin = 100
	left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
	left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
	left_fit[1]*nonzeroy + left_fit[2] + margin))) 

	right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
	right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
	right_fit[1]*nonzeroy + right_fit[2] + margin)))  

	# Again, extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds] 
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds]
	# Fit a second order polynomial to each
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)	

	
	# Visualization
	# Generate x and y values for plotting
	ploty = np.linspace(0, transformed_image.shape[0]-1, transformed_image.shape[0])
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	y_eval = transformed_image.shape[0]
	# Define conversions in x and y from pixels space to meters
	ym_per_pix = 30/720 # meters per pixel in y dimension
	xm_per_pix = 3.7/700 # meters per pixel in x dimension

	# Fit the polynomials to x,y in world space
	left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
	right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
	# Calculate the new radii of curvature
	left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
	right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
	curverad = (left_curverad + right_curverad)/2
	
	# Vehicle position in respect to the center
	camera_position = (left_fitx[-1] + right_fitx[-1])/2
	car_offset = (camera_position - transformed_image.shape[1]/2)*xm_per_pix
	side_position = 'left'	
	if car_offset <= 0:
		side_position = 'right'
	
	# Create an image to draw the lines on
	warp_zero = np.zeros_like(transformed_image).astype(np.uint8)
	color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

	# Recast the x and y points into usable format for cv2.fillPoly()
	pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
	pts = np.hstack((pts_left, pts_right))
	# Draw the lane onto the warped blank image
	cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

	# Warp the blank back to original image space using inverse perspective matrix (Minv)
	Minv = warper(img)[2]
	newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0])) 
	# Combine the result with the original image
	undistorted = undistorted_image(img, mtx, dist)
	result = cv2.addWeighted(undistorted, 1, newwarp, 0.3, 0)

	# Draw the text showing curvature and offset
	cv2.putText(result, 'Radius of Curvature = '+str(round(curverad,3)) + 'm', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
	cv2.putText(result, 'Car is '+str(abs(round(car_offset,3))) + 'm ' + side_position + ' of center', (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)	
	
	return result


output = 'project_video_output.mp4'

clip1 = VideoFileClip('project_video.mp4')

video_clip = clip1.fl_image(pipeline_final) #NOTE: this function expects color images!!
video_clip.write_videofile(output, audio=False)
	

























