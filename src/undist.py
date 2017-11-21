import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle


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

# Iterate over test_images folder and undistort the test images
for fname in images:	
	# Read in each image
	test_image = mpimg.imread(fname)
	undistorted = undistorted_image(test_image, mtx, dist)

	# Plot test images and their undistorted version
	f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))
	f.tight_layout()	
	ax1.imshow(test_image)
	ax1.set_title('Test Image', fontsize=12)
	ax2.imshow(undistorted)
	ax2.set_title('Undistorted Test Image', fontsize=12)
	plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
	plt.show()

######################################################################################################


def abs_sobel_thresh(img, orient='x', thresh=(0,255)):
	# Calculate directional gradient threshold
	
	# Take the derivative in x		
	# Sobel x
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
	abs_sobelx = np.absolute(sobelx)
	scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

	# Threshold x gradient
	sxbinary = np.zeros_like(scaled_sobel)
	sxbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

	return sxbinary


def hls_select(img, thresh=(0, 255)):
	# Calculate color channel threshold	
	
	# Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
	# Apply a threshold to the S channel
    S = hls[:,:,2]
	# Threshold color channel
    hls_binary = np.zeros_like(S)
    hls_binary[(S > thresh[0]) & (S <= thresh[1])] = 1

    return hls_binary

# Apply threshold functions to undistorted test images
for fname in images:
	# Read in each image and distortion-correct it
	test_image = mpimg.imread(fname)
	# Undistort image	
	image = undistorted_image(test_image, mtx, dist)
	
	hls_binary = hls_select(image, thresh=(170, 255))
	sx_binary = abs_sobel_thresh(image, orient='x', thresh=(20, 100))

	# Stack each channel to view their individual contributions in green and blue respectively
	# This returns a stack of the two binary images, whose components I can see as different colors
	color_binary = np.dstack(( np.zeros_like(sx_binary), sx_binary, hls_binary)) * 255

	# Combine the two binary thresholds
	combined_binary = np.zeros_like(sx_binary)
	combined_binary[(hls_binary == 1) | (sx_binary == 1)] = 1
	
	# Plotting the result
	f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,7))
	f.tight_layout()
	# The green is the gradient threshold component and the blue is the colr channel threshold component	
	ax1.set_title('Stacked thresholds', fontsize=12)
	ax1.imshow(color_binary)
	ax2.set_title('Combined S channel and gradient threshold', fontsize=12)
	ax2.imshow(combined_binary, cmap='gray')
	plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
	plt.show()

######################################################################################################


def warper(img):
	# Function that performs perspective transform on a given image
	
	image = undistorted_image(img, mtx, dist)
	img_size = (image.shape[1], image.shape[0])	
	
	hls_binary = hls_select(image, thresh=(150, 255))
	sx_binary = abs_sobel_thresh(image, orient='x', thresh=(20, 100))
	
	combined_binary = np.zeros_like(sx_binary)
	combined_binary[(hls_binary == 1) | (sx_binary == 1)] = 1

	src = np.float32([[230, 700],[1035, 700],[690, 460],[570, 460]])
	dst = np.float32([[280, 720],[980, 720],[980, 0],[280, 0]])
	
	M = cv2.getPerspectiveTransform(src, dst)
	Minv = cv2.getPerspectiveTransform(dst, src)
	warped = cv2.warpPerspective(combined_binary, M, img_size, flags=cv2.INTER_LINEAR)
	

	# Visualize transformation
#	f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,7))
#	f.tight_layout()
	
#	ax1.set_title('Binary image', fontsize=12)
#	ax1.imshow(combined_binary, cmap='gray')
#	ax2.set_title("Bird's eye view binary image", fontsize=12)
#	ax2.imshow(warped, cmap='gray')
#	plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
#	plt.show()

	return warped, M, Minv


######################################################################################################

# Iterate over test images
for test_image in images:
	test_image = mpimg.imread(test_image)
	transformed_image = warper(test_image)[0]
		
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

	# Visualization
	# Generate x and y values for plotting
	ploty = np.linspace(0, transformed_image.shape[0]-1, transformed_image.shape[0])
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
	out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
	plt.imshow(out_img)
	plt.plot(left_fitx, ploty, color='yellow')
	plt.plot(right_fitx, ploty, color='yellow')
	plt.xlim(0, 1280)
	plt.ylim(720, 0)
	plt.show()
	
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
	car_offset = (left_fit_cr[-1] + right_fit_cr[-1])/2 - (transformed_image.shape[1]/2)*xm_per_pix
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
	Minv = warper(test_image)[2]
	newwarp = cv2.warpPerspective(color_warp, Minv, (test_image.shape[1], test_image.shape[0])) 
	# Combine the result with the original image
	undistorted = undistorted_image(test_image, mtx, dist)
	result = cv2.addWeighted(undistorted, 1, newwarp, 0.3, 0)

	# Draw the text showing curvature and offset
	cv2.putText(result, 'Radius of Curvature = '+str(round(curverad,3)) + 'm', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
	cv2.putText(result, 'Car is '+str(abs(round(car_offset,3))) + 'm ' + side_position + ' of center', (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)	
	
	plt.imshow(result)
	plt.show()


	
	

























