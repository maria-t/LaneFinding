import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle

# Read in and make a list of all calibration images located in the camera_cam folder
images = glob.glob('../camera_cal/calibration*.jpg')

# Arrays to hold object and image points respectively from all the images
objpoints = [] #3D points in real world space
imgpoints = [] #2D points in image plane

# 9x6 chessboard images
# Prepare object points, like (0,0,0), (1,0,0), ..., (8,5,0)
# Object points same across images
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2) # x,y coordinates

for fname in images:
	# Read in each image
	img = cv2.imread(fname)

	# Convert image to grayscale
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	# Find the chessboard corners
	ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

	# If corners are found, add object points and image points
	if ret == True:
		imgpoints.append(corners)
		objpoints.append(objp)

		# Draw and display the corners on the images
		corners_img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
		plt.imshow(corners_img)
		plt.show()
	
# Function that performs camera calibration and computes the distortion coefficients
def cal_undistort(img, objpoints, imgpoints):
	# Use cv2.calibrateCamera() and cv2.undistort()
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)	

	dst = cv2.undistort(img, mtx, dist, None, mtx)
	
	# Saving mtx and dist parameters for later
	with open('cam.pickle', 'wb')	as f:
		cam_pickle = {'mtx': mtx, 'dist' : dist}
		pickle.dump(cam_pickle,f)	

	return dst 	

# Display undistorted images
for fname in images:	
	# Read in each image
	img = cv2.imread(fname)
	undistorted = cal_undistort(img, objpoints, imgpoints)
	
	# Plot original images and their undistorted
	f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))
	f.tight_layout()	
	ax1.imshow(img)
	ax1.set_title('Original Image', fontsize=12)
	ax2.imshow(undistorted)
	ax2.set_title('Undistorted Image', fontsize=12)
	plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
	plt.show()


