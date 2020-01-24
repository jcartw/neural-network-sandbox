# From: "Ordering coordinates clockwise with Python and OpenCV" by Adrian Rosebrock
# Link: https://www.pyimagesearch.com/2016/03/21/ordering-coordinates-clockwise-with-python-and-opencv/

# import the necessary packages
from __future__ import print_function
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
from utils import order_points_old, order_points

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--new", type=int, default=-1,
	help="whether or not the new order points should should be used")
args = vars(ap.parse_args())

# load our input image, convert it to grayscale, and blur it slightly
image = cv2.imread("example.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # conver to grayscale (415 x 600) (0-255)
gray = cv2.GaussianBlur(gray, (7, 7), 0) # blur image
# show the image
# cv2.imshow("Image", gray)
# cv2.waitKey(0)

# perform edge detection, then perform a dilation + erosion to
# close gaps in between object edges
edged = cv2.Canny(gray, 50, 100) # apply Canny Edge Detection
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)
# show the image
# cv2.imshow("Image", edged) # 415 x 600 grayscale image (0 or 255)
# cv2.waitKey(0)

# find contours in the edge map
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts) # python list of 6 x [variable] (ragged)

# cnts dimensions: 6 x [174, 54, 66, 62, 221, 384]
# cnts[0][0] = [[474 241]]

# sort the contours from left-to-right and initialize the bounding box
# point colors
(cnts, _) = contours.sort_contours(cnts) # new dimensions: 6 x [66, 221, 62, 174, 384, 54]
colors = ((0, 0, 255), (240, 0, 159), (255, 0, 0), (255, 255, 0))

# loop over the contours individually
for (i, c) in enumerate(cnts):

	# if the contour is not sufficiently large, ignore it
	if cv2.contourArea(c) < 100:
		continue

	# compute the rotated bounding box of the contour, then
	# draw the contours
	box = cv2.minAreaRect(c)
	box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box) # 4x2 array (floats)
	box = np.array(box, dtype="int") # 4x2 np.ndarray (ints)
	cv2.drawContours(image, [box], -1, (0, 255, 0), 2) # draw box outlines (i.e. connect vertices)

	# show the original coordinates
	print("Object #{}:".format(i + 1))
	print("Shape: {0}:".format(box.shape))
	print(box)

	# order the points in the contour such that they appear
	# in top-left, top-right, bottom-right, and bottom-left
	# order, then draw the outline of the rotated bounding
	# box
	rect = order_points_old(box) # 4x2 np.ndarray (floats)
	# rect = order_points(box) # 4x2 np.ndarray (floats)

	# check to see if the new method should be used for
	# ordering the coordinates
	if args["new"] > 0:
		rect = perspective.order_points(box)

	# show the re-ordered coordinates
	print("Reordered Coordinates")
	print(rect.astype("int"))
	print("")

	# loop over the original points and draw them
	for ((x, y), color) in zip(rect, colors):
		cv2.circle(image, (int(x), int(y)), 5, color, -1)

	# draw the object num at the top-left corner
	cv2.putText(image, "Object #{}".format(i + 1),
		(int(rect[0][0] - 15), int(rect[0][1] - 15)),
		cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

	# show the image
	cv2.imshow("Image", image)

cv2.waitKey(0)