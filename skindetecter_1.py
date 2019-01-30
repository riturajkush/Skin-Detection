from pyimagesearch import imutils
import numpy as np
import argparse
import cv2
import sys

lower = np.array([0, 40, 70], dtype = "uint8") 
upper = np.array([20, 255, 255], dtype = "uint8")



camera = cv2.VideoCapture(0)

while True:
	# grab the current frame
	'''The grabbed  variable is simply a boolean flag,
        indicating if the frame was successfully read or not.
        The frame  is the frame itself.'''
	(grabbed, frame) = camera.read()

	# resize the frame, convert it to the HSV color space,
	# and determine the HSV pixel intensities that fall into
	# the speicifed upper and lower boundaries
	frame = imutils.resize(frame, width = 600)
	converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	skinMask = cv2.inRange(converted, lower, upper)

	# apply a series of erosions and dilations to the mask
	# using an elliptical kernel
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
	skinMask = cv2.erode(skinMask, kernel, iterations = 2)
	skinMask = cv2.dilate(skinMask, kernel, iterations = 2)

	# blur the mask to help remove noise, then apply the
	# mask to the frame
	skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
	skin = cv2.bitwise_and(frame, frame, mask = skinMask)

	cv2.imshow("images", np.hstack([frame, skin]))

	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

camera.release()
cv2.destroyAllWindows()
