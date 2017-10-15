import numpy as np
import cv2
from matplotlib import pyplot as plt


img = cv2.imread('/media/aymen/DATA/datasets/UECFOOD100/1/69.jpg')

# Initiate STAR detector
orb = cv2.ORB_create(nfeatures=100000, scoreType=cv2.ORB_FAST_SCORE)

# find the keypoints with ORB
kp = orb.detect(img, None)

# compute the descriptors with ORB
kp, des = orb.compute(img, kp)
# print (des, kp)
# draw only keypoints location,not size and orientation
img2 = img
cv2.drawKeypoints(img, outImage=img2, keypoints=kp, color=(0, 255, 0))
plt.imshow(img2), plt.show()
