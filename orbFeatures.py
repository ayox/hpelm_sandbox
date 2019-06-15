import numpy as np
from numpy import array
from scipy.misc import imresize
import cv2
from matplotlib import pyplot as plt
from cutImages import cut_image
from random import shuffle


def read_feature(img):
    # img = cv2.imread(image_path)
    arr = array(img)
    # Initiate STAR detector
    orb = cv2.ORB_create(nfeatures=100000, scoreType=cv2.ORB_FAST_SCORE)
    # find the keypoints with ORB
    kp = orb.detect(arr, None)
    # compute the descriptors with ORB
    kp, des = orb.compute(arr, kp)
    # des = imresize(des, [100, 32])
    # print (type(kp), kp)
    # draw only keypoints location,not size and orientation
    # img2 = arr
    # cv2.drawKeypoints(arr, outImage=img2, keypoints=kp, color=(0, 255, 0))
    # plt.imshow(img2), plt.show()
    while des.shape[0] < 1000:
        des.append(des)
    # shuffle(des)
    return des[:1000, :]

# des = read_feature(img=cut_image('/media/aymen/DATA/datasets/UECFOOD100/1/81.jpg'))
# print (type(des), np.amax(des), des, des.shape)
