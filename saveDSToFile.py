import numpy as np
from numpy import array
from scipy.misc import imresize
from random import shuffle
from os import listdir
from os.path import isfile, join
from cutImages import cut_image


def read_images(path):
    fullList = []
    for c in range(1, 101):
        mypath = path + str(c) + "/"
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        print (len(onlyfiles), c)
        for x in onlyfiles:
            if x == 'bb_info.txt':
                continue
            img = cut_image(mypath + x)
            arr = array(img)
            arr = imresize(arr, [32, 32, 3])
            r = arr[:, :, 0].flatten()
            g = arr[:, :, 1].flatten()
            b = arr[:, :, 2].flatten()
            label = [c]
            out = np.array(list(label) + list(r) + list(g) + list(b), np.int)
            new = out / 255.
            new[0] = c
            fullList.append(new)
    shuffle(fullList)
    imagesList = np.array(fullList, dtype=float)
    print ("images shape: {}".format(imagesList.shape))
    np.save('foodDS_resized_cropped32_float', imagesList)
    return imagesList


read_images("/media/aymen/DATA/datasets/UECFOOD100/")
# print (l)
# print(np.load('imagesList.bin.npy'))
