import numpy as np
from numpy import array
from scipy.misc import imresize
from os import listdir
from os.path import isfile, join
from cutImages import cut_image
from orbFeatures import read_feature
from PIL import Image
from sklearn.utils import shuffle
from tqdm import tqdm
import cPickle

saved_folder = 'images/'


def saveFolderToFiles_UECFOOD100(path, type, img_size):
    for c in tqdm(range(1, 7)):
        folder_images = []
        folder_images_path = saved_folder + str(c) + type + str(img_size)
        mypath = path + str(c) + "/"
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        # print (len(onlyfiles), c)
        for x in tqdm(onlyfiles):
            if x == 'bb_info.txt':
                continue
            # img = cut_image(mypath + x)
            # arr = array(img)
            try:
                # arr = read_feature(img)
                image_path = mypath + x
                im = Image.open(image_path)
                arr = array(im)
                # print ((mypath + x), type(arr), arr.shape)
                # arr = from features
                arr = imresize(arr, [img_size, img_size, 3])
                # r = arr[:, :, 0].flatten()
                # g = arr[:, :, 1].flatten()
                # b = arr[:, :, 2].flatten()
                label = [c]
                # out = np.array(list(label) + list(r) + list(g) + list(b), np.int)
                out = np.array(list(label) + list(arr.flatten()), np.int)
                new = out / 255.
                new[0] = c
                folder_images.append(new)
            except:
                continue
        np.save(folder_images_path, folder_images)


def read_images_UECFOOD100(path, type, img_size):
    fullList = []
    for c in tqdm(range(1, 101)):
        mypath = path + str(c) + "/"
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        # print (len(onlyfiles), c)
        for x in tqdm(onlyfiles):
            if x == 'bb_info.txt':
                continue
            # img = cut_image(mypath + x)
            # arr = array(img)
            try:
                # arr = read_feature(img)
                image_path = mypath + x
                im = Image.open(image_path)
                arr = array(im)
                # print ((mypath + x), type(arr), arr.shape)
                # arr = from features
                arr = imresize(arr, [img_size, img_size, 3])
                # r = arr[:, :, 0].flatten()
                # g = arr[:, :, 1].flatten()
                # b = arr[:, :, 2].flatten()
                label = [c]
                # out = np.array(list(label) + list(r) + list(g) + list(b), np.int)
                out = np.array(list(label) + list(arr.flatten()), np.int)
                new = out / 255.
                new[0] = c
                fullList.append(new)
            except:
                continue
    shuffle(fullList)
    images_list = np.array(fullList, dtype=float)
    print ("images shape: {}".format(images_list.shape))
    np.save('UECFOOD100_' + type + str(img_size), images_list)
    return images_list


def read_images_food101(path):
    return None


saveFolderToFiles_UECFOOD100("/media/aymen/DATA/datasets/UECFOOD100-test/", 'test', 244)
