from hpelm import ELM
import numpy as np
import time
from PIL import Image
from numpy import array
from scipy.misc import imresize
from random import shuffle
from os import listdir
from os.path import isfile, join


def read_images(path):
    fullList = []
    for c in range(1, 6):
        mypath = path + str(c) + "/"
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        print (len(onlyfiles), c)
        for x in onlyfiles:
            img = Image.open(mypath + x)
            arr = array(img)
            arr = imresize(arr, [32, 32, 3])
            r = arr[:, :, 0].flatten()
            g = arr[:, :, 1].flatten()
            b = arr[:, :, 2].flatten()
            label = [c]
            out = np.array(list(label) + list(r) + list(g) + list(b), np.int)
            fullList.append(out)
    shuffle(fullList)
    imagesList = np.array(fullList, dtype=int)
    print ("images shape: {}".format(imagesList.shape))
    return imagesList


imagesList = read_images("/media/aymen/DATA/datasets/food-for-HPELM/")

batch_size = 200
hidden_num = 2000
_inputs = 3072
_outputs = 6
print("batch_size : {}".format(batch_size))
print("hidden_num : {}".format(hidden_num))
start_time = time.time()
k = 1
elm = ELM(_inputs, _outputs, batch=batch_size)
elm.add_neurons(hidden_num, "tanh")
while k <= 5:
    print("batch : {}".format(k))
    data = imagesList
    print (data.shape, data.dtype)
    train_x = np.array(data[:(batch_size * k), 1:], dtype="int")
    train_y = np.array(data[:(batch_size * k), 0])
    print ("X", train_x.shape)
    print ("Y", train_y.shape)
    train_y = np.eye(np.max(train_y) + 1)[train_y]
    elm.train(train_x, train_y)
    k += 1

end_time = time.time()

testImagesList = read_images("/media/aymen/DATA/datasets/food-for-HPELM/test/")
print (testImagesList[:, 1:])
test_x = np.array(testImagesList[:, 1:], dtype="int")
print (testImagesList[:, 0])
test_y = np.array(testImagesList[:, 0])
test_y = np.eye(np.max(test_y) + 1)[test_y]
#
Y = elm.predict(test_x)
predict = []
for y in Y:
    predict.append(y.argmax())

gt = []
for y in test_y:
    gt.append(y.argmax())

save = []
for _ in np.arange(0, len(gt)):
    k = True if gt[_] == predict[_] else False
    save.append(k)
#
print(" %s seconds" % (end_time - start_time))
print("accuracy: {0}".format(np.mean(save)))
