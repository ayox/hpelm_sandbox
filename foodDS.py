from hpelm import ELM
import numpy as np
import time

# from PIL import Image
# from numpy import array
# from scipy.misc import imresize
# from random import shuffle
# from os import listdir
# from os.path import isfile, join
# from cutImages import cut_image


# def read_images(path):
#     fullList = []
#     for c in range(1, 100):
#         mypath = path + str(c) + "/"
#         onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
#         print (len(onlyfiles), c)
#         for x in onlyfiles:
#             if x == 'bb_info.txt':
#                 continue
#             img = cut_image(mypath + x)
#             arr = array(img)
#             arr = imresize(arr, [32, 32, 3])
#             r = arr[:, :, 0].flatten()
#             g = arr[:, :, 1].flatten()
#             b = arr[:, :, 2].flatten()
#             label = [c]
#             out = np.array(list(label) + list(r) + list(g) + list(b), np.int)
#             fullList.append(out)
#     shuffle(fullList)
#     imagesList = np.array(fullList, dtype=int)
#     print ("images shape: {}".format(imagesList.shape))
#     return imagesList
np.set_printoptions(threshold=100)
print('lodaing... ')
imagesList = np.load('foodDS_resized_cropped32_float.npy')
epoch = 3
batch_size = imagesList.__len__() / epoch
hidden_num = 2000
_inputs = 3072
_outputs = 101
print("batch_size : {}".format(batch_size))
print("hidden_num : {}".format(hidden_num))
start_time = time.time()
k = 1
elm = ELM(_inputs, _outputs, batch=batch_size)
elm.add_neurons(hidden_num, "tanh")
old_batch_size_k = 0
while k <= epoch:
    print("batch : {}".format(k))
    data = imagesList
    print (data.shape, data.dtype)
    train_x = np.array(data[old_batch_size_k + 1:(batch_size * k), 1:], dtype="float")
    train_y = np.array(data[old_batch_size_k + 1:(batch_size * k), 0], dtype="int")
    old_batch_size_k = batch_size * k
    print ("X", train_x.shape)
    print ("Y", train_y.shape)
    train_y = np.eye(np.max(train_y) + 1)[train_y]
    elm.train(train_x, train_y)
    k += 1

end_time = time.time()

testImagesList = np.load('foodDS_resized_cropped32_test_float.npy')
test_x = np.array(testImagesList[:, 1:], dtype="float")
print (testImagesList[:, 0], testImagesList.__len__())
test_y = np.array(testImagesList[:, 0], dtype="int")
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
