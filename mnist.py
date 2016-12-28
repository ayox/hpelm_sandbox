from hpelm import ELM
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import time

print("downloading")
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

batch_size = 10000
k = batch_size
_inputs = 28 * 28 * 1
_outputs = 10
elm = ELM(_inputs, _outputs, batch=batch_size)
elm.add_neurons(1000, "sigm")
print("batch : {}".format(k))
start_time = time.time()
while k <= mnist.train.num_examples:
    print("batch : {}".format(k))
    train_x, train_y = mnist.train.next_batch(batch_size)
    elm.train(train_x, train_y)
    k += batch_size
end_time = time.time()
print('done')
Y = elm.predict(mnist.test.images)

predict = []
for y in Y:
    predict.append(y.argmax())

gt = []
for y in mnist.test.labels:
    gt.append(y.argmax())

save = []
for _ in np.arange(0, len(gt)):
    k = True if gt[_] == predict[_] else False
    save.append(k)

print(" %s seconds" % (end_time - start_time))
print("accuracy: {0}".format(np.mean(save)))
