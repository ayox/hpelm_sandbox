from hpelm import ELM
import numpy as np
import time


def unpickle(fileName):
    import cPickle
    fo = open(fileName, 'rb')
    dictList = cPickle.load(fo)
    fo.close()
    return dictList


batch_size = 10000
hidden_num = 1000

_inputs = 3072
_outputs = 10
print("batch_size : {}".format(batch_size))
print("hidden_num : {}".format(hidden_num))
start_time = time.time()
k = 1
elm = ELM(_inputs, _outputs, batch=batch_size)
elm.add_neurons(hidden_num, "tanh")
while k <= 5:
    print("batch : {}".format(k))
    data = unpickle("./data/cifar-10-batches-py/data_batch_" + str(k))
    print (data)
    train_x = np.array(data['data'], dtype="int")
    train_y = np.array(data['labels'])
    train_y = np.eye(np.max(train_y) + 1)[train_y]
    elm.train(train_x, train_y)
    k += 1

end_time = time.time()

test = unpickle("./data/cifar-10-batches-py/test_batch")
test_x = np.array(test['data'], dtype="int")
test_y = np.array(test['labels'])
test_y = np.eye(np.max(test_y) + 1)[test_y]

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

print(" %s seconds" % (end_time - start_time))
print("accuracy: {0}".format(np.mean(save)))
