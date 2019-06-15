from numpy import genfromtxt
import numpy as np

my_data = genfromtxt('UECFOOD_100.csv', delimiter=',')
new_data = np.zeros([14585, 1001])
new_data[:, 0] = my_data[:, 1000]
new_data[:, 1:] = my_data[:, :1000]

print(new_data.shape, new_data[:, 0], new_data[:, 1000])


def save_ds(array):
    np.save('UECFOOD_100_1000Features_train', array[0:int(14585 * 0.75), :])
    np.save('UECFOOD_100_1000Features_test', array[int(14585 * 0.75):, :])


save_ds(new_data)
