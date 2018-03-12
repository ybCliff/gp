import numpy as np

a = np.zeros((5, 5, 2048))
b = np.zeros((5, 5, 2048))
tmp = np.concatenate((a, b), axis=0)
print(tmp.shape)
