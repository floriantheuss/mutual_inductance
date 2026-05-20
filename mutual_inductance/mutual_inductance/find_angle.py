import numpy as np


X = 3e-8
Y = 1e-6

angle = np.angle(X+1j*Y)
print(angle*180/np.pi)
print(90-angle*180/np.pi)