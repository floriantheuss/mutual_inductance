import numpy as np


# X and Y are the in-phase and quadrature lock-in amplifier outputs (in volts)
X = 3e-8
Y = 1e-6

# Compute the mixing angle between X and Y channels caused by cable phase shifts etc.
angle = np.angle(X+1j*Y)
print(angle*180/np.pi)
print(90-angle*180/np.pi)