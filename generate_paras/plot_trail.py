import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.interpolate import interp1d

modified_input = np.load('modified_inputs/modified_input{}.npy'.format(sys.argv[1]))

x=modified_input[:,0]
y=modified_input[:,1]


plt.scatter(x, y, s=0.1)
#plt.scatter(x,y, s=0.1)
plt.axis([0, 190, 0, 110])
plt.gca().invert_yaxis()
plt.show()
