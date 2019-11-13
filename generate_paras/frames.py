import numpy as np
#topleft, topright, bottomleft
frames = [[[27.89,14.03],[217.93,15.34],[27.21,123.97]]
          ]
frames = np.array(frames)
np.save('frames.npy', frames)
