'''
Created on 20 jan. 2017

@author: fremorti
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as col
import matplotlib.cm as cm

#define color map
cdict = {'red': ((0.0, 0.0, 0.0),
                 (0.1, 0.1, 0.1),
                 (0.2, 0.2, 0.2),
                     (0.3, 0.3, 0.3),
                     (0.4, 0.4, 0.4),
                     (0.5, 0.5, 0.5),
                     (0.6, 0.6, 0.6),
                     (0.7, 0.7, 0.7),
                     (0.8, 0.8, 0.8),
                     (0.9, 0.9, 0.9),
                     (1.0, 1.0, 1.0)),
         'green': ((0.0, 1.0, 1.0),
                   (0.1, 0.9, 0.9),
                     (0.2, 0.8, 0.8),
                     (0.3, 0.7, 0.7),
                     (0.4, 0.6, 0.6),
                     (0.5, 0.5, 0.5),
                     (0.6, 0.4, 0.4),
                     (0.7, 0.3, 0.3),
                     (0.8, 0.2, 0.2),
                     (0.9, 0.1, 0.1),
                     (1.0, 0.0, 0.0)),
         'blue': ((0.0, 0.0, 0.0),
                  (0.25, 0.0, 0.0),
                  (0.6, 0.0, 0.0),
                  (0.75, 0.0, 0.0), 
                  (1.0, 0.0, 0.0))}
cmap1 = col.LinearSegmentedColormap('my_colormap',cdict,N=256,gamma=0.75)
cm.register_cmap(name='GR', cmap=cmap1)



fig, ax = plt.subplots()

image = np.random.random(size=(32, 32))/2+0.5
print(image)
ax.imshow(image, cmap='magma', interpolation='nearest')
plt.axis('off')
plt.show()