import numpy as np
from scipy.spatial import Delaunay
from scipy.io import loadmat


#Matplotlib plotting
import matplotlib.pyplot as plt
import matplotlib.tri as tri


data = loadmat('2D_sedDavg.mat')

time = data['time']
dist = data['dist']
quant = data['quant']

time = time - np.min(time)

t = tri.Triangulation(time.flatten(),dist.flatten()) 

plt.figure(1)
plt.triplot(t, 'ko')
plt.plot(time,dist, 'ks')
plt.show()


