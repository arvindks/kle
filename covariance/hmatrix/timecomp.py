from hmatrix import Hmatrix
from time import time

import numpy as np
sizes = [ 500, 1000, 5000, 10000, 50000, 100000] #, 500000]



import sys
n =  sys.argv[1]


if n == '12':
	def kernel(R): return np.exp(-R)
elif n == '32':
	def kernel(R): return (1+np.sqrt(3)*R)*np.exp(-np.sqrt(3)*R)
elif n == '52':
	def kernel(R): return (1+np.sqrt(5)*R + 5.*R/3.)*np.exp(-np.sqrt(5)*R)




multtime = []
setuptime = []
for N in sizes:

	indx = np.arange(N)

	pts = np.random.rand(N,2)
	start = time()
	Q = Hmatrix(pts, kernel, indx, eps = 1.e-9, rkmax = 32)
	_stime = time()-start
	setuptime.append(_stime)
	print "Hmatrix setup of size %g took %g secs" %(N,_stime)


	x = np.random.randn(N,1)
	y = 0*x

	_time = 0.
	for i in np.arange(3):
		start = time()
		Q.mult(x,y)
		_time += time()-start
	multtime.append(_time/3.)

	print "Hmatrix of size %g took %g secs" %(N,_time)


from matplotlib import pyplot as plt
import matplotlib

plt.close('all')

matplotlib.rcParams['xtick.labelsize'] = 15.
matplotlib.rcParams['xtick.labelsize'] = 15.

from scipy.io import savemat
savemat('time' + n + '.mat',{'N':sizes, 'setuptime':setuptime, 'multtime':multtime})

