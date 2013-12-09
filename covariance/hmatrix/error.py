from aca import *
from tree import *
from hmatrix import *
import numpy as np
N = 2000

pts = np.random.rand(N,3)
indx = np.arange(N)

yd = np.zeros((np.size(indx),),dtype = 'd')
yh = np.zeros((np.size(indx),),dtype = 'd')

def kernel(R):	return (1.+np.sqrt(3)*R)*np.exp(-np.sqrt(3)*R)


mat = GenerateDenseMatrix(pts,indx,indx,kernel)
x = np.random.rand(np.size(indx),)

yd = np.dot(mat,x)


error = []
for k in np.arange(1,15):
	Q = Hmatrix(pts, kernel, indx = indx, indy = None, rkmax = k, eps = 1.e-15, verbose = False)		
	yh *= 0	
	Q.mult(x,yh)

	error.append(np.linalg.norm(yd-yh)/np.linalg.norm(yd))
import matplotlib
matplotlib.rcParams['xtick.labelsize'] = 15.
matplotlib.rcParams['ytick.labelsize'] = 15.


import matplotlib.pyplot as plt
plt.close('all')
plt.figure()
plt.semilogy(error)
plt.xlabel('Block rank, k', fontsize = 16)
plt.ylabel(r'$|| Qx-Q_{\cal H}x || / || Qx||$', fontsize = 16)
plt.title('Relative error with rank', fontsize = 20)
plt.savefig('blockrankerr.png')



