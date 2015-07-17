import numpy as np

from kle import KLE
from covariance import CovarianceMatrix, Matern



def ploteigenvectors(kle):

	mesh 	= kle.mesh
	l    	= kle.l
	v 	= kle.v

	
	#plot eigenvectors
	eigvec = True
	if eigvec:

		eign = [0,2,4,6,9,11,17,19]
		from matplotlib.tri import Triangulation

		vmin = np.min(v[:,eign])
		vmax = np.max(v[:,eign])

		pts = kle.mesh.coordinates()
		tri = Triangulation(pts[:,0], pts[:,1], triangles = mesh.cells())
		
		fig, axes = plt.subplots(nrows = 2, ncols = 4)
		for ax, i in zip(axes.flat,eign):
			im = ax.tripcolor(tri, v[:,i].flatten(), \
				vmin = vmin, vmax = vmax, shading = 'gouraud', \
				cmap = plt.cm.rainbow)
			ax.set_xticks([])
			ax.set_yticks([])

		fig.suptitle('Eigenvectors of KLE', fontsize = 20)
		fig.subplots_adjust(right = 0.8)
		cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
		fig.colorbar(im, cax = cbar_ax)
		

	plt.savefig('figs/square.png')	

	plt.show()
	return

if __name__ == '__main__':

	from dolfin import *
	mesh = UnitSquare(20,20)	#really a 101 x 101 grid
	mesh.coordinates()[:] = 2.*mesh.coordinates()-1.
	
	kernel = Matern(p = 0,	l = 1.)	#Exponential covariance kernel

	kle  = KLE(mesh, kernel, verbose = True)
	kle.compute_eigendecomposition(k = 20)

	print kle.l

	import matplotlib.pyplot as plt
	plt.close('all')
	

	#ploteigenvectors(kle)
	#Generate realizations
	real = kle.realizations()

	

