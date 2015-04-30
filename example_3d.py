from kle import KLE
from covariance import Matern
from dolfin import * 
import numpy as np
def view(kle):

	#Write in pvd format. Can be opened in Paraview or MayaVi

	file = File("figs/kle.pvd","compressed")
	v = Function(kle.V)
	for i in np.arange(kle.l.size):
		v.vector()[:] = np.copy(kle.v[:,i])
		file << (v,float(i)) 

	return

if __name__ == '__main__':
	
	mesh = Mesh("meshes/aneurysm.xml.gz")

	kernel = Matern(p = 2, l = 20.)

	kle = KLE(mesh, kernel, verbose = True)
	kle.compute_eigendecomposition(k = 6)

	view(kle)

