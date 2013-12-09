import numpy as np
from matplotlib import pyplot as plt
from dolfin import *

def Nystrom(kernel, n = 20, a = 1.):
    
    l = legendre(n)
    x = a*l.weights[:,0].flatten()
    w = l.weights[:,1]
    
    X, Y = np.meshgrid(x,x)
    Q = kernel(np.abs(X-Y))

    Q = np.dot(np.diag(np.sqrt(w)),np.dot(Q, np.diag(np.sqrt(w))))
    
    l, v = np.linalg.eigh(Q)
    
    l = l[::-1]
    v = v[:,::-1]
    
    return l,v
    
    

def KLE1D(a = 1, c = 1, k = 10):

	from kle import KLE 
	mesh = UnitInterval(1000)
	mesh.coordinates()[:] = 2.*a*mesh.coordinates()[:] - a

        def kernel(R):  return np.exp(-c*R)


        kle = KLE(mesh,kernel)

        from time import time

       	start = time() 
	la, va = kle.compute_eigendecomposition(k = k)
        la = la[::-1]
	va = va[:,::-1]
        print "Time for arpack %g " %(time()-start)



	return la, va


if __name__ == '__main__':


	x = 0.
	la = analytical1d(x,k = 10,c = 1.,a = 1.)
	ln, _ = KLE1D(k = 10, c = 1., a = 1.)	


	print la
	print ln
