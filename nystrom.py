import numpy as np
import matplotlib.pyplot as plt
from scipy.special import legendre

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
    
if __name__ == '__main__':
    a = 1.
    def kernel(r):    return np.exp(-a*r)
    
    l,v = Nystrom(kernel)
    
    
    