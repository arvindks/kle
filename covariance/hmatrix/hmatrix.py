from tree import *
from aca  import *
import numpy as np

__all__ = ["Hmatrix"]


class Hmatrix:        
	"""
	Implementation of the Hierarchical matrix class. 

        Parameters:
        -----------
        pts:    (n,dim) ndarray
                all the points
        
	kernel: Kernel object
                see covariance/kernel.py for more details

        indx:   (nx,) ndarray
                indices of the points

        indy:   (ny,) ndarray, optional.
                indices of the points
                
                
        rkmax:  int. default = 32.
                Maximum rank of the low rank approximation

        eps:    double. default = 1.e-9.
                Approximate relative error of the low rank approximation
                Computing frobenius norm is O(n^2) so it is avoided.


	Methods:
	--------
	mult()
	transpmult()
	
	Attributes:
	-----------
	ctreex, ctreey:	Cluster Tree objects
	btree:		Block Cluster Tree
	

	Notes:
	------
	Details of the implementation including benchmarking is available in [2]. 

        References:
        -----------
        .. [1] Sergej Rjasanow, Olaf Steinbach, The fast solution of boundary integral equations. Mathematical and analytical techniques with applications to engin
eering. Springer 2007, New York.    
	.. [2]  A.K. Saibaba, Fast solvers for geostatistical inverse problems and uncertainty quantification, PhD Thesis 2013, Stanford University. 


        """
	def __init__(self, pts, kernel, indx, indy = None, rkmax = 32, eps = 1.e-9, verbose = False):	
		self.pts 	= pts
		self.indx 	= indx
		self.indy	= indy
		self.kernel	= kernel
		self.verbose 	= verbose

		from time import time	
	
		#Construct cluster tree	
		dim = np.size(pts,1)
		self.ctreex = Cluster(dim = dim, level = 0) 
		start = time()
		self.ctreex.assign_points_bisection(pts, self.indx)
		if verbose:
			print "Time to construct cluster tree is %g " % (time() - start)
		
		if indy == None:	
			self.ctreey = self.ctreex
		else:
			self.ctreey = Cluster(dim = dim, level = 0) 
	                start = time()
        	        self.ctreey.assign_points_bisection(pts, self.indy)
                	if verbose:
                        	print "Time to construct cluster tree is %g " % (time() - start)


		#Construct Block Cluster tree
		self.btree = BlockCluster(level = 0)
		self.btree.construct_block_tree(self.ctreex, self.ctreey) 
		if verbose:
			print "Time to construct block cluster tree is %g " % (time() - start)
		self.btree.construct_low_rank_representation(self.pts, self.kernel, rkmax, eps) 
		if verbose:
			print "Time to construct low rank representation is %g " % (time() - start)

		return
	def mult(self, x, y, verbose = False):
		"""
		Matrix-vector product with the H-matrix
		
		Parameters:
		-----------
		x:	(n,) ndarray
		y:	(n,) ndarray
		verbose: bool, False
		
		"""
		from time import time 
		start = time()
		self.btree.mult(x, y, self.pts, self.kernel)
		if verbose:
			print "Time for mat-vec is %g" %(time() - start)

		return	
	def transpmult(self, x, y, verbose = False):
		"""
		Transpose matrix-vector product with the H-matrix
		
		Parameters:
		-----------
		x:	(n,) ndarray
		y:	(n,) ndarray
		verbose: bool, False
		
		"""

		
		from time import time 
		start = time()
		self.btree.transpmult(x, y, self.pts, self.kernel)	
		if verbose:
			print "Time for mat-vec is %g" %(time() - start)

		return

	def _memoryusage(self):
		leaflist = []
		costlist = []
		self.btree.construct_space_filling_curve(leaflist,costlist)
		
		memory = float(np.sum(costlist)*8.)/(1024.**2.)
		return memory	


if __name__ == '__main__':

	import sys

	if len(sys.argv) < 2:
		N = 100
	else:
		N = int(sys.argv[1])

	pts = np.random.rand(N,2)
	indx = np.arange(N)

	def kernel(R):
		return np.exp(-R)

	indy = None #np.arange(N/10)
	Q = Hmatrix(pts, kernel, indx = indx, indy = indy, verbose = True)		
	
	direct = True
	if N > 5000:
		direct = False
	

	x = np.random.rand(np.size(indx),)
	yd = np.zeros((np.size(indx),),dtype = 'd')
	yh = np.zeros((np.size(indx),),dtype = 'd')


	print "Memory usage in MB %g" %(Q._memoryusage())
	Q.mult(x,yh)

	if direct == True:
		mat = GenerateDenseMatrix(pts,indx,indx,kernel)
		yd = np.dot(mat,x)
		print "Error is %g" %(np.linalg.norm(yd-yh)/np.linalg.norm(yd))
