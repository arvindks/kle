

import numpy as np
from exceptions import NotImplementedError, KeyError

class _Residual:
        def __init__(self):
                self.res = []

        def __call__(self, rk):
                self.res.append(rk)

        def itercount(self):
                return len(self.res)

        def clear(self):
                self.res = []



class CovarianceMatrix:
	"""
	Implementation of covariance matrix corresponding to covariance kernel

	Parameters:
	-----------
	method:		string, {'Dense','FFT','Hmatrix}
			decides what method to invoke

	pts:		n x dim 
			Location of the points

	kernel:		Kernel object
			See covariance/kernel.py 

	nugget:		double, optional. default = 0.
			A hack to improve the convergence of the iterative solver when inverting for the covariance matrix. See solve() for more details.	

	Attributes:
	-----------
	shape:	(N,N)
		N is the size of matrix
	P:	NxN csr_matrix
		corresponding to the preconditioner
	

	Methods:
	--------
	matvec()
	rmatvec()
	reset()
	itercount()
	build_preconditioner()
	solve()
		
	
	Notes:
	------

	Implementation of three different methods 

	1. Dense Matrix 
	2. FFT based operations if kernel is stationary or translation invariant and points are on a regular grid
	3. Hierarchical Matrix - works for arbitrary kernels on irregular grids

	Details of this implementation (including errors and benchmarking) are provided in chapter 2 in [1]. For details on the algorithms see references within. 

	Compatible with scipy.sparse.LinearOperator. For example, Q = CovarianceMatrix(...);	Qop = aslineroperator(Q)
	References:
	----------
	.. [1] A.K. Saibaba, Fast solvers for geostatistical inverse problems and uncertainty quantification, PhD Thesis 2013, Stanford University 

	
	Examples:
	---------
	import numpy as np
	pts = np.random.randn(1000,2)
	def kernel(r): return np.exp(-r)
	
	Qd = CovarianceMatrix('Dense', pts, kernel)	
	Qh = CovarianceMatrix('Hmatrix', pts, kernel, rkmax = 32, eps = 1.e-6)	
		
	"""
	def __init__(self, method, pts, kernel, nugget = 0.0, **kwargs):
		self.method = method
		self.kernel = kernel
		self.pts    = pts
	
			
		try:
			verbose = kwargs['verbose']
		except KeyError:
			verbose = False
		self.verbose = verbose

		if method == 'Dense':
		        from dense import GenerateDenseMatrix
			self.mat = GenerateDenseMatrix(pts, kernel)
			
			self.pts = pts
	
		
		elif method == 'FFT':
		        from toeplitz import CreateRow, ToeplitzProduct
			xmin 	= kwargs['xmin']
			xmax 	= kwargs['xmax']	
			N    	= kwargs['N']
			theta	= kwargs['theta']		
			
			self.N      = N
			self.row, pts  = CreateRow(xmin,xmax,N,kernel,theta)		
		
			self.pts    = pts
	
		elif method == 'Hmatrix':
		        
                        from hmatrix import Hmatrix	
			n   	= np.size(pts,0)
			ind 	= np.arange(n)

			rkmax = 32 if 'rkmax' not in kwargs else kwargs['rkmax']
			eps   = 1.e-9 if 'eps' not in kwargs else kwargs['eps']

			self.H 	= Hmatrix(pts, kernel, ind, verbose = verbose, rkmax = rkmax, eps = eps)	
			
		else:
			raise NotImplementedError

		self.P		= None
		n = pts.shape[0]
		self.shape 	= (n,n)
		self.nugget	= nugget
		self.dtype      = 'd'
		self.count = 0

		self.solvmatvecs = 0

	def matvec(self, x):
		"""
		Computes the matrix-vector product
		
		Parameters:
		-----------

		x: (n,)	ndarray
			a vector of size n

		Returns:
		-------
		
		y: (n,) ndarray
			

		Notes:
		------
		
		The result of this calculation are dependent on the method chosen. All methods except 'Hmatrix' are exact.			


		"""

		method = self.method
		if method == 'Dense':
			y = np.dot(self.mat,x)

		elif method == 'FFT':
			y = ToeplitzProduct(x, self.row, self.N)

		elif method == 'Hmatrix':
			y = np.zeros_like(x, dtype = 'd')
			self.H.mult(x, y, self.verbose)
		
		y += self.nugget*y	


		self.count += 1
		
		return y

	def rmatvec(self, x):
		"""
		Computes the matrix transpose-vector product
		
		Parameters:
		-----------

		x: (n,)	ndarray
			a vector of size n

		Returns:
		-------
		
		y: (n,) ndarray
			

		Notes:
		------
		
		the result of this calculation are dependent on the method chosen. All methods except 'Hmatrix' are exact. 
		Because of symmetry it is almost the same as matvec, except 'Hmatrix' which is numerically different			


		"""

		method = self.method
		if method == 'Dense':
			y = np.dot(self.mat.T,x)

		elif method == 'FFT':
			y = ToeplitzProduct(x, self.row, self.N)

		elif method == 'Hmatrix':
			y = np.zeros_like(x, dtype = 'd')
			self.H.transpmult(x, y, self.verbose)
		
		y += self.nugget*y	
		self.count += 1
		
		return y

	def reset(self):
		"""	
		Resets the counter of matvecs and solves
		"""
		
		self.count = 0
		self.solvmatvecs = 0
		return

	def itercount(self):
		"""
		Returns the counter of matvecs
		"""
		return self.count

	def build_preconditioner(self, k = 100):
		"""
		Implementation of the preconditioner based on changing basis.

		Parameters:
		-----------
		k:	int, optional. default = 100
			Number of local centers in the preconditioner. Controls the sparity of the preconditioner. 
			
	
		Notes:
		------
		Implementation of the preconditioner based on local centers. 
		The parameter k controls the sparsity and the effectiveness of the preconditioner. 
		Larger k is more expensive but results in fewer iterations. 
		For large ill-conditioned systems, it was best to use a nugget effect to make the problem better conditioned. 

		To Do: implementation based on local centers and additional points. Will remove the hack of using nugget effect.		


		References:
		-----------
		
		"""
	
		from time import time
		
		from scipy.spatial import cKDTree
		from scipy.spatial.distance import pdist, cdist
		
		from scipy.linalg import solve
		from scipy.sparse import csr_matrix	

		#If preconditioner already exists, then nothing to do	
		if self.P != None:	return
	
	
		pts = self.pts
		kernel = self.kernel
	
		N = pts.shape[0]

	        #Build the tree
       	 	start = time()
        	tree = cKDTree(pts, leafsize = 32)
        	end = time()

		if self.verbose:
        		print "Tree building time = %g" % (end-start)

        	#Find the nearest neighbors of all the points
        	start = time()
        	dist, ind = tree.query(pts,k = k)
        	end = time()
		if self.verbose:
	        	print "Nearest neighbor computation time = %g" % (end-start)

        	Q = np.zeros((k,k),dtype='d')
        	y = np.zeros((k,1),dtype='d')

        	row = np.tile(np.arange(N), (k,1)).transpose()
       		col = np.copy(ind)
        	nu = np.zeros((N,k),dtype='d')

        	y[0] = 1.
        	start = time()
        	for i in np.arange(N):
                	Q = kernel(cdist(pts[ind[i,:],:],pts[ind[i,:],:]))
                	nui = np.linalg.solve(Q,y)
                	nu[i,:] = np.copy(nui.transpose())
        	end = time()
		
		if self.verbose:	print "Building preconditioner took  = %g" % (end-start)
	
		ij = np.zeros((N*k,2), dtype = 'i')
        	ij[:,0] = np.copy(np.reshape(row,N*k,order='F').transpose() )
        	ij[:,1] = np.copy(np.reshape(col,N*k,order='F').transpose() )

        	data = np.copy(np.reshape(nu,N*k,order='F').transpose())
        	self.P = csr_matrix((data,ij.transpose()),shape=(N,N), dtype = 'd')

		return

	
	def solve(self, b, maxiter = 1000, tol = 1.e-10):
		"""
		Compute Q^{-1}b
		
		Parameters:
		-----------
		b:  	(n,) ndarray
		 	given right hand side
		maxiter: int, optional. default = 1000
			Maximum number of iterations for the linear solver

		tol:	float, optional. default = 1.e-10
			Residual stoppingtolerance for the iterative solver
				
		Notes:
		------
		If 'Dense' then inverts using LU factorization otherwise uses iterative solver 
			- MINRES if used without preconditioner or GMRES with preconditioner. 
			Preconditioner is not guaranteed to be positive definite.

		"""
		
		
				
		if self.method == 'Dense':
			from scipy.linalg import solve
			x = solve(self.mat, b)

		else:
			from scipy.sparse.linalg import gmres, aslinearoperator, minres

			P = self.P
			Aop = aslinearoperator(self)
		
			residual = _Residual()
			if P != None:
				x, info = gmres(Aop, b, tol = tol, restart = 30, maxiter = 1000, callback = residual, M = P)
			else:
				x, info = minres(Aop, b, tol = tol, maxiter = maxiter, callback = residual )
			self.solvmatvecs += residual.itercount()
			if self.verbose:	
				print "Number of iterations is %g and status is %g"% (residual.itercount(), info)
		return x

	
if __name__ == '__main__':
	n = 5000
	pts = np.random.rand(n, 2)

	def kernel(R):
		return np.exp(-R)

	Q = CovarianceMatrix('Hmatrix',pts,kernel,verbose = True, nugget = 1.e-4)	


	x = np.ones((n,), dtype = 'd')
	y = Q.matvec(x)


	Q.BuildPreconditioner(k = 30, view = False)
	Q.verbose = False
	xd = Q.solve(y)

	print np.linalg.norm(x-xd)/np.linalg.norm(x)
	
