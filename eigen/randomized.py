"""Randomized eigenvalue calculation"""

from scipy.sparse.linalg import LinearOperator, aslinearoperator 
from scipy.sparse import identity
from scipy.linalg import qr, eig, eigh 

import numpy as np

from time import time

__all__ = ['randomhep', 'nystrom', 'randsvd', 'randomghep']

def randomhep(A, k, p = 20, twopass = False):
	"""
	Randomized algorithm for Hermitian eigenvalue problems
	Returns k largest eigenvalues computed using the randomized algorithm
	
	
	Parameters:
	-----------

	A : {SparseMatrix,DenseMatrix,LinearOperator} n x n
		Hermitian matrix operator whose eigenvalues need to be estimated
	k :  int, 
		number of eigenvalues/vectors to be estimated
	p :  int, optional
		oversampling parameter which can improve accuracy of resulting solution
		Default: 20
		
	
	twopass : bool, 
		determines if matrix-vector product is to be performed twice
		Default: False
	
	
	Returns:
	--------
	
	w : ndarray, (k,)
		eigenvalues arranged in descending order
	u : ndarray, (n,k)
		eigenvectors arranged according to eigenvalues
	

	References:
	-----------

	.. [1] Halko, Nathan, Per-Gunnar Martinsson, and Joel A. Tropp. "Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions." SIAM review 53.2 (2011): 217-288.


	Examples:
	---------

	>>> import numpy as np
	>>> A = np.diag(0.95**np.arange(100))
	>>> w, v = RandomizedHEP(A, 10, twopass = True)

	"""


	#Get matrix sizes
	m, n = A.shape
	
	Aop = aslinearoperator(A)


	#For square matrices only
	assert m == n

	#Oversample
	k = k + p 


	#Generate gaussian random matrix 
	Omega = np.random.randn(n,k)
	
	Y = np.zeros((m,k), dtype = 'd')
	for i in np.arange(k):
		Y[:,i] = Aop.matvec(Omega[:,i])
	
	q,_ = qr(Y, mode = 'economic')

	if twopass == True:
		B = np.zeros((k,k),dtype = 'd')
		for i in np.arange(k):
			Aq = Aop.matvec(q[:,i])	
			for j in np.arange(k):
				B[i,j] = np.dot(q[:,j].T,Aq)
			
	else:
		from scipy.linalg import inv, pinv,svd, pinv2
		temp  = np.dot(Omega.T, Y)
		temp2 = np.dot(q.T,Omega)
		temp3 = np.dot(q.T,Y)
		
		B = np.dot(pinv2(temp2.T), np.dot(temp, pinv2(temp2)))
		Binv = np.dot(pinv(temp3.T),np.dot(temp, pinv2(temp3)))	

		B = inv(Binv)
		
	#Eigen subproblem
	w, v = eigh(B)

	#Reverse eigenvalues in descending order
	w = w[::-1]

	#Compute eigenvectors		
	u = np.dot(q, v[:,::-1])	

	k -= p
	return w[:k], u[:,:k]

def nystrom(A, k, p = 20, twopass = False):
	"""Randomized algorithm for Hermitian eigenvalue problems
	
	Parameters:
	
	A 	= LinearOperator n x n
			hermitian matrix operator whose eigenvalues need to be estimated
	k	= int, 
			number of eigenvalues/vectors to be estimated
	twopass = bool, 
			determines if matrix-vector product is to be performed twice
	
	Returns:
	
	w	= double, k
			eigenvalues
	u 	= n x k 
			eigenvectors
	
	"""


	#Get matrix sizes
	m, n = A.shape
	
	#For square matrices only
	assert m == n

	#Oversample
	k = k + p 

	#Generate gaussian random matrix 
	Omega = np.random.randn(n,k)
	
	Y = np.zeros((m,k), dtype = 'd')
	for i in np.arange(k):
		Y[:,i] = A.matvec(Omega[:,i])
	
	q,_ = qr(Y, mode = 'economic')

	Aq = np.zeros((n,k), dtype = 'd')
	for i in np.arange(k):
		Aq[:,i] = A.matvec(q[:,i])	
			
	T = np.dot(q.T, Aq)

	from scipy.linalg import cholesky, svd, inv

	R = cholesky(inv(T), lower = True)
	B = np.dot(Aq, R)
	u, s, _ = svd(B) 

	k -=  p
	return s[:k]**2., u[:,:k]

def RandomizedSVD(A, k, p = 20):
	"""Randomized algorithm for Hermitian eigenvalue problems
	
	Parameters:
	
	A 	= LinearOperator n x n
			operator whose singular values need to be estimated
	k	= int, 
			number of eigenvalues/vectors to be estimated
	
	Returns:
	
	
	"""


	#Get matrix sizes
	m, n = A.shape
	
	#For square matrices only
	assert m == n

	#Oversample
	k = k + p 

	#Generate gaussian random matrix 
	Omega = np.random.randn(n,k)
	
	Y = np.zeros((m,k), dtype = 'd')
	for i in np.arange(k):
		Y[:,i] = A.matvec(Omega[:,i])
	
	q,_ = qr(Y, mode = 'economic')


	Atq = np.zeros((n,k), dtype = 'd')
	for i in np.arange(k):
		Atq[:,i] = A.rmatvec(q[:,i])

	from scipy.linalg import svd
	u, s, vt = svd(Atq.T, full_matrices = False)
	
	diff = Y - np.dot(q, np.dot(q.T,Y))	
	err = np.max(np.apply_along_axis(np.linalg.norm, 0, diff))

	from math import pi
	print "A posterior error is ", 10.*np.sqrt(2./pi)*err

	k = k - p
	return np.dot(q, u[:,:k]), s[:k], vt[:k,:].T

def randomghep(A, B, k, p = 20, BinvA = None, twopass = True, \
		verbose = False, error = False):
	"""
		Randomized algorithm for Generalized Hermitian Eigenvalue problem
		A approx (BU) * Lambda *(BU)^*

		Computes k largest eigenvalues and eigenvectors
		
		Modified from randomized algorithm for EV/SVD of A

	"""

	m, n = A.shape
	assert m == n	
	
	#Oversample
	k = k + p

	#Initialize quantities
	Omega 	= np.random.randn(n,k)
	Yh   	= np.zeros_like(Omega, dtype = 'd')
	Y   	= np.zeros_like(Omega, dtype = 'd')

	start = time()
	#Form matrix vector products with C = B^{-1}A
	if BinvA is None:
		for i in np.arange(k):
			Yh[:,i] = A.matvec(Omega[:,i])
			Y[:,i]  = B.solve(Yh[:,i])	
	else:
		for i in np.arange(k):
			Y[:,i]  = BinvA.matvec(Omega[:,i])


	matvectime = time()-start	
	if verbose:	
		print "Matvec time in eigenvalue calculation is %g " %(matvectime) 

	#Compute Y = Q*R such that Q'*B*Q = I, R can be discarded
	start = time()
	q, Bq, _  = Aorthonormalize(B, Y, verbose = False)

	Borthtime = time()-start
	if verbose:	
		print "B-orthonormalization time in eigenvalue calculation is %g " \
			%(Borthtime) 
	T = np.zeros((k,k), dtype = 'd')	

	
	start = time()
	if twopass == True:
		for i in np.arange(k):
			Aq = A.matvec(q[:,i])
			for j in np.arange(k):
				T[i,j] = np.dot(Aq,q[:,j])
	else:

		for i in np.arange(k):
			Yh[:,i] = B.matvec(Y[:,i])
		
		from scipy.linalg import inv
		OAO = np.dot(Omega.T, Yh)
		QtBO = np.dot(Bq.T, Omega)
		T = np.dot(inv(QtBO.T), np.dot(OAO, inv(QtBO)))

	eigcalctime = time()-start
	if verbose:	
		print "Calculating eigenvalues took %g" %(eigcalctime)
		print "Total time taken for Eigenvalue calculations is %g" %\
			 (matvectime + Borthtime + eigcalctime)

	#Eigen subproblem
	w, v = eigh(T)

	#Reverse eigenvalues in descending order
	w = w[::-1]

	#Compute eigenvectors		
	u = np.dot(q, v[:,::-1])	
	k = k - p

	if error:
		#Compute error estimate
		r = 5 
		O = np.random.randn(n,r)
		err = np.zeros((r,), dtype = 'd')
		AO = np.zeros((n,r), dtype = 'd')
		BinvAO = np.zeros((n,r), dtype = 'd')
		for i in np.arange(r):
			AO[:,i]     = A.matvec(O[:,i])
			BinvAO[:,i] = B.solve(AO[:,i])
			diff =	BinvAO[:,i] - np.dot(q,np.dot(q.T,AO[:,i]))		
			err[i] = np.sqrt(np.dot(diff.T, B.matvec(diff)) )
	
		BinvNorm = np.max(np.apply_along_axis(np.linalg.norm, 0, q))
		alpha = 10.

		from math import pi
		print "Using r = %i and alpha = %g" %(r,alpha)
		print "Error in B-norm is %g" %\
			(alpha*np.sqrt(2./pi)*BinvNorm*np.max(err)/np.max(w))

	
	return w[:k], u[:,:k]

        
if __name__ == '__main__':

	n = 100
	x = np.linspace(0,1,n)
	X, Y = np.meshgrid(x, x)
	Q = np.exp(-np.abs(X-Y))
	
	class LowRank:
		def __init__(self, Q, n):
			self.Q = Q 
			self.shape = (n,n)

		def matvec(self, x):
			return np.dot(Q,x)
	

	mat = LowRank(Q,n)
	matop = aslinearoperator(mat)
	
	l,v = RandomizedHEP(matop, k = 10, twopass = False)	

	le, ve = eig(Q)
	le = np.real(le)	

	#Test A-orthonormalize
	z = np.random.randn(n,10)
	q,_,r = Aorthonormalize(matop,z,verbose = False)



	class Identity:
		def __init__(self, n):
			self.shape = (n,n)

		def matvec(self, x):
			return x
		def solve(self, x):
			return x

	id_ = Identity(n)
	
	l_, v_ = randomghep(matop, id_, 10)

    

