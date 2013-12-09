"""Randomized eigenvalue calculation"""

from scipy.sparse.linalg import LinearOperator, aslinearoperator 
from scipy.sparse import identity
from scipy.linalg import qr, eig, eigh 

import numpy as np

from time import time

__all__ = ['RandomizedHEP', 'Nystrom', 'RandomizedSVD', 'RandomizedGHEP', 'RandomizedGHEP2']

def RandomizedHEP(A, k, p = 20, twopass = False):
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

	if twopass == True:
		B = np.zeros((k,k),dtype = 'd')
		for i in np.arange(k):
			Aq = A.matvec(q[:,i])	
			for j in np.arange(k):
				B[i,j] = np.dot(q[:,j].transpose(),Aq)
			
	else:
		from scipy.linalg import inv, pinv,svd, pinv2
		#B = np.dot( np.dot(q.T, Y), inv(np.dot(q.T, Omega)) )
		

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

	k = k - p
	return w[:k], u[:,:k]

def Nystrom(A, k, p = 20, twopass = False):
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
		



	k = k - p
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

def Aorthonormalize(A, Z, verbose = False):
	""" Produce q'*A*q = I with inner product <x,y>_A = y'* A * x 
	    using Modified Gram-Schmidt
	"""
	
	#Get sizes
	n = np.size(Z,0);	k = np.size(Z,1)
	
	#Initialize
	Aq = np.zeros_like(Z, dtype  = 'd')
	q  = np.zeros_like(Z, dtype = 'd')
	r  = np.zeros((k,k), dtype = 'd')
		

	z  = Z[:,0]
	Aq[:,0] = A.matvec(z)	


	r[0,0] = np.sqrt(np.dot(z.T, Aq[:,0]))
	q[:,0] = Z[:,0]/r[0,0]

	Aq[:,0] /= r[0,0]

	for j in np.arange(1,k):
		q[:,j] = np.copy(Z[:,j])
		
		for i in np.arange(j):
			r[i,j] = np.dot(q[:,j].T,Aq[:,i])
			q[:,j] -= r[i,j]*q[:,i]


		Aq[:,j] = A.matvec(q[:,j])
		r[j,j]  = np.sqrt(np.dot(q[:,j].T,Aq[:,j]))

		#If element becomes too small, terminate
		if np.abs(r[j,j]) < 1.e-12:
			kt = j-1;	
			print "A-orthonormalization broke down"
			break
		
		q[:,j]  /= r[j,j]	
		Aq[:,j] /= r[j,j]	

	if verbose:
		#Verify Q'*A*Q = I
		T = np.zeros((k,k), dtype = 'd')			
		T = np.dot(q.T, Aq)
		
		#print T	
		print np.linalg.norm(T - np.eye(k, dtype = 'd'), ord = 2)		

		#Verify Q*R = Y
		print np.linalg.norm(np.dot(q,r) - Z, 2)


	return q, Aq, r 


def RandomizedGHEP(A, B, k, p = 20, BinvA = None, twopass = True, verbose = False, error = True):
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
			Yh[:,i] = B.matvec(Yh[:,i])


	matvectime = time()-start	
	if verbose:	print "Matvec time in eigenvalue calculation is %g " %(matvectime) 

	#Compute Y = Q*R such that Q'*B*Q = I, R can be discarded
	start = time()
	q, Bq, _  = Aorthonormalize(B, Y, verbose = False)

	Borthtime = time()-start
	if verbose:	print "B-orthonormalization time in eigenvalue calculation is %g " %(Borthtime) 
	T = np.zeros((k,k), dtype = 'd')	

	
	start = time()
	
	if twopass == True:
		for i in np.arange(k):
			Aq = A.matvec(q[:,i])
			for j in np.arange(k):
				T[i,j] = np.dot(Aq,q[:,j])
	else:
		from scipy.linalg import inv
		OAO = np.dot(Omega.T, Yh)
		QtBO = np.dot(Bq.T, Omega)
		T = np.dot(inv(QtBO.T), np.dot(OAO, inv(QtBO)))

	eigcalctime = time()-start
	if verbose:	print "Calculating eigenvalues took %g" %(eigcalctime)
	if verbose:	print "Total time taken for Eigenvalue calculations is %g" % (matvectime + Borthtime + eigcalctime)


	#Eigen subproblem
	w, v = eigh(T)

	#Reverse eigenvalues in descending order
	w = w[::-1]

	#Compute eigenvectors		
	u = np.dot(q, v[:,::-1])	
	k = k - p

	if error:
		#Compute error estimate
		r = 15 
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
		print "Error in B-norm is %g" %(alpha*np.sqrt(2./pi)*BinvNorm*np.max(err)/np.max(w))

	
	return w[:k], u[:,:k]


def RandomizedGHEP2(A, B, k, p = 20, twopass = True, error = True):
	"""
		Randomized algorithm for Generalized Hermitian Eigenvalue problem
		A * U = B * U * Lambda

		Computes k largest eigenvalues and eigenvectors
		
		Modified from randomized algorithm for EV/SVD of A

	"""

	m,n = A.shape
	assert m == n	
	
	#Oversample
	k = k + p

	#Initialize quantities
	Omega 	= np.random.randn(n,k)
	Y   	= np.zeros_like(Omega, dtype = 'd')
	
	from time import time

	start = time()
	#Form matrix vector products with C = B^{-1}A
	for i in np.arange(k):
		Y[:,i] = A.matvec(Omega[:,i])

	matvectime = time()-start	
	print "Matvec time in eigenvalue calculation is %g " %(matvectime) 

	
	matvec = lambda x: B.solve(x)
	Binv = LinearOperator((n,n), matvec = matvec, dtype = 'd')		

	#Compute Y = Q*R such that Q'*B*Q = I, R can be discarded
	start = time()
	Bq, q, r  = Aorthonormalize(Binv, Y, verbose = False)
	Borthtime = time()-start
	print "B-orthonormalization time in eigenvalue calculation is %g " %(Borthtime) 

	start = time()

	T = np.zeros((k,k), dtype = 'd')	

	if twopass == True:
		for i in np.arange(k):
			Aq = A.matvec(q[:,i])
			for j in np.arange(k):
				T[i,j] = np.dot(Aq,q[:,j])

		#Eigen subproblem
		w, v = eigh(T)
		#Reverse eigenvalues in descending order
		w = w[::-1]
		
		
	else:
		from scipy.linalg import inv, svd, pinv, pinv2
		#T = np.dot(np.dot(q.T, Y),inv(np.dot(Bq.T, Omega)))
	
		OAO = np.dot(Omega.T, Y)
		QtBO = np.dot(Bq.T, Omega)
		T = np.dot(inv(QtBO.T), np.dot(OAO, inv(QtBO)))

		w, v = eigh(T); w = w[::-1]

	eigcalctime = time()-start
	print "Calculating eigenvalues took %g" %(eigcalctime)

	print "Total time taken for Eigenvalue calculations is %g" % (matvectime + Borthtime + eigcalctime)
	
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

		from math import pi
		print "Error in B-norm is %g" %(10.*np.sqrt(2./pi)*BinvNorm*np.max(err))


	
		
	#Compute eigenvectors		
	u = np.dot(q, v[:,::-1])	

	k = k - p
	w = w[:k]
	u = u[:,:k]
	
	return w, u


def LowRankConversion(W,B):
	"""
	Convert low rank matrix WW^T = UDU^T, where U^TBU = I
	Returns U,D
	Untested
	"""

	w,_, r = Aorthonormalize(B,W, verbose = False)
	mat = np.dot(r,r.T)

	from scipy.linalg import svd
	u,s,_ = svd(mat, compute_uv = True)

	return np.dot(w,u), s
	
def AddSymmetricLowRankMatrices(U, d1, V, d2, B, tol = 1.e-10):
       """
       A = UD_1U^T + VD_2V^T = WDW^T
       Truncates the singular values smaller than tol
       Returns W,D
       """	  	
       U = np.dot(U,np.diag(np.sqrt(d1)))
       V = np.dot(V,np.diag(np.sqrt(d2)))
       
       W = np.hstack((U,V))
       
       w, d = LowRankConversion(W,B)		
       ind = np.flatnonzero(d/d[0] > tol)						
        	   
       return w[:,ind], d[ind]
        
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
	
	l_, v_ = RandomizedGHEP(matop, id_, 10)

	#print l_, le[:10]


	V = np.random.randn(n,10)
	I = id_
	u,d = LowRankConversion(V,mat)
	d = np.diag(d)   
	print np.linalg.norm(np.dot(V,V.T)-np.dot(u,np.dot(d,u.T)),2)

        U = np.random.randn(n,10);   V = np.random.randn(n,10);
	u1,d1 = LowRankConversion(U,mat);   u2, d2 = LowRankConversion(V,mat);
	
	w, d = AddSymmetricLowRankMatrices(u1,d1,u2,d2,mat)
	print np.linalg.norm(np.dot(w,np.dot(np.diag(d),w.T)) -np.dot(U,U.T) - np.dot(V,V.T),2)
    
    

