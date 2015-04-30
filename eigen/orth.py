import numpy as np
from scipy.linalg import qr
from scipy.sparse.linalg import LinearOperator, aslinearoperator

__all__ = ['mgs','mgs_stable','cholqr','precholqr']

def mgs(A, Z, verbose = False):
	""" 
	Returns QR decomposition of Z. Q and R satisfy the following relations 
	in exact arithmetic

	1. QR    	= Z
	2. Q^*AQ 	= I
	3. Q^*AZ	= R 
	4. ZR^{-1}	= Q
	
	Uses Modified Gram-Schmidt for computing the A-orthogonal QR factorization

	Parameters
	----------
	A : {sparse matrix, dense matrix, LinearOperator}
		An array, sparse matrix, or LinearOperator representing
		the operation ``A * x``, where A is a real or complex square matrix.

	Z : ndarray
	
	verbose : bool, optional
		  Displays information about the accuracy of the resulting QR


	Returns
	-------	

	q : ndarray
		The A-orthogonal vectors 

	Aq : ndarray
		The A^{-1}-orthogonal vectors

	r : ndarray
		The r of the QR decomposition


	See Also
	--------
	mgs_stable : Modified Gram-Schmidt with re-orthogonalization
	precholqr  : Based on CholQR 	

	References
	----------
	.. [1]  B. Lowery and J. Langou, 
		Stability Analysis of QR factorization in an Oblique 
		Inner Product http://arxiv.org/abs/1401.5171

	Examples
	--------
	
	>>> import numpy as np
	>>> A = np.diag(np.arange(1,101)) 	
	>>> Z = np.random.randn(100,10)
	>>> q, Aq, r = mgs(A, Z, verbose = True)
		

	"""
	
	#Get sizes
	n = np.size(Z,0);	k = np.size(Z,1)
	
	#Convert into linear operator
	Aop = aslinearoperator(A)


	#Initialize
	Aq = np.zeros_like(Z, dtype  = 'd')
	q  = np.zeros_like(Z, dtype = 'd')
	r  = np.zeros((k,k), dtype = 'd')
		

	z  = Z[:,0]
	Aq[:,0] = Aop.matvec(z)	


	r[0,0] = np.sqrt(np.dot(z.T, Aq[:,0]))
	q[:,0] = Z[:,0]/r[0,0]

	Aq[:,0] /= r[0,0]

	for j in np.arange(1,k):
		q[:,j] = Z[:,j]
		for i in np.arange(j):
			r[i,j] = np.dot(q[:,j].T,Aq[:,i])
			q[:,j] -= r[i,j]*q[:,i]


		Aq[:,j] = Aop.matvec(q[:,j])
		r[j,j]  = np.sqrt(np.dot(q[:,j].T,Aq[:,j]))

		#If element becomes too small, terminate
		if np.abs(r[j,j]) < 1.e-14:
			k = j;	
			
			q = q[:,:kt]
			Aq = Aq[:,:kt]
			r = r[:kt,:kt]

			print "A-orthonormalization broke down"
			break
		
		q[:,j]  /= r[j,j]	
		Aq[:,j] /= r[j,j]	

	q = q[:,:k]
	Aq = Aq[:,:k]
	r = r[:k,:k]

	if verbose:
		#Verify Q*R = Y
		print "||QR-Y|| is ", np.linalg.norm(np.dot(q,r) - Z[:,:k], 2)
		
		#Verify Q'*A*Q = I
		T = np.dot(q.T, Aq)
		print "||Q^TAQ-I|| is ", np.linalg.norm(T - np.eye(k, dtype = 'd'), ord = 2)		

		#verify Q'AY = R 
		print "||Q^TAY-R|| is ", np.linalg.norm(np.dot(Aq.T,Z[:,:k]) - r,2)

		#Verify YR^{-1} = Q
		print "||YR^{-1}-Q|| is ", np.linalg.norm(np.linalg.solve(r.T,Z[:,:k].T).T-q,2)



	return q, Aq, r 


def mgs_stable(A, Z, verbose = False):
	""" 
	Returns QR decomposition of Z. Q and R satisfy the following relations 
	in exact arithmetic

	1. QR    	= Z
	2. Q^*AQ 	= I
	3. Q^*AZ	= R 
	4. ZR^{-1}	= Q
	
	Uses Modified Gram-Schmidt with re-orthogonalization (Rutishauser variant)
	for computing the A-orthogonal QR factorization

	Parameters
	----------
	A : {sparse matrix, dense matrix, LinearOperator}
		An array, sparse matrix, or LinearOperator representing
		the operation ``A * x``, where A is a real or complex square matrix.

	Z : ndarray
	
	verbose : bool, optional
		  Displays information about the accuracy of the resulting QR
		  Default: False

	Returns
	-------	

	q : ndarray
		The A-orthogonal vectors 

	Aq : ndarray
		The A^{-1}-orthogonal vectors

	r : ndarray
		The r of the QR decomposition


	See Also
	--------
	mgs : Modified Gram-Schmidt without re-orthogonalization
	precholqr  : Based on CholQR 	


	References
	----------
	.. [1] A.K. Saibaba, J. Lee and P.K. Kitanidis, Randomized algorithms for Generalized
		Hermitian Eigenvalue Problems with application to computing 
		Karhunen-Loe've expansion http://arxiv.org/abs/1307.6885
	
	.. [2] W. Gander, Algorithms for the QR decomposition. Res. Rep, 80(02), 1980
	
	Examples
	--------
	
	>>> import numpy as np
	>>> A = np.diag(np.arange(1,101)) 	
	>>> Z = np.random.randn(100,10)
	>>> q, Aq, r = mgs_stable(A, Z, verbose = True)

	"""



	#Get sizes
	m = np.size(Z,0);	n = np.size(Z,1)
	
	#Convert into linear operator
	Aop = aslinearoperator(A)
	
	#Initialize
	Aq = np.zeros_like(Z, dtype  = 'd')
	q  = np.zeros_like(Z, dtype = 'd')
	r  = np.zeros((n,n), dtype = 'd')
		
	reorth = np.zeros((n,), dtype = 'd')
	eps = np.finfo(np.float64).eps

	q = np.copy(Z)

	for k in np.arange(n):
		Aq[:,k] = Aop.matvec(q[:,k])
		t = np.sqrt(np.dot(q[:,k].T,Aq[:,k]))
	
		nach = 1;	u = 0;
		while nach:
			u += 1
			for i in np.arange(k):
				s = np.dot(Aq[:,i].T,q[:,k])
				r[i,k] += s
				q[:,k] -= s*q[:,i];
			
			Aq[:,k] = Aop.matvec(q[:,k])	
			tt = np.sqrt(np.dot(q[:,k].T,Aq[:,k]))
			if tt > t*10.*eps and tt < t/10.:
				nach = 1;	t = tt;
			else:
				nach = 0;
				if tt < 10.*eps*t:	tt = 0.
			

		reorth[k] = u
		r[k,k] = tt
		tt = 1./tt if np.abs(tt*eps) > 0. else 0.
		q[:,k]  *= tt
		Aq[:,k] *= tt
	

	if verbose:
		#Verify Q*R = Y
		print "||QR-Y|| is ", np.linalg.norm(np.dot(q,r) - Z, 2)
		
		#Verify Q'*A*Q = I
		T = np.dot(q.T, Aq)
		print "||Q^TAQ-I|| is ", np.linalg.norm(T - np.eye(n, dtype = 'd'), ord = 2)		

		#verify Q'AY = R 
		print  "||Q^TAY-R|| is ", np.linalg.norm(np.dot(Aq.T,Z) - r,2)

		#Verify YR^{-1} = Q
		val = np.inf
		try:
			val = np.linalg.norm(np.linalg.solve(r.T,Z.T).T-q,2)
		except LinAlgError:
			print "Singular"
		print "||YR^{-1}-Q|| is ", val

	return q, Aq, r 


def cholqr(A, Z, verbose = False):
	""" 
	Returns QR decomposition of Z. Q and R satisfy the following relations 
	in exact arithmetic

	1. QR    	= Z
	2. Q^*AQ 	= I
	3. Q^*AZ	= R 
	4. ZR^{-1}	= Q
	
	Uses Chol QR algorithm proposed in [1] for computing the A-orthogonal 
	QR factorization. 'precholqr' function has better orthogonality properties

	Parameters
	----------
	A : {sparse matrix, dense matrix, LinearOperator}
		An array, sparse matrix, or LinearOperator representing
		the operation ``A * x``, where A is a real or complex square matrix.

	Z : ndarray
	
	verbose : bool, optional
		  Displays information about the accuracy of the resulting QR
		  Default: False

	Returns
	-------	

	q : ndarray
		The A-orthogonal vectors 

	Aq : ndarray
		The A^{-1}-orthogonal vectors

	r : ndarray
		The r of the QR decomposition


	See Also
	--------
	mgs : Modified Gram-Schmidt without re-orthogonalization
	mgs_stable : Modified Gram-Schmidt with re-orthogonalization


	References
	----------
		
	.. [1]  B. Lowery and J. Langou, 
		Stability Analysis of QR factorization in an Oblique 
		Inner Product http://arxiv.org/abs/1401.5171

	.. [2] A.K. Saibaba, J. Lee and P.K. Kitanidis, Randomized algorithms for Generalized
		Hermitian Eigenvalue Problems with application to computing 
		Karhunen-Loe've expansion http://arxiv.org/abs/1307.6885


	Examples
	--------
	
	>>> import numpy as np
	>>> A = np.diag(np.arange(1,101)) 	
	>>> Z = np.random.randn(100,10)
	>>> q, Aq, r = cholqr(A, Z, verbose = True)

	"""


	#Convert into linear operator
	Aop = aslinearoperator(A)

	B = np.apply_along_axis(lambda x: Aop.matvec(x), 0, Z)
	C = np.dot(Z.T, B)
	
	r = np.linalg.cholesky(C).T
	q = np.linalg.solve(r.T,Z.T).T
	Aq = np.linalg.solve(r.T,B.T).T

	if verbose:
		
		#Verify Q*R = Y
		print "||QR-Y|| is ", np.linalg.norm(np.dot(q,r) - Z, 2)
		
		#Verify Q'*A*Q = I
		T = np.dot(q.T, Aq)
		n = T.shape[1]
		print "||Q^TAQ-I|| is ", \
			np.linalg.norm(T - np.eye(n, dtype = 'd'), ord = 2)		

		#verify Q'AY = R 
		print "||Q^TAY-R|| is ", np.linalg.norm(np.dot(Aq.T,Z) - r,2)

		#Verify YR^{-1} = Q
		val = np.inf
		try:
			val = np.linalg.norm(np.linalg.solve(r.T,Z.T).T-q,2)
		except LinAlgError:
			print "||YR^{-1}-Q|| is ", "Singular"
		print "||YR^{-1}-Q|| is ", val

	return q, Aq, r 		

def precholqr(A, Z, verbose = False):

	""" 
	Returns QR decomposition of Z. Q and R satisfy the following relations 
	in exact arithmetic

	1. QR    	= Z
	2. Q^*AQ 	= I
	3. Q^*AZ	= R 
	4. ZR^{-1}	= Q
	
	Uses Pre Chol QR algorithm proposed in [1] for computing 
	the A-orthogonal QR factorization

	Parameters
	----------
	A : {sparse matrix, dense matrix, LinearOperator}
		An array, sparse matrix, or LinearOperator representing
		the operation ``A * x``, where A is a real or complex square matrix.

	Z : ndarray
	
	verbose : bool, optional
		  Displays information about the accuracy of the resulting QR
		  Default: False

	Returns
	-------	

	q : ndarray
		The A-orthogonal vectors 

	Aq : ndarray
		The A^{-1}-orthogonal vectors

	r : ndarray
		The r of the QR decomposition


	See Also
	--------
	mgs : Modified Gram-Schmidt without re-orthogonalization
	mgs_stable : Modified Gram-Schmidt with re-orthogonalization


	References
	----------
		
	.. [1]  B. Lowery and J. Langou, 
		Stability Analysis of QR factorization in an Oblique 
		Inner Product http://arxiv.org/abs/1401.5171

	.. [2] A.K. Saibaba, J. Lee and P.K. Kitanidis, Randomized algorithms for Generalized
		Hermitian Eigenvalue Problems with application to computing 
		Karhunen-Loe've expansion http://arxiv.org/abs/1307.6885


	Examples
	--------
	
	>>> import numpy as np
	>>> A = np.diag(np.arange(1,101)) 	
	>>> Z = np.random.randn(100,10)
	>>> q, Aq, r = precholqr(A, Z, verbose = True)

	"""


	y, s = qr(Z, mode = 'economic')
	q, Aq, u = cholqr(A, y, False)
	r = np.dot(u,s)

	if verbose:
		
		#Verify Q*R = Y
		print "||QR-Y|| is ", np.linalg.norm(np.dot(q,r) - Z, 2)
		
		#Verify Q'*A*Q = I
		T = np.dot(q.T, Aq)
		n = T.shape[1]
		print "||Q^TAQ-I|| is ", \
			np.linalg.norm(T - np.eye(n, dtype = 'd'), ord = 2)		

		#verify Q'AY = R 
		print  "||Q^TAY-R|| is ", np.linalg.norm(np.dot(Aq.T,Z) - r,2)

		#Verify YR^{-1} = Q
		val = np.inf
		try:
			val = np.linalg.norm(np.linalg.solve(r.T,Z.T).T-q,2)
		except LinAlgError:
			print "||YR^{-1}-Q|| is ", "Singular"
		print "||YR^{-1}-Q|| is ", val

	return q, Aq, r	
	

if __name__ == '__main__':

	m, n  = 100, 5

	A = np.diag(np.arange(1,m+1))
	Z = np.random.randn(m,n)

	print "Testing MGS"
	mgs(A, Z, verbose = True)
	print "Testing MGS-R"
	mgs_stable(A, Z, verbose = True)
	print "Testing CholQR"
	cholqr(A, Z, verbose = True)
	print "Testing PreCholQR"
	precholqr(A, Z, verbose = True)	
