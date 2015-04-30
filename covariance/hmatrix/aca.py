import numpy as np

__all__ = ['GenerateDenseMatrix', 'ACA', 'ACApp']

#Compute O(N^2) interactions
def GenerateDenseMatrix(pts, indx, indy, kernel):

	nx = indx.size
	ny = indy.size	

	ptsx = pts[indx,:]
	ptsy = pts[indy,:]

	if nx == 1:	ptsx = ptsx[np.newaxis,:]
	if ny == 1:	ptsy = ptsy[np.newaxis,:]

	
	dim = np.size(pts,1)
	R = np.zeros((nx,ny),'d')


	for i in np.arange(dim):
		X, Y = np.meshgrid(ptsx[:,i],ptsy[:,i])
		R += (X.transpose()-Y.transpose())**2.

	return kernel(np.sqrt(R))			

#This is the expensive version, only for debugging purposes
def ACA(pts, indx, indy, kernel, rkmax, eps):
	"""
	Adaptive Cross Approximation

	Parameters:
	-----------
	pts:	(n,dim) ndarray
		all the points
	
	indx:	(nx,) ndarray
		indices of the points

	indy:	(ny,) ndarray
		indices of the points
		
	kernel:	Kernel object
		see covariance/kernel.py for more details
	
	rkmax: 	int
		Maximum rank of the low rank approximation

	eps:	double
		Relative error of the low rank approximation

	Returns:
	--------
	
	A,B:	(nx,k) and (ny,k) ndarray
		such that Q approx AB^T		

	References:
	-----------
	Sergej Rjasanow, Olaf Steinbach, The fast solution of boundary integral equations. 
		Mathematical and analytical techniques with applications to engineering. Springer 2007, New York.	

	"""

	#Generate matrix
	R = GenerateDenseMatrix(pts, indx, indy, kernel)

	normR = np.linalg.norm(R, 'fro')

	nx = indx.size
	ny = indy.size

	A = np.zeros((nx,rkmax),'d')
	B = np.zeros((ny,rkmax),'d')


	kmax = rkmax
	
	for k in np.arange(rkmax):	
		#Find largest pivot indices
		ind = np.unravel_index(np.argmax(np.abs(R)),(nx,ny))
		
		#Largest pivot
		gamma = 1./R[ind]
		
		u, v = gamma*R[:,ind[1]], R[ind[0],:]
		
		A[:,k] = np.copy(u)
		B[:,k] = np.copy(v.transpose())

		R -= np.outer(u,v)
			
		if np.linalg.norm(R,'fro') <= eps*normR:
			kmax = k		
			break
		
	return A[:,:kmax], B[:,:kmax]	


#Implementation of the partially pivoted Adaptive Cross Approximation
def ACApp(pts, indx, indy, kernel, rkmax, eps):
        """
        Partially pivoted Adaptive Cross Approximation

        Parameters:
        -----------
        pts:    (n,dim) ndarray
                all the points
        
        indx:   (nx,) ndarray
                indices of the points

        indy:   (ny,) ndarray
                indices of the points
                
        kernel: Kernel object
                see covariance/kernel.py for more details
        
        rkmax:  int
                Maximum rank of the low rank approximation

        eps:    double
                Approximate relative error of the low rank approximation
		Computing frobenius norm is O(n^2) so it is avoided.
        Returns:
        --------
        
        A,B:    (nx,k) and (ny,k) ndarray
                such that Q approx AB^T         

        References:
        -----------
        Sergej Rjasanow, Olaf Steinbach, The fast solution of boundary integral equations. 
	Mathematical and analytical techniques with applications to engineering. Springer 2007, New York.    

        """
	nx = np.size(indx)
	ny = np.size(indy)

	A = np.zeros((nx,rkmax),'d')
	B = np.zeros((ny,rkmax),'d')

	rows = np.zeros((rkmax+1,),'i')	
	
	#Initialize
	row  = np.min(np.arange(nx))

	#Norm
	norm = 0.

	#Maximum rank
	kmax = rkmax 
	
	for k in np.arange(rkmax):

		#generate row
		b = GenerateDenseMatrix(pts,indx[row],indy,kernel)
		B[:,k] = np.copy(b.ravel())
		for nu in np.arange(k):
			B[:,k] -= A[row,nu]*B[:,nu] 

	
		#maximum row entry
		col = np.argmax(np.abs(B[:,k]))

		#Compute delta
		delta = B[col,k]
		if np.abs(delta) < 1.e-16:
			kmax = k 
			break	
		
		B[:,k] /= delta
		
		#Generate column
		a = GenerateDenseMatrix(pts,indx,indy[col],kernel)	 

		A[:,k] = np.copy(a.ravel() )
	
		for nu in np.arange(k):
			A[:,k] -= A[:,nu]*B[col,nu]


		#Next row
		diff = np.setdiff1d(np.arange(nx),rows[:k+1])	
		if np.size(diff) == 0:
			break
		row = diff[np.argmin(np.abs(A[diff,k]))]
		rows[k+1] = row

		#update norm
		for nu in np.arange(k):
			norm += 2.*np.dot(A[:,k],A[:,nu])*np.dot(B[:,k],B[:,nu])

		
		ukvk = np.linalg.norm(A[:,k])**2.*np.linalg.norm(B[:,k])**2.
		norm += ukvk
	
		if ukvk <= eps*np.sqrt(norm):
			kmax = k
			break
	
	return A[:,:kmax], B[:,:kmax]
		

		
		
if __name__ == '__main__':
		
	#Test for ACA	
	N = 1000
	#pts = np.linspace(0,1,N)[:,np.newaxis]
	pts = np.random.rand(N,2)
	pts[N/2:,0] += 2.
	pts[N/2:,1] += 2.

	indx = np.arange(N/2)
	indy = np.setdiff1d(np.arange(N),indx)

	#Kernel
	def kernel(R):
		return np.exp(-R)

	
	rkmax = int(N/2) 
	eps = 1.e-12
	print rkmax, eps

	from time import time

	start = time()
	mat = GenerateDenseMatrix(pts,indx,indy,kernel)
	print "Time for full construction is %g" %(time()-start)

	
	start = time()
	A,B = ACApp(pts,indx,indy,kernel,rkmax,eps)
	print "Time for ACA construction is %g" %(time()-start)

	#Check against svd 
	from scipy.linalg import svdvals as svd
	s = svd(mat)
	s = s/s[0]
        ind = np.extract(s > 1.e-6, s)	


	print "Error is ", np.linalg.norm(mat-np.dot(A,B.transpose()))/np.linalg.norm(mat)
	print np.size(A,1), ind.size 
	


					
