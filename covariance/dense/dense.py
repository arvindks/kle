import numpy as np

__all__ = ['GenerateDenseMatrix']
#Compute O(N^2) interactions    
def GenerateDenseMatrix(pts, kernel):
        """
	Explicitly generate dense matrix from entries
	
	Parameters:
	-----------
	
	pts:	(n,dim)
		Points in the domain

	kernel:	Kernel object

	
	Returns:
	--------
	
	mat:	n x n dense matrix
		has entries such that mat(i,j) = kernel(pts[i,:],pts[j,:]
	
	
	"""                        
        nx = np.size(pts,0)   
	
	ptsx = pts
        if nx == 1:     ptsx = pts[np.newaxis,:]
	 
                        
        dim = np.size(pts,1)    
        R = np.zeros((nx,nx),'d')
                        

        for i in np.arange(dim):
                X, Y = np.meshgrid(ptsx[:,i],ptsx[:,i])
                R += (X.transpose()-Y.transpose())**2. 
                
        return kernel(np.sqrt(R))         
