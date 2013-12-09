import numpy as np
from aca import *

#Iterator
from itertools import product


__all__ = ["Cluster", "BlockCluster"]

class Cluster:
	"""	
	Implementation of a cluster tree

	Methods:
	--------
	assign_points_bisection()
	diameter()	
		

	"""
	def __init__(self, dim = 2, level = 0, parent = None, children = []):
		#Physical dimension of the points	
		self.dim = dim	

		#Level of the tree
		self.level	= level

		#Parent
		self.parent  	= parent
		
		#Siblings
		self.children = []

	def assign_points_bisection(self, pts, indices):
		"""
		Divide points by geometric bisection
		"""
	
		n = np.size(indices)
	
		self.indices = indices
		self.n       = n		

			
		dim = self.dim

		#Bounding boxes
		self.alpha = np.zeros((dim,), 'd')
		self.beta  = np.zeros((dim,), 'd')

		for i in np.arange(dim):
			self.alpha[i] = np.min(pts[indices,i])
			self.beta[i]  = np.max(pts[indices,i])

		#Split along direction of maximum
		jmax = np.argmax(self.beta-self.alpha)
	
		#Divide the region into two halves	
		gamma = 0.5*(self.beta[jmax] + self.alpha[jmax])



		#If cluster size is too small return
		if n < 64:
			return
		

		#otherwise split clusters 
		indl = np.extract(pts[indices,jmax] <= gamma, indices)
		indr = np.extract(pts[indices,jmax] > gamma, indices)
	

		#Create new clusters and recursively call them
		for i in np.arange(2):
			child = Cluster(dim = self.dim, level = self.level+1, parent = self)
			self.children.append(child)

		self.children[0].assign_points_bisection(pts, indl)
		self.children[1].assign_points_bisection(pts, indr)

		return

	#Diameter of each cluster
	def diameter(self):
		return np.linalg.norm(self.beta-self.alpha)
	
	
#Distance between clusters			
def distance(tau, sigma):
	assert isinstance(tau, Cluster) and isinstance(sigma, Cluster)
		
	dist = 0.
	for k in np.arange(tau.dim):
		dist += np.max((0., tau.alpha[k]-sigma.beta[k]))**2. + np.max((0.,sigma.alpha[k]-tau.beta[k]))**2.
		
	return np.sqrt(dist)

#Are the clusters sufficiently "far" away?
def admissible(tau, sigma, eta = 0.75):
	assert isinstance(tau, Cluster) and isinstance(sigma, Cluster)
		
	if np.min((tau.diameter(),sigma.diameter())) <= eta*distance(tau,sigma):
		return True
	else:
		return False		


#Block cluster class
class BlockCluster:
	"""
	Implementation of Block Cluster

	"""
	def __init__(self, level = 0, parent = None, tau = None, sigma = None):

		#Level
		self.level 	= level

		#Heritage
		self.parent 	= parent
		
		#Clusters
		self.tau 	= tau
		self.sigma	= sigma


		#Direct or not 
		self.admissible = False
	
		#List of children
		self.children 		= []
		self.clusterpairs 	= []
	
	def construct_block_tree(self, tau, sigma):

		self.tau = tau
		self.sigma = sigma
			
		
		if not admissible(tau, sigma) and len(tau.children) != 0 and len(sigma.children) != 0:	

			for Clx, Cly in product(tau.children,sigma.children):
				block = BlockCluster(self.level+1,self)
				block.construct_block_tree( Clx, Cly)
				self.children.append(block)		
				self.clusterpairs.append((Clx,Cly))
	
		else:
			#Determine whether or not to compute interactions
			if admissible(tau, sigma):
				self.admissible = True	
			return	

		return			


	def construct_space_filling_curve(self, leaflist, costlist):
		if len(self.children) != 0:
			for child in self.children:
				child.construct_space_filling_curve(leaflist,costlist)
		else:
			leaflist.append(self)
			
			nx = np.size(self.tau.indices)
			ny = np.size(self.sigma.indices)
		
			
			if self.admissible:
				rk = np.size(self.A,1)
				costlist.append(2.*rk*(nx+ny ) )
			#else:
			#	costlist.append(nx*ny)

	def construct_low_rank_representation(self, pts, kernel, rkmax, eps):
		if len(self.children) != 0:
			for child in self.children:
				child.construct_low_rank_representation(pts,kernel,rkmax,eps)
		else:
			self.rkmax = rkmax
			self.eps   = eps
			if self.admissible:
				self.A, self.B = ACApp(pts, self.tau.indices, self.sigma.indices, kernel, rkmax, eps)	
			else:
				self.mat = GenerateDenseMatrix(pts, self.tau.indices, self.sigma.indices, kernel)
	
		return	
	

	def mult(self, x, y, pts, kernel):
		"""
		Matrix-vector product
		
		Parameters:
		-----------
		x:	(n,) ndarray
			Vector to be multiplied

		y:	(n,) ndarray	
			Vector after multiplication

		pts:	(n,dim) ndarray
			

		kernel: Kernel object
			

		"""
		if len(self.children) != 0:
	
			#Iterate through children
			for child in self.children:
				child.mult(x, y, pts, kernel)
		else:
			if not self.admissible:  #Direct interactions
				y[self.tau.indices] += np.dot(self.mat,x[self.sigma.indices])
			else:		#Use ACA approximations
				y[self.tau.indices] += np.dot(self.A, np.dot(self.B.T,x[self.sigma.indices]))					
		return
				
	def transpmult(self, x, y, pts, kernel):
		"""
		Transpose matrix-vector product
		
		Parameters:
		-----------
		x:	(n,) ndarray
			Vector to be multiplied

		y:	(n,) ndarray	
			Vector after multiplication

		pts:	(n,dim) ndarray
			

		kernel: Kernel object
			

		"""

	
		
		if len(self.children) != 0:

                        #Iterate through children
                        for child in self.children:
                                child.transpmult(x, y, pts, kernel)
                else:
                        if not self.admissible:  #Direct interactions
                                mat = GenerateDenseMatrix(pts, self.tau.indices, self.sigma.indices, kernel)
                                y[self.sigma.indices] += np.dot(mat.transpose(),x[self.tau.indices])
                        else:           #Use ACA approximations
                                y[self.sigma.indices] += np.dot(self.B, np.dot(self.A.T,x[self.tau.indices]))                                
		return 	 	
			



if __name__ == '__main__':
	tree = Cluster()			
			
	N = 10000
	
	from math import pi
	theta = 2.*pi*np.arange(N)/float(N)

	#Circle
	pts = np.hstack((np.cos(theta)[:,np.newaxis],np.sin(theta)[:,np.newaxis]))

	#Cardoid
	a = 1.
	x = a*(2.*np.cos(theta)-np.cos(2.*theta))
	y = a*(2.*np.sin(theta)-np.sin(2.*theta))
	pts = np.hstack((x[:,np.newaxis],y[:,np.newaxis]))

	indices = np.arange(N)	
	
	from time import time

	start = time()
	tree.AssignPointsBisection(pts, indices)
	print "Assigning points to clusters took %g" % (time()-start)			

	PlotTree(tree, 0.75, 7)	
