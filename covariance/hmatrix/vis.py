from tree import *

#Plotting functions
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.patches import Rectangle as rect
from matplotlib.collections import PatchCollection

import numpy as np

#Functions to visualize clusters in 2D
def VisualizeCluster2D(node, maxlevels, ax):
	if node.level != maxlevels:
		#Split along direction of maximum
		jmax = np.argmax(node.beta-node.alpha)
	
		#Divide the region into two halves	
		gamma = 0.5*(node.beta[jmax] + node.alpha[jmax])
		
		xmin, xmax, ymin, ymax = node.alpha[0], node.beta[0], node.alpha[1], node.beta[1]
		
		#Plot bounding boxes
		ax.plot([xmin,xmax],[ymin,ymin], 'k-', linewidth = 1.)
		ax.plot([xmax,xmax],[ymin,ymax], 'k-', linewidth = 1.)
		ax.plot([xmax,xmin],[ymax,ymax], 'k-', linewidth = 1.)
		ax.plot([xmin,xmin],[ymax,ymin], 'k-', linewidth = 1.)

		if jmax == 0:
			ax.plot([gamma,gamma],[ymin,ymax],'k-', linewidth = 1.)
		else:
			ax.plot([xmin,xmax],[gamma,gamma],'k-', linewidth = 1.)

		for child in node.children:
			VisualizeCluster2D(child, maxlevels,ax)
				
				
	else:
		return
				
	return

def VisualizeClusterTree2D(tree, pts, rows, cols):

	maxlevels = rows*cols
	
	

	dim = np.size(pts,1)
	assert dim == 2

	#Compute bounding boxes
	alpha = np.zeros((dim,), 'd')
	beta  = np.zeros((dim,), 'd')

	N   = np.size(pts,0)
	indices = np.arange(N)


	fig, axarray = plt.subplots(rows, cols) 
	

	for i in np.arange(dim):
		alpha[i] = np.min(pts[indices,i])
		beta[i]  = np.max(pts[indices,i])
	
	xmin, xmax, ymin, ymax = alpha[0], beta[0], alpha[1], beta[1]
	
	level = 0
	for i in np.arange(rows):
		for j in np.arange(cols):
			#Plot points
			
			axarray[i,j].plot(pts[:,0],pts[:,1],'b-', markersize = 1.)
			fig.gca().aspect('equal')		

			axarray[i,j].set_xticks([])
			axarray[i,j].set_yticks([])
	
			VisualizeCluster2D(tree, level, axarray[i,j])	
			level += 1
	plt.savefig('figs/clustertree' + str(maxlevels) +'.png') 

	return

#Plot the cluster tree as a tree diagram
def plotnode(node, xmin, xmax, ymin, ymax, factor, markersize, maxlevels):
	
	mid = 0.5*(xmin+xmax)
	plt.plot(mid,ymax, 's', markersize = markersize)
	if len(node.children) == 0 or node.level == maxlevels:
		return
	else:
		#plot connecting lines
		plt.plot([mid, 0.5*(xmin+mid)], [ymax,ymin], '-k', linewidth = 1.)	
		plt.plot([mid, 0.5*(mid+xmax)], [ymax,ymin], '-k', linewidth = 1.)	
		
		#Recurse
		plotnode(node.children[0], xmin, mid, ymin - factor*(ymax-ymin), ymin, factor, 0.5*markersize, maxlevels)
		plotnode(node.children[1], mid, xmax, ymin - factor*(ymax-ymin), ymin, factor, 0.5*markersize, maxlevels)
	
	
	return	

def PlotTree(tree, factor, maxlevels):
	plt.figure(0)
	plt.clf()
	
	plt.xticks([])
	plt.yticks([])
	plt.axis([0,1,0,1.1])
	
	
	a = (1.- factor)/(1.- factor**(maxlevels+1.))

	plotnode(tree, 0., 1., 1.-a , 1., factor, 32, maxlevels)

	plt.draw()
		
	plt.savefig('figs/tree' + str(maxlevels) + '.png') 


	return


#Plot block cluster tree as a 2D hierarchical object
def plotblockclusternode(plt, block, xmin, ymin, width, height, maxlevel, ax):
	if block.level == maxlevel or len(block.children) == 0:
		
		col = 'b' if block.admissible else 'r'
		
		r = rect((xmin, ymin), width = width, height = height, color = col, alpha =0.8, visible = True, fill = True)
		r.set_edgecolor('k')
	
		#plt.gca().add_patch(r)
		ax.add_patch(r)
		return
	else:
		ht = height
		wd = width

		fract = float(block.tau.children[0].n)/float(block.tau.n)
		fracs = float(block.sigma.children[0].n)/float(block.sigma.n)

		fract = float(block.tau.children[0].n)/float(block.tau.n)
		fracs = float(block.sigma.children[0].n)/float(block.sigma.n)

		#[1 0; 0 0]	quadrant
		x = xmin
		y = ymin + (1.-fract)*ht
		plotblockclusternode(plt, block.children[0], x, y, fracs*wd, fract*ht, maxlevel, ax)
		#[0 1; 0 0]	quadrant
		x = xmin + fracs*wd
		y = ymin + (1.-fract)*ht
		plotblockclusternode(plt, block.children[1], x, y, (1.-fracs)*wd, fract*ht,maxlevel, ax)
		#[0 0; 1 0]
		x = xmin
		y = ymin
		plotblockclusternode(plt, block.children[2], x, y, fracs*wd, (1.-fract)*ht,maxlevel,ax)
		#[0 0; 0 1]
		x = xmin + fracs*wd
		y = ymin
		plotblockclusternode(plt, block.children[3], x, y, (1.-fracs)*wd, (1.-fract)*ht,maxlevel,ax)
		
	return	

def plotblockclustertree(btree, rows, cols):
	
	level = 0
	
	_, axarray = plt.subplots(rows, cols)


	for i in np.arange(rows):
		for j in np.arange(cols):
		
			axarray[i,j].set_xticks([])
			axarray[i,j].set_yticks([])
	
			plotblockclusternode(plt, btree,0.,0.,1.,1.,level, axarray[i,j])
			level += 1
		

	plt.savefig('figs/bcluster'+str(rows*cols)+'.png')
	
	return


if __name__ == '__main__':
	tree = Cluster()			
			
	N = 20000
	
	from math import pi
	theta = 2.*pi*np.arange(N)/float(N)

	#pts = np.random.rand(N,2)

	#Circle
	pts = np.hstack((np.cos(theta)[:,np.newaxis],np.sin(theta)[:,np.newaxis]))

	#Cardoid
	#a = 1.
	#x = a*(2.*np.cos(theta)-np.cos(2.*theta))
	#y = a*(2.*np.sin(theta)-np.sin(2.*theta))
	#pts = np.hstack((x[:,np.newaxis],y[:,np.newaxis]))

	#from dolfin import *
	#mesh = Mesh("dolfin-2.xml.gz")
	#pts = mesh.coordinates()

	#Epitrichoid
	#a = b = c = 1 
	#x = (a + b)*np.cos(theta) - c*np.cos((a/b + theta)*theta)
	#y = (a + b)*np.sin(theta) - c*np.sin((a/b + 1)*theta)
	#pts = np.hstack((x[:,np.newaxis],y[:,np.newaxis]))


	#N = 100
	#from dolfin import *
	#mesh = UnitCircle(N)
	#pts = mesh.coordinates()
	#N = pts.shape[0]


	#Straight line 
	#x = np.linspace(0,1,N)
	#y = np.zeros((N,1))
	#pts = np.hstack((x[:,np.newaxis],y))

		
	plt.close('all')
	
	indices = np.arange(N)	
	
	from time import time

	start = time()
	tree.AssignPointsBisection(pts, indices)
	print "Assigning points to clusters took %g" % (time()-start)			

	#VisualizeClusterTree2D(tree, pts, 3 , 2 )

	##PlotTree(tree, 0.75, 7)	
	btree = BlockCluster(level = 0)
        btree.ConstructBlockTree(tree, tree)

	plotblockclustertree(btree, rows = 3, cols = 3)	
	
