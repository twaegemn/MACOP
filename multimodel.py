import numpy as np

class multimodel(object):
	''' Copy and initialize some parameters '''
	def __init__(self,inp_dim_x,inp_dim_q,dim, eta_s, eta_g):
		self.dim = dim
		self.inp_dim_x = inp_dim_x
		self.inp_dim_q = inp_dim_q

		self.eta_s = eta_s
		self.eta_g = eta_g

		self.projection = 0.1*np.random.randn(self.dim,self.inp_dim_x+self.inp_dim_q)#+self.inp_dim_q

	''' Simple softmax function '''
	def softmax(self,x,x_q):
		A = np.exp(-np.dot(self.projection,np.atleast_2d(np.hstack((x,x_q))).T))#+ma_q
		B = np.sum(A)
		h = A/(B+0.000001)	# Add small number to prevent division by zero
		
		return h.T, self.projection

	''' Find best performing controller and calculate a target controller distribution (softmax regression)'''
	def updateProjection(self,model_e,model_outputs,softmax, des, x, x_q):
		hardmax = softmax.copy().flatten()
		hbest = hardmax[np.argmax(softmax)]

		hardmax[:] = softmax.copy().flatten()/(1.- hbest)*(1.-1./self.dim)
		hardmax[np.argmax(softmax)] = 1./self.dim+0.01*np.random.randn() # Add small noise to improve exploration

		# Update projection matrix according to this new distribution
		self.projection = self.projection + self.eta_g*self.projection/np.linalg.norm(self.projection)
		self.projection = self.projection + self.eta_s*np.dot(np.atleast_2d(softmax-hardmax).T,np.atleast_2d(np.hstack((x,x_q))))

		return hardmax;
