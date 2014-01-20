import numpy as np
from scipy import linalg

# Create parameter container
class Parameters:
    pass

class model(object):

	def __init__(self,para):	
		self.parms = Parameters()
		self.parms.N = para.N
		self.parms.Nin = para.Nin
		self.parms.Nout= para.Nout
		self.parms.inp_sc= para.inp_sc
		self.parms.spectr_rad= para.spectr_rad
		self.parms.leak_rate= para.leak_rate
		self.parms.inpBias = para.inpBias

		self.parms.Pscaling = para.Pscaling
		self.parms.OutScaling = para.OutScaling

		self.parms.alpha = para.alpha

		self.parms.t1 = para.t1
		self.parms.dt = para.dt

		self.parms.y_d = para.y_d

		self.P = np.eye(self.parms.N)*self.parms.Pscaling#100
		self.inpBias = self.parms.inpBias * np.random.randn(self.parms.N,1)

		wtemp = np.random.randn(self.parms.N,self.parms.N)
		self.W = self.parms.spectr_rad*wtemp/max(abs(linalg.eigvals(wtemp)))
		self.V = np.random.randn(self.parms.N,self.parms.Nin)*np.dot(np.ones((self.parms.N,1)),self.parms.inp_sc) 

		self.Woutp = np.random.randn(self.parms.Nout,self.parms.N)

		self.state_A = np.random.rand(self.parms.N,1)*2.-1.
		self.state_B = self.state_A
		self.orgstate = self.state_A
		
	''' Update the output weights of the model (RLS) '''
	def update(self,dyn_states,dyn_output,softmax,warmup=0):
		self.state_A = self.update_res_states(self.state_A, dyn_states)

		self.P = (self.P - (np.dot(np.dot(np.dot(self.P, self.state_A), self.state_A.T), 
			self.P) / (self.parms.alpha + np.dot(np.dot(self.state_A.T, self.P), self.state_A)))) / self.parms.alpha
		
		dw = np.dot(self.P,self.state_A).T/(1.+np.dot(np.dot(self.state_A.T,self.P),self.state_A)) 

		e = (np.dot(self.Woutp, self.state_A)) - self.parms.OutScaling*np.atleast_2d(dyn_output).T

		if warmup==0:
			self.Woutp = self.Woutp - e * softmax * dw
			self.dw = dw

		return e

	def predict(self, dyn_states):
		self.state_B = self.update_res_states(self.state_B, dyn_states)
		return np.dot(self.Woutp, self.state_B)/self.parms.OutScaling

	def predictA(self):
		return np.dot(self.Woutp, self.state_A)/self.parms.OutScaling

	''' Update the reservoir states '''
	def update_res_states(self, res_state, dyn_states):
		res_state = res_state * (1. - self.parms.leak_rate) + self.parms.leak_rate * np.tanh(np.dot(self.W, res_state) + np.dot(self.V, dyn_states) + self.inpBias)
		res_state[-1] = 1
		# When linear models are needed uncomment the following line (e.g., less expressive models)
		#res_state = np.dot(self.V, dyn_states) 
	
		return res_state

	''' Reset the states if needed '''
	def resetState(self):
		self.state_A = self.orgstate
		self.state_B = self.orgstate
