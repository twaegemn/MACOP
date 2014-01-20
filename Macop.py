import numpy as np
import datetime
import time

import scipy.io as sio
import sys

import multimodel as mmdl
import model as mdl

class Macop(object):

	def __init__(self,parms):
		self.setParms(parms)
		self.inited = 0
		self.forceLimits = parms.outpLimits

	def init(self):
		self.y_d = self.parms.y_d.copy()

		# Load Models
		self.models = list()
		for nm in xrange(int(self.parms.num_models)):
			self.models.append(mdl.model(self.parms))

		# Load multimodel setup
		self.multimodels = mmdl.multimodel(float(self.parms.NinSoftmax),float(self.parms.Nout),self.parms.num_models,self.parms.eta_s, self.parms.eta_g)

		self.sdpos = np.zeros((self.parms.Nout,))

		# Reading and setting al the initial parameters
		t1 = self.parms.t1
		dt = self.parms.dt
		delta = self.parms.delta
	   
		self.sysInp = np.zeros((self.parms.sNin, t1 / dt + delta + 1))
		self.sysInp2 = np.zeros((self.parms.Nout, t1 / dt + delta + 1))
		self.sysOutp = np.zeros((self.parms.Nout, t1 / dt + delta + 1))

		self.eeVal = self.sysInp
		self.inited = 1

		self.model_e = np.zeros((self.parms.Nout, self.parms.num_models))
		self.model_outputs = np.zeros((self.parms.Nout, self.parms.num_models,t1 / dt + delta + 1))
		self.h = np.zeros((self.parms.num_models,t1 / dt + delta + 1))
		self.model_doutputs = np.zeros((self.parms.Nout, self.parms.num_models))

		self.i = delta
 
		self.t = 0

	''' Reset all controllers if needed '''
	def reset(self):
		self.i = self.parms.delta
 		self.t = 0
		
		for nm in xrange(int(self.parms.num_models)):
			self.models[nm].resetState()

	''' Update/Run the MACOP controller. This function should be called iteratively '''
	def run(self,actualState,warmup=0):
		
		self.y_d = self.parms.y_d
		delta = self.parms.delta

		self.t = self.t + 1

		# Convert inp to a range between -1 and 1
		self.sysInp[:, self.i] = actualState # Assumed that this is done (use helper function bellow if not)
		self.sysInp2[:, self.i] = self.sysOutp[:, self.i-1]

		# Construct input to all the models
		inputA = np.atleast_2d(np.array([(self.sysInp[:, self.i - delta]).tolist(), (self.sysInp[:, self.i]).tolist()]).flatten()).T
		inputB = np.atleast_2d(np.array([(self.sysInp[:, self.i]).tolist(), (self.y_d[:, self.i + delta]).tolist()]).flatten()).T

		# Which model is more probable for this sample than the others?
		self.h[:,self.i], proj = self.multimodels.softmax(self.sysInp[0, self.i],self.sysInp2[:, self.i])

		winner = self.h[:,self.i].flatten()

		# Train each model (most probable more than the others)
		for nm in xrange(int(self.parms.num_models)):
			inpA = np.vstack((inputA,np.atleast_2d(self.model_outputs[:,nm,self.i-delta-1]).T))
			inpB = np.vstack((inputB,np.atleast_2d(self.model_outputs[:,nm,self.i-1]).T))

			# Update controllers and scale learning rate
			self.model_e[:,nm] = self.models[nm].update(inpA,self.sysOutp[:, self.i-delta],winner[nm],warmup=warmup).flatten()
			self.model_outputs[:,nm,self.i] =  self.models[nm].predict(inpB).flatten()

			#print self.model_e[:,nm].flatten()
			# Ensure that all model outputs are within the desired range [-1,1]
			self.model_outputs[self.model_outputs[:,nm,self.i]>1.,nm,self.i]=1.
			self.model_outputs[self.model_outputs[:,nm,self.i]<-1.,nm,self.i]=-1.

			# Mix output
			self.sysOutp[:, self.i] = self.sysOutp[:, self.i] + winner[nm]*(self.model_outputs[:,nm,self.i])

		# Modify the softmax based on the certainty of a model
		##TODO: Be sure that the outputs are used as desired in the mixing system (what are your inputs to the mixing)
		ht = self.multimodels.updateProjection(self.model_e, self.model_outputs[:,:,self.i],self.h[:,self.i].flatten(), self.y_d[:, self.i], self.sysInp[:, self.i],self.sysInp2[:, self.i])

		self.sysOutp[self.sysOutp[:, self.i]>1., self.i] = 1.
		self.sysOutp[self.sysOutp[:, self.i]<-1., self.i] = -1.

		# Convert joints (between -1 and 1) back to their corresponding range
		n_sdpos = self.deNormJoints(self.sysOutp[:, self.i])

		self.i += 1
			
		return n_sdpos

	''' HELPER FUNCTIONS '''

	def isInitd(self):
		return self.inited

	def setParms(self, parms):
		self.parms = parms

	def rescaleValue(self, value, minv, maxv):
		return (maxv-minv)/2.*value+((maxv-minv)/2.+minv)

	def unscaleValue(self, value, minv, maxv):
		return (value-((maxv-minv)/2.+minv))/((maxv-minv)/2.)

	def limValue(self, val, minv, maxv):
		print min(max(val,minv),maxv)
		return min(max(val,minv),maxv)
	
	def getShiftInfo(self,data):
		return np.min(data,axis=1), np.max(data,axis=1)

	def shiftData(self,val,minv,maxv):
		return val#-((maxv-minv)/2.+minv)

	def unshiftData(self,val,minv,maxv):
		return val#+((maxv-minv)/2.+minv)

	def getEEVal(self):
		return self.eeVal

	def lowpass(self,data,l):
		for i in xrange(data.shape[1]-1):
			data[:,i+1] = (1.-1./l)*data[:,i] + data[:,i+1]/l
		return data

	def deNormJoints(self,joints):
		
		njoints = np.zeros((1,self.parms.Nout)).flatten()
		for i in xrange(self.parms.Nout):
			njoints[i] = self.rescaleValue(joints[i],self.forceLimits[0,i],self.forceLimits[1,i])
		return njoints

	def normJoints(self,joints):
		
		njoints = np.zeros((1,self.parms.Nout)).flatten()
		for i in xrange(self.parms.Nout):
			njoints[i] = self.unscaleValue(joints[i],self.forceLimits[0,i],self.forceLimits[1,i])
		return njoints

	def saveState(self,finalJointPos):
		tstamp = str(int(time.time()))
		fname = ''+tstamp+'_saved_'+str(int(self.parms.num_models))
		persModels.saveModels(fname, self.models, self.multimodels, finalJointPos)

	def updateDesTraj(self,y_d):
		self.y_d = y_d.copy()

