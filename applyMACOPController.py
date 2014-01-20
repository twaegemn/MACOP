import matplotlib.pyplot as plt
import numpy as np
import datetime
import time
import thread

import ctypes
import scipy.io as sio

import Macop
import omronControl

import sys

plt.ion()

class DataCollector(object):
	pass

dataCollector = DataCollector()

import parSettings

# Load parameters from parSettings.py
parms = parSettings.parms

# Define the threads as functions ####################################
def ctrlThread(controller, dataCollector, parms):
	print "MCtrl Thread started"
	while(not dataCollector.isInitd):
		pass

	while(not dataCollector.finished):
		tic = datetime.datetime.now()

		##TODO: Set the inputs to MACOP (e.g, from sensor thread)
		# state = dataCollector.sensVal
		dataCollector.contrOutput = controller.run(state)

		contDelay = (datetime.datetime.now()-tic).microseconds/1000000.
		timeDist = (parms.controllerStep-contDelay)

		if(timeDist<0):
			print "MC_overrun"		

		time.sleep(max(0.,timeDist))

def sensorThread(tracker, dataCollector, parms):
	print "Sensor thread started..."
	while(not dataCollector.finished):
		tic = datetime.datetime.now()
		
		##TODO: Put your sensor acquisition here
		# dataCollector.sensVal = 
		
		contDelay = (datetime.datetime.now()-tic).microseconds/1000000.
		timeDist = (parms.sensStep-contDelay)
		
		if(timeDist<0):
			print "SE_overrun"		
		
		time.sleep(max(0.,timeDist))

def subThePlots(pl,data,num):
	for i in xrange(data.shape[0]):#
		pl[i].set_xdata((np.arange(0,num)))
		pl[i].set_ydata((data[i,0:num]))

def pltThread(dataCollector, parms):
	while(not dataCollector.finished):
		if dataCollector.i > 1:
			if dataCollector.i%400==0: # Only plot every 400 iterations
				subThePlots(dataCollector.pl2,dataCollector.angle_traj,dataCollector.i+parms.delta)
				subThePlots(dataCollector.pl3,dataCollector.state_traj[0:3,:],dataCollector.i+parms.delta)
				subThePlots(dataCollector.pl4,dataCollector.state_traj[3:6,:],dataCollector.i+parms.delta)	
				plt.xlim([max(0,dataCollector.i-parms.delta-300.),dataCollector.i-parms.delta])
				plt.ylim([-1,1])
				plt.draw()

		time.sleep(parms.controllerStep)

##TODO: Define robot ############################################
# robot =

## DEFINE OBJECTS ##################################
# Initiate MACOP controller with parameters

controller = Macop.Macop(parms)
controller.init()

## INIT OBJECTS ####################################
dataCollector.finished = 0
dataCollector.isInitd = 0
dataCollector.contrOutput = np.zeros((parms.Nout,)) # Initiate first MACOP output

parms.realTime = 0. # Initiate real time counter

## START THREADS
ctrThread = thread.start_new_thread(ctrlThread,(controller, dataCollector,parms))
#plThread = thread.start_new_thread(pltThread,(dataCollector,parms))

## INIT ROBOT TO REGION
robot.run()
time.sleep(1)

## MAIN ROBOT CONTROL LOOP ###############################
print "Main control thread started ..."
for j in xrange(1):
	i = 0
	while parms.realTime<parms.maxTime:
		tic = datetime.datetime.now()

		# Get current MACOP output
		npos = dataCollector.contrOutput
		
		# Filter if needed (some robots can not handle too jerky control signals)
		#npos_t = npos+nposv*parms.mainLoopStep
		#nposv = -parms.critDamp*npos+(1.-parms.critDamp*parms.mainLoopStep)*(nposv_t+parms.critDamp*npos)*np.exp(-parms.critDamp*parms.mainLoopStep);
		#npos = npos_t

		# Scale, limit and/or resize control output
		# npos = (npos*300.+400.)*50.
		# npos[npos>650.*50.] = 650.*50
		# npos[npos<150.*50.] = 150.*50.

		##TODO: Send command to robot
		# robot.writeData(npos.tolist(),address="C0005")

		# Initiate MACOP because thread was waiting until now
		dataCollector.isInitd = 1 

		# Collect info (observations)
		#dataCollector.sens_traj[:,i] = dataCollector.sensVal
		#dataCollector.angle_traj[:,i] = npos

		parms.realTime = parms.realTime + parms.mainLoopStep

		i = i+1

		# Calculate delay and wait if needed
		contDelay = (datetime.datetime.now()-tic).microseconds/1000000.
		timeDist = (parms.mainLoopStep-contDelay)
		
		if(timeDist<0):
			print "RC_overrun: ", contDelay

		time.sleep(max(0.,timeDist))

dataCollector.finished = 1
## Stop robot #########################################
print "Control stopping ...."
time.sleep(1)
robot.stop()
print "DONE"
