import numpy as np
import scipy
import scipy.signal

# Create parameter container
class Parameters(object):
    pass

parms = Parameters()

# MACOP PARAMETERS
parms.num_models = 3.       # Number of controllers in MACOP
parms.eta_g = 0.0002        # Model gain
parms.eta_s = 0.00008       # Model damping
parms.NinSoftmax = 3.       # Number of parameters to take into account in softmax (see MACOP.py and multimodel.py)

parms.Pscaling = 100.       # RLS Scaling
parms.OutScaling = 20.      # Scale the desired output to modify precision and clamping time

# RESERVOIR PARAMETERS (Same for every controller)
parms.N = 300.              # Number of neurons
parms.sNin = 3              # Number of different inputs
parms.Nout= 3               # Number of outputs
parms.Nin= (parms.sNin)*2.+parms.Nout # Actual number of inputs (time shifted and feedback included)
parms.inp_sc=.3*np.ones((1,parms.Nin))  # All inputs have same input scaling. However it is possible to customize this.
                                        # parms.inp_sc[:,[2,6]] = 0.1;

parms.outpLimits = np.array([[-1.,-1.,-1.],   # Under limit of MACOP output for every output
                           [ 1., 1., 1.]])  # Upper limit of MACOP output for every output

parms.spectr_rad=1.         # Spectral radius
parms.leak_rate=1.          # Leak rate
parms.inpBias = 0.1         #.Input bias

parms.delta = 3             # Delta for control trick (time shift of input)
parms.alpha = 0.9999        # Forgetting factor RLS, in range [0.9999, 0.999999]

# GENERAL PARAMETERS (THREADS, etc)
parms.maxTime = 10000.
parms.sensStep = 0.005      # Time step of sensor thread (large enough to have no over runs) [ms]
parms.controllerStep = 0.1  # Time step of control (MACOP) thread (large enough to have no over runs) [ms]
parms.mainLoopStep = 0.015  #.Time step of actual robot control

# Filter parameters (if used)
parms.critDamp = 0.8
parms.lpass = 0.2

parms.t1 = parms.maxTime
parms.dt = parms.controllerStep

##TODO: Target signal that needs to be tracked
#parms.y_d = 
