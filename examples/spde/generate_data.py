# #!/usr/bin/env python


# # path = os.path.dirname(os.path.abspath(__file__))
# # path = os.path.dirname(path)
# # print(path)

# from solver import SteadyStateHeat2DSolver 
# #from heateqsolver import SteadyStateHeat2DSolver

# import argparse 
# import sys
# import os
# import numpy as np
# import os
# import GPy
# import matplotlib.pyplot as plt
# from fipy import *
# from scipy.interpolate import griddata
# from pdb import set_trace as keyboard
# import time

# #parse command line arguments 
# parser = argparse.ArgumentParser()
# parser.add_argument('-N', dest = 'N', type = int, 
#                     default = 1000, help  = 'Number of samples of the random inputs')
# parser.add_argument('-nx', dest = 'nx', type =  int, 
#                     default = 32, help = 'Number of FV cells in the x direction.')
# parser.add_argument('-ny', dest = 'ny', type = int, 
#                     default = 32, help = 'Number of FV cells in the y direction.')
# parser.add_argument('-lx', dest = 'lx', type = float, 
#                     default = 0.02, help = 'Lengthscale of the random field along the x direction.')
# parser.add_argument('-ly', dest = 'ly', type = float, 
#                     default = 0.02, help = 'Lengthscale of the random field along the y direction.')
# parser.add_argument('-var', dest = 'var', type = float, 
#                     default = 1., help = 'Signal strength (variance) of the random field.')
# parser.add_argument('-k', dest = 'k', type = str, 
#                     default = 'rbf', help = 'Type of covariance kernel (rbf, exp, mat32 or mat52)')
# args = parser.parse_args()
# kernels = {'rbf':GPy.kern.RBF, 'exp':GPy.kern.Exponential, 
#            'mat32':GPy.kern.Matern32, 'mat52':GPy.kern.Matern52} # dictionary of kernels

# num_samples = args.N
# nx = args.nx
# ny = args.ny
# ellx = args.lx
# elly = args.ly
# variance = args.var 
# k_ = args.k
# assert k_ in kernels.keys()
# kern = kernels[k_]

# #define a mean function
# def mean(x):
#     """
#     Mean of the permeability field. 

#     m(x) = 0. 
#     """
#     n = x.shape[0]
#     return np.zeros((n, 1))

# def q(x):
#     n = x.shape[0]
#     s = np.zeros((n))
#     return s

# #data directory
# cwd = os.getcwd()
# data='data'
# datadir = os.path.abspath(os.path.join(cwd, data))
# if not os.path.exists(datadir):
#     os.makedirs(datadir)

# #GPy kernel
# k=kern(input_dim = 2,
#        lengthscale = [ellx, elly],
#        variance = variance,
#        ARD = True)

# ##define the solver object
# solver = SteadyStateHeat2DSolver(nx=nx, ny=ny)
# cellcenters = solver.mesh.cellCenters.value.T
# np.save(os.path.join(datadir, 'cellcenters.npy'), cellcenters)

# #get source field 
# source = q(cellcenters)

# #get covariance matrix and compute its Cholesky decomposition
# m=mean(cellcenters)
# C=k.K(cellcenters) + 1e-6*np.eye(cellcenters.shape[0])
# L=np.linalg.cholesky(C)

# #define matrices to save results 
# inputs = np.zeros((num_samples, nx, ny))
# outputs = np.zeros((num_samples, nx, ny))

# start = time.time()
# #generate samples
# for i in xrange(num_samples):
#     #display
#     if (i+1)%100 == 0:
#         print ("Generating sample "+str(i+1))
    
#     #generate a sample of the random field input
#     z =np.random.randn(cellcenters.shape[0], 1)
#     f = m + np.dot(L, z)   
#     sample = np.exp(f[:, 0])

#     #solve the PDE  
#     solver.set_coeff(C=sample)   #set diffusion coefficient. 
#     solver.set_source(source=source)   #set source term. 
#     solver.solve()  

#     #save data 
#     inputs[i] = f.reshape((nx, ny))
#     outputs[i] = solver.phi.value.reshape((nx, ny))

# #end timer
# finish = time.time() - start
# print ("Time (sec) to generate "+str(num_samples)+" samples : " +str(finish))

# #save data
# datafile = k_+"_lx_"+str(ellx).replace('.', '')+\
#             "_ly_"+str(elly).replace('.', '')+\
#             "_v_"+str(variance).replace('.', '')+".npz"

# np.savez(os.path.join(datadir, datafile), inputs=inputs,\
#                                           outputs=outputs,\
#                                           nx=nx, ny=ny, lx=ellx, ly=elly,\
#                                           var=variance)


######################### Generating data with time evaluation #########################
#!/usr/bin/env python

import argparse 
import sys
import os
import numpy as np
import GPy
import matplotlib.pyplot as plt
from fipy import *
from scipy.interpolate import griddata
from pdb import set_trace as keyboard
import time
from solver import SteadyStateHeat2DSolver 

# Parse command line arguments 
parser = argparse.ArgumentParser()
parser.add_argument('-N', dest='N', type=int, 
                    default=2048, help='Number of samples of the random inputs')
parser.add_argument('-nx', dest='nx', type=int, 
                    default=32, help='Number of FV cells in the x direction.')
parser.add_argument('-ny', dest='ny', type=int, 
                    default=32, help='Number of FV cells in the y direction.')
parser.add_argument('-lx', dest='lx', type=float, 
                    default=0.02, help='Lengthscale of the random field along the x direction.')
parser.add_argument('-ly', dest='ly', type=float, 
                    default=0.02, help='Lengthscale of the random field along the y direction.')
parser.add_argument('-var', dest='var', type=float, 
                    default=1., help='Signal strength (variance) of the random field.')
parser.add_argument('-k', dest='k', type=str, 
                    default='rbf', help='Type of covariance kernel (rbf, exp, mat32 or mat52)')
args = parser.parse_args()
kernels = {'rbf': GPy.kern.RBF, 'exp': GPy.kern.Exponential, 
           'mat32': GPy.kern.Matern32, 'mat52': GPy.kern.Matern52}  # Dictionary of kernels

num_samples = args.N
nx = args.nx
ny = args.ny
ellx = args.lx
elly = args.ly
variance = args.var 
k_ = args.k
assert k_ in kernels.keys()
kern = kernels[k_]

def mean(x):
    """
    Mean of the permeability field. 

    m(x) = 0. 
    """
    n = x.shape[0]
    return np.zeros((n, 1))

def q(x):
    n = x.shape[0]
    s = np.zeros((n))
    return s

# Data directory
cwd = os.getcwd()
data = 'data'
datadir = os.path.abspath(os.path.join(cwd, data,'test'))
if not os.path.exists(datadir):
    os.makedirs(datadir)

# GPy kernel    
k = kern(input_dim=2,
         lengthscale=[ellx, elly],
         variance=variance,
         ARD=True)

# Define the solver object
solver = SteadyStateHeat2DSolver(nx=nx, ny=ny)
cellcenters = solver.mesh.cellCenters.value.T
np.save(os.path.join(datadir, 'cellcenters.npy'), cellcenters)

# Get source field 
source = q(cellcenters)

# Get covariance matrix and compute its Cholesky decomposition
m = mean(cellcenters)
C = k.K(cellcenters) + 1e-6 * np.eye(cellcenters.shape[0])
L = np.linalg.cholesky(C)

# Define matrices to save results 
inputs = np.zeros((num_samples, nx, ny))
outputs = np.zeros((num_samples, nx, ny))

# Start timer for total process
start_total = time.time()

# Generate samples
for i in range(num_samples):
    # Display progress
    if (i + 1) % 100 == 0:
        print(f"Generating sample {i + 1}")

    # Start timer for each sample
    start_sample = time.time()

    # Generate a sample of the random field input
    z = np.random.randn(cellcenters.shape[0], 1)
    f = m + np.dot(L, z)
    sample = np.exp(f[:, 0])

    # Solve the PDE
    solver.set_coeff(C=sample)  # Set diffusion coefficient. 
    solver.set_source(source=source)  # Set source term. 
    solver.solve()

    # Save data 
    inputs[i] = f.reshape((nx, ny))
    outputs[i] = solver.phi.value.reshape((nx, ny))

    # End timer for each sample
    elapsed_sample = time.time() - start_sample
    print(f"Sample {i + 1} solved in {elapsed_sample:.2f} seconds")

# End timer for total process
elapsed_total = time.time() - start_total
print(f"Time (sec) to generate {num_samples} samples: {elapsed_total:.2f}")

# Save data
datafile = f"{k_}_lx_{str(ellx).replace('.', '')}_ly_{str(elly).replace('.', '')}_v_{str(variance).replace('.', '')}.npz"
np.savez(os.path.join(datadir, datafile), inputs=inputs,
         outputs=outputs,
         nx=nx, ny=ny, lx=ellx, ly=elly,
         var=variance)


