#!/usr/bin/env python


# path = os.path.dirname(os.path.abspath(__file__))
# path = os.path.dirname(path)
# print(path)

from solver_with_geometric_conditions import SteadyStateHeat2DSolver 

import argparse 
import sys
import os
import numpy as np
import os
import GPy
import matplotlib.pyplot as plt
from fipy import *
from scipy.interpolate import griddata
from pdb import set_trace as keyboard
import time

#parse command line arguments 
parser = argparse.ArgumentParser()
parser.add_argument('-N', dest = 'N', type = int, 
                    default = 10, help  = 'Number of samples of the random inputs')
parser.add_argument('-nx', dest = 'nx', type =  int, 
                    default = 32, help = 'Number of FV cells in the x direction.')
parser.add_argument('-ny', dest = 'ny', type = int, 
                    default = 32, help = 'Number of FV cells in the y direction.')
parser.add_argument('-lx', dest = 'lx', type = float, 
                    default = 0.02, help = 'Lengthscale of the random field along the x direction.')
parser.add_argument('-ly', dest = 'ly', type = float, 
                    default = 0.02, help = 'Lengthscale of the random field along the y direction.')
parser.add_argument('-var', dest = 'var', type = float, 
                    default = 1., help = 'Signal strength (variance) of the random field.')
parser.add_argument('-k', dest = 'k', type = str, 
                    default = 'rbf', help = 'Type of covariance kernel (rbf, exp, mat32 or mat52)')
parser.add_argument('-geo', type = str, default = 'vertical', help = 'Type of separation wanted (vertical, horizontal)')
args = parser.parse_args()
kernels = {'rbf':GPy.kern.RBF, 'exp':GPy.kern.Exponential, 
           'mat32':GPy.kern.Matern32, 'mat52':GPy.kern.Matern52} # dictionary of kernels

num_samples = args.N
nx = args.nx
ny = args.ny
ellx = args.lx
elly = args.ly
variance = args.var 
separation = args.geo
k_ = args.k
assert k_ in kernels.keys()
kern = kernels[k_]

#define a mean function
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

#data directory
cwd = os.getcwd()
data='data'
store_directory = 'geometric_shape' 
datadir = os.path.abspath(os.path.join(cwd, data, store_directory, separation))
if not os.path.exists(datadir):
    os.makedirs(datadir)

#GPy kernel
k=kern(input_dim = 2,
       lengthscale = [ellx, elly],
       variance = variance,
       ARD = True)

##define the solver object

solver = SteadyStateHeat2DSolver(
    nx=32, ny=32,
    value_left=1., value_right=0.,
    value_top=1., value_bottom=0.,
    division_axis=separation  # Change to "horizontal" for horizontal split
)
cellcenters = solver.mesh.cellCenters.value.T
np.save(os.path.join(datadir, 'cellcenters.npy'), cellcenters)

#get source field 
source = q(cellcenters)

#get covariance matrix and compute its Cholesky decomposition
m=mean(cellcenters)
C=k.K(cellcenters) + 1e-6*np.eye(cellcenters.shape[0])
L=np.linalg.cholesky(C)

#define matrices to save results 
inputs = np.zeros((num_samples, nx, ny))
outputs = np.zeros((num_samples, nx, ny))

start = time.time()
#generate samples
for i in range(num_samples):
    #display
    if (i+1)%100 == 0:
        print ("Generating sample "+str(i+1))
    
    #generate a sample of the random field input
    z =np.random.randn(cellcenters.shape[0], 1)
    print("value of cellcenter", z.shape)
    print("value of L", L.shape)
    f = m + np.dot(L, z)   
    sample = np.exp(f[:, 0])
    print("value of sample", sample.shape)
    
    ########### modification of the value inside the geometry

    # modification of the value of f

    f_like = np.zeros_like(f)
    print("value of f", f.shape)

    for e in range(len(f_like)):
        if e < len(f_like)/2:
            f_like[e] = 0.2
        else:
            f_like[e] = 0.8    

    #modification of C
    C = np.zeros_like(f[:, 0])  # Ensure same size as f[:, 0]

    print("value of C", C.shape[0])

    for j in range(len(C)):
        if j < len(C)/2:
            C[j] = 0.2
        else:
            C[j] = 0.8

    # Conversion from vertical to horizontal
    if separation == 'vertical':
        C = C.reshape((solver.nx, solver.ny)).T.flatten()

    # Plot of the generated C
    C_plot = C.reshape((solver.nx, solver.ny))
    plt.figure()
    plt.imshow(C_plot.T, origin='lower', cmap='viridis')
    plt.colorbar(label='$T(x)$')
    plt.title('Initial value of C')


    #solve the PDE  
    # solver.set_coeff(C=sample)   #set diffusion coefficient. 

    solver.set_coeff(C)
    solver.set_source(source=source)   #set source term. 
    solver.solve()  
    
    print("value of f", f.shape)
    print("shape of inputs", inputs.shape)
    print("value of i ", i)

    # visualization of f (Input)
    f_plot = f_like.reshape((solver.nx, solver.ny))
    plt.figure()
    plt.imshow(f_plot.T, origin='lower', cmap='viridis')
    plt.colorbar(label='$f$')
    plt.title('Input Field f value')

    # visualization of phi (Output associated)
    phi_plot = solver.phi.value.reshape((solver.nx, solver.ny))
    plt.figure()
    plt.imshow(phi_plot.T, origin='lower', cmap='viridis')
    plt.colorbar(label='$f$')
    plt.title('Output Field Phi')
    plt.show()

    #save data 
    inputs[i] = f.reshape((nx, ny))
    outputs[i] = solver.phi.value.reshape((nx, ny))

#end timer
finish = time.time() - start
print ("Time (sec) to generate "+str(num_samples)+" samples : " +str(finish))

# Visualize the input data
plt.imshow(solver.C.value.reshape((solver.nx, solver.ny)), origin='lower', cmap='viridis')
plt.colorbar(label='$C(x)$')
plt.title('input C')




# Visualize the generated data
plt.figure()
plt.imshow(solver.phi.value.reshape((solver.nx, solver.ny)), origin='lower', cmap='viridis')
plt.colorbar(label='$T(x)$')
plt.title('Initial Phi Field')
plt.show()

#save data
datafile = k_+"_lx_"+str(ellx).replace('.', '')+\
            "_ly_"+str(elly).replace('.', '')+\
            "_v_"+str(variance).replace('.', '')+".npz"

np.savez(os.path.join(datadir, datafile), inputs=inputs,\
                                          outputs=outputs,\
                                          nx=nx, ny=ny, lx=ellx, ly=elly,\
                                          var=variance)