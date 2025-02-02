#!/usr/bin/env python

import argparse
import numpy as np
import os
from examples.spde.solver import SteadyStateHeat2DSolver  # Ensure this path is correct for your setup
import GPy
import torch
from scripts.surrogate_pytorch import DeepUQSurrogate  # Ensure surrogate_pytorch has PyTorch model
import matplotlib.pyplot as plt
import time

'''
This file is used to generate data for the heat equation solver. The data is generated by solving the heat equation
'''

def parse_argument():
    '''
    Part of the code that handle the parsing of the argument
    '''
    parser = argparse.ArgumentParser(description="Run heat equation solver with random field inputs.")
    parser.add_argument('-N', type=int, default=4096*10, help="Number of samples")
    parser.add_argument('-nx', type=int, default=32, help="Number of grid cells in x direction")
    parser.add_argument('-ny', type=int, default=32, help="Number of grid cells in y direction")
    parser.add_argument('-lx', type=float, default=0.02, help="Lengthscale x")
    parser.add_argument('-ly', type=float, default=0.02, help="Lengthscale y")
    parser.add_argument('-var', type=float, default=1.0, help="Variance of random field")
    parser.add_argument('-k', type=str, default="rbf", help="Kernel type for GP model")
    parser.add_argument('-lf', type=int, default=4096, help="Number of samples you want to store per file")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_argument()

    # Define the kernel and other parameters
    # Define the kernel and other parameters
    kernels = {'rbf': GPy.kern.RBF, 'exp': GPy.kern.Exponential, 
            'mat32': GPy.kern.Matern32, 'mat52': GPy.kern.Matern52}

    num_samples = args.N
    lenght_file = args.lf
    nx = args.nx
    ny = args.ny
    lx = args.lx
    ly = args.ly
    variance = args.var
    kernel_type = args.k
    assert kernel_type in kernels
    kernel = kernels[kernel_type](input_dim=2, lengthscale=[lx, ly], variance=variance, ARD=True)

        
    # Initialize solver
    solver = SteadyStateHeat2DSolver(nx=nx, ny=ny)
    cellcenters = solver.mesh.cellCenters.value.T

    # Placeholder arrays to save input/output samples
    inputs = np.zeros((lenght_file, nx, ny))
    outputs = np.zeros((lenght_file, nx, ny))

    # Run the solver and generate data
    for i in range(num_samples):
        print(f"Generating sample {i + 1}/{num_samples}")
        
        # Generate random field sample
        mean = np.zeros((cellcenters.shape[0], 1))
        C = kernel.K(cellcenters) + 1e-6 * np.eye(cellcenters.shape[0])
        L = np.linalg.cholesky(C)
        random_field = mean + np.dot(L, np.random.randn(cellcenters.shape[0], 1))
        sample = np.exp(random_field[:, 0])

        # Set up and solve
        solver.set_coeff(C=sample)
        solver.set_source(source=np.zeros_like(sample))  # Assuming zero source
        solver.solve()
        if (i+1) % lenght_file ==0: 

            path_file = os.path.abspath(__file__)

            # Create a 'data' directory in the same directory as the current script
            data_directory = os.path.join(os.path.dirname(path_file), "data", "file_4096")
            os.makedirs(data_directory, exist_ok=True)

            # Create the full path to the output file
            output_file = os.path.join(data_directory, f"data_{kernel_type}_samples_{num_samples}_nx_{nx}_ny_{ny}_{i//lenght_file}_{lenght_file}.npz")

            # Save the data
            np.savez(output_file, inputs=inputs, outputs=outputs)
            print(f"Data saved to {output_file}")
            print("Output file saved in:", os.path.abspath(output_file))

            inputs = np.zeros((lenght_file, nx, ny))
            outputs = np.zeros((lenght_file, nx, ny))     

        # Save input-output pairs
        inputs[i % lenght_file] = random_field.reshape((nx, ny))
        outputs[i % lenght_file] = solver.phi.value.reshape((nx, ny))
    
    



