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
from torchsummary import summary
from torch.optim.lr_scheduler import CosineAnnealingLR

#This file is used to train the surrogate model with the generated data on the file in the diretory file_split, to evaluate the model, run evaluate_model.py

# Extract important parameters from the name of the dataset
def extract_parameters(name):
    name = name.split('/')[-1].split('.')[0]
    name = name.split('_')
    N = int(name[-5])
    nx = int(name[-3])
    ny = int(name[-1])
    return N, nx, ny

if __name__ == '__main__':
    directory_path = rf"C:\python_workspace\3A\ProjetSOIA\deep-uq-paper-master\data\file_4096"

    # Initialize and train the surrogate model
    nx = ny = 32
    D = nx * ny
    print('value of D= nx*ny', D)
    L = 3        # Number of encoding layers
    d = 128      # Encoding dimension

    # Training parameters
    epochs = 80
    T_max = epochs
    batch_size = 512
    loss_values = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    surrogate = DeepUQSurrogate(D=D, L=L, d=d, output_size=D).to(device)

    # Define optimizer and loss function
    # optimizer = torch.optim.Adam(surrogate.parameters(), lr=1e-3)

    optimizer = torch.optim.Adam(surrogate.parameters(), lr=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=1e-5)
    criterion = torch.nn.MSELoss()

    # Print the model summary
    summary(surrogate, input_size=(D,))


    # Iterate over all files in the directory
    for filename in os.listdir(directory_path):
        data_path = os.path.join(directory_path, filename)

        model_path = os.path.join(os.path.dirname(__file__), "models", "iteration_model_4096_test.pth")
        if os.path.exists(model_path):
            surrogate.load_state_dict(torch.load(model_path))
            print(f"Model loaded from {model_path}")
            print(f"learning from {data_path}")


        # Load the generated dataset
        data = np.load(data_path)
        inputs = data['inputs'].reshape(-1, nx * ny)
        outputs = data['outputs'].reshape(-1, nx * ny)

        inputs = torch.from_numpy(inputs).float()
        outputs = torch.from_numpy(outputs).float()

        # Split the data into training and test sets (80% train, 20% test)
        num_samples = inputs.size(0)
        train_size = int(0.8 * num_samples)
        test_size = num_samples - train_size

        # Split the dataset
        train_inputs, test_inputs = torch.split(inputs, [train_size, test_size])
        train_outputs, test_outputs = torch.split(outputs, [train_size, test_size])        

        # Reinitialize optimizer and scheduler for each file
        optimizer = torch.optim.Adam(surrogate.parameters(), lr=1e-3)
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=1e-5)

        start_time = time.time()
        print('start training')

        # Track training time
        start_time = time.time()    
        # Training loop with loss tracking for plotting
        for epoch in range(epochs):
            for i in range(0, train_inputs.size(0), batch_size):
                # Move batch data to GPU only as needed
                x_batch = train_inputs[i:i + batch_size].to(device)
                y_batch = train_outputs[i:i + batch_size].to(device)

                # Forward pass
                y_pred = surrogate(x_batch)

                # Compute loss
                loss = criterion(y_pred, y_batch)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Free up memory for each batch
                del x_batch, y_batch, y_pred
                torch.cuda.empty_cache()

            # Log and measure time every 5 epochs
            if (epoch + 1) % 5 == 0:
                end_time = time.time()
                elapsed_time = end_time - start_time
                loss_val = criterion(surrogate(train_inputs.to(device)), train_outputs.to(device)).item()

                # Record loss and reset start time
                print(f"Epoch {epoch+1}, Loss: {loss_val}, Time for last 5 epochs: {elapsed_time:.2f} seconds")
                loss_values.append(loss_val)
                start_time = time.time()

        print("Training complete.")
        # Define the path where the model will be saved
        directory_model_path = os.path.dirname(model_path)
        os.makedirs(directory_model_path, exist_ok=True)

        # Save the model
        torch.save(surrogate.state_dict(), model_path)
        print(f"Model saved to {model_path}")