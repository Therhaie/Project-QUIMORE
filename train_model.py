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
This file is to train the surrogate model using the generated data'''

# def parse_argument():
#     '''
#     Part of the code that handle the parsing of the argument
#     '''
#     parser = argparse.ArgumentParser(description="Run the training of the model.")
#     parser.add_argument('-D', type=str, required=True, help="Link to the data file") 
#     args = parser.parse_args()
#     return args

# Extract important parameters from the name of the dataset
def extract_parameters(name):
    '''
    This function is used to extract the parameters from the name of the dataset
    '''
    name = name.split('/')[-1].split('.')[0]
    name = name.split('_')
    #print(name)
    N = int(name[-5])
    nx = int(name[-3])
    ny = int(name[-1])

    return N, nx, ny

if __name__ == '__main__':
    # args = parse_argument()
    # data_path = args.D
    #data_path = "/mnt/c/python_workspace/3A/ProjetSOIA/deep-uq-paper-master/data/data_rbf_samples_500_nx_32_ny_32.npz"
    data_path = rf"C:\python_workspace\3A\ProjetSOIA\deep-uq-paper-master\data\data_rbf_samples_10000_nx_32_ny_32.npz"
    N, nx, ny = extract_parameters(data_path)
    print(f"Number of samples: {N}, nx: {nx}, ny: {ny}")

    # Load the generated dataset

    # data_path = os.path.join(os.path.dirname(__file__), "data", "data_rbf_samples.npz")
    # print(data_path)
    # data = np.load("data_rbf_samples.npz")

    data=np.load(data_path)
    # Reshape inputs and outputs for training (flat vectors)
    inputs = data['inputs'].reshape(-1, nx * ny)
    outputs = data['outputs'].reshape(-1, nx * ny)

    # Convert to PyTorch tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    inputs = torch.from_numpy(inputs).float().to(device)
    outputs = torch.from_numpy(outputs).float().to(device)

    # Split the data into training and test sets (80% train, 20% test)
    num_samples = inputs.size(0)
    train_size = int(0.8 * num_samples)
    test_size = num_samples - train_size

    # Split the dataset
    train_inputs, test_inputs = torch.split(inputs, [train_size, test_size])
    train_outputs, test_outputs = torch.split(outputs, [train_size, test_size])

    # Initialize and train the surrogate model
    D = nx * ny  # Dimensionality of input
    print('value of D= nx*ny',D)
    L = 3        # Number of encoding layers
    d = 128      # Encoding dimension

    surrogate = DeepUQSurrogate(D=D, L=L, d=d, output_size=D).to(device) # be careful with the output_size shape of the 

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(surrogate.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    # Training parameters
    epochs = 100
    batch_size = 16 # 32 not enough memory
    loss_values = []

    # Track training time
    start_time = time.time()

    # Training loop with loss tracking for plotting
    for epoch in range(epochs):
        for i in range(0, inputs.size(0), batch_size):
            x_batch = inputs[i:i + batch_size]
            y_batch = outputs[i:i + batch_size]
            
            # Forward pass
            y_pred = surrogate(x_batch)
            
            # Compute loss
            loss = criterion(y_pred, y_batch)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Log and measure time every 20 epochs
        if (epoch + 1) % 20 == 0:
            end_time = time.time()
            elapsed_time = end_time - start_time
            loss_val = criterion(surrogate(inputs), outputs).item()
            
            # Record loss and reset start time
            print(f"Epoch {epoch+1}, Loss: {loss_val}, Time for last 20 epochs: {elapsed_time:.2f} seconds")
            loss_values.append(loss_val)
            start_time = time.time()  # Reset timer for the next 20 epochs

    print("Training complete.")

    # Define the path where the model will be saved
    model_path = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(model_path, exist_ok=True)
    model_path = os.path.join(model_path, "trained_model.pth") 

    # Plot the evolution of the loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(0, epochs, 20), loss_values, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Evolution of Training Loss")
    plt.legend()
    plt.show()

    # Need to modify the name of the output model to be able to extract importante information if needed to be used / reloaed later

    # Save the model
    torch.save(surrogate.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Plot 5 random images from the test set: ground truth vs predictions
    with torch.no_grad():
        # Get predictions for the test set
        test_predictions = surrogate(test_inputs).reshape(-1, nx, ny)
        test_ground_truth = test_outputs.reshape(-1, nx, ny)

        # Select 5 random samples
        indices = np.random.choice(test_predictions.size(0), 5, replace=False)

        # Plot ground truth vs predictions for 5 random samples
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        for idx, i in enumerate(indices):
            # Ground truth
            axes[0, idx].imshow(test_ground_truth[i].cpu().numpy(), cmap="viridis")
            axes[0, idx].set_title(f"Ground Truth {i+1}")
            axes[0, idx].axis("off")
            
            # Prediction
            axes[1, idx].imshow(test_predictions[i].cpu().numpy(), cmap="viridis")
            axes[1, idx].set_title(f"Prediction {i+1}")
            axes[1, idx].axis("off")

        plt.suptitle("Ground Truth vs Predictions (Test Set)")
        plt.tight_layout()
        plt.show()
