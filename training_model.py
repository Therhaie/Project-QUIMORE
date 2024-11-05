import tensorflow as tf
from scripts.surrogate_summary  import DeepUQSurrogate  # surrogate_summary.py is the same as surrogate but with a summary method added
import numpy as np
import os

# Load the generated dataset
data_path = os.path.join(os.path.dirname(__file__), "data", "data_rbf_samples.npz")
print(data_path)
data = np.load("data_rbf_samples.npz")
inputs = data['inputs'].reshape(-1, nx * ny)  # Flatten for training
outputs = data['outputs'].reshape(-1, nx * ny)  # Flatten for training

# Initialize and train the surrogate model
D = nx * ny  # Dimensionality of input
L = 3        # Number of encoding layers (example)
d = 128      # Encoding dimension (example)

surrogate = DeepUQSurrogate(D=D, L=L, d=d)

# Create training loop
epochs = 500  # Set epochs for training
batch_size = 32

# Training
for epoch in range(epochs):
    for i in range(0, inputs.shape[0], batch_size):
        x_batch = inputs[i:i+batch_size]
        y_batch = outputs[i:i+batch_size]
        surrogate.sess.run(surrogate.step, feed_dict={surrogate.x: x_batch, surrogate.ytrue: y_batch})

    if epoch % 50 == 0:
        loss_val = surrogate.sess.run(surrogate.loss, feed_dict={surrogate.x: inputs, surrogate.ytrue: outputs})
        print(f"Epoch {epoch}, Loss: {loss_val}")

print("Training complete.")

#To use the trained model for prediction
# Assuming new data 'new_data' with the same shape as 'inputs'
# predicted_output = surrogate.predict(new_data) 
# print("Predicted output:", predicted_output)


