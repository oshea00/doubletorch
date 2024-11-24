"""
This module implements a simple neural network using PyTorch to learn the 
mapping of 8-bit binary numbers to their doubled values, modulo 256.

Classes:
    NeuralNet: Defines a neural network with one hidden layer.

Functions:
    number_to_binary(num): Converts a number to its 8-bit binary representation.
    binary_to_number(binary): Converts an 8-bit binary representation to a number.
    generate_dataset(size): Generates a dataset of 8-bit binary numbers and their doubled values.

Hyperparameters:
    learning_rate: The learning rate for the optimizer.
    epochs: The number of training epochs.

Usage:
    The script initializes the dataset, model, loss function, and optimizer.
    It then trains the model using the specified number of epochs and prints the loss every 100 epochs.
    Finally, it tests the model with an example input and prints the result.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define the neural network
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(8, 16)  # Input layer to hidden layer
        self.fc2 = nn.Linear(16, 8)  # Hidden layer to output layer

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))  # Sigmoid activation for hidden layer
        x = torch.sigmoid(self.fc2(x))  # Sigmoid activation for output layer
        return x

# Function to convert number to 8-bit binary representation
def number_to_binary(num):
    return np.array([int(x) for x in f"{num:08b}"], dtype=np.float32)

# Function to convert 8-bit binary to a number
def binary_to_number(binary):
    return int("".join(str(int(b)) for b in binary), 2)

# Generate dataset
def generate_dataset(size=256):
    inputs = []
    outputs = []
    for i in range(size):
        input_binary = number_to_binary(i % 256)
        output_binary = number_to_binary((i * 2) % 256)  # Multiply by 2, mod 256
        inputs.append(input_binary)
        outputs.append(output_binary)
    return np.array(inputs), np.array(outputs)

# Hyperparameters
learning_rate = 0.01
epochs = 100000

# Initialize the dataset
inputs, outputs = generate_dataset()
inputs = torch.tensor(inputs)
outputs = torch.tensor(outputs)

# Initialize the model, loss function, and optimizer
model = NeuralNet()
criterion = nn.MSELoss()  # Mean Squared Error loss
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    # Forward pass
    predictions = model(inputs)
    loss = criterion(predictions, outputs)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print progress
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

# Test the model
test_input = number_to_binary(45)  # Example: 15 in binary is 00001111
test_input_tensor = torch.tensor(test_input).unsqueeze(0)  # Add batch dimension
with torch.no_grad():
    test_output = model(test_input_tensor).squeeze(0).round()  # Round to nearest binary
test_output_number = binary_to_number(test_output.numpy())

print(f"Input: {binary_to_number(test_input)} (binary: {test_input})")
print(f"Output: {test_output_number} (binary: {test_output.numpy()})")
