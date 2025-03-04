import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import torchvision
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Grayscale, Normalize, Compose
from torchvision.utils import make_grid
import pennylane as qml
from pennylane import numpy as np
import torch.optim as optim
import os
import matplotlib.pyplot as plt
%matplotlib inline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformation for preprocessing
transform = Compose([
    Grayscale(num_output_channels=1),  # Convert to grayscale
    ToTensor(),  # Convert image to PyTorch tensor
    Normalize((0.5,), (0.5,))  # Normalize to range [-1, 1]
])


# Load CIFAR-10 dataset
train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)

batch_size = 256


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class QuanvolutionalLayer(nn.Module):
    def __init__(self, filter_size=2, n_layers=1, stride=2, padding=0):
        super(QuanvolutionalLayer, self).__init__()
        self.filter_size = filter_size
        self.n_layers = n_layers
        self.stride = stride
        self.padding = padding

        # Define a quantum device
        self.dev = qml.device("default.qubit", wires=filter_size**2)

        # Generate random quantum circuit parameters
        self.rand_params = np.random.uniform(0, 2*np.pi, size=(n_layers, filter_size**2))

    def quantum_circuit(self, inputs):
        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs):
            for i in range(self.filter_size**2):
                qml.RY(inputs[i] * np.pi, wires=i)

            qml.templates.RandomLayers(self.rand_params, wires=list(range(self.filter_size**2)))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.filter_size**2)]

        return torch.tensor(circuit(inputs), dtype=torch.float32).to(inputs.device)  # Move to correct device

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        if self.padding != 0:
            x = torch.nn.functional.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='constant', value=0)

        output_size = ((height + 2 * self.padding - self.filter_size) // self.stride) + 1
        output = torch.zeros(batch_size, self.filter_size**2, output_size, output_size, device=x.device)  # Move to same device

        for i in range(output_size):
            for j in range(output_size):
                row_start = i * self.stride
                col_start = j * self.stride
                patch = x[:, :, row_start:row_start+self.filter_size, col_start:col_start+self.filter_size]
                patch = patch.contiguous().reshape(batch_size, -1)  

                q_results = torch.stack([self.quantum_circuit(p) for p in patch])

                output[:, :, i, j] = q_results  # Store at corrected index

        return output
    
    
    
class Quanvolutional_Convolutional_NeuralNetwork(nn.Module):
    def __init__(self):
        super(Quanvolutional_Convolutional_NeuralNetwork, self).__init__()
        self.quanv1 = QuanvolutionalLayer(filter_size=3, n_layers=1, stride=2, padding=1)
        self.conv1 = nn.Conv2d(in_channels=9, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)  
        self.fc2 = nn.Linear(128, 10)  # Final classification (CIFAR-10 has 10 classes)

    def forward(self, x):
        x = self.quanv1(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)          
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = F.relu(x)  
        x = self.fc2(x)
        
        return x 
    
    
model = Quanvolutional_Convolutional_NeuralNetwork()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)



def train(model, train_loader, criterion, optimizer, epochs=30):
    model.train()  # Set model to training mode

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)  # Move to GPU if available

            optimizer.zero_grad()  # Reset gradients

            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)  # Get class with highest probability
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        # Print epoch stats
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {total_loss/len(train_loader):.4f} | Accuracy: {correct/total:.4f}")

    print("Training complete!")


def test(model, test_loader, criterion):
    model.eval()  # Set model to evaluation mode
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():  # No gradient calculation needed
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            test_loss += criterion(outputs, labels).item()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    print(f"Test Loss: {test_loss/len(test_loader):.4f} | Test Accuracy: {correct/total:.4f}")



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
model.to(device)  # Move model to GPU

# Train the model
train(model, train_loader, criterion, optimizer, epochs=30)


# Evaluate the model
test(model, test_loader, criterion)


model_path = "Models"
os.makedirs(model_path, exist_ok=True)

final_model = f"QC__model_cifr_n_1_p_1_s_1.pth"
final_model_path = os.path.join(model_path, final_model)

torch.save(model.state_dict(), final_model_path)