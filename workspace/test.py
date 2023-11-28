"""
Usage: 
  test.py <model_arch> <weights_path>

Arguments:
    <model_arch>        Must be one of: [googlenet, densenet, resnet]
    <weights_path>      Weights of the model architecture
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
import torchvision
from torch.utils.data import DataLoader
from docopt import docopt
from models import *

args=docopt(__doc__)

# Hyperparamenters
batch_size = 4

# Dataset transformation
transform = transforms.Compose([
    transforms.Resize(256),         
    transforms.CenterCrop(224),     
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the test dataset (replace with your dataset)
test_data = datasets.ImageFolder('dataset2', transform=transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

# Load the pre-trained model from a directory
model_path = args['<weights_path>']

# Load model
if args['<model_arch>'] == 'googlenet':
    model = GoogleNet(525)
elif args['<model_arch>'] == 'densenet':
    model = DenseNet(525)
elif args['<model_arch>'] == 'resnet':
    model = ResNet(525)

model.load_state_dict(torch.load(model_path))
#model.summary()
# Set the model to evaluation mode
model.eval()

# Set GPU support
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Evaluation
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Accuracy on the test dataset: {accuracy:.2f}%")
