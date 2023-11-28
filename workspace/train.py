"""
Usage: 
  train.py <model>

Arguments:
    <model>     Must be one of: [googlenet, densenet, resnet]
"""


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets
from torch.utils.tensorboard import SummaryWriter
import time
from datetime import datetime
from docopt import docopt
import os
from models import *

args=docopt(__doc__)

# Definir época de entrenamiento
def train_one_epoch(epoch_index):
    running_loss = 0.
    last_loss = 0.
    total_loss = 0.
    batch_counter = 0
    for i, data in enumerate(train_loader):
        # Every data instance is an input + label pair
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # Zero your gradients for every batch!
        optimizer.zero_grad()
        # Make predictions for this batch
        outputs = model(inputs)        
        # Compute the loss and its gradients
        loss = criterion(outputs, labels)
        loss.backward()
        # Adjust learning weights
        optimizer.step()
        # Gather data and report
        total_loss += loss.item()
        running_loss += loss.item()
        if i % 100 == 99:
            last_loss = running_loss / 100 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            #tb_x = epoch_index * len(train_loader) + i + 1
            #tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.
        batch_counter += 1
    total_loss /= batch_counter
    return total_loss
# Definir las transformaciones para el conjunto de datos
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Definir hiperparámetros
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
project_name = 'birds525'
batch_size = 32
learning_rate = 0.001
num_epochs = 10
model_type = args['<model>']

# Create project
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
dir_name = 'checkpoints/train_' + model_type + '_' + project_name + '_' + timestamp
os.makedirs(dir_name)

# Cargar dataset
train_data = datasets.ImageFolder('dataset/train', transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid_data = datasets.ImageFolder('dataset/valid', transform=transform)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=True)
# Reportar split sizes
print('Training set has {} instances'.format(len(train_data)))
print('Validation set has {} instances'.format(len(valid_data)))

# Inicializar el modelo

if args['<model>'] == 'googlenet':
    model = GoogleNet(num_classes=len(train_data.classes)).to(device)
elif args['<model>'] == 'densenet':
    model = DenseNet(num_classes=len(train_data.classes)).to(device)
elif args['<model>'] == 'resnet':
    model = ResNet(num_classes=len(train_data.classes)).to(device)
    
# Definir la función de pérdida y el optimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Entrenar el modelo

best_vloss = 1_000_000.

# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
""" writer = SummaryWriter('checkpoints/runs/birds_trainer_{}'.format(timestamp)) """
epoch_number = 0
start_time = time.time()
for epoch in range(num_epochs):
    print('EPOCH {}:'.format(epoch_number + 1))    
    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    start_epoch = time.time()
    avg_loss = train_one_epoch(epoch_number)
    end_epoch = time.time()
    print(f'  Epoca finalizada en {end_epoch - start_epoch} segundos')
    # We don't need gradients on to do reporting
    model.train(False)
    running_vloss = 0.0
    with torch.no_grad():
        for i, vdata in enumerate(valid_loader):
            vinputs, vlabels = vdata
            vinputs, vlabels = vinputs.to(device), vlabels.to(device)
            voutputs = model(vinputs)
            vloss = criterion(voutputs, vlabels)
            running_vloss += vloss
    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = dir_name + '/model_{}_{}.pth'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)
    # Next epoch
    epoch_number += 1

end_time = time.time()
print(f'Tiempo de entrenamiento: {end_time - start_time} segundos')
torch.save(model.state_dict(), dir_name + '/final_model.pth')

