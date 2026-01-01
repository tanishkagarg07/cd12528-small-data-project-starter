# Starter code for Part 1 of the Small Data Solutions Project
# 

#Set up image data for train and test

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms 
from TrainModel import train_model
from TestModel import test_model
from torchvision import models


# use this mean and sd from torchvision transform documentation
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# -----------------------------
# Set up Transforms
# -----------------------------

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# -----------------------------
# Set up DataLoaders
# -----------------------------

batch_size = 10
num_workers = 4

train_dataset = datasets.ImageFolder("data/train", transform=train_transform)
val_dataset   = datasets.ImageFolder("data/val", transform=val_transform)
test_dataset  = datasets.ImageFolder("data/test", transform=test_transform)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
)

# class names
class_names = train_dataset.classes

# -----------------------------
# VGG16 Transfer Learning
# -----------------------------

model = models.vgg16(pretrained=True)

# Freeze feature layers
for param in model.features.parameters():
    param.requires_grad = False

# Replace classifier
num_features = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_features, 3)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# -----------------------------
# Training Hyperparameters
# -----------------------------

num_epochs = 5
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
train_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

# -----------------------------
# Train and Test
# -----------------------------

def main():
    trained_model = train_model(
        model,
        criterion,
        optimizer,
        train_lr_scheduler,
        train_loader,
        val_loader,
        num_epochs=num_epochs
    )

    test_model(test_loader, trained_model, class_names)

if __name__ == '__main__':
    main()
    print("done")
