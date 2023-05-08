import pandas as pd
from PIL import Image
import numpy as np
import os
import regex as re
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import DatasetFolder
from sklearn.model_selection import train_test_split
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import time
from tqdm import tqdm
import sys

class ImageDataFrameDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_path = os.path.join(self.root_dir, row['filename'])
        image = Image.open(image_path)
        
        if self.transform:
            image = self.transform(image)
        
        label = row['gender']        
        return idx, image, label


def main():

    train_name = sys.argv[1]

    image_data = []
    counter = 0

    # loop over all the files in the folder
    for filename in os.listdir(train_name):    
        # open the image file and convert it to a NumPy array
        if (filename == ".DS_Store"):
            continue
        with Image.open(os.path.join(train_name, filename)) as img:
        # img_array = np.asarray(img)
            
            parts = filename.split("_") 
            age = int(parts[0])
            gender = int(parts[1])
            ethnicity = int(parts[2])

            image_data.append([filename, img, age, gender, ethnicity])
            counter += 1
            # if counter % 10000 == 0:
            #     print(counter)

    # create a DataFrame from the image data
    train_image_df = pd.DataFrame(image_data, columns=["filename", "image", "age", "gender", "ethnicity"])


    image_data = []
    counter = 0

    # set the path to the folder containing the images
    # path = "images/"

    # loop over all the files in the folder
    for filename in os.listdir("UTKFace_test"):    
        # open the image file and convert it to a NumPy array
        if (filename == ".DS_Store"):
            continue
        with Image.open(os.path.join("UTKFace_test", filename)) as img:
            
            parts = filename.split("_") 
            age = int(parts[0])
            gender = int(parts[1])
            ethnicity = int(parts[2])

            image_data.append([filename, img, age, gender, ethnicity])
            counter += 1
            # if counter % 10000 == 0:
            #     print(counter)

    # create a DataFrame from the image data
    test_image_df = pd.DataFrame(image_data, columns=["filename", "image", "age", "gender", "ethnicity"])

    train_dataset = ImageDataFrameDataset(
    dataframe=train_image_df,
    root_dir=train_name,
    transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    )

    test_dataset = ImageDataFrameDataset(
        dataframe=test_image_df,
        root_dir='UTKFace_test',
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    )

    # Load the ResNet18 model
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)

    # Define the loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    # Create data loaders for the train and test sets
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Train the model
    num_epochs = 11
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    start_time = time.time()

    for epoch in range(num_epochs):
        pbar = tqdm(train_loader, position=0, leave=True)

        for batch_idx, (idx, data, target) in enumerate(pbar):
            pbar.set_description(f'Epoch {epoch}')
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            pbar.set_postfix({'Training Loss': f'{loss.item():.4f}'})

       # Evaluate the model on the test set after each epoch
        test_loss = 0
        correct = 0
        misclassified_filenames = []

        with torch.no_grad():
            for idx, (data_idx, data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += loss_fn(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

                # Save misclassified filenames
                misclassified = (pred != target.view_as(pred)).nonzero(as_tuple=True)[0]
                misclassified_filenames.extend(test_image_df.iloc[data_idx[misclassified].cpu()]['filename'].tolist())

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        print(f"Elapsed time: {time.time() - start_time:.4f} seconds")
        print(f'Epoch {epoch}: Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')

        # Save misclassified filenames after each epoch
        with open(f'misclassified_filenames_epoch_{epoch}.txt', 'w') as f:
            for filename in misclassified_filenames:
                f.write(f"{filename}\n")


# -----------------------------------
if __name__ == '__main__':
    main()

