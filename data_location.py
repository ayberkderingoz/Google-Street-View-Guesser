import csv
import os
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random
import numpy as np
#import path
"""
Implementation of Sign Language Dataset
"""
 
class Location_Isolated(Dataset):
    def __init__(self, image_folder, coords_csv):
        self.image_folder = image_folder
        self.coords_csv = coords_csv

        # Create a list of image file names
        self.image_files = os.listdir(self.image_folder)

        # Read the coordinates from the CSV file
        self.coordinates = self._read_coordinates()

        # Define data transformation (you can customize this)
        self.transform = transforms.Compose([
            transforms.Resize((640, 640)),  # Resize image to a specific size
            transforms.ToTensor(),          # Convert to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
        ])

    def _read_coordinates(self):
        coordinates = {}
        with open(self.coords_csv, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for idx, row in enumerate(reader):
                if len(row) >= 2:
                    x, y = map(float, row)
                    image_filename = f"{idx}.png"
                    coordinates[image_filename] = (x, y)
                else:
                # Handle cases where there are not enough values in the row
                    print(f"Warning: Skipping row {idx} due to insufficient values.")
        return coordinates

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        image_filename = self.image_files[idx]
        image_path = os.path.join(self.image_folder, image_filename)
        image = Image.open(image_path)

        # Apply transformations
        image = self.transform(image)


        # Get corresponding coordinates from the CSV file
        label = self.coordinates.get(image_filename)

        # Convert label to a tensor
        if label is None:
            label = [0.0, 0.0]  # Replace with appropriate placeholder values

        # Convert label to a tensor
        label = torch.FloatTensor(label)


        return {'data': image, 'label': label}