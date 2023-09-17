import torch
from torchvision import transforms
from PIL import Image
import numpy as np

# Define the model architecture (replace with your model definition)
from resnet50 import resnet50
model = resnet50()

# Load the pre-trained model checkpoint (replace with your checkpoint path)
checkpoint_path = "C:/Users/Ayberk/Dev/Google-Street-View-Guesser/checkpoint/street_view_train/resnet50_epoch100.pth"
checkpoint = torch.load(checkpoint_path)  # Load on CPU
# model.load_state_dict(checkpoint['model_state_dict'])
model.load_state_dict(checkpoint)
model.eval()  # Set the model to evaluation mode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    model.cuda()

model = model.to(device)


# Load the specific image for testing (replace with your image path)
image_path = "C:/Users/Ayberk/Dev/validation2/50004.png"
image = Image.open(image_path)
transform = transforms.Compose([
    transforms.Resize((640, 640)),  # Resize image to a specific size
    transforms.ToTensor(),          # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])
image = transform(image)
image = image.to(device)

# Pass the image through the model
with torch.no_grad():



    output = model(image.unsqueeze(0))

# Interpret the model's predictions (modify as needed)
predictions = output.cpu().numpy()

# Print or process the predictions as needed
print("Predictions:", predictions)

# You can further process and analyze the predictions based on your specific task.
# For example, if it's a
