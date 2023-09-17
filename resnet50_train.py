import os
import sys
from datetime import datetime
import logging
import torch


import os
import sys
from datetime import datetime
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from resnet50 import resnet50
from data_location import Location_Isolated
from train import train_epoch
from validation import val_epoch
from collections import OrderedDict

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self):
        super(LabelSmoothingCrossEntropy, self).__init__()
    def forward(self, x, target, smoothing=0.1):
        confidence = 1. - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss.mean()

# Path setting
exp_name = 'street_view_train'
data_path = "C:/Users/Ayberk/Dev/train2/"
#data_path = "../bitirme_dataset/train/minicik_train/"
#data_path = "../bitirme_dataset/train/train_set_vfbha39/train/"
data_path2 = "C:/Users/Ayberk/Dev/validation2/"
label_train_path = "C:/Users/Ayberk/Dev/coords_new.csv"
label_val_path = "C:/Users/Ayberk/Dev/coords_new.csv"
model_path = "checkpoint/{}".format(exp_name)
log_path = "log/resnet50.log".format(exp_name, datetime.now())
sum_path = "runs/resnet50_{}_{:%Y-%m-%d_%H-%M-%S}".format(exp_name, datetime.now())
model_path = "checkpoint/{}".format(exp_name)
if not os.path.exists(model_path):
    os.mkdir(model_path)
if not os.path.exists(os.path.join('results', exp_name)):
    os.mkdir(os.path.join('results', exp_name))
phase = 'Train'
# Log to file & tensorboard writer
logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.FileHandler(log_path), logging.StreamHandler()])
logger = logging.getLogger('SLR')
logger.info('Logging to file...')
writer = SummaryWriter(sum_path)


# Use specific gpus
# os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6,7"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Hyperparams
num_classes = 2
epochs = 100
batch_size = 3
learning_rate = 1e-3#1e-3 Train 1e-4 Finetune
weight_decay = 1e-4 #1e-4
log_interval = 100
sample_size = 640

attention = False
drop_p = 0.0
hidden1, hidden2 = 512, 256


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# Train with 3DCNN
if __name__ == '__main__':
    # Load data
    transform = transforms.Compose([transforms.Resize([sample_size, sample_size]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5])])
    train_set = Location_Isolated(image_folder=data_path, coords_csv=label_train_path,)
    val_set = Location_Isolated(image_folder=data_path2, coords_csv=label_val_path,)
    logger.info("Dataset samples: {}".format(len(train_set)+len(val_set)))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    # Create model

    model = resnet50(pretrained=True, num_classes=2)
    # load pretrained



    print(model)


    
    model = model.to(device)
    # Run the model parallelly
    if torch.cuda.device_count() > 1:
        logger.info("Using {} GPUs".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    # Create loss criterion & optimizer
    criterion = nn.MSELoss()
    #criterion = LabelSmoothingCrossEntropy()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=0.001)

    # Start training
    if phase == 'Train':
        logger.info("Training Started".center(60, '#'))
        for epoch in range(epochs):
            print(epoch)
            print('lr: ', get_lr(optimizer))
            # Train the model
            train_epoch(model, criterion, optimizer, train_loader, device, epoch, logger, log_interval, writer)
            
            # Validate the model
            val_loss = val_epoch(model, criterion, val_loader, device, epoch, logger, writer)
            scheduler.step(val_loss)
            
            # Save model
            torch.save(model.state_dict(), os.path.join(model_path, "resnet50_epoch{:03d}.pth".format(epoch+1)))
            logger.info("Epoch {} Model Saved".format(epoch+1).center(60, '#'))
    elif phase == 'Test':
        logger.info("Testing Started".center(60, '#'))
        val_loss = val_epoch(model, criterion, val_loader, device, 0, logger, writer, phase=phase, exp_name=exp_name)

    logger.info("Finished".center(60, '#'))