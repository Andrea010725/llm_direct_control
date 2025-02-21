"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os
from tkinter.messagebox import NO
import torch
import data_setup, engine, utils

from torchvision import transforms
import model_lcd
# import model_lcd_v2
from torch.utils.data import DataLoader

# Setup hyperparameters
NUM_EPOCHS = 15
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001
NUM_WORKERS = os.cpu_count()
# NUM_WORKERS = 

# Setup directories
train_dir = "/home/PJLAB/guyi/Documents/code/lane_change/data/train"
test_dir = "/home/PJLAB/guyi/Documents/code/lane_change/data/test"

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create transforms
data_transform = transforms.Compose([
#   transforms.Resize((32, 128)),
  transforms.ToTensor()
])

# Create DataLoaders with help from data_setup.py
# train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
#     train_dir=train_dir,
#     test_dir=test_dir,
#     transform=data_transform,
#     batch_size=BATCH_SIZE
# )

# use custom data_loader
train_data_custom = data_setup.ImageFolderCustom(targ_dir=train_dir, transform=data_transform)
# print(train_data_custom.class_to_idx)

train_dataloader = DataLoader(dataset=train_data_custom, # use custom created train Dataset
                                     batch_size=BATCH_SIZE, # how many samples per batch?
                                     num_workers=NUM_WORKERS, # how many subprocesses to use for data loading? (higher = more)
                                     shuffle=True) # shuffle the data?

test_data_custom = data_setup.ImageFolderCustom(targ_dir=test_dir, transform=data_transform)

test_dataloader = DataLoader(dataset=test_data_custom, # use custom created test Dataset
                                    batch_size=1,
                                    num_workers=NUM_WORKERS, 
                                    shuffle=True) # don't usually need to shuffle testing data

# Create model with help from model_builder.py
model = model_lcd.LCD().cuda()
# model = model_lcd_v2.LCD_v2().cuda()

# model.load_state_dict(torch.load("./1016_grain_v1.pth"))
model.load_state_dict(torch.load("/home/PJLAB/guyi/Documents/code/lane_change/model_v1_9.pth"))

# model_builder.TinyVGG(
#     input_shape=3,
#     hidden_units=HIDDEN_UNITS,
#     output_shape=len(class_names)
# ).to(device)

# Set loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE)

# Start training with help from engine.py
engine.test(model=model, dataloader=test_dataloader, vis_opt=False, device=device)

# # Save the model with help from postprocess_utils.py
# postprocess_utils.save_model(model=model,
#                  target_dir="models",
#                  model_name="05_going_modular_script_mode_tinyvgg_model.pth")
