import numpy as np
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim


from model import FCN8sNet
from data_generator import Dataset

import matplotlib.pyplot as plt

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

in_res = (224, 224)
out_res = (224, 224)
num_landmarks = 29
num_epochs = 5
batch_size = 8

xnet = FCN8sNet(in_res=in_res, num_landmarks=num_landmarks)
xnet = xnet.to(device, dtype=torch.float)

train_dataset = Dataset("data/cofw_annotations.json", "data/cofw/images",
                                    inres=in_res, outres=out_res, is_train=True)

val_dataset = Dataset("data/cofw_annotations.json", "data/cofw/images",
                                    inres=in_res, outres=out_res, is_train=False)

num_train = train_dataset.get_dataset_size()
num_val = val_dataset.get_dataset_size()

print('[INFO] Training size: {}'.format(num_train))
print('[INFO] Validation size: {}'.format(num_val))

#Loss function
loss = nn.MSELoss()
#Optimizer
optimizer = optim.Adam(xnet.parameters(), lr=0.001)

# Loop over epochs

trn_loss = []
val_loss = []

for epoch in range(num_epochs):

	train_generator = train_dataset.generator(batch_size=batch_size, sigma=1, is_shuffle=True, epoch_end=False)
	val_generator = val_dataset.generator(batch_size=1, sigma=1, is_shuffle=False, epoch_end=False)

	pbar = tqdm(total=num_train//batch_size, desc='epoch {}/{}: '.format(epoch, num_epochs))
	# Training
	for x_batch, y_batch in train_generator:
		optimizer.zero_grad()
		# Transfer to GPU
		x_batch = Variable(torch.from_numpy(x_batch))
		y_batch = Variable(torch.from_numpy(y_batch))
		x_batch, y_batch = x_batch.to(device, dtype=torch.float), y_batch.to(device, dtype=torch.float)

		# Model computations 
		# Forward pass, backward pass, optimize
		y_pred = xnet(x_batch)
		loss_val = loss(y_pred, y_batch)
		loss_val.backward()
		optimizer.step()

		pbar.update(1)
	pbar.close()

	trn_loss.append(loss_val.data)
	# Validation 
	test_loss = 0
	with torch.set_grad_enabled(False):
		for x_batch, y_batch in val_generator:
			x_batch = Variable(torch.from_numpy(x_batch))
			y_batch = Variable(torch.from_numpy(y_batch))
			x_batch, y_batch = x_batch.to(device, dtype=torch.float), y_batch.to(device, dtype=torch.float)

			hm_pred = xnet(x_batch)
			b_loss = loss(hm_pred, y_batch)

			test_loss += b_loss.item()

	val_loss.append(test_loss/num_val)

	print("[INFO] train loss is: {}, val loss is: {}".format(loss_val.data, test_loss/num_val))
        
torch.save(xnet.state_dict(), 'fcn_landmark.pth')
