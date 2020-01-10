import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim


class FCN8sNet(nn.Module):
	def __init__(self, in_res, num_landmarks):
		super(FCN8sNet, self).__init__()

		self.in_res = in_res
		self.num_landmarks = num_landmarks

		self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
		self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
		self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
		self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
		self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
		self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
		self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
		self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
		self.conv5_1 = nn.Conv2d(512, 4096, kernel_size=7, stride=1, padding=3)
		self.conv5_2 = nn.Conv2d(4096, 4096, kernel_size=1, stride=1, padding=0)
		self.conv6_1 = nn.Conv2d(512, self.num_landmarks, kernel_size=1, stride=1, padding=0)
		self.conv7_1 = nn.Conv2d(256, self.num_landmarks, kernel_size=1, padding=0)

		self.convtrans_1 = nn.ConvTranspose2d(4096, self.num_landmarks, kernel_size=4, stride=4, bias=False)
		self.convtrans_2 = nn.ConvTranspose2d(self.num_landmarks, self.num_landmarks, kernel_size=2, stride=2, bias=False)
		self.convtrans_3 = nn.ConvTranspose2d(self.num_landmarks, self.num_landmarks, kernel_size=8, stride=8, bias=False)

		self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        

	def forward(self, x):
        
		# define VGG for encoder part 
		# block 1
		x = F.relu(self.conv1_1(x))
		x = F.relu(self.conv1_2(x))
		x = self.pool(x)
		fm1 = x

		# block 2
		x = F.relu(self.conv2_1(x))
		x = F.relu(self.conv2_2(x))
		x = self.pool(x)
		fm2 = x

		# block 3
		x = F.relu(self.conv3_1(x))
		x = F.relu(self.conv3_2(x))
		x = F.relu(self.conv3_2(x))
		x = self.pool(x)
		fm3 = x

		# block 4
		x = F.relu(self.conv4_1(x))
		x = F.relu(self.conv4_2(x))
		x = F.relu(self.conv4_2(x))
		x = self.pool(x)
		fm4 = x

		# block 5
		x = F.relu(self.conv4_2(x))
		x = F.relu(self.conv4_2(x))
		x = F.relu(self.conv4_2(x))
		x = self.pool(x)
		fm5 = x

		# define decoder part 
		o = F.relu(self.conv5_1(x))
		o = F.relu(self.conv5_2(o))

		con1 = self.convtrans_1(o)

		con2 = F.relu(self.conv6_1(fm4))
		con2 = self.convtrans_2(con2)

		con3 = F.relu(self.conv7_1(fm3))

		o = con1 + con2 + con3
		o = self.convtrans_3(o)

		#o = o.view(-1, self.in_res * self.in_res * self.num_landmarks)

		return (o)


