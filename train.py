import math
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt 
from model import VRNN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""implementation of the Variational Recurrent
Neural Network (VRNN) from https://arxiv.org/abs/1506.02216
using unimodal isotropic gaussian distributions for 
inference, prior, and generating models."""

from data_jsb import datasets as jsbdatasets

def train(epoch):
	train_loss = 0
	for batch_idx, (data, _) in enumerate(train_loader):
		
		#transforming data
		#data = Variable(data)
		#to remove eventually
		# data = Variable(data.squeeze().transpose(0, 1))
		data = data.transpose(0, 1)  # (num_seq, batch_size, num_notes)
		
		# rescale data
		# data = (data - data.min()) / (data.max() - data.min())
		
		#forward + backward + optimize
		optimizer.zero_grad()
		kld_loss, nll_loss, _, _ = model(data)
		loss = kld_loss + nll_loss
		loss.backward()
		optimizer.step()

		#grad norm clipping, only in pytorch version >= 1.10
		# nn.utils.clip_grad_norm(model.parameters(), clip)

		#printing
		if batch_idx % print_every == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\t KLD Loss: {:.6f} \t NLL Loss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
				100. * batch_idx / len(train_loader),
				kld_loss.item() / batch_size,
				nll_loss.item() / batch_size))

			sample = model.sample(28)
			plt.imshow(sample.numpy())
			plt.pause(1e-6)

		train_loss += loss.item()


	print('====> Epoch: {} Average loss: {:.4f}'.format(
		epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
	"""uses test data to evaluate 
	likelihood of the model"""
	
	mean_kld_loss, mean_nll_loss = 0, 0
	for i, (data, _) in enumerate(test_loader):
		#data = Variable(data)
		data = Variable(data.squeeze().transpose(0, 1))
		data = (data - data.min()) / (data.max() - data.min())

		kld_loss, nll_loss, _, _ = model(data)
		mean_kld_loss += kld_loss.item()
		mean_nll_loss += nll_loss.item()

	mean_kld_loss /= len(test_loader.dataset)
	mean_nll_loss /= len(test_loader.dataset)

	print('====> Test set loss: KLD Loss = {:.4f}, NLL Loss = {:.4f} '.format(
		mean_kld_loss, mean_nll_loss))


#hyperparameters
x_dim = 88
h_dim = 100
z_dim = 16
n_layers =  1
n_epochs = 300
clip = 10
learning_rate = 1e-3
batch_size = 32
seed = 128
print_every = 100
save_every = 10
bound = 'ELBO' # 'IWAE', 'FIVO'

#manual seed
torch.manual_seed(seed)
plt.ion()

#init model + optimizer + datasets
dataset = 'jsb'
train_loader = jsbdatasets.create_pianoroll_dataset('data_jsb/%s.pkl'%dataset, 'train', batch_size)
valid_loader = jsbdatasets.create_pianoroll_dataset('data_jsb/%s.pkl'%dataset, 'valid', batch_size)
test_loader = jsbdatasets.create_pianoroll_dataset('data_jsb/%s.pkl'%dataset, 'test', batch_size)

model = VRNN(x_dim, h_dim, z_dim, n_layers).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(1, n_epochs + 1):
	
	#training + testing
	train(epoch)
	test(epoch)

	#saving model
	if epoch % save_every == 1:
		fn = 'saves/vrnn_state_dict_'+str(epoch)+'.pth'
		torch.save(model.state_dict(), fn)
		print('Saved model to '+fn)