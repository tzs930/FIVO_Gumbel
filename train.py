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

USEWANDB=False
if USEWANDB:
	import wandb
	wandb.init(project="fivo_vrnn")

def train(epoch):
	train_loss = 0
	for batch_idx, (data, _, lengths) in enumerate(train_loader):		
		max_length = lengths.max()
		data = data.transpose(0, 1)  # (num_seq, batch_size, num_notes)
		data = data[:max_length]
		# rescale data
		# data = (data - data.min()) / (data.max() - data.min())
		
		#forward + backward + optimize
		optimizer.zero_grad()
		# fivo_loss, kld_loss, nll_loss, _, _, all_fivo_loss = model(data)
		fivo_loss = model(data)
		# loss = kld_loss + nll_loss
		loss = fivo_loss
		# with torch.autograd.set_detect_anomaly(True):
		loss.backward()
		optimizer.step()

		#grad norm clipping, only in pytorch version >= 1.10
		nn.utils.clip_grad_norm(model.parameters(), clip)

		#printing
		if batch_idx % print_every == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\t NLL: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss / len(train_loader)))
			# print('Train Epoch: {} [{}/{} ({:.0f}%)]\t KLD Loss: {:.6f} \t NLL Loss: {:.6f}'.format(
			# 	epoch, batch_idx * len(data), len(train_loader.dataset),
			# 	100. * batch_idx / len(train_loader),
			# 	kld_loss.item(),
			# 	nll_loss.item()))
			
			# sample = model.sample(28)
			# plt.imshow(sample.numpy())
			# plt.pause(1e-6)

		train_loss += loss.item()


	print('====> Epoch: {} Average loss: {:.4f}'.format(
		epoch, train_loss / len(train_loader.dataset)))
	
	if USEWANDB:
		# wandb.log({'train_KLD': kld_loss}, step=epoch)
		wandb.log({'train_NMLL': loss}, step=epoch)
		# wandb.log({'train_loss': train_loss / len(train_loader.dataset)}, step=epoch)


def evaluate(epoch):
	"""uses test data to evaluate 
	likelihood of the model"""
	
	# mean_kld_loss, mean_nll_loss = 0, 0
	mean_loss = 0
	for i, (data, _, _) in enumerate(valid_loader):
		#data = Variable(data)
		data = Variable(data.squeeze().transpose(0, 1))
		# data = (data - data.min()) / (data.max() - data.min())

		# kld_loss, nll_loss, _, _ = model(data)
		loss = model(data)
		# mean_kld_loss += kld_loss.item()
		# mean_nll_loss += nll_loss.item()
    
	# mean_kld_loss /= len(valid_loader.dataset)
	# mean_nll_loss /= len(valid_loader.dataset)
	loss /= len(valid_loader.dataset)
	
	if USEWANDB:
		wandb.log({'valid_NLL': loss}, step=epoch)
		# wandb.log({'valid_KLD': mean_kld_loss}, step=epoch)
		# wandb.log({'valid_NLL': mean_nll_loss}, step=epoch)
		# wandb.log({'valid_loss': mean_kld_loss + mean_nll_loss}, step=epoch)

	print('====> Valid set loss: NMLL = {:.4f}'.format(loss))
	# print('====> Valid set loss: KLD Loss = {:.4f}, NLL Loss = {:.4f} '.format(
	# 	mean_kld_loss, mean_nll_loss))

# def test(epoch):
# 	"""uses test data to evaluate 
# 	likelihood of the model"""
	
# 	mean_kld_loss, mean_nll_loss = 0, 0
# 	for i, (data, _) in enumerate(test_loader):
# 		#data = Variable(data)
# 		data = Variable(data.squeeze().transpose(0, 1))
# 		data = (data - data.min()) / (data.max() - data.min())

# 		kld_loss, nll_loss, _, _ = model(data)
# 		mean_kld_loss += kld_loss.item()
# 		mean_nll_loss += nll_loss.item()

# 	mean_kld_loss /= len(test_loader.dataset)
# 	mean_nll_loss /= len(test_loader.dataset)
	
# 	if USEWANDB:
# 		wandb.log({'test_KLD': mean_kld_loss}, step=epoch)
# 		wandb.log({'test_NLL': mean_nll_loss}, step=epoch)
# 		wandb.log({'test_loss': mean_kld_loss + mean_nll_loss}, step=epoch)

# 	print('====> Test set loss: KLD Loss = {:.4f}, NLL Loss = {:.4f} '.format(
# 		mean_kld_loss, mean_nll_loss))


#hyperparameters
x_dim = 88
h_dim = 32
z_dim = 32
n_layers =  1
n_epochs = 100
clip = 10
learning_rate = 3e-5
batch_size = 8
num_particles = 4
seed = 128
print_every = 100
valid_every = 5
save_every = 100
bound = 'ELBO' # 'IWAE', 'FIVO'

#manual seed
torch.manual_seed(seed)
# plt.ion()

#init model + optimizer + datasets
dataset = 'jsb'
train_loader = jsbdatasets.create_pianoroll_dataset('data_jsb/%s.pkl'%dataset, 'train', batch_size)
valid_loader = jsbdatasets.create_pianoroll_dataset('data_jsb/%s.pkl'%dataset, 'valid', batch_size)
test_loader = jsbdatasets.create_pianoroll_dataset('data_jsb/%s.pkl'%dataset, 'test', batch_size)

model = VRNN(x_dim, h_dim, z_dim, n_layers, num_particles).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(1, n_epochs + 1):
	
	#training + testing
	train(epoch)

	if epoch % valid_every == 0:
		evaluate(epoch)

	#saving model
	if epoch % save_every == 0:
		fn = 'saves/vrnn_state_dict_'+str(epoch)+'.pth'
		torch.save(model.state_dict(), fn)
		print('Saved model to '+fn)

# test(epoch)