import math
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torchvision import datasets, transforms
from torch.autograd import Variable
#import matplotlib.pyplot as plt 
from model import VRNN, IWAE, FIVO
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""implementation of the Variational Recurrent
Neural Network (VRNN) from https://arxiv.org/abs/1506.02216
using unimodal isotropic gaussian distributions for 
inference, prior, and generating models."""

from data_jsb import datasets as jsbdatasets

USEWANDB=True
if USEWANDB:
	import wandb	

def generate_seq_mask(lengths, seq_len, batch_size):
	mask = torch.zeros(batch_size, seq_len)
	for i, leng in enumerate(lengths):
		mask[i][:leng] = 1.
	return mask.T.to(device)

def train(epoch):
	train_loss = 0
	cum_num_train = 0
	loghatlist = []
	iwaelist = []

	for batch_idx, (data, _, lengths) in enumerate(train_loader):		
		max_length = lengths.max()
		data = data.transpose(0, 1)  # (num_seq, batch_size, num_notes)
		# data = data[:max_length]
		mask = generate_seq_mask(lengths, data.shape[0], data.shape[1])
		# rescale data
		# data = (data - data.min()) / (data.max() - data.min())
		
		#forward + backward + optimize
		optimizer.zero_grad()
		# kld_loss, nll_loss, _, _ = model(data, mask)
		# loss = kld_loss + nll_loss

		fivo_loss, logphat_total, _, kl, iwae_bound = model(data, mask, num_particles)
		loss = fivo_loss
		# with torch.autograd.set_detect_anomaly(True):
		loss.backward()
		optimizer.step()

		logp_per_timestep = logphat_total / lengths
		iwae_bound_per_t = iwae_bound / lengths
		mean_logp_per_timestep = logp_per_timestep.mean().item()
		mean_iwae = iwae_bound_per_t.mean().item()
		#grad norm clipping, only in pytorch version >= 1.10
		nn.utils.clip_grad_norm_(model.parameters(), clip)

		for i, loghat_ in enumerate(logp_per_timestep):
			loghatlist.append(loghat_.item())
			iwaelist.append(iwae_bound_per_t[i].item())

		#printing
		if batch_idx % print_every == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Sequence LL: {:.6f} KL: {:.6f} timestep LL : {:.6f},  timestep IWAE : {:.6f}'.format(
				epoch, batch_idx * batch_size, len(train_loader.dataset), 
				100. * batch_idx / len(train_loader), 
				-loss.item() / data.size(1), kl.mean(), mean_logp_per_timestep, mean_iwae
				)
				)			

		train_loss += loss.item()
		cum_num_train += data.size(1)		

	avg_train_loss = train_loss / cum_num_train
	print('====> Epoch: {} Average LL per sequence: {:.4f}, Average LL per timestep: {:.4f}, IWAE per timestep : {:.6f}'.format(
		epoch, -avg_train_loss, np.mean(loghatlist), np.mean(iwaelist)))
	
	if USEWANDB:
		wandb.log({'train_FIVO(4)_t': np.mean(loghatlist)}, step=epoch)
		wandb.log({'train_IWAE(4)_t': np.mean(loghatlist)}, step=epoch)
		wandb.log({'train_KL': kl.mean()}, step=epoch)
		# wandb.log({'train_KLD': kld_loss}, step=epoch)
		# wandb.log({'train_NLL': loss}, step=epoch)
		# wandb.log({'train_loss': train_loss / len(train_loader.dataset)}, step=epoch)


def evaluate(epoch):
	"""uses test data to evaluate 
	likelihood of the model"""
	
	mean_kld_loss, mean_nll_loss = 0, 0
	mean_loss = 0
	kl_acc = 0
	loghatlist = []
	iwaelist = []

	for i, (data, _, lengths) in enumerate(valid_loader):
		#data = Variable(data)		
		# data = Variable(data.squeeze().transpose(0, 1), requires_grad=False)
		data = data.transpose(0, 1)
		mask = generate_seq_mask(lengths, data.shape[0], data.shape[1])
		# data = (data - data.min()) / (data.max() - data.min())

		# kld_loss, nll_loss, _, _ = model(data, mask)
		loss, loghat, _, kl, iwae_bound = model(data, mask, num_particles)
		
		kl_acc += kl.mean()
		mean_loss += loss

		logp_per_timestep = loghat / lengths
		iwae_bound_per_t = iwae_bound / lengths
		
		for i, loghat_ in enumerate(logp_per_timestep):
			loghatlist.append(loghat_.item())
			iwaelist.append(iwae_bound_per_t[i].item())

		# mean_kld_loss += kld_loss.item()
		# mean_nll_loss += nll_loss.item()
    
	# mean_kld_loss /= len(valid_loader.dataset)
	# mean_nll_loss /= len(valid_loader.dataset)
	mean_loss /= len(valid_loader.dataset)
	kl_acc /= len(valid_loader.dataset)
	mean_loghat_per_timestep = np.mean(loghatlist)
	mean_iwae_per_timestep = np.mean(iwaelist)
	
	if USEWANDB:
		# wandb.log({'valid_NLL': loss}, step=epoch)
		wandb.log({'valid_FIVO(4)_t': mean_loghat_per_timestep}, step=epoch)
		wandb.log({'valid_IWAE(4)_t': mean_iwae_per_timestep}, step=epoch)
		# wandb.log({'valid_KLD': mean_kld_loss}, step=epoch)
		# wandb.log({'valid_NLL': mean_nll_loss}, step=epoch)
		# wandb.log({'valid_loss': mean_kld_loss + mean_nll_loss}, step=epoch)

	print('====> Valid set loss: Avg. Marginal LL = {:.4f}, KL = {:.4f}, Marginal LL per timestep = {:.4f}, IWAE bound per timestep = {:.4f}'
				.format(-mean_loss, kl_acc, mean_loghat_per_timestep, mean_iwae_per_timestep))
	# print('====> Valid set loss: KLD Loss = {:.4f}, NLL Loss = {:.4f} '.format(
	# 	mean_kld_loss, mean_nll_loss))

	return mean_loss


def test(epoch):
	"""uses test data to evaluate 
	likelihood of the model"""
	
	loss_total = 0
	loghatlist = []
	iwaelist = []

	for i, (data, _, lengths) in enumerate(test_loader):
		#data = Variable(data)
		# data = Variable(data.squeeze().transpose(0, 1), requires_grad=False)
		data = data.transpose(0, 1)
		mask = generate_seq_mask(lengths, data.shape[0], data.shape[1])
		# data = (data - data.min()) / (data.max() - data.min())

		loss, loghat, _, _, iwae_bound = model(data, mask, num_eval_particles)
		loss_total += loss

		logp_per_timestep = loghat / lengths
		iwae_bound_per_t = iwae_bound / lengths

		for i, loghat_ in enumerate(logp_per_timestep):
			loghatlist.append(loghat_.item())
			iwaelist.append(iwae_bound_per_t[i].item())
		# mean_kld_loss += kld_loss.item()
		# mean_nll_loss += nll_loss.item()

	loss_total /= len(test_loader.dataset)	
	mean_loghat_per_timestep = np.mean(loghatlist)
	mean_iwae_per_timestep = np.mean(iwaelist)

	if USEWANDB:
		wandb.log({'test_FIVO(32)_t': mean_loghat_per_timestep}, step=epoch)
		wandb.log({'test_IWAE(32)_t': mean_iwae_per_timestep}, step=epoch)
	# 	wandb.log({'test_KLD': mean_kld_loss}, step=epoch)
	# 	wandb.log({'test_NLL': mean_nll_loss}, step=epoch)
	# 	wandb.log({'test_loss': mean_kld_loss + mean_nll_loss}, step=epoch)

	print('====> Test set loss:  Avg. Marginal LL = {:.4f}, Avg. Marginal LL per timestep = {:.4f}, IWAE bound per timestep = {:.4f}'
					.format(-loss_total, mean_loghat_per_timestep, mean_iwae_per_timestep))


#hyperparameters
x_dim = 88
h_dim = 32
z_dim = 32
n_layers =  1
n_epochs = 500
clip = 10
learning_rate = 3e-5
batch_size = 4
eval_batch_size = 1
num_particles = 8
num_eval_particles = 64
seed = 128
print_every = 10
valid_every = 5
save_every = 100
bound = 'FIVO' # 'IWAE', 'FIVO'
wandb.init(project="fivo_vrnn", name='%s_%d' % (bound, num_particles))

#manual seed
torch.manual_seed(seed)
min_loss = 10000000.
# plt.ion()

#init model + optimizer + datasets
dataset = 'jsb'
train_loader = jsbdatasets.create_pianoroll_dataset('data_jsb/%s.pkl'%dataset, 'train', batch_size)
valid_loader = jsbdatasets.create_pianoroll_dataset('data_jsb/%s.pkl'%dataset, 'valid', batch_size)
test_loader = jsbdatasets.create_pianoroll_dataset('data_jsb/%s.pkl'%dataset, 'test', eval_batch_size)

# model = IWAE(x_dim, h_dim, z_dim, n_layers).to(device)
if bound == 'IWAE':
	model = IWAE(x_dim, h_dim, z_dim, n_layers).to(device)
elif bound == 'FIVO':
	model = FIVO(x_dim, h_dim, z_dim, n_layers).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
if USEWANDB:
	wandb.watch(model)

for epoch in range(1, n_epochs + 1):	
	#training + testing
	train(epoch)

	if epoch % valid_every == 0:
		val_loss = evaluate(epoch)
		if val_loss < min_loss:
			print('== Best valid loss! Start Testing.. ')
			min_loss = val_loss
			test(epoch)

	#saving model
	if epoch % save_every == 0:
		fn = 'saves/vrnn_state_dict_'+str(epoch)+'.pth'

		torch.save(model.state_dict(), fn)
		print('Saved model to '+fn)

test(epoch)
