import math
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torchvision import datasets, transforms
from torch.autograd import Variable
#import matplotlib.pyplot as plt 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torch.distributions import MultivariateNormal, Bernoulli
"""implementation of the Variational Recurrent
Neural Network (VRNN) from https://arxiv.org/abs/1506.02216
using unimodal isotropic gaussian distributions for 
inference, prior, and generating models."""


class VRNN(nn.Module):
	def __init__(self, x_dim, h_dim, z_dim, n_layers, bias=False):
		super(VRNN, self).__init__()

		self.x_dim = x_dim
		self.h_dim = h_dim
		self.z_dim = z_dim
		self.n_layers = n_layers

		self.num_z_particles = 4

		#feature-extracting transformations
		self.phi_x = nn.Sequential(
			nn.Linear(x_dim, h_dim),
			nn.ReLU(),
			nn.Linear(h_dim, h_dim),
			nn.ReLU())
		self.phi_z = nn.Sequential(
			nn.Linear(z_dim, h_dim),
			nn.ReLU())

		#encoder
		self.enc = nn.Sequential(
			nn.Linear(h_dim + h_dim, h_dim),
			nn.ReLU(),
			nn.Linear(h_dim, h_dim),
			nn.ReLU())
		self.enc_mean = nn.Linear(h_dim, z_dim)
		self.enc_std = nn.Sequential(
			nn.Linear(h_dim, z_dim),
			nn.Softplus())

		#prior
		self.prior = nn.Sequential(
			nn.Linear(h_dim, h_dim),
			nn.ReLU())
		self.prior_mean = nn.Linear(h_dim, z_dim)
		self.prior_std = nn.Sequential(
			nn.Linear(h_dim, z_dim),
			nn.Softplus()) 

		#decoder
		self.dec = nn.Sequential(
			nn.Linear(h_dim + h_dim, h_dim),
			nn.ReLU(),
			nn.Linear(h_dim, h_dim),
			nn.ReLU())
		self.dec_std = nn.Sequential(
			nn.Linear(h_dim, x_dim),
			nn.Softplus())
		#self.dec_mean = nn.Linear(h_dim, x_dim)
		self.dec_mean = nn.Sequential(
			nn.Linear(h_dim, x_dim),
			nn.Sigmoid())

		#recurrence
		# self.rnn = nn.GRU(h_dim + h_dim, h_dim, n_layers, bias)
		self.rnn = nn.LSTM(h_dim + h_dim, h_dim, n_layers, bias)


	def forward(self, x, mask):

		all_enc_mean, all_enc_std = [], []
		all_dec_mean, all_dec_std = [], []
		kld_loss = 0
		nll_loss = 0

		h = Variable(torch.zeros(self.n_layers, x.size(1), self.h_dim)).to(device)
		c = Variable(torch.zeros(self.n_layers, x.size(1), self.h_dim)).to(device)

		for t in range(x.size(0)):
			# Inference
			phi_x_t = self.phi_x(x[t])

			#encoder
			enc_t = self.enc(torch.cat([phi_x_t, h[-1]], 1))
			enc_mean_t = self.enc_mean(enc_t)
			enc_std_t = self.enc_std(enc_t)

			#prior
			prior_t = self.prior(h[-1])
			prior_mean_t = self.prior_mean(prior_t)
			prior_std_t = self.prior_std(prior_t) + 1.

			#sampling and reparameterization
			z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
			phi_z_t = self.phi_z(z_t)

			#decoder
			dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
			dec_mean_t = self.dec_mean(dec_t)
			dec_std_t = self.dec_std(dec_t)

			#recurrence
			_, (h,c) = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), (h,c))

			#computing losses
			kld_loss += torch.sum(self._kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t) * mask[t])  # [num_seq_len, batch]
			#nll_loss += self._nll_gauss(dec_mean_t, dec_std_t, x[t])
			nll_loss += torch.sum(self._nll_bernoulli(dec_mean_t, x[t]) * mask[t])

			all_enc_std.append(enc_std_t)
			all_enc_mean.append(enc_mean_t)
			all_dec_mean.append(dec_mean_t)
			all_dec_std.append(dec_std_t)

		return kld_loss, nll_loss, \
			(all_enc_mean, all_enc_std), \
			(all_dec_mean, all_dec_std)


	def sample(self, seq_len):

		sample = torch.zeros(seq_len, self.x_dim)

		h = Variable(torch.zeros(self.n_layers, 1, self.h_dim))
		c = Variable(torch.zeros(self.n_layers, 1, self.h_dim))

		for t in range(seq_len):

			#prior
			prior_t = self.prior(h[-1])
			prior_mean_t = self.prior_mean(prior_t)
			prior_std_t = self.prior_std(prior_t)

			#sampling and reparameterization
			z_t = self._reparameterized_sample(prior_mean_t, prior_std_t)
			phi_z_t = self.phi_z(z_t)
			
			#decoder
			dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
			dec_mean_t = self.dec_mean(dec_t)
			#dec_std_t = self.dec_std(dec_t)

			phi_x_t = self.phi_x(dec_mean_t)

			#recurrence
			_, (h, c) = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), (h, c))

			sample[t] = dec_mean_t.data
	
		return sample


	def reset_parameters(self, stdv=1e-1):
		for weight in self.parameters():
			weight.data.normal_(0, stdv)


	def _init_weights(self, stdv):
		pass


	def _reparameterized_sample(self, mean, std):
		"""using std to sample"""
		eps = torch.FloatTensor(std.size()).normal_()
		eps = Variable(eps).to(device)
		return eps.mul(std).add_(mean)


	def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
		"""Using std to compute KLD"""

		kld_element =  (2 * torch.log(std_2) - 2 * torch.log(std_1) + 
			(std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
			std_2.pow(2) - 1)
		return 0.5 * torch.sum(kld_element, dim=-1)


	def _nll_bernoulli(self, theta, x):
		# mean w.r.t. batch, sum w.r.t input dimensions
		return - torch.sum(x*torch.log(theta) + (1-x)*torch.log(1-theta), dim=-1)


	def _nll_gauss(self, mean, std, x):
		pass


class FIVO(nn.Module):
	def __init__(self, x_dim, h_dim, z_dim, n_layers, num_particles, bias=False):
		super(FIVO, self).__init__()

		self.x_dim = x_dim
		self.h_dim = h_dim
		self.z_dim = z_dim
		self.n_layers = n_layers
		self.num_zs = num_particles

		#feature-extracting transformations
		self.phi_x = nn.Sequential(
			nn.Linear(x_dim, h_dim),
			nn.ReLU(),
			nn.Linear(h_dim, h_dim),
			nn.ReLU())
		self.phi_z = nn.Sequential(
			nn.Linear(z_dim, h_dim),
			nn.ReLU())

		#encoder
		self.enc = nn.Sequential(
			nn.Linear(h_dim + h_dim, h_dim),
			nn.ReLU(),
			nn.Linear(h_dim, h_dim),
			nn.ReLU())
		self.enc_mean = nn.Linear(h_dim, z_dim)
		self.enc_std = nn.Sequential(
			nn.Linear(h_dim, z_dim),
			nn.Softplus())

		#prior
		self.prior = nn.Sequential(
			nn.Linear(h_dim, h_dim),
			nn.ReLU())
		self.prior_mean = nn.Linear(h_dim, z_dim)
		self.prior_std = nn.Sequential(
			nn.Linear(h_dim, z_dim),
			nn.Softplus()) 

		#decoder
		self.dec = nn.Sequential(
			nn.Linear(h_dim + h_dim, h_dim),
			nn.ReLU(),
			nn.Linear(h_dim, h_dim),
			nn.ReLU())
		self.dec_std = nn.Sequential(
			nn.Linear(h_dim, x_dim),
			nn.Softplus())
		#self.dec_mean = nn.Linear(h_dim, x_dim)
		self.dec_mean = nn.Sequential(
			nn.Linear(h_dim, x_dim),
			nn.Sigmoid())

		#recurrence
		# self.rnn = nn.GRU(h_dim + h_dim, h_dim, n_layers, bias)
		# self.rnns = [nn.LSTM(h_dim + h_dim, h_dim, n_layers, bias) for _ in range(self.num_zs)]
		self.rnn = nn.LSTM(h_dim + h_dim, h_dim, n_layers, bias)


	def forward(self, x, mask):

		all_enc_mean, all_enc_std = [], []
		all_dec_mean, all_dec_std = [], []
		all_fivo_loss = []
		
		kld_loss = 0
		nll_loss = 0
		fivo_loss = 0		

		# hs = [Variable(torch.zeros(self.n_layers, x.size(1), self.h_dim)).to(device) for _ in range(self.num_zs)]
		# cs = [Variable(torch.zeros(self.n_layers, x.size(1), self.h_dim)).to(device) for _ in range(self.num_zs)]

		h = Variable(torch.zeros(self.n_layers, x.size(1) * self.num_zs, self.h_dim)).to(device)
		c = Variable(torch.zeros(self.n_layers, x.size(1) * self.num_zs, self.h_dim)).to(device)

		ws, zs = [], [] # [num_seq, num_particles, (batch_size, embed_size)]
		log_hat_ps = [0.]
		log_hat_p_acc = torch.zeros(x.size(1)).to(device)

		logw0 = torch.log(torch.ones([x.size(1), self.num_zs], requires_grad=False).to(device) / float(self.num_zs))
		logweight = logw0		
		
		# ws = torch.ones(x.size(0), self.num_zs, x.size(1), requires_grad=False) / float(self.num_zs)
		# ws = [Variable(torch.ones(self.num_zs, x.size(1)) / float(self.num_zs), requires_grad=False) for _ in range(x.size(0)) ]# [num_seq, num_zs, batch_size]

		# with torch.autograd.set_detect_anomaly(True):			
		for t in range(x.size(0)):
			# if (x[t] == 0).all():
			# 	continue

			# Inference
			# xts = x[t].tile(self.num_zs, 1)   # [batch_size * num_particle, 88]
			xts = x[t].repeat((1, self.num_zs)).reshape((x.size(1)*self.num_zs, x.size(-1)))
			phi_x_ts = self.phi_x(xts)			# [batch_size * num_particle, embed_size]
			# phi_x_ts = phi_x_t.repeat(self.num_zs, 1)

			#sampling and reparameterization
			# logwnew = logwold.clone()
			# hat_p = 0.
			# h = h.reshape(self.n_layers, self.num_zs, x.size(1), self.h_dim) 

			# may boost learning speed by using parellelism
			# phi_z_t_is = []
			enc_t = self.enc(torch.cat([phi_x_ts, h[-1]], 1))  
			enc_mean_t = self.enc_mean(enc_t) 
			enc_std_t = self.enc_std(enc_t)

			encoder_dist = MultivariateNormal(enc_mean_t, scale_tril=torch.diag_embed(enc_std_t))

			prior_t = self.prior(h[-1])
			prior_mean_t = self.prior_mean(prior_t)
			prior_std_t = self.prior_std(prior_t) + 1.

			prior_dist = MultivariateNormal(prior_mean_t, scale_tril=torch.diag_embed(prior_std_t))

			z_t_is = encoder_dist.rsample()  # reparametrizable
			
			phi_z_ts = self.phi_z(z_t_is)
			# phi_z_t_is.append(phi_z_t)

			dec_t = self.dec(torch.cat([phi_z_ts, h[-1]], 1))
			dec_mean_t = self.dec_mean(dec_t) 
			decoder_dist = Bernoulli(probs=dec_mean_t)

			prior_logprob_ti = prior_dist.log_prob( z_t_is.detach() ) + 1e-7
			encoder_logprob_ti = encoder_dist.log_prob( z_t_is.detach()  ) + 1e-7
			decoder_logprob_ti = decoder_dist.log_prob(xts).sum(-1) + 1e-7
						
			log_alpha_ti = prior_logprob_ti + decoder_logprob_ti - encoder_logprob_ti # [batch_size, ]

			log_alpha_ti = log_alpha_ti.reshape(x.size(1), -1)  # [batch_size, num_particles]
			hat_p = torch.exp(logweight + log_alpha_ti) 		# [batch_size, num_particles]
			logweight = logweight + log_alpha_ti.detach()      
				
			# log_hat_p = log_hat_ps[t] + torch.log(hat_p)
			log_hat_p_acc += (torch.log(hat_p) * mask[t][None].T).mean(-1)
			logweight = logweight - torch.log(hat_p.detach())
			# logwold = logwnew
			# log_hat_ps.append(log_hat_p_acc)

			# recurrence
			# h = h.reshape(self.n_layers, self.num_zs * x.size(1), self.h_dim)
			
			# phi_z_ts = torch.cat(phi_z_t_is, dim=0)

			_, (h, c) = self.rnn(torch.cat([phi_x_ts, phi_z_ts], 1).unsqueeze(0), (h, c))				

			if torch.isnan(log_hat_p_acc).any():
				import IPython; IPython.embed()			
			#computing losses						
			# kld_loss /= self.num_zs
			# nll_loss /= self.num_zs

		fivo_loss = -log_hat_p_acc.sum()

		# return fivo_loss, kld_loss, nll_loss, \
		# 	(all_enc_mean, all_enc_std), \
		# 	(all_dec_mean, all_dec_std), \
		# 	log_hat_ps
		return fivo_loss


	def sample(self, seq_len):

		sample = torch.zeros(seq_len, self.x_dim)

		h = Variable(torch.zeros(self.n_layers, 1, self.h_dim))
		c = Variable(torch.zeros(self.n_layers, 1, self.h_dim))

		for t in range(seq_len):

			#prior
			prior_t = self.prior(h[-1])
			prior_mean_t = self.prior_mean(prior_t)
			prior_std_t = self.prior_std(prior_t)

			#sampling and reparameterization
			z_t = self._reparameterized_sample(prior_mean_t, prior_std_t)
			phi_z_t = self.phi_z(z_t)
			
			#decoder
			dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
			dec_mean_t = self.dec_mean(dec_t)
			#dec_std_t = self.dec_std(dec_t)

			phi_x_t = self.phi_x(dec_mean_t)

			#recurrence
			_, (h, c) = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), (h, c))

			sample[t] = dec_mean_t.data
	
		return sample


	def reset_parameters(self, stdv=1e-1):
		for weight in self.parameters():
			weight.data.normal_(0, stdv)


	def _init_weights(self, stdv):
		pass


	def _reparameterized_sample(self, mean, std):
		"""using std to sample"""
		eps = torch.FloatTensor(std.size()).normal_()
		eps = Variable(eps)
		return eps.mul(std).add_(mean)


	def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
		"""Using std to compute KLD"""
		eps = 1e-6
		kld_element =  (2 * torch.log(std_2 + eps) - 2 * torch.log(std_1 + eps) + 
			(std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
			std_2.pow(2) - 1)

		return 0.5 * torch.sum(torch.sum(kld_element, dim=-1), dim=-1)

	def _kld_gauss_dist(self, mean_1, std_1, mean_2, std_2):
		cov1 = torch.diag_embed(std_1)
		cov2 = torch.diag_embed(std_2)
		diag_normal1 = torch.distributions.MultivariateNormal(loc=mean_1, scale_tril=cov1)
		diag_normal2 = torch.distributions.MultivariateNormal(loc=mean_2, scale_tril=cov2)
		kl = torch.distributions.kl_divergence(diag_normal1, diag_normal2)

		# if torch.isnan(kl).any():
		# 	import IPython; IPython.embed()

		return torch.sum(kl)

    # def _kl_lossfunc(self, mean1, std1, mean2, std2):	
  	# 	return tf.sum(0.5 * (-1.0 - torch.log(std1) + r_logvar + ((q_mean - r_mean) ** 2 + tf.exp(q_logvar)) / tf.exp(r_logvar)), dim=-1)

	def _nll_bernoulli(self, theta, x):
		# mean w.r.t. batch, sum w.r.t input dimensions
		return - torch.sum(torch.sum(x*torch.log(theta) + (1-x)*torch.log(1-theta), dim=-1), dim=-1)


	def _nll_gauss(self, mean, std, x):
		pass
