import math
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torch.distributions import MultivariateNormal, Bernoulli
"""implementation of the Variational Recurrent
Neural Network (VRNN) from https://arxiv.org/abs/1506.02216
using unimodal isotropic gaussian distributions for 
inference, prior, and generating models."""


class VRNN(nn.Module):
	def __init__(self, x_dim, h_dim, z_dim, n_layers, num_particles, bias=False):
		super(VRNN, self).__init__()

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
		log_hat_p_acc = 0.

		logw0 = torch.log(torch.ones(self.num_zs, x.size(1), requires_grad=False) / float(self.num_zs))
		logweight = logw0		
		
		# ws = torch.ones(x.size(0), self.num_zs, x.size(1), requires_grad=False) / float(self.num_zs)
		# ws = [Variable(torch.ones(self.num_zs, x.size(1)) / float(self.num_zs), requires_grad=False) for _ in range(x.size(0)) ]# [num_seq, num_zs, batch_size]

		# with torch.autograd.set_detect_anomaly(True):			
		for t in range(x.size(0)):
			if (x[t] == 0).all():
				continue

			# Inference
			phi_x_t = self.phi_x(x[t])

			#sampling and reparameterization
			# logwnew = logwold.clone()
			hat_p = 0.
			h = h.reshape(self.n_layers, self.num_zs, x.size(1), self.h_dim)				

			# may boost learning speed by using parellelism
			phi_z_t_is = []
			for i in range(self.num_zs):
				#encoder
				enc_t = self.enc(torch.cat([phi_x_t, h[-1][i]], 1))
				enc_mean_t = self.enc_mean(enc_t)
				enc_std_t = self.enc_std(enc_t) + 1e-7
				
				encoder_dist = MultivariateNormal(enc_mean_t, scale_tril=torch.diag_embed(enc_std_t))

				#prior
				prior_t = self.prior(h[-1][i])
				prior_mean_t = self.prior_mean(prior_t)
				prior_std_t = self.prior_std(prior_t) + 1e-2

				prior_dist = MultivariateNormal(prior_mean_t, scale_tril=torch.diag_embed(prior_std_t))
			
				z_t_i = encoder_dist.rsample()  # reparametrizable
				# z_t_i = self._reparameterized_sample(enc_mean_t, enc_std_t)	# (batch_size, embed_size)

				phi_z_t = self.phi_z(z_t_i)
				phi_z_t_is.append(phi_z_t)
				
				#decoders
				dec_t = self.dec(torch.cat([phi_z_t, h[-1][i]], 1))
				dec_mean_t = self.dec_mean(dec_t)
				# dec_std_t = self.dec_std(dec_t)

				decoder_dist = Bernoulli(probs=dec_mean_t)

				# calculate p(z^i_t|h_{i-1}),  p(x_t | h_{i-1}, z^i_t), q(z^i_t)
				prior_logprob_ti = prior_dist.log_prob( z_t_i.detach() )
				encoder_logprob_ti = encoder_dist.log_prob( z_t_i.detach() )
				decoder_logprob_ti = decoder_dist.log_prob(x[t]).sum(-1)
				
				# calculate \alpha
				log_alpha_ti = prior_logprob_ti + decoder_logprob_ti - encoder_logprob_ti # [batch_size, ]				
				
				# hat_p += torch.exp(logwold[i] + log_alpha_ti)
				# logwnew[i] = logwold[i] + log_alpha_ti.detach()
				hat_p += torch.exp(logweight + log_alpha_ti)
				logweight[i] = logweight[i] + log_alpha_ti.detach()					

				# nll_loss += self._nll_gauss(dec_mean_t, dec_std_t, x[t])
				# kld_loss += self._kld_gauss_dist(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)
				# nll_loss += self._nll_bernoulli(dec_mean_t, x[t])
				
			# log_hat_p = log_hat_ps[t] + torch.log(hat_p)
			log_hat_p_acc += torch.log(hat_p) * mask[t]
			logweight = logweight - torch.log(hat_p.detach())
			# logwold = logwnew
			log_hat_ps.append(log_hat_p_acc)

			# recurrence
			h = h.reshape(self.n_layers, self.num_zs * x.size(1), self.h_dim)
			phi_x_ts = phi_x_t.repeat(self.num_zs, 1)
			phi_z_ts = torch.cat(phi_z_t_is, dim=0)

			_, (h, c) = self.rnn(torch.cat([phi_x_ts, phi_z_ts], 1).unsqueeze(0), (h, c))				
			
			#computing losses						
			# kld_loss /= self.num_zs
			# nll_loss /= self.num_zs

		fivo_loss = -log_hat_p_acc.sum()

		if torch.isnan(fivo_loss):
			import IPython; IPython.embed()

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