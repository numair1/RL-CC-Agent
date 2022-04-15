from py_interface import *
from ctypes import *
import os
import torch
import argparse
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt


class net(nn.Module):
	def __init__(self):
		super(net, self).__init__()
		self.layers = nn.Sequential(
			nn.Linear(7, 20),
			nn.ReLU(),
			nn.Linear(20, 20),
			nn.ReLU(),
			nn.Linear(20, 4),
		)

	def forward(self, x):
		return self.layers(x)


class DQN(object):
	def __init__(self):
		self.eval_net = net()
		self.target_net = net()
		self.learn_step = 0
		self.batchsize = 32
		self.observer_shape = 7
		self.target_replace = 100
		self.memory_counter = 0
		self.memory_capacity = 1000
		self.memory = np.zeros((2000, 2*7+2))	# s, a, r, s'
		self.optimizer = torch.optim.Adam(
			self.eval_net.parameters(), lr=0.0001)
		self.loss_func = nn.MSELoss()

	def choose_action(self, x):
		x = torch.Tensor(x)
		if np.random.uniform() > 0.99 ** self.memory_counter:	# choose best
			action = self.eval_net.forward(x)
			action = torch.argmax(action, 0).numpy()
		else:	# explore
			action = np.random.randint(0, 4)
		return action

	def store_transition(self, s, a, r, s_):
		index = self.memory_counter % self.memory_capacity
		self.memory[index, :] = np.hstack((s, [a, r], s_))
		self.memory_counter += 1

	def learn(self, ):
		self.learn_step += 1
		if self.learn_step % self.target_replace == 0:
			self.target_net.load_state_dict(self.eval_net.state_dict())
		sample_list = np.random.choice(self.memory_capacity, self.batchsize)
		# choose a mini batch
		sample = self.memory[sample_list, :]
		s = torch.Tensor(sample[:, :self.observer_shape])
		a = torch.LongTensor(
			sample[:, self.observer_shape:self.observer_shape+1])
		r = torch.Tensor(
			sample[:, self.observer_shape+1:self.observer_shape+2])
		s_ = torch.Tensor(sample[:, self.observer_shape+2:])
		q_eval = self.eval_net(s).gather(1, a)
		q_next = self.target_net(s_).detach()
		q_target = r + 0.8 * q_next.max(1, True)[0].data

		loss = self.loss_func(q_eval, q_target)

		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
