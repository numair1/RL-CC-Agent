# Code borrowed from Orca repository
import numpy as np
class Normalizer():
	def __init__(self, S_DIM):
		self.n = 0
		self.mean = np.zeros(S_DIM)
		self.mean_diff = np.zeros(S_DIM)
		self.var = np.zeros(S_DIM)
		self.dim = S_DIM
		self.min = np.zeros(S_DIM)
		self.mean_r = 0
		self.mean_diff_r = 0
		self.var_r = 0
		self.n_r = 0

	def observe(self, x):
		self.n += 1
		last_mean = np.copy(self.mean)
		self.mean += (x-self.mean)/self.n
		self.mean_diff += (x-last_mean)*(x-self.mean)
		self.var = self.mean_diff/self.n

	def normalize(self, inputs):
		obs_std = np.sqrt(self.var)
		a=np.zeros(self.dim)
		if self.n > 2:
			a=(inputs - self.mean)/obs_std
			for i in range(0,self.dim):
				if a[i] < self.min[i]:
					self.min[i] = a[i]
			return a
		else:
			return np.zeros(self.dim)

	def observe_reward(self, x):
		self.n_r += 1
		last_mean = np.copy(self.mean_r)
		self.mean_r += (x-self.mean_r)/self.n_r
		self.mean_diff_r += (x-last_mean)*(x-self.mean_r)
		self.var_r = self.mean_diff_r/self.n_r

	def normalize_reward(self, input):
		obs_std = np.sqrt(self.var_r)
		a=0
		if self.n_r > 2:
			a=input/obs_std
		return a
