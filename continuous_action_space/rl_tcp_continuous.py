from __future__ import division
import gym
import numpy as np
import torch
from torch.autograd import Variable
import os
import psutil
import gc
import train
import buffer
from py_interface import *
from ctypes import *
import argparse
import torch.nn as nn
import matplotlib.pyplot as plt
import RL_env_setup_continuous as rlesc

# Parse relevant command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--result', action='store_true',
					help='whether output figures')
parser.add_argument('--output_dir', type=str,
					default='./result', help='output figures path')
parser.add_argument('--use_rl', action='store_true',
					help='whether use rl algorithm')
args = parser.parse_args()

# Set up logging and saving
if args.result:
	for res in res_list:
		globals()[res] = []
	if args.output_dir:
		if not os.path.exists(args.output_dir):
			os.mkdir(args.output_dir)

# Set up parameters for NN training
MAX_EPISODES = 5000
MAX_STEPS = 1000
MAX_BUFFER = 1000000
MAX_TOTAL_REWARD = 300
# Connect the relevant variables here
S_DIM = 5
A_DIM = 1
A_MAX = 1.8

print(' State Dimensions :- ', S_DIM)
print(' Action Dimensions :- ', A_DIM)
print(' Action Max :- ', A_MAX)

exp = Experiment(1234, 4096, 'rl-tcp', '../../../')
exp.run(show_output=0)
# Initialize trainer and momery replay
ram = buffer.MemoryBuffer(MAX_BUFFER)
trainer = train.Trainer(S_DIM, A_DIM, A_MAX, ram)
r_list = []
try:
	for i in range(MAX_EPISODES):
		print("EPISODE: ", i)
		exp.reset()
		Init(1234, 4096)
		var = Ns3AIRL(1234, rlesc.TcpRlEnv, rlesc.TcpRlAct)
		#ns3Settings = {'error_p': 1.0}
		pro = exp.run(show_output=False)
		while not var.isFinish():
			with var as data:
				if not data:
					break
				#print('EPISODE :- ', _ep)
				simTime_us = data.env.simTime_us
				ssThresh = data.env.ssThresh
				cWnd = data.env.cWnd
				segmentsAcked = data.env.segmentsAcked
				segmentSize = data.env.segmentSize
				bytesInFlight = data.env.bytesInFlight
				throughput = data.env.throughput
				rtt = data.env.rtt
				print("---------------STATE----------------")
				print("simTime_us: ", simTime_us)
				print("ssThres: ", ssThresh, "\n", "cWnd: ", cWnd)
				print("segmentsAcked: ", segmentsAcked, "\n", "segmentSize: ", segmentSize)
				print("bytesInFlight: ", bytesInFlight)
				print("throughput: ", throughput / (2 ** 17), "\n", "rtt: ", rtt / (10 ** 3))
				print("---------------STATE----------------")
				if args.result:
					for res in res_list:
						globals()[res].append(globals()[res[:-2]])

				if rtt <= 0 or not args.use_rl:
					new_cWnd = 1
					new_ssThresh = 1
					# IncreaseWindow
					if (cWnd < ssThresh):
						# slow start
						if (segmentsAcked >= 1):
							new_cWnd = cWnd + segmentSize
					if (cWnd >= ssThresh):
						# congestion avoidance
						if (segmentsAcked > 0):
							adder = 1.0 * (segmentSize * segmentSize) / cWnd
							adder = int(max(1.0, adder))
							new_cWnd = cWnd + adder
					# GetSsThresh
					new_ssThresh = int(max(2 * segmentSize, bytesInFlight / 2))
					data.act.new_cWnd = new_cWnd
					data.act.new_ssThresh = new_ssThresh
				else:
					observation = [ssThresh, cWnd, segmentsAcked, segmentSize, bytesInFlight]
					state = np.float32(observation)
					action = trainer.get_exploration_action(state)
					print('Action:' , action)
					print(int((2**action)*cWnd))
					data.act.new_cWnd = int((2**action)*cWnd)
					data.act.new_ssThresh = int(max(2 * segmentSize, bytesInFlight / 2))

					simTime_us_post = data.env.simTime_us
					ssThresh_post = data.env.ssThresh
					cWnd_post = data.env.cWnd
					segmentsAcked_post = data.env.segmentsAcked
					segmentSize_post = data.env.segmentSize
					bytesInFlight_post = data.env.bytesInFlight
					throughput_post = data.env.throughput
					rtt_post = data.env.rtt

					if simTime_us != simTime_us_post or ssThresh != ssThresh_post or cWnd != cWnd_post \
						or segmentsAcked != segmentsAcked_post or segmentSize != segmentSize_post \
						or bytesInFlight != bytesInFlight_post or throughput != throughput_post or rtt != rtt_post:
						print("NOT EQUAL")

					new_observation = [ssThresh, cWnd, segmentsAcked, segmentSize, bytesInFlight]

					print("---------STATE AFTER ACTION-----------")
					print("simTime_us: ", simTime_us_post)
					print("ssThres: ", ssThresh_post, "\n", "cWnd: ", cWnd_post)
					print("segmentsAcked: ", segmentsAcked_post, "\n", "segmentSize: ", segmentSize_post)
					print("bytesInFlight: ", bytesInFlight_post)
					print("throughput: ", throughput_post / (2 ** 17), "\n", "rtt: ", rtt_post / (10 ** 3))
					print("---------STATE AFTER ACTION-----------")

					if rtt > 0:

						reward = throughput / rtt
						print("REWARD: ", reward)
						r_list.append(reward)
						s_ = [ssThresh, cWnd, segmentsAcked,
							  segmentSize, bytesInFlight, throughput, rtt]
						done = False
						info = []

						if done:
							new_state = None
						else:
							new_state = np.float32(new_observation)
							# push this exp in ram
							ram.add(state, action, reward, new_state)
						observation = new_observation

					# perform optimization
					trainer.optimize()
				if done:
					break

		# check memory consumption and clear memory
		gc.collect()
		# process = psutil.Process(os.getpid())
		# print(process.memory_info().rss)
		#if _ep%100 == 0:
		#	trainer.save_models(_ep)
except KeyboardInterrupt:
	exp.kill()
	del exp
if args.result:
	y = r_list
	x = range(len(y))
	plt.clf()
	plt.plot(x, y, label=res[:-2], linewidth=1, color='r')
	plt.xlabel('Step Number')
	plt.title('Information of Reward')
	plt.savefig('{}.png'.format(os.path.join(args.output_dir, 'reward')))
	for res in res_list:
		y = globals()[res]
		x = range(len(y))
		plt.clf()
		plt.plot(x, y, label=res[:-2], linewidth=1, color='r')
		plt.xlabel('Step Number')
		plt.title('Information of {}'.format(res[:-2]))
		plt.savefig('{}.png'.format(os.path.join(args.output_dir, res[:-2])))
print('Completed episodes')
