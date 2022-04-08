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

# Parse relevant command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--result', action='store_true',
                    help='whether output figures')
parser.add_argument('--output_dir', type=str,
                    default='./result', help='output figures path')
parser.add_argument('--use_rl', action='store_true',
                    help='whether use rl algorithm')
args = parser.parse_args()


# Define variables for RL env
class TcpRlEnv(Structure):
    _pack_ = 1
    _fields_ = [
        ('nodeId', c_uint32),
        ('socketUid', c_uint32),
        ('envType', c_uint8),
        ('simTime_us', c_int64),
        ('ssThresh', c_uint32),
        ('cWnd', c_uint32),
        ('segmentSize', c_uint32),
        ('segmentsAcked', c_uint32),
        ('bytesInFlight', c_uint32),
    ]


class TcpRlAct(Structure):
    _pack_ = 1
    _fields_ = [
        ('new_ssThresh', c_uint32),
        ('new_cWnd', c_uint32)
    ]

# Create RL env
Init(1234, 4096)
var = Ns3AIRL(1234, TcpRlEnv, TcpRlAct)
res_list = ['ssThresh_l', 'cWnd_l', 'segmentsAcked_l',
            'segmentSize_l', 'bytesInFlight_l']

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

exp = Experiment(1234, 4096, 'rl-tcp', '../../')
exp.run(show_output=0)
# Initialize trainer and momery replay
ram = buffer.MemoryBuffer(MAX_BUFFER)
trainer = train.Trainer(S_DIM, A_DIM, A_MAX, ram)
r_list = []
try:
	while not var.isFinish():
		with var as data:
			if not data:
				break
			#print('EPISODE :- ', _ep)
			ssThresh = data.env.ssThresh
			cWnd = data.env.cWnd
			segmentsAcked = data.env.segmentsAcked
			segmentSize = data.env.segmentSize
			bytesInFlight = data.env.bytesInFlight
			if args.result:
				for res in res_list:
					globals()[res].append(globals()[res[:-2]])
			observation = [ssThresh, cWnd, segmentsAcked, segmentSize, bytesInFlight]
			state = np.float32(observation)
			action = trainer.get_exploration_action(state)
			print('Action:' , action)
			print(int((2**action)*cWnd))
			data.act.new_cWnd = int((2**action)*cWnd)
			data.act.new_ssThresh = int(max(2 * segmentSize, bytesInFlight / 2))

			new_observation = [data.env.ssThresh, data.env.cWnd, data.env.segmentsAcked,\
								data.env.segmentSize, data.env.bytesInFlight]
			reward = segmentsAcked - bytesInFlight - cWnd
			r_list.append(reward)
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
