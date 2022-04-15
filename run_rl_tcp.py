# -*- Mode: python; py-indent-offset: 4; indent-tabs-mode: nil; coding: utf-8; -*-
# Copyright (c) 2020 Huazhong University of Science and Technology, Dian Group
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 2 as
# published by the Free Software Foundation;
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
#
# Author: Pengyu Liu <eic_lpy@hust.edu.cn>
#		 Hao Yin <haoyin@uw.edu>

from py_interface import *
from ctypes import *
import os
import torch
import argparse
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import RL_env_setup
import RL_agent

parser = argparse.ArgumentParser()
parser.add_argument('--result', action='store_true',
					help='whether output figures')
parser.add_argument('--output_dir', type=str,
					default='./result', help='output figures path')
parser.add_argument('--use_rl', action='store_true',
					help='whether use rl algorithm')


res_list = ['ssThresh_l', 'cWnd_l', 'segmentsAcked_l',
			'segmentSize_l', 'bytesInFlight_l', 'throughput_l', 'rtt_l']
args = parser.parse_args()

if args.result:
	for res in res_list:
		globals()[res] = []
	if args.output_dir:
		if not os.path.exists(args.output_dir):
			os.mkdir(args.output_dir)

if args.use_rl:
	dqn = RL_agent.DQN()
r_list = []
exp = Experiment(1235, 4096, 'rl-tcp', '../../')
try:
	for i in range(100):
		print(i)
		exp.reset()
		Init(1234, 4096)
		var = Ns3AIRL(1234, RL_env_setup.TcpRlEnv, RL_env_setup.TcpRlAct)
		#ns3Settings = {'error_p': 1.0}
		pro = exp.run(show_output=False)
		while not var.isFinish():
			with var as data:
				if not data:
					break
		#		 print(var.GetVersion())
				ssThresh = data.env.ssThresh
				cWnd = data.env.cWnd
				segmentsAcked = data.env.segmentsAcked
				segmentSize = data.env.segmentSize
				bytesInFlight = data.env.bytesInFlight
				throughput = data.env.throughput
				rtt = data.env.rtt
				if args.result:
					for res in res_list:
						globals()[res].append(globals()[res[:-2]])
						#print(globals()[res][-1])

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
					data.act.new_cWnd = 100000
					data.act.new_ssThresh = new_ssThresh
				else:
					s = [ssThresh, cWnd, segmentsAcked, segmentSize, bytesInFlight, throughput, rtt]
					a = dqn.choose_action(s)
					if a & 1:
						new_cWnd = cWnd + segmentSize
					else:
						if(cWnd > 0):
							new_cWnd = cWnd + int(max(1, (segmentSize * segmentSize) / cWnd))
						elif cWnd == 0:
							new_cWnd = 1
					if a < 3:
						new_ssThresh = 2 * segmentSize
					else:
						new_ssThresh = int(bytesInFlight / 2)
					# print("Expoch Actions")
					# print('newCwnd',new_cWnd)
					# print('new_ssThresh', new_ssThresh)
					data.act.new_cWnd = new_cWnd
					data.act.new_ssThresh = new_ssThresh

					ssThresh = data.env.ssThresh
					cWnd = data.env.cWnd
					segmentsAcked = data.env.segmentsAcked
					segmentSize = data.env.segmentSize
					bytesInFlight = data.env.bytesInFlight
					throughput = data.env.throughput
					rtt = data.env.rtt
					# modify the reward
					# r = segmentsAcked - bytesInFlight - cWnd
					if rtt > 0:
						r = throughput / rtt
						r_list.append(r)
						s_ = [ssThresh, cWnd, segmentsAcked,
							  segmentSize, bytesInFlight, throughput, rtt]
						dqn.store_transition(s, a, r, s_)
					if dqn.memory_counter > dqn.memory_capacity:
						dqn.learn()
		pro.wait()
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
