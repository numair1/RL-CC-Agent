import numpy as np
import gc
import train
import buffer
from py_interface import *
import argparse
import RL_env_setup_continuous as rlesc
import normalizer
import utils
import graph
import math
import torch
import pickle as pkl

torch.manual_seed(2)
# Parse relevant command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--result', action='store_true', help='whether output figures')
args = parser.parse_args()

# Set up parameters for NN training
MAX_EPISODES = 10
MAX_STEPS = 1000
MAX_BUFFER = 100000
MAX_TOTAL_REWARD = 300
# Connect the relevant variables here
S_DIM = 5
A_DIM = 1
A_MAX = 2

print(' State Dimensions :- ', S_DIM)
print(' Action Dimensions :- ', A_DIM)
print(' Action Max :- ', A_MAX)

exp = Experiment(1234, 4096, 'rl-tcp', '../../../')
exp.run(show_output=0)
# Initialize trainer and momery replay
ram = buffer.MemoryBuffer(MAX_BUFFER)
trainer = train.Trainer(S_DIM, A_DIM, A_MAX, ram)

throughputs_train = []
actions_train = []
rewards_train = []

throughputs_cs = []
actions_cs = []
rewards_cs = []

throughputs_online = []
actions_online = []
rewards_online = []
standardizer = normalizer.Normalizer(S_DIM)
try:
	for i in range(MAX_EPISODES):
		print("EPISODE: ", i)
		exp.reset()
		Init(1234, 4096)
		var = Ns3AIRL(1234, rlesc.TcpRlEnv, rlesc.TcpRlAct)
		if i > 8:
			ns3Settings = {'bottleneck_bandwidth': "2Mbps", 'bottleneck_delay': "5ms"}
			pro = exp.run(setting = ns3Settings, show_output=False)
		else:
			pro = exp.run(show_output=False)
		observation = []
		state = []
		reward_counter = 0  # used to calculate average reward
		reward_sum = 0.0
		unnormalized_state = []

		cur_throughputs_train = []
		cur_actions_train = []
		cur_rewards_train = []

		cur_throughputs_cs = []
		cur_actions_cs = []
		cur_rewards_cs = []

		cur_throughputs_online = []
		cur_actions_online = []
		cur_rewards_online = []

		j = 0
		while not var.isFinish():
			with var as data:
				if not data:
					break
				j += 1
				# these 2 are unused by our RLL algorithm but used for TCP
				ssThresh, segmentSize = data.env.ssThresh, data.env.segmentSize

				# these 5 are used by our RLL algorithm
				cWnd, segmentsAcked, bytesInFlight, throughput, rtt = \
				data.env.cWnd, data.env.segmentsAcked, data.env.bytesInFlight, data.env.throughput, data.env.rtt

				observation = [cWnd, segmentsAcked, bytesInFlight, throughput, rtt]

				if i <= 8: # training
					standardizer.observe(observation)
					standardized_observation = standardizer.normalize(observation)
					if throughput == 0:
						standardized_observation[-1] = 50.0  # some very large rtt
						reward = -observation[0]  # cWnd
					else:
						reward = throughput / rtt

					standardizer.observe_reward(reward)
					standardized_reward = standardizer.normalize_reward(reward)

					new_state = np.float32(standardized_observation)
					if len(state) != 0:  # len(state) == 0 on first iteration of while loop
						if new_cWnd != observation[0]:
							if observation[0] != 0 and unnormalized_state[0] != 0:
								action = math.log2(observation[0] / unnormalized_state[0])
								action = np.asarray([action])
								ram.add(state, action, standardized_reward, new_state)
								cur_throughputs_train.append(throughput)
								cur_actions_train.append(action)
								if ram.len > 512:
									trainer.optimize()

					state = new_state
					unnormalized_state = observation
					action = trainer.get_exploration_action(state)
					new_cWnd = int((2**action)*cWnd)
					new_ssThresh = int(max(2 * segmentSize, bytesInFlight / 2))

					data.act.new_cWnd = new_cWnd
					data.act.new_ssThresh = new_ssThresh

					cur_rewards_train.append(standardized_reward)

				elif i > 8 and i < 9: # clean slate
					observation = [cWnd, segmentsAcked, bytesInFlight, throughput, rtt]
					standardizer.observe(observation)
					standardized_observation = standardizer.normalize(observation)
					standardized_observation = np.float32(standardized_observation)

					if throughput == 0:
						action = 0.0
						reward = -observation[0]  # cWnd
					else:
						action = trainer.get_exploration_action(standardized_observation)
						reward = throughput / rtt

					standardizer.observe_reward(reward)
					standardized_reward = standardizer.normalize_reward(reward)

					new_cWnd = int((2**action)*cWnd)
					new_ssThresh = int(max(2 * segmentSize, bytesInFlight / 2))

					data.act.new_cWnd = new_cWnd
					data.act.new_ssThresh = new_ssThresh

					cur_throughputs_cs.append(throughput)
					cur_actions_cs.append(action)
					cur_rewards_cs.append(standardized_reward)

				elif i >= 9:  # online
					if j % 5 != 0 and j > 20:  # TCP is supposed to act
						if throughput != 0:
							assert segmentsAcked > 0
							new_cWnd, new_ssThresh = utils.TCP(cWnd, ssThresh, segmentsAcked, segmentSize, bytesInFlight)

							action = min(max(math.log2(new_cWnd/cWnd), -2.0), 2.0)

							data.act.new_cWnd = new_cWnd
							data.act.new_ssThresh = new_ssThresh
						else:
							action = 0.0
					else:
						standardizer.observe(observation)
						standardized_observation = standardizer.normalize(observation)
						if throughput == 0:
							standardized_observation[-1] = 50.0  # some very large rtt
							reward = -observation[0]  # cWnd
						else:
							reward = throughput / rtt

						standardizer.observe_reward(reward)
						standardized_reward = standardizer.normalize_reward(reward)

						new_state = np.float32(standardized_observation)
						if len(state) != 0:  # len(state) == 0 on first iteration of while loop
							if new_cWnd != observation[0]:
								action = math.log2(observation[0] / unnormalized_state[0])
								action = np.asarray([action])
							ram.add(state, action, standardized_reward, new_state)
							trainer.optimize()

						state = new_state
						unnormalized_state = observation
						action = trainer.get_exploration_action(state)
						new_cWnd = int((2**action)*cWnd)
						new_ssThresh = int(max(2 * segmentSize, bytesInFlight / 2))

						data.act.new_cWnd = new_cWnd
						data.act.new_ssThresh = new_ssThresh

					cur_throughputs_online.append(throughput)
					cur_actions_online.append(action)
					cur_rewards_online.append(standardized_reward)
		if i <= 8:  # training
			throughputs_train.append(cur_throughputs_train)
			actions_train.append(cur_actions_train)
			rewards_train.append(cur_rewards_train)
		elif i > 8 and i < 9:  # clean slate
			throughputs_cs.append(cur_throughputs_cs)
			actions_cs.append(cur_actions_cs)
			rewards_cs.append(cur_rewards_cs)
		else: #  i >= 95:  # online
			throughputs_online.append(cur_throughputs_online)
			actions_online.append(cur_actions_online)
			rewards_online.append(cur_rewards_online)
		# check memory consumption and clear memory
		gc.collect()
	trainer.save_models(MAX_EPISODES)
except KeyboardInterrupt:
	exp.kill()
	del exp

np.save('./data/throughputs_train', throughputs_train)
np.save('./data/actions_train', actions_train)
np.save('./data/rewards_train', rewards_train)

np.save('./data/throughputs_cs', throughputs_cs)
np.save('./data/actions_cs', actions_cs)
np.save('./data/rewards_cs', rewards_cs)

np.save('./data/throughputs_online', throughputs_online)
np.save('./data/actions_online', actions_online)
np.save('./data/rewards_online', rewards_online)

print('Completed episodes')
