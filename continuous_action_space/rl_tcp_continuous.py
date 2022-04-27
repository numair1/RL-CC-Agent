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
MAX_EPISODES = 100
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

avg_rewards = []  # a list of average reward per episode
throughputs = []  # a list of throughputs
actions = []
standardizer = normalizer.Normalizer(S_DIM)

throughputs_2d = []
actions_2d = []
rewards_2d = []

throughputs_online = []  # a list of throughputs
rtts_online = []
throughputs_clean = []  # a list of throughputs
rtts_clean = []
try:
	for i in range(MAX_EPISODES):
		print("EPISODE: ", i)
		exp.reset()
		Init(1234, 4096)
		var = Ns3AIRL(1234, rlesc.TcpRlEnv, rlesc.TcpRlAct)
		if i > 89:
			ns3Settings = {'bottleneck_bandwidth': "2Mbps", 'bottleneck_delay': "5ms"}
			pro = exp.run(setting = ns3Settings, show_output=False)
		else:
			pro = exp.run(show_output=False)
		observation = []
		state = []
		reward_counter = 0  # used to calculate average reward
		reward_sum = 0.0
		unnormalized_state = []

		cur_throughputs_online = []  # a list of throughputs
		cur_rtts_online = []
		cur_throughputs_clean = []  # a list of throughputs
		cur_rtts_clean = []

		curr_throughput = []
		curr_reward = []
		curr_action = []
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

				if i <= 89:
					throughputs.append(throughput)
					curr_throughput.append(throughput)
					standardizer.observe(observation)
					standardized_observation = standardizer.normalize(observation)
					if throughput == 0:
						standardized_observation[-1] = 50.0  # some very large rtt
						reward = -observation[0]  # cWnd
					else:
						reward = throughput / rtt

					standardizer.observe_reward(reward)
					standardized_reward = standardizer.normalize_reward(reward)

					reward_counter += 1
					reward_sum += standardized_reward
					curr_reward.append(reward)
					new_state = np.float32(standardized_observation)
					if len(state) != 0:  # len(state) == 0 on first iteration of while loop
						if new_cWnd != observation[0]:
							action = math.log2(observation[0] / unnormalized_state[0])
							action = np.asarray([action])
						ram.add(state, action, standardized_reward, new_state)
						if ram.len > 512:
							trainer.optimize()

					state = new_state
					unnormalized_state = observation
					action = trainer.get_exploration_action(state)
					actions.append(action)
					curr_action.append(action)
					new_cWnd = int((2**action)*cWnd)
					new_ssThresh = int(max(2 * segmentSize, bytesInFlight / 2))

					data.act.new_cWnd = new_cWnd
					data.act.new_ssThresh = new_ssThresh

				elif i > 89 and i < 95: # clean slate
					cur_throughputs_clean.append(throughput)
					if rtt > 0:
						cur_rtts_clean.append(rtt)

					observation = [cWnd, segmentsAcked, bytesInFlight, throughput, rtt]
					standardizer.observe(observation)
					standardized_observation = standardizer.normalize(observation)
					standardized_observation = np.float32(standardized_observation)

					if throughput == 0:
						continue
					action = trainer.get_exploration_action(standardized_observation)
					new_cWnd = int((2**action)*cWnd)
					new_ssThresh = int(max(2 * segmentSize, bytesInFlight / 2))

					data.act.new_cWnd = new_cWnd
					data.act.new_ssThresh = new_ssThresh

				elif i >= 95:  # online
					if j % 5 != 0 and j > 20:  # TCP is supposed to act
						if throughput != 0:
							assert segmentsAcked > 0
							new_cWnd, new_ssThresh = utils.TCP(cWnd, ssThresh, segmentsAcked, segmentSize, bytesInFlight)
							throughputs.append(throughput)
							actions.append(new_cWnd)
							data.act.new_cWnd = new_cWnd
							data.act.new_ssThresh = new_ssThresh
						continue

					throughputs.append(throughput)
					standardizer.observe(observation)
					standardized_observation = standardizer.normalize(observation)
					if throughput == 0:
						standardized_observation[-1] = 50.0  # some very large rtt
						reward = -observation[0]  # cWnd
					else:
						reward = throughput / rtt

					standardizer.observe_reward(reward)
					standardized_reward = standardizer.normalize_reward(reward)

					reward_counter += 1
					reward_sum += standardized_reward
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
					actions.append(action)
					new_cWnd = int((2**action)*cWnd)
					new_ssThresh = int(max(2 * segmentSize, bytesInFlight / 2))

					cur_throughputs_online.append(throughput)
					if rtt > 0:
						cur_rtts_online.append(rtt)

					data.act.new_cWnd = new_cWnd
					data.act.new_ssThresh = new_ssThresh
		if i > 89 and i < 95:
			throughputs_clean.append(cur_throughputs_clean)
			rtts_clean.append(cur_rtts_clean)
		elif i >= 95:
			throughputs_online.append(cur_throughputs_online)
			rtts_online.append(cur_rtts_online)
		avg_rewards.append(reward_sum / reward_counter)
		throughputs_2d.append(curr_throughput)
		actions_2d.append(curr_action)
		rewards_2d.append(curr_reward)
		# check memory consumption and clear memory
		gc.collect()
	trainer.save_models(MAX_EPISODES)
except KeyboardInterrupt:
	exp.kill()
	del exp

with open('./data/clean_slate/throughputs.pickle', 'wb') as fh:
    pkl.dump(throughputs_clean, fh)
with open('./data/clean_slate/rtts.pickle', 'wb') as fh:
    pkl.dump(rtts_clean, fh)
with open('./data/online/throughputs.pickle', 'wb') as fh:
    pkl.dump(throughputs_online, fh)
with open('./data/online/rtts.pickle', 'wb') as fh:
    pkl.dump(rtts_online, fh)

if args.result:
	graph.graph_avg_rewards(avg_rewards)
	graph.graph_throughputs(throughputs)
	graph.graph_actions(actions)
	np.save('reward_2d', rewards_2d)
	np.save('actions_2d', actions_2d)
	np.save('throughputs_2d', throughputs_2d)
print('Completed episodes')
