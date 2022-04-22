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
# import rl_tcp_continuous

# Parse relevant command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--result', action='store_true', help='whether output figures')
args = parser.parse_args()

# Set up parameters for NN training
NUM_EPISODES = 5
MAX_BUFFER = 1000000
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
trainer.load_models(5)

avg_rewards = []  # a list of average reward per episode
throughputs = []  # a list of throughputs
actions = []
standardizer = normalizer.Normalizer(S_DIM)
try:
	for i in range(NUM_EPISODES):
		print("EPISODE: ", i)
		exp.reset()
		Init(1234, 4096)
		var = Ns3AIRL(1234, rlesc.TcpRlEnv, rlesc.TcpRlAct)
		#ns3Settings = {'error_p': 1.0}
		pro = exp.run(show_output=False)
		observation = []
		state = []
		reward_counter = 0  # used to calculate average reward
		reward_sum = 0.0
		unnormalized_state = []
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

				if j % 10 != 0 and j > 20:  # TCP is supposed to act
					if throughput != 0:
						print("TCP AGENT ACTED")
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
					print("IF BRANCH")
					standardized_observation[-1] = 50.0  # some very large rtt
					reward = -observation[0]  # cWnd
				else:
					print("ELSE BRANCH")
					reward = throughput / rtt

				standardizer.observe_reward(reward)
				standardized_reward = standardizer.normalize_reward(reward)

				print("RAW REWARD: ", reward)
				print("STANDARDIZED REWARD: ", standardized_reward)
				reward_counter += 1
				reward_sum += standardized_reward
				new_state = np.float32(standardized_observation)
				if len(state) != 0:  # len(state) == 0 on first iteration of while loop
					if new_cWnd != observation[0]:
						action = math.log2(observation[0] / unnormalized_state[0])
						action = np.asarray([action])
					ram.add(state, action, standardized_reward, new_state)
					trainer.optimize()
				print('------------------------------------')
				print('Raw Observation')
				print(observation)
				print('State')
				print(state)
				print('New State')
				print(new_state)
				print('------------------------------------')

				state = new_state
				unnormalized_state = observation
				action = trainer.get_exploration_action(state)
				actions.append(action)
				new_cWnd = int((2**action)*cWnd)
				new_ssThresh = int(max(2 * segmentSize, bytesInFlight / 2))
				if new_cWnd >= 1e8:  # 100 million. 100,000,000
					print("BREAKING")
					break

				print('AGENT ACTION:' , action)
				print('new_cwnd:', new_cWnd, '\n')

				data.act.new_cWnd = new_cWnd
				data.act.new_ssThresh = new_ssThresh

		avg_rewards.append(reward_sum / reward_counter)
		# check memory consumption and clear memory
		gc.collect()
except KeyboardInterrupt:
	exp.kill()
	del exp
if args.result:
	graph.graph_avg_rewards(avg_rewards)
	graph.graph_throughputs(throughputs)
	graph.graph_actions(actions)
print('Completed episodes')
