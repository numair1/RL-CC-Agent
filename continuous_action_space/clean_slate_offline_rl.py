import numpy as np
import gc
import train
import buffer
from py_interface import *
import argparse
import RL_env_setup_continuous as rlesc
import normalizer
import graph
import pickle as pkl

# Parse relevant command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--result', action='store_true', help='whether output figures')
args = parser.parse_args()

# Set up parameters for NN training
MAX_EPISODES = 1
MAX_STEPS = 1000
MAX_BUFFER = 1000000
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
trainer.load_models(100)

throughputs = []  # a list of throughputs
rtts = []
standardizer = normalizer.Normalizer(S_DIM)
try:
	for i in range(MAX_EPISODES):
		print("EPISODE: ", i)
		exp.reset()
		Init(1234, 4096)
		var = Ns3AIRL(1234, rlesc.TcpRlEnv, rlesc.TcpRlAct)
		#ns3Settings = {'error_p': 1.0}
		pro = exp.run(show_output=False)
		cur_throughputs = []
		cur_rtts = []
		while not var.isFinish():
			with var as data:
				if not data:
					break
				# these 2 are unused by our RLL algorithm but used for TCP
				ssThresh, segmentSize = data.env.ssThresh, data.env.segmentSize

				# these 5 are used by our RLL algorithm
				cWnd, segmentsAcked, bytesInFlight, throughput, rtt = \
				data.env.cWnd, data.env.segmentsAcked, data.env.bytesInFlight, data.env.throughput, data.env.rtt

				cur_throughputs.append(throughput)
				if rtt > 0:
					cur_rtts.append(rtt)

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

		# check memory consumption and clear memory
		gc.collect()
		throughputs.append(cur_throughputs)
		rtts.append(cur_rtts)
except KeyboardInterrupt:
	exp.kill()
	del exp

with open('./data/clean_slate/throughputs.pickle', 'wb') as fh:
    pkl.dump(throughputs, fh)
with open('./data/clean_slate/rtts.pickle', 'wb') as fh:
	pkl.dump(rtts, fh)

if args.result:
	graph.graph_avg_rewards(avg_rewards)
	graph.graph_throughputs(throughputs)
	graph.graph_actions(actions)
print('Completed episodes')
