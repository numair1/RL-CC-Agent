from py_interface import *
import argparse
import RL_env_setup_continuous as rlesc
import utils
import graph
import pickle as pkl

# Parse relevant command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--result', action='store_true', help='whether output figures')
args = parser.parse_args()

# Set up parameters for NN training
MAX_EPISODES = 5

exp = Experiment(1234, 4096, 'rl-tcp', '../../../')
exp.run(show_output=0)

throughputs = []  # a list of throughputs
rtts = []
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

				# new_cWnd, new_ssThresh = utils.TCP(cWnd, ssThresh, segmentsAcked, segmentSize, bytesInFlight)

				cur_throughputs.append(throughput)
				if rtt > 0:
					cur_rtts.append(rtt)

				# data.act.new_cWnd = new_cWnd
				# data.act.new_ssThresh = new_ssThresh
		throughputs.append(cur_throughputs)
		rtts.append(cur_rtts)
except KeyboardInterrupt:
	exp.kill()
	del exp

with open('./data/TCP/throughputs.pickle', 'wb') as fh:
    pkl.dump(throughputs, fh)
with open('./data/TCP/rtts.pickle', 'wb') as fh:
    pkl.dump(rtts, fh)

if args.result:
	graph.graph_throughputs(throughputs)
print('Completed episodes')
