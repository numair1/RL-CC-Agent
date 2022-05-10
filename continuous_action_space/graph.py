import matplotlib.pyplot as plt
import pickle
import numpy as np

def graph_helper(lst, xlabel, ylabel, title, filename):
	plt.clf()
	plt.plot(range(len(lst)), lst)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(title)
	plt.savefig('./graphs/' + filename + '.png')

# average rewards per episode
def graph_avg_rewards():
	with open('./data/rewards_train', 'rb') as fh:
		rewards_train = pickle.load(fh)
	with open('./data/rewards_cs', 'rb') as fh:
		rewards_cs = pickle.load(fh)
	with open('./data/rewards_online', 'rb') as fh:
		rewards_online = pickle.load(fh)

	avg_rewards = [np.average(episode) for episode in rewards_train + rewards_cs + rewards_online]

	graph_helper(avg_rewards, 'Episode', 'Reward', 'Average Reward per Episode', 'avg_reward')

# througput over time
def graph_throughputs():
	with open('./data/throughputs_train', 'rb') as fh:
		throughputs_train = pickle.load(fh)
	with open('./data/throughputs_cs', 'rb') as fh:
		throughputs_cs = pickle.load(fh)
	with open('./data/throughputs_online', 'rb') as fh:
		throughputs_online = pickle.load(fh)

	throughputs = np.hstack(np.array(throughputs_train + throughputs_cs + throughputs_online, dtype=object))
	graph_helper(throughputs, 'Time', 'Throughput (Mbps)', 'Throughput over time', 'throughput')

# action over time
def graph_actions():
	with open('./data/actions_train', 'rb') as fh:
		actions_train = pickle.load(fh)
	with open('./data/actions_cs', 'rb') as fh:
		actions_cs = pickle.load(fh)
	with open('./data/actions_online', 'rb') as fh:
		actions_online = actions_online = pickle.load(fh)

	actions = np.hstack(np.array(actions_train + actions_cs + actions_online, dtype=object))
	graph_helper(actions, 'Time', 'Action', 'Action over time', 'action')


graph_avg_rewards()
graph_throughputs()
graph_actions()
