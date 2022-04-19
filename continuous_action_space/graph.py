import matplotlib.pyplot as plt

# average rewards per episode
def graph_avg_rewards(avg_rewards):
	graph_helper(avg_rewards, 'Episode', 'Reward', 'Average Reward per Episode', 'avg_reward')

# througput over time
def graph_throughputs(throughputs):
	graph_helper(throughputs, 'Time', 'Throughput (Mbps)', 'Throughput over time', 'throughput')

# througput over time
def graph_actions(actions):
	graph_helper(actions, 'Time', 'Action', 'Action over time', 'action')

def graph_helper(lst, xlabel, ylabel, title, filename):
	plt.clf()
	plt.plot(range(len(lst)), lst)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(title)
	plt.savefig('./graphs/' + filename + '.png')
