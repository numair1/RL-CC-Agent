import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np

with open('./data/TCP/throughputs.pickle', 'rb') as fh:
	tcp_throughputs = pkl.load(fh)

with open('./data/TCP/rtts.pickle', 'rb') as fh:
	tcp_rtts = pkl.load(fh)

with open('./data/clean_slate/throughputs.pickle', 'rb') as fh:
	clean_slate_throughputs = pkl.load(fh)

with open('./data/clean_slate/rtts.pickle', 'rb') as fh:
	clean_slate_rtts = pkl.load(fh)

with open('./data/online/throughputs.pickle', 'rb') as fh:
	online_throughputs = pkl.load(fh)

with open('./data/online/rtts.pickle', 'rb') as fh:
	online_rtts = pkl.load(fh)

def avg_jagged_2d_array(arr):
	summ = 0
	lenn = 0
	for lst in arr:
		summ += sum(lst)
		lenn += len(lst)
	avg = summ / lenn
	return avg

tcp_throughputs_avg = avg_jagged_2d_array(tcp_throughputs)
tcp_rtts_avg = avg_jagged_2d_array(tcp_rtts)

clean_slate_throughputs_avg = avg_jagged_2d_array(clean_slate_throughputs)
clean_slate_rtts_avg = avg_jagged_2d_array(clean_slate_rtts)

online_throughputs_avg = avg_jagged_2d_array(online_throughputs)
online_rtts_avg = avg_jagged_2d_array(online_rtts)

# Averaged Normalized Throughput over Averaged Normalized Delay
plt.clf()
plt.scatter(tcp_throughputs_avg, tcp_rtts_avg, c='b', label='TCP Cubic')
plt.scatter(clean_slate_throughputs_avg, clean_slate_rtts_avg, c='r', label='Clean Slate')
plt.scatter(online_throughputs_avg, online_rtts_avg, c='g', label='Online')
plt.xlabel('Averaged Normalized Delay')
plt.ylabel('Averaged Normalized Throughput')
plt.title('Delay over Throughput for 3 CC models')
plt.legend()
plt.savefig('./graphs/throughput_delay.png')
