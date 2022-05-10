import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np

with open('./data/throughputs_tcp.pkl', 'rb') as fh:
	tcp_throughputs = pkl.load(fh)
with open('./data/rtts_tcp.pkl', 'rb') as fh:
	tcp_rtts = pkl.load(fh)

with open('./data/throughputs_online.pkl', 'rb') as fh:
	online_throughputs = pkl.load(fh)
with open('./data/rtts_online.pkl', 'rb') as fh:
	online_rtts = pkl.load(fh)

def avg_jagged_2d_array(arr):
	summ = 0.0
	lenn = 0.0
	for lst in arr:
		summ += sum(lst)
		lenn += len(lst)
	if lenn == 0:
		avg = 0
	else:
		avg = summ / lenn
	return avg

tcp_throughputs_avg = avg_jagged_2d_array(tcp_throughputs) / 125000
tcp_rtts_avg = avg_jagged_2d_array(tcp_rtts) / 1000

online_throughputs_avg = avg_jagged_2d_array(online_throughputs) / 125000
online_rtts_avg = avg_jagged_2d_array(online_rtts) / 1000

# Averaged Normalized Throughput over Averaged Normalized Delay
plt.clf()
plt.scatter(tcp_rtts_avg, tcp_throughputs_avg, c='b', label='TCP Cubic')
plt.scatter(online_rtts_avg, online_throughputs_avg, c='g', label='Online')
plt.xlabel('Averaged Delay (ms)')
plt.ylabel('Averaged Throughput (Mbps)')
plt.title('Delay over Throughput for 2 CC models')
plt.legend()
plt.margins(0.5)
plt.savefig('./graphs/throughput_delay.png', bbox_inches="tight")
