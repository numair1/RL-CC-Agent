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
	summ = 0.0
	lenn = 0.0
	for lst in arr:
		summ += sum(lst)
		lenn += len(lst)
	avg = summ / lenn
	return avg

tcp_throughputs_avg = avg_jagged_2d_array(tcp_throughputs) / 125000
tcp_rtts_avg = avg_jagged_2d_array(tcp_rtts) / 1000

clean_slate_throughputs_avg = avg_jagged_2d_array(clean_slate_throughputs) / 125000
clean_slate_rtts_avg = avg_jagged_2d_array(clean_slate_rtts) / 1000

online_throughputs_avg = avg_jagged_2d_array(online_throughputs) / 125000
online_rtts_avg = avg_jagged_2d_array(online_rtts) / 1000

# tcp_throughputs_avg = np.average(tcp_throughputs[0])# / 125000
# tcp_rtts_avg = np.average(tcp_rtts[0]) / 1000
#
# clean_slate_throughputs_avg = avg_jagged_2d_array(clean_slate_throughputs)# / 125000
# clean_slate_rtts_avg = avg_jagged_2d_array(clean_slate_rtts) / 1000
#
# online_throughputs_avg = np.average(online_throughputs[0])# / 125000
# online_rtts_avg = np.average(online_rtts[0]) / 1000

# Averaged Normalized Throughput over Averaged Normalized Delay
plt.clf()
plt.scatter(tcp_rtts_avg, tcp_throughputs_avg, c='b', label='TCP Cubic')
# plt.scatter(clean_slate_throughputs_avg, clean_slate_rtts_avg, c='r', label='Clean Slate')
plt.scatter(online_rtts_avg, online_throughputs_avg, c='g', label='Online')
plt.xlabel('Averaged Delay (ms)')
plt.ylabel('Averaged Throughput (Mbps)')
plt.title('Delay over Throughput for 2 CC models')
plt.legend()
plt.margins(0.5)
plt.savefig('./graphs/throughput_delay.png', bbox_inches="tight")
