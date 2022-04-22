import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np

with open('./data/TCP/throughputs.pickle', 'rb') as fh:
	tcp_throughputs = pkl.load(fh)

with open('./data/TCP/rtts.pickle', 'rb') as fh:
	tcp_rtts = pkl.load(fh)

# with open('./data/clean_slate/throughputs.pickle', 'rb') as fh:
# 	clean_slate_throughputs = pkl.load(fh)
#
# with open('./data/combined/throughputs.pickle', 'rb') as fh:
# 	combined_throughputs = pkl.load(fh)

tcp_throughputs = np.average(tcp_throughputs, axis=0)
# clean_slate_throughputs = np.average(clean_slate_throughputs, axis=0)
# combined_throughputs = np.average(combined_throughputs, axis=0)

tcp_throughputs = [np.average(tcp_throughputs[i:i+10]) for i in range(0, len(tcp_throughputs) - 11, 10)]
tcp_throughputs = np.asarray(tcp_throughputs)

plt.clf()
plt.plot(range(len(tcp_throughputs)), tcp_throughputs)
plt.xlabel('Timethroughputs TCP')
plt.ylabel('throughputs TCP')
plt.title('Throughputs over Time')
plt.savefig('./graphs/throughputs.png')

avgs = [np.average(tcp_rtts), np.average(tcp_throughputs)]

# Averaged Normalized Throughput over Averaged Normalized Delay
plt.clf()
plt.scatter(avgs[0], avgs[1], c='b', label='TCP')
# plt.scatter(combined[0], combined[1], c='r', label='Combined')
# plt.scatter(clean_slate[0], clean_slate[1], c='g', label='Clean Slate')
plt.xlabel('Averaged Normalized Delay')
plt.ylabel('Averaged Normalized Throughput')
plt.title('Delay over Throughput for 3 CC models')
plt.legend()
plt.savefig('./graphs/throughput_delay.png')
