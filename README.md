# Online Reinforcement Learning For Congestion Control

This is the repository that contains the code for our project on utilizing Reinforcement Learning (RL) to develop congrestion control algoeithms. The approach we utilize is inspired by a previous approach called Orca (https://github.com/Soheil-ab/Orca) that combines the fine grained control of existing TCP CC algorithms with the flexibility of coarse grained RL algorithms. We extend this approach by turning this into an online algorithm, i.e. the corase grained RL agent continues to learn even after it has been deployed. We hopes this would allow for a congestion control algorithm that would better generalize to unseen network conditions. 


Here are the instructions to reproduce our results. We performed our experiments on a Ubuntu 20.04 system with Python 3.8. We provide setup instructions that we use. First, we must install the following dependencies:

1. NS3 - We followed the instructions outlined here: https://karimmd.github.io/post/tutorial/ns3_installation/
2. NS3-ai - To install ns3-ai, we followed the instructions outlined the their GitHub repo (https://github.com/hust-diangroup/ns3-ai). A brief recap can be found below:
```
cd ns-allinone-3.33/ns-3.33/contrib
git clone https://github.com/hust-diangroup/ns3-ai.git
cd ../
./waf configure
./waf
cd ns-allinone-3.33/ns-3.33/ns3-ai/py_interface
pip3 install . --user
```
3. Relevant python packages
```
pip3 install torch torchvision torchaudio
pip3 install matplotlib
```

4. Clone our repo into the scratch folder
```
cd ns-allinone-3.33/ns-3.33/scratch
git clone https://github.com/numair1/RL-CC-Agent.git
mv RL-CC-Agent rl-tcp
cd ../
./waf configure
./waf

```

5. Now to run our experiment, run the following code
```
cd scratch/rl-tcp/continuous_action_space
python3 rl_tcp_continuous.py --result
```
NOTE: This will take a while and there can be memlock issues becuase the ns3-ai simulator is buggy with shared memory usage.

After the training loop is done, the necessary training graphs will be in the ./graphs folder. Now to generate the throughput vs delay graphs and compare with TCP, run the following code:
```
python3 tcp.py
python3 graph_throughputs.py
```
This will store the graph in the same ./graphs folder

