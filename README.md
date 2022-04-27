Here are the instructions to reproduce our results. We performed our experiments on a Ubuntu 20.04 system with Python 3.8. We provide setup instructions that we use. First, we must install the following dependencies:

1. NS3 - We followed the instructions outlined here: https://karimmd.github.io/post/tutorial/ns3_installation/
2. 
3. NS3-ai - To install ns3-ai, we followed the instructions outlined the their GitHub repo (https://github.com/hust-diangroup/ns3-ai). A brief recap can be found below:
```
cd /ns-allinone-3.33/ns-3.33/contrib
git clone https://github.com/hust-diangroup/ns3-ai.git
cd ../
./waf configure
./waf
cd /ns-allinone-3.33/ns-3.33/ns3-ai/py_interface
pip3 install . --user
```
