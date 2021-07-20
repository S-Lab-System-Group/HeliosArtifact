+ `philly_trace.csv`  

It is used to compare with our datacenter workloads.

We transfer the original Philly trace file into `.csv` format and select the same period of job logs as described in ["Analysis of Large-Scale Multi-Tenant GPU Clusters for DNN Training Workloads"](https://www.usenix.org/system/files/atc19-jeon.pdf) (ATC’19). 

The official public data can be download from [philly-traces](https://github.com/msr-fiddle/philly-traces). 

+ `philly_trace_B.csv`  

Failed jobs would be retried for a fixed number of times in Philly. If we process Philly trace by regarding each attempt as an individual job, we generate `philly_trace_B.csv`.