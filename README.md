# Artifact for SC '21
[![DOI](https://zenodo.org/badge/387809469.svg)](https://zenodo.org/badge/latestdoi/387809469)

This repository contains the artifact for the SC '21 paper "*Characterization and Prediction of Deep LearningWorkloads in Large-Scale GPU Datacenters*". It includes following four parts:

+ `enviornment`: The experimental environment  in ***Appendix: Artifact Description/Artifact Evaluation***. 

+ `data`: Helios traces download from [HeliosData](https://github.com/S-Lab-System-Group/HeliosData).

+ `analysis`: It contains scripts for analyzing traces.

+ `framework`: It contains `QSSF Service` and `CES Service` scripts



> **Note that only the `Venus` trace is public available now. Other traces are being censored. We will release them as soon as possible.**

## Detailed Introduction

### `enviornment`
Provide details on the experimental environment as shown in ***Appendix: Artifact Description/Artifact Evaluation***. 

+ `collect_environment.sh`: Gather execution environment information for GPU compute node and analysis platform.

+ `env_analysis_platform`: Execution environment information for trace analysis platform.

+ `env_datacenter_node`: Execution environment information for GPU compute node in our datacenter (from Volta Cluster).

+ ***Summary***

    |         | Analysis Platform   | Datacenter Node          |
    | ------- | ------------------- | ------------------------ |
    | System  | Ubuntu 20.04 LTS    | CentOS 7.4               |
    | CPU     | Intel Core i9-10900 | 2 x Intel Xeon Gold 6146 |
    | Memory  | 32GB DDR4           | 376GB DDR4               |
    | GPU     | GeForce RTX 2080 Ti | 8 x Tesla V100-SXM2      |
    | Network | Ethernet            | InfiniBand EDR           |

### `data`
Initially, this folder is ***NOT exist***. You need to download and unzip the dataset from [HeliosData](https://github.com/S-Lab-System-Group/HeliosData). After that, this folder structure should be: 


```
📦data
 ┣ 📂Earth
 ┃ ┣ 📜cluster_gpu_number.csv
 ┃ ┗ 📜cluster_log.csv
 ┣ 📂Saturn
 ┃ ┣ 📜cluster_gpu_number.csv
 ┃ ┗ 📜cluster_log.csv
 ┣ 📂Uranus
 ┃ ┣ 📜cluster_gpu_number.csv
 ┃ ┗ 📜cluster_log.csv
 ┗ 📂Venus
 ┃ ┣ 📜cluster_gpu_number.csv
 ┃ ┗ 📜cluster_log.csv
```

> **Note that only the `Venus` trace is public available now.**


### `analysis`
Contains parsing and plotting code to analyze traces.

+ **compare with Philly trace**: Figure 1: Comparisons of job characteristics between Helios and Philly.

+ **cluster characterization**: Figure 2: Daily pattern of the cluster usage in Helios.
    
    Figure 3: Monthly trends of cluster activities in Helios.

    Figure 4: The boxplot of utilization distributions for thetop 10 largest VCs of Earth in May (sorted by size).

+ **job characterization**: Figure 5: CDF of GPU (a) and CPU (b) job duration.

    Figure 6: The CDFs of job sizes (in GPU number) with the number of jobs (a) and GPU time (b).

    Figure 7: Distribution of jobs by their final statuses.



+ **user characterization**: Figure 8: The CDFs of users that consume the cluster resources in terms of (a) GPU Time (b) CPU Time.

    Figure 9: (a) CDFs of users w.r.t. GPU job queuing delay. (b)Distributions of user GPU job completion ratios.



### `framework`
An prediction-based GPU resource management framework. 

This folder contains `QSSF Service` and `CES Service` scripts and related data.



## Quick Start
These scripts have been tested on Ubuntu 20.04 with Python 3.8 (on the analysis platform).

Here are the ***step-by-step*** instructions for artifact.
### Preparing

1.  Download Helios artifact and data repository.
    ```bash
    git clone git@github.com:S-Lab-System-Group/HeliosArtifact.git
    cd HeliosArtifact

    git clone git@github.com:S-Lab-System-Group/HeliosData.git
    mv ./HeliosData/data ./
    ```

2. Check software dependencies:
   
   For the `analysis` part, JupyterLab / JupyterNotebook is needed.

   For the other python libraries used in this project, you can check `requirements.txt`.


### Reproducing `analysis`

3.  Prepare and parse the trace files for analyzing.
    ```bash
    cd analysis
    python ./trace_parser.py --cluster-list 'Venus'
    ```
4.  After generating all required data, you can analyze traces through `.ipynb` files within 4 sub-folders of `analysis`:**1_compare with Philly trace**, **2_cluster characterization**, **3_job characterization**, **4_user characterization**.

    These Jupyter Notebook scripts are used for generating figures of the trace analysis part of the paper.

> **Note that only the `Venus` trace is public available now. Thus, some generated figures are incomplete comparing with the paper version.**


### Reproducing `framework`


####  `QSSF Service`
  
5. Before executing the simulation of QSSF service, data preparation is needed.

   It generates VC configuration and job trace for each cluster.

    ```bash
    cd framework/QSSF\ Service/data
    bash prepare_data.sh 
    ```

6. Then, you can run all scheduling policies on **Philly** trace using `sweep` mode, as below:
   
    ```bash
    cd ..
    python simulator.py -e='Philly' -t='./data/Philly' --sweep 
    ```

   See `run.sh` for more usage examples on **Helios**. Note that since we do not release job name information, the `estimator` and `qssf policy` are not available for **Helios**.
   


7. After the program is executed, you can check the result in the `log` folder. The job log and time sequence of each VC are provided separately.

8. Besides, we provide simulation analysis and plot script in `plot`.

   You can generate Figure 13 in the paper through this script. 

####  `CES Service`

9. Run CES simulation on **Helios**:
   
    ```bash
    cd framework/CES\ Service
    python CES_Helios.py
    ```

    You can specify different cluster in the script and adjust the different configurations of CES service by the `hyperparameter` function.


10. Similarly, run CES simulation on **Philly**:
   
    ```bash
    python CES_Philly.py
    ```

11. From the code output and generated figures `helios_ces` (Figure 14) & `philly_ces` (Figure 15), we can analyze the CES service performance in detail.
