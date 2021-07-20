#! /bin/sh
#! /bin/sh

# Linux data-gathering commands; adjust as necessary for your platform.
#
# Be sure to remove any information from the output that would violate
# SC's double-blind review policies.

:<<!
For Anlysis Platform,  sh ./collect_environment.sh 2>&1 | tee env_analysis_platform.txt
!

set -x
inxi -F -c0
nvidia-smi
pip list


:<<!
For Datacenter Node,  sh ./collect_environment.sh 2>&1 | tee env_datacenter_node.txt
!

set -x
cat /etc/redhat-release
lscpu || cat /proc/cpuinfo
cat /proc/meminfo
nvidia-smi
lspci