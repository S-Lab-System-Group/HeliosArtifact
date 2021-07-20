from .placer.consolidate import ConsolidatePlacement
from .placer.consolidateFirst import ConsolidateFirstPlacement
from .placer.random import RandomPlacement
import pandas as pd
import os


class Policy:
    def __init__(self, trace, vc, placement, log_dir, logger, start_ts):
        self._placement = placement
        self._vc = vc
        self._vc_name = vc.vc_name
        self._log_dir = log_dir
        self.trace = trace.vc_trace(vc.vc_name)
        self.logger = logger
        self.start_ts = start_ts

        self.total_job_num = self.trace.job_num()
        self.que_list = []  # Pending Jobs
        self.run_list = []  # Running Jobs
        self.end_job_num = 0
        self.time = start_ts

        # Time Sequence Recoder
        self.time_list = []
        self.total_gpu_num = []
        self.idle_gpu_num = []
        self.pend_gpu_num = []
        self.run_job_num = []
        self.pend_job_num = []
        self.pend_job_num_less_8 = []
        self.total_node_num = []
        self.consolidate_node_num = []
        self.shared_node_num = []

    def job_placer(self, job):
        if self._placement == 'consolidate':
            return ConsolidatePlacement(self._vc).place(job)
        if self._placement == 'random':
            return RandomPlacement(self._vc).place(job)
        if self._placement == 'consolidateFirst':
            return ConsolidateFirstPlacement(self._vc).place(job)
        raise NotImplementedError

    def ckpt_overhead(self, job):
        gpu_num = job.__getitem__('gpu_num')
        if gpu_num == 1:
            return 7
        elif gpu_num <= 8:
            return 30
        else:
            return 60

    def runtime_log(self):
        self.logger.info(
            f'{self._vc_name} | Time: {int(self.time)} | Total Job: {self.total_job_num} | End job: {self.end_job_num} | Running job: {len(self.run_list)} | Pending job: {len(self.que_list)}')

    '''Simulation Result Recorder'''

    def log_recorder(self, policy_name):
        if not os.path.exists(os.path.join(self._log_dir, self._vc_name)):
            os.mkdir(os.path.join(self._log_dir, self._vc_name))

        df = pd.DataFrame(self.trace.job_list)

        if len(df) == 0:
            print('No Job in VC: ', self._vc_name)
            raise NotImplementedError

        df['jct'] = df['end_time'] - df['submit_time']
        avg_jct = round(df['jct'].mean(), 2)
        avg_que = round(df['queue'].mean(), 2)
        self.logger.info(
            f'{self._vc_name} | Average JCT: {avg_jct} | Average Queue: {avg_que}')

        df.to_csv(
            f'{self._log_dir}/{self._vc_name}/{policy_name}_{self._placement}_{self._vc_name}_log.csv', index=False)

        # Time Sequence
        seq_dict = {'time': self.time_list,
                    'total_gpu_num': self.total_gpu_num,
                    'idle_gpu_num': self.idle_gpu_num,
                    'pending_gpu_num': self.pend_gpu_num,
                    'running_gpujob_num': self.run_job_num,
                    'pending_gpujob_num': self.pend_job_num,
                    'pending_job_num_less_8': self.pend_job_num_less_8,
                    'total_node_num': self.total_node_num,
                    'consolidate_node_num': self.consolidate_node_num,
                    'shared_node_num': self.shared_node_num,
                    }
        seq = pd.DataFrame(seq_dict)
        seq['gpu_utilization'] = ((seq['total_gpu_num'] - seq['idle_gpu_num']) /
                                  seq['total_gpu_num']).round(3)
        seq.to_csv(
            f'{self._log_dir}/{self._vc_name}/{policy_name}_{self._placement}_{self._vc_name}_seq.csv', index=False)
        # gpu_utilization

    def pend_job_num_small(self):
        job_num = 0
        for job in self.que_list:
            if job.__getitem__('gpu_num') < 8:
                job_num += 1
        return job_num

    def seq_recorder(self):
        self.time_list.append(self.time)
        self.total_gpu_num.append(self._vc.total_gpus)
        self.idle_gpu_num.append(self._vc.vc_free_gpus())
        self.pend_gpu_num.append(sum(job.__getitem__('gpu_num')
                                     for job in self.que_list))
        self.run_job_num.append(len(self.run_list))
        self.pend_job_num.append(len(self.que_list))
        self.pend_job_num_less_8.append(self.pend_job_num_small())
        self.total_node_num.append(self._vc.node_num)
        self.consolidate_node_num.append(self._vc.consolidate_node_num())
        self.shared_node_num.append(self._vc.shared_node_num())
