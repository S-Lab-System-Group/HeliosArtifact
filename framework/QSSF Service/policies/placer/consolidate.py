class ConsolidatePlacement:
    def __init__(self, vc):
        self.name = 'consolidate'
        self.vc = vc
        self.avail_nodes = self.vc.avail_node_list()

    '''
        Enforce consolidate placement
        Node list selection
        -- job_gpu_num <= 8
        -- job_gpu_num > 8  and job_gpu_num % 8 == 0
        -- job_gpu_num > 8  and job_gpu_num % 8 != 0
    '''

    def consolidateSelect(self, job_gpu_num):
        alloc_nodes = []
        if job_gpu_num <= 8:
            nodes = sorted(self.avail_nodes,
                           key=lambda x: x.free_gpus, reverse=False)
            for node in nodes:
                if node.free_gpus >= job_gpu_num:
                    alloc_nodes.append((node, job_gpu_num))
                    return True, alloc_nodes
            return False, alloc_nodes
        else:
            nodes = sorted(self.avail_nodes,
                           key=lambda x: x.free_gpus, reverse=True)
            if job_gpu_num % 8 == 0:
                node_num = job_gpu_num // 8
                for node in nodes:
                    if node.free_gpus < 8:
                        return False, alloc_nodes

                    if node.free_gpus == 8 and node_num > 0:
                        alloc_nodes.append((node, 8))
                        node_num -= 1

                    if node_num == 0:
                        return True, alloc_nodes
            else:
                node_num = (job_gpu_num // 8) + 1
                for node in nodes:
                    if node.free_gpus == 8 and node_num > 1:
                        alloc_nodes.append((node, 8))
                        node_num -= 1
                        continue

                    if node.free_gpus >= (job_gpu_num % 8) and node_num == 1:
                        alloc_nodes.append((node, job_gpu_num % 8))
                        node_num -= 1
                        return True, alloc_nodes

                    return False, alloc_nodes

    def place(self, job):
        vc_free_gpu_num = self.vc.vc_free_gpus()
        job_gpu_num = job['gpu_num']

        # Total Free GPU Check
        if vc_free_gpu_num < job_gpu_num:
            return False

        # TODO: Support for 4 GPU Nodes
        if self.vc._num_gpus_per_node != 8:
            raise NotImplementedError

        select_flag, alloc_nodes = self.consolidateSelect(job_gpu_num)

        ''' Placement '''
        if select_flag:
            for (node, req_gpu) in alloc_nodes:
                node.allocate_gpu(req_gpu)
                job['nodes'].append({node.node_name: req_gpu})
            return True
        else:
            return False
