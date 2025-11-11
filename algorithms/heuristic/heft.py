import numpy as np
from typing import Dict, List, Tuple

class HEFTScheduler:
    """
    Heterogeneous Earliest Finish Time (HEFT) Algorithm
    
    References:
        H. Topcuoglu et al., "Performance-effective and low-complexity 
        task scheduling for heterogeneous computing," IEEE TPDS, 2002.
    """
    
    def __init__(self, env):
        """
        Initialize HEFT scheduler
        
        Args:
            env: Task offloading environment
        """
        self.env = env
        
    def compute_upward_rank(self, dag: Dict) -> np.ndarray:
        """
        Compute upward rank for all tasks (priority metric)
        
        Args:
            dag: Task graph dictionary
            
        Returns:
            Array of upward ranks for each task
        """
        num_tasks = dag['num_tasks']
        ranks = np.zeros(num_tasks)
        
        # Build successors list with communication costs
        successors = [[] for _ in range(num_tasks)]
        for edge in dag['edges']:
            successors[edge['source']].append({
                'target': edge['target'],
                'comm_cost': edge['data']
            })
        
        # Build computation cost matrix W (tasks x processors)
        num_processors = 2 + self.env.num_edge_servers  # local + cloud + edges
        W = np.zeros((num_tasks, num_processors))
        
        for i in range(num_tasks):
            task = dag['tasks'][i]
            cycles = task['cycles']
            
            # Local execution (processor 0)
            W[i, 0] = cycles / self.env.local_freq
            
            # Cloud execution (processor 1)
            W[i, 1] = cycles / self.env.cloud_freq
            
            # Edge servers (processors 2+)
            for j in range(self.env.num_edge_servers):
                W[i, 2 + j] = cycles / self.env.edge_freq[j]
        
        # Compute average computation cost for each task
        w_bar = np.mean(W, axis=1)
        
        def compute_rank(task_id: int) -> float:
            """Recursively compute upward rank"""
            if ranks[task_id] > 0:
                return ranks[task_id]
            
            # Average computation cost for this task
            avg_comp_cost = w_bar[task_id]
            
            if len(successors[task_id]) == 0:
                # Exit task - rank is just computation cost
                ranks[task_id] = avg_comp_cost
            else:
                # Non-exit task: w_bar + max(c_bar + rank(successor))
                max_succ_rank = 0
                
                for succ in successors[task_id]:
                    succ_id = succ['target']
                    
                    # Recursively compute successor rank
                    succ_rank = compute_rank(succ_id)
                    
                    # Average communication cost
                    # Assume average bandwidth for communication
                    avg_comm = succ['comm_cost'] / self.env.bandwidth_up
                    
                    # Find maximum among all successors
                    max_succ_rank = max(max_succ_rank, avg_comm + succ_rank)
                
                ranks[task_id] = avg_comp_cost + max_succ_rank
            
            return ranks[task_id]
        
        # Compute ranks for all tasks
        for i in range(num_tasks):
            compute_rank(i)
        
        return ranks
    
    def compute_earliest_finish_time(self, task_id: int, processor_id: int,
                                     dag: Dict, schedule: Dict,
                                     W: np.ndarray) -> Tuple[float, float]:
        """
        Compute earliest start time and finish time for a task on a processor
        
        Args:
            task_id: Task index
            processor_id: Processor index (0=local, 1=cloud, 2+=edge)
            dag: Task graph
            schedule: Current schedule state
            W: Computation cost matrix
            
        Returns:
            (earliest_start_time, earliest_finish_time) tuple
        """
        task = dag['tasks'][task_id]
        
        # Find predecessors
        predecessors = []
        for edge in dag['edges']:
            if edge['target'] == task_id:
                predecessors.append({
                    'source': edge['source'],
                    'data': edge['data']
                })
        
        # Compute data ready time (max over all predecessors)
        data_ready_time = 0.0
        
        for pred in predecessors:
            pred_id = pred['source']
            pred_finish = schedule['finish_times'][pred_id]
            pred_processor = schedule['assignments'][pred_id]
            
            # Communication cost (0 if same processor)
            if pred_processor == processor_id:
                comm_cost = 0.0
            else:
                # Communication time = data_size / bandwidth
                comm_cost = pred['data'] / self.env.bandwidth_up
            
            # Data ready time is when predecessor finishes + communication
            ready_time = pred_finish + comm_cost
            data_ready_time = max(data_ready_time, ready_time)
        
        # Processor available time
        processor_available = schedule['processor_available'][processor_id]
        
        # Earliest start time = max(data_ready_time, processor_available)
        earliest_start = max(data_ready_time, processor_available)
        
        # Computation time from matrix W
        comp_time = W[task_id, processor_id]
        
        # Earliest finish time
        earliest_finish = earliest_start + comp_time
        
        return earliest_start, earliest_finish
    
    def schedule(self, dag: Dict) -> Tuple[List[Tuple[int, int]], float, float]:
        """
        Generate HEFT schedule for a DAG
        
        Args:
            dag: Task graph dictionary
            
        Returns:
            Tuple of (schedule, makespan, energy)
            schedule: List of (task_id, processor_id) tuples
            makespan: Total completion time
            energy: Total energy consumption
        """
        num_tasks = dag['num_tasks']
        num_processors = 2 + self.env.num_edge_servers  # local + cloud + edges
        
        # Build computation cost matrix W
        W = np.zeros((num_tasks, num_processors))
        
        for i in range(num_tasks):
            task = dag['tasks'][i]
            cycles = task['cycles']
            
            # Local execution (processor 0)
            W[i, 0] = cycles / self.env.local_freq
            
            # Cloud execution (processor 1)
            W[i, 1] = cycles / self.env.cloud_freq
            
            # Edge servers (processors 2+)
            for j in range(self.env.num_edge_servers):
                W[i, 2 + j] = cycles / self.env.edge_freq[j]
        
        # Phase 1: Task Prioritization - Compute upward ranks
        ranks = self.compute_upward_rank(dag)
        
        # Sort tasks by rank in decreasing order
        sorted_tasks = np.argsort(ranks)[::-1]
        
        # Phase 2: Processor Selection
        # Initialize schedule state
        schedule = {
            'assignments': [-1] * num_tasks,
            'start_times': [0.0] * num_tasks,
            'finish_times': [0.0] * num_tasks,
            'processor_available': [0.0] * num_processors
        }
        
        result_schedule = []
        total_energy = 0.0
        
        # Schedule each task in priority order
        for task_id in sorted_tasks:
            task = dag['tasks'][task_id]
            
            # Find processor with earliest finish time
            best_processor = 0
            best_est = float('inf')
            best_eft = float('inf')
            
            for processor_id in range(num_processors):
                est, eft = self.compute_earliest_finish_time(
                    task_id, processor_id, dag, schedule, W
                )
                
                if eft < best_eft:
                    best_eft = eft
                    best_est = est
                    best_processor = processor_id
            
            # Assign task to best processor
            schedule['assignments'][task_id] = best_processor
            schedule['start_times'][task_id] = best_est
            schedule['finish_times'][task_id] = best_eft
            schedule['processor_available'][best_processor] = best_eft
            
            result_schedule.append((task_id, best_processor))
            
            # Calculate energy consumption
            if best_processor == 0:  # Local execution
                # Energy = kappa * cycles * freq^2
                energy = self.env.kappa * task['cycles'] * (self.env.local_freq ** 2)
                
            elif best_processor == 1:  # Cloud offloading
                # Energy = transmission time * transmission power
                trans_time = task['data_size'] / self.env.bandwidth_up
                # Also consider result transmission back
                result_size = task['data_size'] * 0.1  # Assume 10% result size
                trans_time += result_size / self.env.bandwidth_down
                energy = trans_time * self.env.cloud_power_tx
                
            else:  # Edge offloading
                # Energy = transmission time * transmission power
                trans_time = task['data_size'] / self.env.bandwidth_up
                # Also consider result transmission back
                result_size = task['data_size'] * 0.1
                trans_time += result_size / self.env.bandwidth_down
                energy = trans_time * self.env.edge_power_tx
            
            total_energy += energy
        
        # Makespan is the maximum finish time
        makespan = max(schedule['finish_times'])
        
        return result_schedule, makespan, total_energy