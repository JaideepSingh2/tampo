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
        
        # Build successors list
        successors = [[] for _ in range(num_tasks)]
        for edge in dag['edges']:
            successors[edge['source']].append({
                'target': edge['target'],
                'comm_cost': edge['data']
            })
        
        def compute_rank(task_id: int) -> float:
            if ranks[task_id] > 0:
                return ranks[task_id]
            
            task = dag['tasks'][task_id]
            
            # Average computation cost across all resources
            avg_comp_cost = (
                task['cycles'] / self.env.local_freq +
                task['cycles'] / self.env.cloud_freq +
                sum(task['cycles'] / freq for freq in self.env.edge_freq)
            ) / (2 + self.env.num_edge_servers)
            
            if len(successors[task_id]) == 0:
                # Exit task
                ranks[task_id] = avg_comp_cost
            else:
                # Max successor rank + avg communication cost
                max_succ = 0
                for succ in successors[task_id]:
                    succ_id = succ['target']
                    succ_rank = compute_rank(succ_id)
                    
                    # Average communication cost
                    avg_comm = succ['comm_cost'] / self.env.bandwidth_up
                    
                    max_succ = max(max_succ, succ_rank + avg_comm)
                
                ranks[task_id] = avg_comp_cost + max_succ
            
            return ranks[task_id]
        
        # Compute ranks for all tasks
        for i in range(num_tasks):
            compute_rank(i)
        
        return ranks
    
    def compute_earliest_finish_time(self, task_id: int, resource_id: int,
                                     dag: Dict, schedule: Dict) -> float:
        """
        Compute earliest finish time for a task on a resource
        
        Args:
            task_id: Task index
            resource_id: Resource index (0=local, 1=cloud, 2+=edge)
            dag: Task graph
            schedule: Current schedule state
            
        Returns:
            Earliest finish time
        """
        task = dag['tasks'][task_id]
        
        # Find predecessors
        predecessors = []
        for edge in dag['edges']:
            if edge['target'] == task_id:
                predecessors.append({
                    'source': edge['source'],
                    'comm_cost': edge['data']
                })
        
        # Data ready time (when all predecessors finish + communication)
        data_ready_time = 0
        for pred in predecessors:
            pred_id = pred['source']
            pred_finish = schedule['finish_times'][pred_id]
            pred_resource = schedule['assignments'][pred_id]
            
            # Add communication cost if different resources
            if pred_resource != resource_id and resource_id != 0:
                comm_time = pred['comm_cost'] / self.env.bandwidth_up
                data_ready_time = max(data_ready_time, pred_finish + comm_time)
            else:
                data_ready_time = max(data_ready_time, pred_finish)
        
        # Resource available time
        resource_available = schedule['resource_available'][resource_id]
        
        # Earliest start time
        earliest_start = max(data_ready_time, resource_available)
        
        # Computation time
        if resource_id == 0:  # Local
            comp_time = task['cycles'] / self.env.local_freq
        elif resource_id == 1:  # Cloud
            comp_time = task['cycles'] / self.env.cloud_freq
        else:  # Edge
            edge_idx = resource_id - 2
            comp_time = task['cycles'] / self.env.edge_freq[edge_idx]
        
        earliest_finish = earliest_start + comp_time
        
        return earliest_finish
    
    def schedule(self, dag: Dict) -> Tuple[List[Tuple[int, int]], float, float]:
        """
        Generate HEFT schedule for a DAG
        
        Args:
            dag: Task graph dictionary
            
        Returns:
            Tuple of (schedule, makespan, energy)
            schedule: List of (task_id, resource_id) tuples
            makespan: Total completion time
            energy: Total energy consumption
        """
        num_tasks = dag['num_tasks']
        num_resources = 1 + 1 + self.env.num_edge_servers  # local + cloud + edges
        
        # Compute priorities
        ranks = self.compute_upward_rank(dag)
        
        # Sort tasks by priority (descending)
        sorted_tasks = np.argsort(ranks)[::-1]
        
        # Initialize schedule state
        schedule = {
            'assignments': [-1] * num_tasks,
            'finish_times': [0.0] * num_tasks,
            'resource_available': [0.0] * num_resources
        }
        
        result_schedule = []
        total_energy = 0.0
        
        # Schedule each task
        for task_id in sorted_tasks:
            task = dag['tasks'][task_id]
            
            # Find resource with earliest finish time
            best_resource = 0
            best_eft = float('inf')
            
            for resource_id in range(num_resources):
                eft = self.compute_earliest_finish_time(task_id, resource_id, 
                                                       dag, schedule)
                if eft < best_eft:
                    best_eft = eft
                    best_resource = resource_id
            
            # Assign task to best resource
            schedule['assignments'][task_id] = best_resource
            schedule['finish_times'][task_id] = best_eft
            schedule['resource_available'][best_resource] = best_eft
            
            result_schedule.append((task_id, best_resource))
            
            # Calculate energy
            if best_resource == 0:  # Local
                energy = self.env.kappa * task['cycles'] * (self.env.local_freq ** 2)
            elif best_resource == 1:  # Cloud
                trans_time = task['data_size'] / self.env.bandwidth_up
                energy = trans_time * self.env.cloud_power_tx
            else:  # Edge
                trans_time = task['data_size'] / self.env.bandwidth_up
                energy = trans_time * self.env.edge_power_tx
            
            total_energy += energy
        
        makespan = max(schedule['finish_times'])
        
        return result_schedule, makespan, total_energy


