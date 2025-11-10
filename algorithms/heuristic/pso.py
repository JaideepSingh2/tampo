import numpy as np
from typing import Dict, List, Tuple
from copy import deepcopy

class PSOScheduler:
    """
    Particle Swarm Optimization for Task Offloading
    
    References:
        J. Kennedy and R. Eberhart, "Particle swarm optimization," 
        IEEE ICNN, 1995.
    """
    
    def __init__(self, env, config: Dict):
        """
        Initialize PSO scheduler
        
        Args:
            env: Task offloading environment
            config: PSO configuration parameters
        """
        self.env = env
        self.num_particles = config.get('num_particles', 50)
        self.max_iterations = config.get('max_iterations', 100)
        self.w = config.get('w', 0.7)  # Inertia weight
        self.c1 = config.get('c1', 1.5)  # Cognitive parameter
        self.c2 = config.get('c2', 1.5)  # Social parameter
        
    def evaluate_fitness(self, position: np.ndarray, dag: Dict, 
                        preference: np.ndarray) -> float:
        """
        Evaluate fitness of a particle position
        
        Args:
            position: Particle position (continuous values)
            dag: Task graph
            preference: [w_delay, w_energy]
            
        Returns:
            Fitness value (lower is better)
        """
        # Convert continuous position to discrete schedule
        num_resources = 1 + 1 + self.env.num_edge_servers
        schedule = np.clip(np.round(position), 0, num_resources - 1).astype(int)
        
        # Simulate execution
        delay, energy = self.simulate_execution(schedule, dag)
        
        # Multi-objective fitness
        fitness = preference[0] * delay + preference[1] * energy
        
        return fitness
    
    def simulate_execution(self, schedule: np.ndarray, dag: Dict) -> Tuple[float, float]:
        """
        Simulate execution of a schedule
        
        Args:
            schedule: Task assignments
            dag: Task graph
            
        Returns:
            (total_delay, total_energy)
        """
        num_tasks = dag['num_tasks']
        finish_times = [0.0] * num_tasks
        total_energy = 0.0
        
        # Build dependency structure
        dependencies = [[] for _ in range(num_tasks)]
        for edge in dag['edges']:
            dependencies[edge['target']].append(edge['source'])
        
        # Execute tasks in topological order
        for task_id in range(num_tasks):
            task = dag['tasks'][task_id]
            resource = schedule[task_id]
            
            # Wait for dependencies
            ready_time = 0.0
            for dep_id in dependencies[task_id]:
                dep_finish = finish_times[dep_id]
                dep_resource = schedule[dep_id]
                
                # Add communication if different resources
                if dep_resource != resource and resource != 0:
                    for edge in dag['edges']:
                        if edge['source'] == dep_id and edge['target'] == task_id:
                            comm_time = edge['data'] / self.env.bandwidth_up
                            ready_time = max(ready_time, dep_finish + comm_time)
                            break
                else:
                    ready_time = max(ready_time, dep_finish)
            
            # Execution time
            if resource == 0:  # Local
                exec_time = task['cycles'] / self.env.local_freq
                energy = self.env.kappa * task['cycles'] * (self.env.local_freq ** 2)
            elif resource == 1:  # Cloud
                exec_time = task['cycles'] / self.env.cloud_freq
                trans_time = task['data_size'] / self.env.bandwidth_up
                energy = trans_time * self.env.cloud_power_tx
                exec_time += trans_time
            else:  # Edge
                edge_idx = resource - 2
                exec_time = task['cycles'] / self.env.edge_freq[edge_idx]
                trans_time = task['data_size'] / self.env.bandwidth_up
                energy = trans_time * self.env.edge_power_tx
                exec_time += trans_time
            
            finish_times[task_id] = ready_time + exec_time
            total_energy += energy
        
        total_delay = max(finish_times)
        return total_delay, total_energy
    
    def optimize(self, dag: Dict, preference: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Run PSO optimization
        
        Args:
            dag: Task graph
            preference: User preference vector
            
        Returns:
            (best_schedule, delay, energy)
        """
        num_tasks = dag['num_tasks']
        num_resources = 1 + 1 + self.env.num_edge_servers
        
        # Initialize swarm
        positions = np.random.uniform(0, num_resources, 
                                     (self.num_particles, num_tasks))
        velocities = np.random.uniform(-1, 1, 
                                      (self.num_particles, num_tasks))
        
        # Personal best
        p_best_positions = deepcopy(positions)
        p_best_fitness = np.array([self.evaluate_fitness(pos, dag, preference) 
                                   for pos in positions])
        
        # Global best
        g_best_idx = np.argmin(p_best_fitness)
        g_best_position = deepcopy(p_best_positions[g_best_idx])
        g_best_fitness = p_best_fitness[g_best_idx]
        
        # Optimization loop
        for iteration in range(self.max_iterations):
            for i in range(self.num_particles):
                # Update velocity
                r1, r2 = np.random.rand(2)
                velocities[i] = (
                    self.w * velocities[i] +
                    self.c1 * r1 * (p_best_positions[i] - positions[i]) +
                    self.c2 * r2 * (g_best_position - positions[i])
                )
                
                # Update position
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], 0, num_resources - 0.01)
                
                # Evaluate fitness
                fitness = self.evaluate_fitness(positions[i], dag, preference)
                
                # Update personal best
                if fitness < p_best_fitness[i]:
                    p_best_fitness[i] = fitness
                    p_best_positions[i] = deepcopy(positions[i])
                    
                    # Update global best
                    if fitness < g_best_fitness:
                        g_best_fitness = fitness
                        g_best_position = deepcopy(positions[i])
        
        # Convert best position to schedule
        best_schedule = np.clip(np.round(g_best_position), 0, 
                               num_resources - 1).astype(int)
        delay, energy = self.simulate_execution(best_schedule, dag)
        
        return best_schedule, delay, energy

