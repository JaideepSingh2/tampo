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
            Fitness value (lower is better for PSO)
        """
        # Convert continuous position to discrete schedule
        num_resources = 1 + 1 + self.env.num_edge_servers
        schedule = np.clip(np.round(position), 0, num_resources - 1).astype(int)
        
        # Simulate execution
        delay, energy = self.simulate_execution(schedule, dag)
        
        # Multi-objective fitness (lower is better)
        fitness = preference[0] * delay + preference[1] * energy
        
        return fitness
    
    def simulate_execution(self, schedule: np.ndarray, dag: Dict) -> Tuple[float, float]:
        """
        Simulate execution of a schedule following DAG dependencies
        
        Args:
            schedule: Task assignments
            dag: Task graph
            
        Returns:
            (total_delay, total_energy)
        """
        num_tasks = dag['num_tasks']
        num_resources = 1 + 1 + self.env.num_edge_servers
        
        # Ensure valid schedule
        schedule = np.clip(schedule, 0, num_resources - 1).astype(int)
        
        # Track finish times for each task
        finish_times = np.zeros(num_tasks)
        total_energy = 0.0
        
        # Build dependency structure (predecessors for each task)
        predecessors = [[] for _ in range(num_tasks)]
        for edge in dag['edges']:
            predecessors[edge['target']].append({
                'task_id': edge['source'],
                'data_size': edge['data']
            })
        
        # Process tasks in order (assumes tasks are in valid topological order)
        for task_id in range(num_tasks):
            task = dag['tasks'][task_id]
            resource = schedule[task_id]
            
            # Calculate ready time (when all dependencies are satisfied)
            ready_time = 0.0
            
            for pred in predecessors[task_id]:
                pred_id = pred['task_id']
                pred_finish = finish_times[pred_id]
                pred_resource = schedule[pred_id]
                
                # Add communication delay if tasks are on different resources
                if pred_resource != resource:
                    # Communication delay = data_size / bandwidth
                    comm_delay = pred['data_size'] / self.env.bandwidth_up
                    ready_time = max(ready_time, pred_finish + comm_delay)
                else:
                    # No communication if on same resource
                    ready_time = max(ready_time, pred_finish)
            
            # Calculate execution time and energy based on resource type
            exec_time, exec_energy = self._calculate_execution(task, resource)
            
            # Task finishes at ready_time + execution_time
            finish_times[task_id] = ready_time + exec_time
            total_energy += exec_energy
        
        # Total delay is the maximum finish time (makespan)
        total_delay = np.max(finish_times)
        
        return total_delay, total_energy
    
    def _calculate_execution(self, task: Dict, resource: int) -> Tuple[float, float]:
        """
        Calculate execution time and energy for a task on a given resource
        
        Args:
            task: Task dictionary with 'cycles' and 'data_size'
            resource: Resource ID (0=local, 1=cloud, 2+=edge)
            
        Returns:
            (execution_time, energy_consumption)
        """
        cycles = task['cycles']
        data_size = task['data_size']
        
        if resource == 0:  # Local execution
            # T_local = C_i / f_UE
            exec_time = cycles / self.env.local_freq
            
            # E_comp = kappa * C_i * (f_UE^2)
            energy = self.env.kappa * cycles * (self.env.local_freq ** 2)
            
        elif resource == 1:  # Cloud offloading
            # Upload time
            upload_time = data_size / self.env.bandwidth_up
            
            # Computation time on cloud
            comp_time = cycles / self.env.cloud_freq
            
            # Download time (assume result is 10% of input)
            download_time = (data_size * 0.1) / self.env.bandwidth_down
            
            # Total execution time
            exec_time = upload_time + comp_time + download_time
            
            # E_comm = P_tx * (upload_time + download_time)
            energy = self.env.cloud_power_tx * (upload_time + download_time)
            
        else:  # Edge offloading (resource >= 2)
            edge_idx = resource - 2
            
            # Ensure edge index is valid
            if edge_idx >= len(self.env.edge_freq):
                edge_idx = 0
            
            # Upload time
            upload_time = data_size / self.env.bandwidth_up
            
            # Computation time on edge server
            comp_time = cycles / self.env.edge_freq[edge_idx]
            
            # Download time
            download_time = (data_size * 0.1) / self.env.bandwidth_down
            
            # Total execution time
            exec_time = upload_time + comp_time + download_time
            
            # E_comm = P_tx * (upload_time + download_time)
            energy = self.env.edge_power_tx * (upload_time + download_time)
        
        return exec_time, energy
    
    def optimize(self, dag: Dict, preference: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Run PSO optimization to find best task offloading schedule
        
        Args:
            dag: Task graph
            preference: User preference vector [w_delay, w_energy]
            
        Returns:
            (best_schedule, delay, energy)
        """
        num_tasks = dag['num_tasks']
        num_resources = 1 + 1 + self.env.num_edge_servers  # local + cloud + edges
        
        # Step 1: Initialize swarm
        # Positions: continuous values in [0, num_resources)
        positions = np.random.uniform(0, num_resources - 0.01, 
                                     (self.num_particles, num_tasks))
        
        # Velocities: small random initial velocities
        velocities = np.random.uniform(-0.5, 0.5, 
                                      (self.num_particles, num_tasks))
        
        # Step 2: Initialize personal best (pbest) for each particle
        p_best_positions = deepcopy(positions)
        p_best_fitness = np.array([
            self.evaluate_fitness(pos, dag, preference) 
            for pos in positions
        ])
        
        # Step 3: Initialize global best (gbest)
        g_best_idx = np.argmin(p_best_fitness)
        g_best_position = deepcopy(p_best_positions[g_best_idx])
        g_best_fitness = p_best_fitness[g_best_idx]
        
        # Track best fitness history
        best_history = [g_best_fitness]
        
        # Step 4: Optimization loop (iterations)
        for iteration in range(self.max_iterations):
            for i in range(self.num_particles):
                # A. Update velocity using PSO formula
                # v(t+1) = w*v(t) + c1*r1*(pbest - x(t)) + c2*r2*(gbest - x(t))
                r1 = np.random.rand(num_tasks)
                r2 = np.random.rand(num_tasks)
                
                # Inertia component
                inertia = self.w * velocities[i]
                
                # Cognitive (personal best) component
                cognitive = self.c1 * r1 * (p_best_positions[i] - positions[i])
                
                # Social (global best) component
                social = self.c2 * r2 * (g_best_position - positions[i])
                
                # Update velocity
                velocities[i] = inertia + cognitive + social
                
                # Velocity clamping (optional but recommended)
                max_velocity = num_resources * 0.2  # 20% of search space
                velocities[i] = np.clip(velocities[i], -max_velocity, max_velocity)
                
                # B. Update position
                # x(t+1) = x(t) + v(t+1)
                positions[i] += velocities[i]
                
                # C. Clamp position to valid range [0, num_resources)
                positions[i] = np.clip(positions[i], 0, num_resources - 0.01)
                
                # D. Evaluate fitness of new position
                current_fitness = self.evaluate_fitness(positions[i], dag, preference)
                
                # E. Update personal best
                if current_fitness < p_best_fitness[i]:
                    p_best_fitness[i] = current_fitness
                    p_best_positions[i] = deepcopy(positions[i])
                    
                    # F. Update global best
                    if current_fitness < g_best_fitness:
                        g_best_fitness = current_fitness
                        g_best_position = deepcopy(positions[i])
            
            # Track best fitness
            best_history.append(g_best_fitness)
            
            # Print progress every 20 iterations
            if (iteration + 1) % 20 == 0:
                # Convert to discrete schedule for reporting
                temp_schedule = np.clip(np.round(g_best_position), 0, 
                                       num_resources - 1).astype(int)
                delay, energy = self.simulate_execution(temp_schedule, dag)
                print(f"  Iteration {iteration + 1}/{self.max_iterations}: "
                      f"Best Fitness = {g_best_fitness:.6f}, "
                      f"Delay = {delay:.4f}s, Energy = {energy:.4f}J")
        
        # Step 5: Convert best continuous position to discrete schedule
        # Use rounding and clamping as described in pso.txt
        best_schedule = np.clip(np.round(g_best_position), 0, 
                               num_resources - 1).astype(int)
        
        # Get final delay and energy
        final_delay, final_energy = self.simulate_execution(best_schedule, dag)
        
        return best_schedule, final_delay, final_energy