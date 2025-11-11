import numpy as np
from typing import Dict, List, Tuple
from copy import deepcopy

class GAScheduler:
    """
    Genetic Algorithm for Task Offloading
    
    References:
        D. E. Goldberg, "Genetic Algorithms in Search, Optimization, 
        and Machine Learning," Addison-Wesley, 1989.
    """
    
    def __init__(self, env, config: Dict):
        """
        Initialize GA scheduler
        
        Args:
            env: Task offloading environment
            config: GA configuration parameters
        """
        self.env = env
        self.population_size = config.get('population_size', 100)
        self.num_generations = config.get('num_generations', 100)
        self.crossover_rate = config.get('crossover_rate', 0.8)
        self.mutation_rate = config.get('mutation_rate', 0.1)
        self.tournament_size = config.get('tournament_size', 5)
        self.elitism_size = config.get('elitism_size', 2)
        
    def evaluate_fitness(self, chromosome: np.ndarray, dag: Dict, 
                        preference: np.ndarray) -> float:
        """
        Evaluate fitness of a chromosome
        
        Args:
            chromosome: Offloading decisions (processor assignments)
            dag: Task graph
            preference: [w_delay, w_energy]
            
        Returns:
            Fitness value (higher is better)
        """
        delay, energy = self.simulate_execution(chromosome, dag)
        
        # Calculate cost (to minimize)
        cost = preference[0] * delay + preference[1] * energy
        
        # Convert to fitness (higher is better)
        # Use inverse with small epsilon to avoid division by zero
        fitness = 1.0 / (cost + 1e-10)
        
        return fitness
    
    def simulate_execution(self, schedule: np.ndarray, dag: Dict) -> Tuple[float, float]:
        """
        Simulate execution of a schedule following DAG dependencies
        
        Args:
            schedule: Task assignments (chromosome)
            dag: Task graph
            
        Returns:
            (total_delay, total_energy)
        """
        num_tasks = dag['num_tasks']
        num_resources = 1 + 1 + self.env.num_edge_servers  # local + cloud + edges
        
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
    
    def tournament_selection(self, population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        """
        Tournament selection - randomly select k individuals and return the best
        
        Args:
            population: Current population
            fitness: Fitness values
            
        Returns:
            Selected individual (parent)
        """
        # Randomly select tournament_size individuals
        tournament_indices = np.random.choice(
            len(population), 
            self.tournament_size, 
            replace=False
        )
        
        # Get their fitness values
        tournament_fitness = fitness[tournament_indices]
        
        # Winner is the one with highest fitness
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        
        return population[winner_idx].copy()
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Single-point crossover
        
        Args:
            parent1: First parent chromosome
            parent2: Second parent chromosome
            
        Returns:
            Two offspring chromosomes
        """
        if np.random.rand() < self.crossover_rate:
            # Choose random crossover point
            point = np.random.randint(1, len(parent1))
            
            # Create offspring by swapping genes at crossover point
            offspring1 = np.concatenate([parent1[:point], parent2[point:]])
            offspring2 = np.concatenate([parent2[:point], parent1[point:]])
        else:
            # No crossover - offspring are copies of parents
            offspring1 = parent1.copy()
            offspring2 = parent2.copy()
        
        return offspring1, offspring2
    
    def mutate(self, chromosome: np.ndarray, num_resources: int) -> np.ndarray:
        """
        Mutation operator - randomly change genes with small probability
        
        Args:
            chromosome: Individual to mutate
            num_resources: Number of available resources
            
        Returns:
            Mutated chromosome
        """
        mutated = chromosome.copy()
        
        for i in range(len(mutated)):
            if np.random.rand() < self.mutation_rate:
                # Replace with random valid resource ID
                mutated[i] = np.random.randint(0, num_resources)
        
        return mutated
    
    def optimize(self, dag: Dict, preference: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Run GA optimization to find best task offloading schedule
        
        Args:
            dag: Task graph (DAG structure)
            preference: User preference vector [w_delay, w_energy]
            
        Returns:
            (best_schedule, delay, energy)
        """
        num_tasks = dag['num_tasks']
        num_resources = 1 + 1 + self.env.num_edge_servers  # local + cloud + edges
        
        # Step 1: Initialize population with random chromosomes
        population = np.random.randint(
            0, 
            num_resources, 
            (self.population_size, num_tasks)
        )
        
        best_chromosome = None
        best_fitness = -float('inf')
        best_history = []
        
        # Step 2: Evolution loop
        for generation in range(self.num_generations):
            # A. Fitness Evaluation
            fitness = np.array([
                self.evaluate_fitness(individual, dag, preference) 
                for individual in population
            ])
            
            # Track best individual
            gen_best_idx = np.argmax(fitness)
            if fitness[gen_best_idx] > best_fitness:
                best_fitness = fitness[gen_best_idx]
                best_chromosome = population[gen_best_idx].copy()
            
            best_history.append(best_fitness)
            
            # Print progress every 20 generations
            if (generation + 1) % 20 == 0:
                delay, energy = self.simulate_execution(best_chromosome, dag)
                print(f"  Generation {generation + 1}/{self.num_generations}: "
                      f"Best Fitness = {best_fitness:.6f}, "
                      f"Delay = {delay:.4f}s, Energy = {energy:.4f}J")
            
            # B. Selection and Reproduction
            new_population = []
            
            # Elitism: Keep the best individuals
            elite_indices = np.argsort(fitness)[-self.elitism_size:]
            for idx in elite_indices:
                new_population.append(population[idx].copy())
            
            # C. Generate rest of population through selection, crossover, and mutation
            while len(new_population) < self.population_size:
                # Selection: Tournament selection
                parent1 = self.tournament_selection(population, fitness)
                parent2 = self.tournament_selection(population, fitness)
                
                # Crossover: Single-point crossover
                offspring1, offspring2 = self.crossover(parent1, parent2)
                
                # Mutation: Random gene changes
                offspring1 = self.mutate(offspring1, num_resources)
                offspring2 = self.mutate(offspring2, num_resources)
                
                new_population.append(offspring1)
                if len(new_population) < self.population_size:
                    new_population.append(offspring2)
            
            # Replace old population
            population = np.array(new_population[:self.population_size])
        
        # Step 3: Return best solution
        final_delay, final_energy = self.simulate_execution(best_chromosome, dag)
        
        return best_chromosome, final_delay, final_energy