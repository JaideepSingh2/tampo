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
        
    def evaluate_fitness(self, chromosome: np.ndarray, dag: Dict, 
                        preference: np.ndarray) -> float:
        """
        Evaluate fitness of a chromosome
        
        Args:
            chromosome: Offloading decisions
            dag: Task graph
            preference: [w_delay, w_energy]
            
        Returns:
            Fitness value (higher is better)
        """
        delay, energy = self.simulate_execution(chromosome, dag)
        
        # Multi-objective fitness (negative because we minimize)
        fitness = -(preference[0] * delay + preference[1] * energy)
        
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
        num_resources = 1 + 1 + self.env.num_edge_servers
        
        # Ensure valid schedule
        schedule = np.clip(schedule, 0, num_resources - 1).astype(int)
        
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
            
            # Execution time and energy
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
    
    def tournament_selection(self, population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        """
        Tournament selection
        
        Args:
            population: Current population
            fitness: Fitness values
            
        Returns:
            Selected individual
        """
        tournament_indices = np.random.choice(len(population), 
                                             self.tournament_size, 
                                             replace=False)
        tournament_fitness = fitness[tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        
        return population[winner_idx].copy()
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Single-point crossover
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Two offspring
        """
        if np.random.rand() < self.crossover_rate:
            point = np.random.randint(1, len(parent1))
            offspring1 = np.concatenate([parent1[:point], parent2[point:]])
            offspring2 = np.concatenate([parent2[:point], parent1[point:]])
        else:
            offspring1 = parent1.copy()
            offspring2 = parent2.copy()
        
        return offspring1, offspring2
    
    def mutate(self, chromosome: np.ndarray, num_resources: int) -> np.ndarray:
        """
        Mutation operator
        
        Args:
            chromosome: Individual to mutate
            num_resources: Number of available resources
            
        Returns:
            Mutated chromosome
        """
        for i in range(len(chromosome)):
            if np.random.rand() < self.mutation_rate:
                chromosome[i] = np.random.randint(0, num_resources)
        
        return chromosome
    
    def optimize(self, dag: Dict, preference: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Run GA optimization
        
        Args:
            dag: Task graph
            preference: User preference vector
            
        Returns:
            (best_schedule, delay, energy)
        """
        num_tasks = dag['num_tasks']
        num_resources = 1 + 1 + self.env.num_edge_servers
        
        # Initialize population
        population = np.random.randint(0, num_resources, 
                                      (self.population_size, num_tasks))
        
        best_chromosome = None
        best_fitness = -float('inf')
        
        # Evolution loop
        for generation in range(self.num_generations):
            # Evaluate fitness
            fitness = np.array([self.evaluate_fitness(ind, dag, preference) 
                               for ind in population])
            
            # Track best
            gen_best_idx = np.argmax(fitness)
            if fitness[gen_best_idx] > best_fitness:
                best_fitness = fitness[gen_best_idx]
                best_chromosome = population[gen_best_idx].copy()
            
            # Create next generation
            new_population = []
            
            # Elitism: keep best individual
            new_population.append(best_chromosome.copy())
            
            # Generate rest of population
            while len(new_population) < self.population_size:
                # Selection
                parent1 = self.tournament_selection(population, fitness)
                parent2 = self.tournament_selection(population, fitness)
                
                # Crossover
                offspring1, offspring2 = self.crossover(parent1, parent2)
                
                # Mutation
                offspring1 = self.mutate(offspring1, num_resources)
                offspring2 = self.mutate(offspring2, num_resources)
                
                new_population.append(offspring1)
                if len(new_population) < self.population_size:
                    new_population.append(offspring2)
            
            population = np.array(new_population)
        
        # Get final results
        delay, energy = self.simulate_execution(best_chromosome, dag)
        
        return best_chromosome, delay, energy



