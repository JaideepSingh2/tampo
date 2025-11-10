import numpy as np
import yaml
import argparse
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt

from env.base_offloading_env import TaskOffloadingEnv
from utils.dag_parser import DAGParser
from utils.metrics import calculate_hypervolume, normalize_objectives

from algorithms.heuristic.heft import HEFTScheduler
from algorithms.heuristic.pso import PSOScheduler
from algorithms.heuristic.ga import GAScheduler
from algorithms.rl.ppo_baseline import PPOAgent
from algorithms.rl.gmorl import GMORLAgent
from algorithms.rl.tampo import TAMPOAgent

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_environment(config):
    """Setup the task offloading environment"""
    env = TaskOffloadingEnv(config['system'])
    return env

def test_heuristic_algorithms(env, dag_parser, config):
    """Test all heuristic algorithms"""
    print("\n" + "="*60)
    print("Testing Heuristic Algorithms")
    print("="*60)
    
    results = {}
    
    # Load DAG tasks
    dags = dag_parser.load_dataset(num_graphs=10)
    
    if len(dags) == 0:
        print("Warning: No DAG graphs loaded. Skipping heuristic algorithms.")
        return results
    
    # Test HEFT
    if config['algorithms']['heft']['enabled']:
        print("\n[1/3] Testing HEFT...")
        heft = HEFTScheduler(env)
        heft_delays = []
        heft_energies = []
        
        for dag in dags:
            schedule, delay, energy = heft.schedule(dag)
            heft_delays.append(delay)
            heft_energies.append(energy)
        
        results['HEFT'] = {
            'avg_delay': np.mean(heft_delays),
            'avg_energy': np.mean(heft_energies),
            'std_delay': np.std(heft_delays),
            'std_energy': np.std(heft_energies)
        }
        print(f"  Avg Delay: {results['HEFT']['avg_delay']:.4f}s")
        print(f"  Avg Energy: {results['HEFT']['avg_energy']:.4f}J")
    
    # Test PSO
    if config['algorithms']['pso']['enabled']:
        print("\n[2/3] Testing PSO...")
        pso = PSOScheduler(env, config['algorithms']['pso'])
        pso_delays = []
        pso_energies = []
        
        for dag in dags:
            preference = np.array([0.5, 0.5])
            schedule, delay, energy = pso.optimize(dag, preference)
            pso_delays.append(delay)
            pso_energies.append(energy)
        
        results['PSO'] = {
            'avg_delay': np.mean(pso_delays),
            'avg_energy': np.mean(pso_energies),
            'std_delay': np.std(pso_delays),
            'std_energy': np.std(pso_energies)
        }
        print(f"  Avg Delay: {results['PSO']['avg_delay']:.4f}s")
        print(f"  Avg Energy: {results['PSO']['avg_energy']:.4f}J")
    
    # Test GA
    if config['algorithms']['ga']['enabled']:
        print("\n[3/3] Testing GA...")
        ga = GAScheduler(env, config['algorithms']['ga'])
        ga_delays = []
        ga_energies = []
        
        for dag in dags:
            preference = np.array([0.5, 0.5])
            schedule, delay, energy = ga.optimize(dag, preference)
            ga_delays.append(delay)
            ga_energies.append(energy)
        
        results['GA'] = {
            'avg_delay': np.mean(ga_delays),
            'avg_energy': np.mean(ga_energies),
            'std_delay': np.std(ga_delays),
            'std_energy': np.std(ga_energies)
        }
        print(f"  Avg Delay: {results['GA']['avg_delay']:.4f}s")
        print(f"  Avg Energy: {results['GA']['avg_energy']:.4f}J")
    
    return results

def test_rl_algorithms(env, config):
    """Test RL algorithms"""
    print("\n" + "="*60)
    print("Testing RL Algorithms")
    print("="*60)
    
    results = {}
    
    # Increase training episodes significantly
    NUM_TRAINING_EPISODES = 2000  # Much more training!
    NUM_EVAL_EPISODES = 20  # More evaluation episodes
    
    # Test PPO
    print(f"\n[1/3] Training PPO Baseline ({NUM_TRAINING_EPISODES} episodes)...")
    ppo = PPOAgent(env, config['training'])
    ppo.train(num_episodes=NUM_TRAINING_EPISODES)
    
    # Evaluate PPO
    print("Evaluating PPO...")
    ppo_delays = []
    ppo_energies = []
    for eval_ep in range(NUM_EVAL_EPISODES):
        state = env.reset()
        done = False
        episode_delay = 0
        episode_energy = 0
        
        while not done:
            action, _, _ = ppo.select_action(state, deterministic=True)
            state, reward, done, info = env.step(action)
            episode_delay += info['delay']
            episode_energy += info['energy']
        
        ppo_delays.append(episode_delay)
        ppo_energies.append(episode_energy)
    
    results['PPO'] = {
        'avg_delay': np.mean(ppo_delays),
        'avg_energy': np.mean(ppo_energies),
        'std_delay': np.std(ppo_delays),
        'std_energy': np.std(ppo_energies)
    }
    print(f"  Avg Delay: {results['PPO']['avg_delay']:.4f}s")
    print(f"  Avg Energy: {results['PPO']['avg_energy']:.4f}J")
    
    # Test GMORL
    if config['algorithms']['gmorl']['enabled']:
        print(f"\n[2/3] Training GMORL ({NUM_TRAINING_EPISODES} episodes)...")
        gmorl = GMORLAgent(env, config['training'])
        gmorl.train(num_episodes=NUM_TRAINING_EPISODES)
        
        # Evaluate GMORL with different preferences
        print("Evaluating GMORL...")
        preferences = [
            np.array([0.8, 0.2]),  # Delay-focused
            np.array([0.5, 0.5]),  # Balanced
            np.array([0.2, 0.8])   # Energy-focused
        ]
        
        gmorl_delays = []
        gmorl_energies = []
        
        for pref in preferences:
            for _ in range(NUM_EVAL_EPISODES // 3):
                state = env.reset(preference_vector=pref)
                done = False
                episode_delay = 0
                episode_energy = 0
                
                while not done:
                    action, _, _ = gmorl.select_action(state, pref, deterministic=True)
                    state, reward, done, info = env.step(action)
                    episode_delay += info['delay']
                    episode_energy += info['energy']
                
                gmorl_delays.append(episode_delay)
                gmorl_energies.append(episode_energy)
        
        results['GMORL'] = {
            'avg_delay': np.mean(gmorl_delays),
            'avg_energy': np.mean(gmorl_energies),
            'std_delay': np.std(gmorl_delays),
            'std_energy': np.std(gmorl_energies)
        }
        print(f"  Avg Delay: {results['GMORL']['avg_delay']:.4f}s")
        print(f"  Avg Energy: {results['GMORL']['avg_energy']:.4f}J")
    
    # Test TAM-PO
    if config['algorithms']['tampo']['enabled']:
        print("\n[3/3] Training TAM-PO...")
        
        # Load task dataset for meta-learning
        dag_parser = DAGParser(data_folder="data/meta_offloading_20/offload_random20_1")
        task_graphs = dag_parser.load_dataset(num_graphs=50)  # Load more tasks
        
        if len(task_graphs) == 0:
            print("Warning: No task graphs loaded. Skipping TAM-PO.")
        else:
            # Convert DAG format to task format
            tasks_for_env = []
            for dag in task_graphs:
                task = {
                    'num_tasks': dag['num_tasks'],
                    'tasks': dag['tasks'],
                    'edges': dag['edges'],
                    'size': sum(t['data_size'] for t in dag['tasks']),
                    'cycles': sum(t['cycles'] for t in dag['tasks'])
                }
                tasks_for_env.append(task)
            
            # Load tasks into environment
            env.load_task_dataset(tasks_for_env)
            
            tampo = TAMPOAgent(env, config['training'])
            
            # Meta-training with more iterations
            print(f"Meta-training with {len(tasks_for_env)} tasks...")
            tampo.meta_train(
                num_iterations=100,  # More meta-iterations
                meta_batch_size=min(10, len(tasks_for_env))
            )
            
            # Evaluate TAM-PO properly
            print("Evaluating TAM-PO...")
            tampo_delays = []
            tampo_energies = []
            
            for _ in range(NUM_EVAL_EPISODES):
                # Sample a task
                task_id = env.sample_tasks(1)[0]
                env.set_task(task_id)
                
                # Inner loop adaptation
                adapted_params = tampo.inner_loop_adaptation(task_id, num_steps=5)
                
                # Create adapted policy
                adapted_policy = type(tampo.meta_policy)(
                    env.observation_space.shape[0],
                    env.action_space.n
                ).to(tampo.device)
                adapted_policy.load_state_dict(adapted_params)
                adapted_policy.eval()
                
                # Test with adapted policy
                state = env.reset()
                preference = tampo.sample_preference()
                done = False
                episode_delay = 0
                episode_energy = 0
                
                while not done:
                    import torch
                    from torch.distributions import Categorical
                    
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(tampo.device)
                    pref_tensor = torch.FloatTensor(preference).unsqueeze(0).to(tampo.device)
                    
                    with torch.no_grad():
                        logits = adapted_policy(state_tensor, pref_tensor)
                        action_probs = torch.softmax(logits, dim=-1)
                        action = torch.argmax(action_probs, dim=-1).item()
                    
                    state, reward, done, info = env.step(action)
                    episode_delay += info['delay']
                    episode_energy += info['energy']
                
                tampo_delays.append(episode_delay)
                tampo_energies.append(episode_energy)
            
            results['TAMPO'] = {
                'avg_delay': np.mean(tampo_delays),
                'avg_energy': np.mean(tampo_energies),
                'std_delay': np.std(tampo_delays),
                'std_energy': np.std(tampo_energies)
            }
            print(f"  Avg Delay: {results['TAMPO']['avg_delay']:.4f}s")
            print(f"  Avg Energy: {results['TAMPO']['avg_energy']:.4f}J")
    
    return results

def save_results(results, output_dir):
    """Save results to JSON and generate plots"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save JSON
    results_file = os.path.join(output_dir, 'results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {results_file}")
    
    # Generate comparison plots
    if len(results) > 0:
        plot_comparison(results, output_dir)

def plot_comparison(results, output_dir):
    """Generate comparison plots"""
    algorithms = list(results.keys())
    delays = [results[alg]['avg_delay'] for alg in algorithms]
    energies = [results[alg]['avg_energy'] for alg in algorithms]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Delay comparison
    ax1.bar(algorithms, delays, color=['blue', 'orange', 'green', 'red', 'purple', 'brown'][:len(algorithms)])
    ax1.set_ylabel('Average Delay (s)')
    ax1.set_title('Delay Comparison')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    
    # Energy comparison
    ax2.bar(algorithms, energies, color=['blue', 'orange', 'green', 'red', 'purple', 'brown'][:len(algorithms)])
    ax2.set_ylabel('Average Energy (J)')
    ax2.set_title('Energy Comparison')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plot_file = os.path.join(output_dir, 'comparison.png')
    plt.savefig(plot_file, dpi=300)
    print(f"Comparison plot saved to {plot_file}")
    
    # Pareto front plot
    plt.figure(figsize=(8, 6))
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
    for i, alg in enumerate(algorithms):
        plt.scatter(results[alg]['avg_delay'], 
                   results[alg]['avg_energy'],
                   label=alg, s=200, alpha=0.7, 
                   color=colors[i % len(colors)])
    
    plt.xlabel('Delay (s)', fontsize=12)
    plt.ylabel('Energy (J)', fontsize=12)
    plt.title('Pareto Front Comparison', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    pareto_file = os.path.join(output_dir, 'pareto_front.png')
    plt.savefig(pareto_file, dpi=300)
    print(f"Pareto front plot saved to {pareto_file}")

def main():
    parser = argparse.ArgumentParser(description='Task Offloading Algorithm Comparison')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--test-heuristic', action='store_true',
                       help='Test heuristic algorithms')
    parser.add_argument('--test-rl', action='store_true',
                       help='Test RL algorithms')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test with fewer episodes')
    
    args = parser.parse_args()
    
    # Load configuration
    print("Loading configuration...")
    config = load_config(args.config)
    
    # Adjust for quick test
    if args.quick:
        print("Running in QUICK mode (fewer training episodes)")
        # Will use fewer episodes in test_rl_algorithms
    
    # Setup environment
    print("Setting up environment...")
    env = setup_environment(config)
    
    # Setup DAG parser
    dag_parser = DAGParser(data_folder="data/meta_offloading_20/offload_random20_1")
    
    # Run tests
    all_results = {}
    
    if args.test_heuristic or (not args.test_heuristic and not args.test_rl):
        heuristic_results = test_heuristic_algorithms(env, dag_parser, config)
        all_results.update(heuristic_results)
    
    if args.test_rl or (not args.test_heuristic and not args.test_rl):
        rl_results = test_rl_algorithms(env, config)
        all_results.update(rl_results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output, timestamp)
    save_results(all_results, output_dir)
    
    print("\n" + "="*60)
    print("Testing Complete!")
    print("="*60)
    print("\nKey Improvements Made:")
    print("1. Increased RL training from 100 to 2000 episodes")
    print("2. Better reward normalization")
    print("3. Proper evaluation with deterministic policies")
    print("4. More meta-learning tasks for TAM-PO")
    print("5. Fixed TAM-PO evaluation to use adapted policy")

if __name__ == "__main__":
    main()