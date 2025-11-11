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
from algorithms.rl.tampo import TAMPOFramework

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_environment(config):
    """Setup the task offloading environment"""
    env = TaskOffloadingEnv(config['system'])
    return env

def get_user_input():
    """Get user input for which algorithms to run"""
    print("\n" + "="*70)
    print(" "*20 + "ALGORITHM SELECTION")
    print("="*70 + "\n")
    
    algorithms = {}
    
    # Heuristic algorithms
    print("HEURISTIC ALGORITHMS:")
    print("-" * 40)
    
    heft = input("Run HEFT? (yes/no): ").strip().lower()
    algorithms['HEFT'] = heft in ['yes', 'y']
    
    pso = input("Run PSO? (yes/no): ").strip().lower()
    algorithms['PSO'] = pso in ['yes', 'y']
    
    ga = input("Run GA? (yes/no): ").strip().lower()
    algorithms['GA'] = ga in ['yes', 'y']
    
    if any([algorithms['HEFT'], algorithms['PSO'], algorithms['GA']]):
        heuristic_tasks = input("Number of test tasks for heuristics (default 10): ").strip()
        algorithms['heuristic_tasks'] = int(heuristic_tasks) if heuristic_tasks else 10
    else:
        algorithms['heuristic_tasks'] = 10
    
    # RL algorithms
    print("\n\nREINFORCEMENT LEARNING ALGORITHMS:")
    print("-" * 40)
    
    ppo_run = input("Run PPO? (yes/no): ").strip().lower()
    algorithms['PPO'] = ppo_run in ['yes', 'y']
    
    gmorl_run = input("Run GMORL? (yes/no): ").strip().lower()
    algorithms['GMORL'] = gmorl_run in ['yes', 'y']
    
    tampo_run = input("Run TAMPO? (yes/no): ").strip().lower()
    algorithms['TAMPO'] = tampo_run in ['yes', 'y']
    
    # Common training/evaluation parameters for RL
    if any([algorithms['PPO'], algorithms['GMORL'], algorithms['TAMPO']]):
        print("\n\nRL TRAINING & EVALUATION PARAMETERS:")
        print("-" * 40)
        
        # Training episodes/iterations
        if algorithms['PPO']:
            ppo_episodes = input("Number of training episodes for PPO (default 100): ").strip()
            algorithms['ppo_episodes'] = int(ppo_episodes) if ppo_episodes else 100
        else:
            algorithms['ppo_episodes'] = 100
        
        if algorithms['GMORL']:
            gmorl_episodes = input("Number of training episodes for GMORL (default 100): ").strip()
            algorithms['gmorl_episodes'] = int(gmorl_episodes) if gmorl_episodes else 100
        else:
            algorithms['gmorl_episodes'] = 100
        
        if algorithms['TAMPO']:
            tampo_iterations = input("Number of meta-iterations for TAMPO (default 100): ").strip()
            algorithms['tampo_iterations'] = int(tampo_iterations) if tampo_iterations else 100
        else:
            algorithms['tampo_iterations'] = 100
        
        # Common evaluation episodes
        eval_episodes = input("Number of evaluation episodes for all RL algorithms (default 20): ").strip()
        algorithms['eval_episodes'] = int(eval_episodes) if eval_episodes else 20
    else:
        algorithms['ppo_episodes'] = 100
        algorithms['gmorl_episodes'] = 100
        algorithms['tampo_iterations'] = 100
        algorithms['eval_episodes'] = 20
    
    # Summary
    print("\n" + "="*70)
    print(" "*25 + "CONFIGURATION SUMMARY")
    print("="*70)
    
    selected = []
    if algorithms['HEFT']:
        selected.append(f"HEFT")
    if algorithms['PSO']:
        selected.append(f"PSO")
    if algorithms['GA']:
        selected.append(f"GA")
    if algorithms['PPO']:
        selected.append(f"PPO ({algorithms['ppo_episodes']} training episodes)")
    if algorithms['GMORL']:
        selected.append(f"GMORL ({algorithms['gmorl_episodes']} training episodes)")
    if algorithms['TAMPO']:
        selected.append(f"TAMPO ({algorithms['tampo_iterations']} meta-iterations)")
    
    if len(selected) == 0:
        print("\n⚠️  No algorithms selected. Exiting...")
        return None
    
    print("\nSelected Algorithms:")
    for i, alg in enumerate(selected, 1):
        print(f"  {i}. {alg}")
    
    if any([algorithms['HEFT'], algorithms['PSO'], algorithms['GA']]):
        print(f"\nHeuristic test tasks: {algorithms['heuristic_tasks']}")
    
    if any([algorithms['PPO'], algorithms['GMORL'], algorithms['TAMPO']]):
        print(f"RL evaluation episodes: {algorithms['eval_episodes']} (common for all RL algorithms)")
    
    print("\n" + "="*70)
    
    confirm = input("\nProceed with this configuration? (yes/no): ").strip().lower()
    if confirm not in ['yes', 'y']:
        print("Configuration cancelled.")
        return None
    
    return algorithms

def test_heft(env, dag_parser, num_tasks=10):
    """Test HEFT algorithm"""
    print("\nTesting HEFT...")
    dags = dag_parser.load_dataset(num_graphs=num_tasks)
    
    if len(dags) == 0:
        print("Warning: No DAG graphs loaded.")
        return None
    
    heft = HEFTScheduler(env)
    delays, energies = [], []
    
    for dag in dags:
        schedule, delay, energy = heft.schedule(dag)
        delays.append(delay)
        energies.append(energy)
    
    result = {
        'avg_delay': np.mean(delays),
        'avg_energy': np.mean(energies),
        'std_delay': np.std(delays),
        'std_energy': np.std(energies)
    }
    print(f"  Avg Delay: {result['avg_delay']:.4f}s")
    print(f"  Avg Energy: {result['avg_energy']:.4f}J")
    return result

def test_pso(env, dag_parser, config, num_tasks=10):
    """Test PSO algorithm"""
    print("\nTesting PSO...")
    dags = dag_parser.load_dataset(num_graphs=num_tasks)
    
    if len(dags) == 0:
        print("Warning: No DAG graphs loaded.")
        return None
    
    pso = PSOScheduler(env, config['algorithms']['pso'])
    delays, energies = [], []
    
    for dag in dags:
        preference = np.array([0.5, 0.5])
        schedule, delay, energy = pso.optimize(dag, preference)
        delays.append(delay)
        energies.append(energy)
    
    result = {
        'avg_delay': np.mean(delays),
        'avg_energy': np.mean(energies),
        'std_delay': np.std(delays),
        'std_energy': np.std(energies)
    }
    print(f"  Avg Delay: {result['avg_delay']:.4f}s")
    print(f"  Avg Energy: {result['avg_energy']:.4f}J")
    return result

def test_ga(env, dag_parser, config, num_tasks=10):
    """Test GA algorithm"""
    print("\nTesting GA...")
    dags = dag_parser.load_dataset(num_graphs=num_tasks)
    
    if len(dags) == 0:
        print("Warning: No DAG graphs loaded.")
        return None
    
    ga = GAScheduler(env, config['algorithms']['ga'])
    delays, energies = [], []
    
    for dag in dags:
        preference = np.array([0.5, 0.5])
        schedule, delay, energy = ga.optimize(dag, preference)
        delays.append(delay)
        energies.append(energy)
    
    result = {
        'avg_delay': np.mean(delays),
        'avg_energy': np.mean(energies),
        'std_delay': np.std(delays),
        'std_energy': np.std(energies)
    }
    print(f"  Avg Delay: {result['avg_delay']:.4f}s")
    print(f"  Avg Energy: {result['avg_energy']:.4f}J")
    return result

def test_ppo(env, config, train_episodes=100, eval_episodes=20):
    """Test PPO algorithm"""
    print(f"\nTraining PPO ({train_episodes} episodes)...")
    ppo = PPOAgent(env, config['training'])
    ppo.train(num_episodes=train_episodes)
    
    print(f"Evaluating PPO ({eval_episodes} episodes)...")
    delays, energies = [], []
    
    for _ in range(eval_episodes):
        state = env.reset()
        done = False
        episode_delay = 0
        episode_energy = 0
        
        while not done:
            action, _, _ = ppo.select_action(state, deterministic=True)
            state, reward, done, info = env.step(action)
            episode_delay += info['delay']
            episode_energy += info['energy']
        
        delays.append(episode_delay)
        energies.append(episode_energy)
    
    result = {
        'avg_delay': np.mean(delays),
        'avg_energy': np.mean(energies),
        'std_delay': np.std(delays),
        'std_energy': np.std(energies)
    }
    print(f"  Avg Delay: {result['avg_delay']:.4f}s")
    print(f"  Avg Energy: {result['avg_energy']:.4f}J")
    return result

def test_gmorl(env, config, train_episodes=100, eval_episodes=20):
    """Test GMORL algorithm"""
    print(f"\nTraining GMORL ({train_episodes} episodes)...")
    gmorl = GMORLAgent(env, config['training'])
    gmorl.train(num_episodes=train_episodes)
    
    print(f"Evaluating GMORL ({eval_episodes} episodes)...")
    preferences = [
        np.array([0.8, 0.2]),  # Delay-focused
        np.array([0.5, 0.5]),  # Balanced
        np.array([0.2, 0.8])   # Energy-focused
    ]
    
    delays, energies = [], []
    
    for pref in preferences:
        for _ in range(eval_episodes // 3):
            state = env.reset(preference_vector=pref)
            done = False
            episode_delay = 0
            episode_energy = 0
            
            while not done:
                action, _, _ = gmorl.select_action(state, pref, deterministic=True)
                state, reward, done, info = env.step(action)
                episode_delay += info['delay']
                episode_energy += info['energy']
            
            delays.append(episode_delay)
            energies.append(episode_energy)
    
    result = {
        'avg_delay': np.mean(delays),
        'avg_energy': np.mean(energies),
        'std_delay': np.std(delays),
        'std_energy': np.std(energies)
    }
    print(f"  Avg Delay: {result['avg_delay']:.4f}s")
    print(f"  Avg Energy: {result['avg_energy']:.4f}J")
    return result

def test_tampo(env, dag_parser, config, train_iterations=100, eval_episodes=20):
    """Test TAM-PO algorithm"""
    print(f"\nTraining TAM-PO ({train_iterations} iterations)...")
    
    # Load task dataset
    task_graphs = dag_parser.load_dataset(num_graphs=50)
    
    if len(task_graphs) == 0:
        print("Warning: No task graphs loaded.")
        return None
    
    # Convert DAG format
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
    
    # Create and train TAM-PO
    tampo_framework = TAMPOFramework(env, config['training'])
    tampo_framework.train(
        num_iterations=train_iterations,
        meta_batch_size=min(10, len(tasks_for_env))
    )
    
    # Evaluate
    print(f"Evaluating TAM-PO ({eval_episodes} episodes)...")
    result = tampo_framework.evaluate(num_episodes=eval_episodes)
    
    print(f"  Avg Delay: {result['avg_delay']:.4f}s")
    print(f"  Avg Energy: {result['avg_energy']:.4f}J")
    
    # Save model
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "tampo_trained.pth")
    tampo_framework.save(model_path)
    
    return result

def save_results(results, output_dir):
    """Save results to JSON and generate plots"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save JSON
    results_file = os.path.join(output_dir, 'results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {results_file}")
    
    # Generate plots if results exist
    if len(results) > 0:
        plot_comparison(results, output_dir)

def plot_comparison(results, output_dir):
    """Generate comparison plots"""
    algorithms = list(results.keys())
    delays = [results[alg]['avg_delay'] for alg in algorithms]
    energies = [results[alg]['avg_energy'] for alg in algorithms]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Delay comparison
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    ax1.bar(algorithms, delays, color=colors[:len(algorithms)], alpha=0.8, edgecolor='black')
    ax1.set_ylabel('Average Delay (s)', fontsize=12, fontweight='bold')
    ax1.set_title('Delay Comparison', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    for i, (alg, delay) in enumerate(zip(algorithms, delays)):
        ax1.text(i, delay, f'{delay:.2f}', ha='center', va='bottom', fontsize=9)
    
    # Energy comparison
    ax2.bar(algorithms, energies, color=colors[:len(algorithms)], alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Average Energy (J)', fontsize=12, fontweight='bold')
    ax2.set_title('Energy Comparison', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    for i, (alg, energy) in enumerate(zip(algorithms, energies)):
        ax2.text(i, energy, f'{energy:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plot_file = os.path.join(output_dir, 'comparison.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to {plot_file}")
    plt.close()
    
    # Pareto front plot
    plt.figure(figsize=(10, 8))
    
    for i, alg in enumerate(algorithms):
        plt.scatter(
            results[alg]['avg_delay'], 
            results[alg]['avg_energy'],
            label=alg, 
            s=300, 
            alpha=0.7, 
            color=colors[i % len(colors)],
            edgecolors='black',
            linewidths=2
        )
        
        plt.annotate(
            alg,
            (results[alg]['avg_delay'], results[alg]['avg_energy']),
            xytext=(10, 10),
            textcoords='offset points',
            fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3)
        )
    
    plt.xlabel('Average Delay (s)', fontsize=14, fontweight='bold')
    plt.ylabel('Average Energy (J)', fontsize=14, fontweight='bold')
    plt.title('Pareto Front: Delay vs Energy Trade-off', fontsize=16, fontweight='bold')
    plt.legend(fontsize=11, loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.4, linestyle='--')
    
    if len(delays) > 0 and len(energies) > 0:
        plt.axvline(x=min(delays), color='green', linestyle='--', alpha=0.5, label='Best Delay')
        plt.axhline(y=min(energies), color='blue', linestyle='--', alpha=0.5, label='Best Energy')
    
    pareto_file = os.path.join(output_dir, 'pareto_front.png')
    plt.savefig(pareto_file, dpi=300, bbox_inches='tight')
    print(f"Pareto front plot saved to {pareto_file}")
    plt.close()

def main():
    # Print banner
    print("\n" + "="*70)
    print(" "*10 + "TASK OFFLOADING ALGORITHM COMPARISON FRAMEWORK")
    print(" "*20 + "Interactive Mode")
    print("="*70 + "\n")
    
    # Get user input
    user_choices = get_user_input()
    
    if user_choices is None:
        return
    
    # Load configuration
    print("\n📋 Loading configuration...")
    config = load_config('configs/default_config.yaml')
    print("✓ Configuration loaded")
    
    # Setup environment
    print("\n🏗️  Setting up environment...")
    env = setup_environment(config)
    print(f"✓ Environment created with {env.num_servers} servers")
    
    # Setup DAG parser
    dag_parser = DAGParser(data_folder="data/meta_offloading_20/offload_random20_1")
    
    # Run selected algorithms
    results = {}
    
    print("\n" + "="*70)
    print("Starting Algorithm Execution")
    print("="*70)
    
    # Heuristic algorithms
    if user_choices['HEFT']:
        result = test_heft(env, dag_parser, user_choices['heuristic_tasks'])
        if result:
            results['HEFT'] = result
    
    if user_choices['PSO']:
        result = test_pso(env, dag_parser, config, user_choices['heuristic_tasks'])
        if result:
            results['PSO'] = result
    
    if user_choices['GA']:
        result = test_ga(env, dag_parser, config, user_choices['heuristic_tasks'])
        if result:
            results['GA'] = result
    
    # RL algorithms
    if user_choices['PPO']:
        result = test_ppo(env, config, user_choices['ppo_episodes'], user_choices['eval_episodes'])
        if result:
            results['PPO'] = result
    
    if user_choices['GMORL']:
        result = test_gmorl(env, config, user_choices['gmorl_episodes'], user_choices['eval_episodes'])
        if result:
            results['GMORL'] = result
    
    if user_choices['TAMPO']:
        result = test_tampo(env, dag_parser, config, user_choices['tampo_iterations'], user_choices['eval_episodes'])
        if result:
            results['TAMPO'] = result
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join('results', timestamp)
    save_results(results, output_dir)
    
    # Print summary
    print("\n" + "="*70)
    print(" "*25 + "EXPERIMENT COMPLETE!")
    print("="*70)
    
    if len(results) > 0:
        print("\n📊 Results Summary:")
        print("-" * 70)
        print(f"{'Algorithm':<15} {'Avg Delay (s)':<15} {'Avg Energy (J)':<15}")
        print("-" * 70)
        for alg, metrics in results.items():
            print(f"{alg:<15} {metrics['avg_delay']:<15.4f} {metrics['avg_energy']:<15.4f}")
        print("-" * 70)
        
        # Find best performers
        best_delay_alg = min(results.items(), key=lambda x: x[1]['avg_delay'])
        best_energy_alg = min(results.items(), key=lambda x: x[1]['avg_energy'])
        
        print(f"\n🏆 Best Delay: {best_delay_alg[0]} ({best_delay_alg[1]['avg_delay']:.4f}s)")
        print(f"🏆 Best Energy: {best_energy_alg[0]} ({best_energy_alg[1]['avg_energy']:.4f}J)")
    
    print(f"\n📁 Results saved to: {output_dir}")
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    main()