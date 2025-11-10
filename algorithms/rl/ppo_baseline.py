import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple
from torch.distributions import Categorical

class PolicyNetwork(nn.Module):
    """Actor network for PPO"""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: List[int] = [256, 256]):
        super(PolicyNetwork, self).__init__()
        
        layers = []
        prev_dim = obs_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, action_dim))
        layers.append(nn.Softmax(dim=-1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class ValueNetwork(nn.Module):
    """Critic network for PPO"""
    
    def __init__(self, obs_dim: int, hidden_dims: List[int] = [256, 256]):
        super(ValueNetwork, self).__init__()
        
        layers = []
        prev_dim = obs_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class PPOAgent:
    """
    Proximal Policy Optimization Agent for Task Offloading
    
    References:
        Schulman et al., "Proximal Policy Optimization Algorithms," 2017
    """
    
    def __init__(self, env, config: Dict):
        """
        Initialize PPO agent
        
        Args:
            env: Task offloading environment
            config: Configuration dictionary
        """
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Hyperparameters
        self.learning_rate = config.get('learning_rate', 3e-4)
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.clip_epsilon = config.get('clip_epsilon', 0.2)
        self.value_coef = config.get('value_coef', 0.5)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        self.ppo_epochs = config.get('ppo_epochs', 10)
        self.batch_size = config.get('batch_size', 64)
        
        # Networks
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        hidden_dims = config.get('hidden_dims', [256, 256])
        
        self.policy = PolicyNetwork(obs_dim, action_dim, hidden_dims).to(self.device)
        self.value = ValueNetwork(obs_dim, hidden_dims).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            list(self.policy.parameters()) + list(self.value.parameters()),
            lr=self.learning_rate
        )
        
        # Storage
        self.reset_storage()
    
    def reset_storage(self):
        """Reset experience storage"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def select_action(self, state: np.ndarray, deterministic: bool = False):
        """
        Select action based on current policy
        
        Args:
            state: Current state
            deterministic: Whether to select deterministically
            
        Returns:
            action, log_prob, value
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs = self.policy(state_tensor)
            value = self.value(state_tensor)
        
        if deterministic:
            action = torch.argmax(action_probs, dim=-1)
        else:
            dist = Categorical(action_probs)
            action = dist.sample()
        
        log_prob = torch.log(action_probs.squeeze(0)[action])
        
        return action.item(), log_prob.item(), value.item()
    
    def store_transition(self, state, action, reward, value, log_prob, done):
        """Store transition in memory"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def compute_gae(self, next_value: float):
        """
        Compute Generalized Advantage Estimation
        
        Args:
            next_value: Value of the next state
            
        Returns:
            advantages, returns
        """
        advantages = []
        returns = []
        
        gae = 0
        next_val = next_value
        
        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + self.gamma * next_val * (1 - self.dones[t]) - self.values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[t]) * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[t])
            
            next_val = self.values[t]
        
        return np.array(advantages), np.array(returns)
    
    def update(self, next_value: float):
        """
        Update policy and value networks
        
        Args:
            next_value: Value of the next state
        """
        # Compute advantages and returns
        advantages, returns = self.compute_gae(next_value)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # PPO update
        for _ in range(self.ppo_epochs):
            # Compute current policy and values
            action_probs = self.policy(states)
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            values = self.value(states).squeeze()
            
            # Policy loss with clipping
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = nn.MSELoss()(values, returns)
            
            # Total loss
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.policy.parameters()) + list(self.value.parameters()),
                self.max_grad_norm
            )
            self.optimizer.step()
        
        # Clear storage
        self.reset_storage()
    
    def train(self, num_episodes: int):
        """
        Train the agent
        
        Args:
            num_episodes: Number of training episodes
        """
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action, log_prob, value = self.select_action(state)
                next_state, reward, done, info = self.env.step(action)
                
                self.store_transition(state, action, reward, value, log_prob, done)
                
                state = next_state
                episode_reward += reward
            
            # Final value
            if done:
                next_value = 0.0
            else:
                _, _, next_value = self.select_action(next_state)
            
            # Update policy
            self.update(next_value)
            
            if episode % 10 == 0:
                print(f"Episode {episode}, Reward: {episode_reward:.2f}")
    
    def save(self, path: str):
        """Save model"""
        torch.save({
            'policy': self.policy.state_dict(),
            'value': self.value.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)
    
    def load(self, path: str):
        """Load model"""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy'])
        self.value.load_state_dict(checkpoint['value'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
