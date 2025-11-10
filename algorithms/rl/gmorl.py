import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple
from torch.distributions import Categorical

class PreferenceConditionedPolicyNetwork(nn.Module):
    """
    Policy network conditioned on user preference vector
    Enables single model to serve multiple user preferences
    """
    
    def __init__(self, obs_dim: int, action_dim: int, 
                 hidden_dims: List[int] = [256, 256]):
        super(PreferenceConditionedPolicyNetwork, self).__init__()
        
        # Preference vector dimension (delay weight, energy weight)
        self.preference_dim = 2
        
        # Input layer takes both observation and preference
        input_dim = obs_dim + self.preference_dim
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, obs, preference):
        """
        Forward pass with preference conditioning
        
        Args:
            obs: State observation
            preference: [w_delay, w_energy] preference vector
        
        Returns:
            Action logits
        """
        # Concatenate observation and preference
        x = torch.cat([obs, preference], dim=-1)
        return self.network(x)

class MultiObjectiveValueNetwork(nn.Module):
    """
    Value network that outputs separate values for each objective
    """
    
    def __init__(self, obs_dim: int, num_objectives: int = 2,
                 hidden_dims: List[int] = [256, 256]):
        super(MultiObjectiveValueNetwork, self).__init__()
        
        self.num_objectives = num_objectives
        
        # Shared layers
        shared_layers = []
        prev_dim = obs_dim + 2  # obs + preference
        
        for hidden_dim in hidden_dims:
            shared_layers.append(nn.Linear(prev_dim, hidden_dim))
            shared_layers.append(nn.ReLU())
            shared_layers.append(nn.LayerNorm(hidden_dim))
            prev_dim = hidden_dim
        
        self.shared_network = nn.Sequential(*shared_layers)
        
        # Separate heads for each objective
        self.objective_heads = nn.ModuleList([
            nn.Linear(prev_dim, 1) for _ in range(num_objectives)
        ])
    
    def forward(self, obs, preference):
        """
        Forward pass
        
        Args:
            obs: State observation
            preference: Preference vector
        
        Returns:
            Vector of values for each objective
        """
        x = torch.cat([obs, preference], dim=-1)
        shared_features = self.shared_network(x)
        
        # Get value for each objective
        values = torch.cat([head(shared_features) for head in self.objective_heads], dim=-1)
        
        return values

class GMORLAgent:
    """
    Generalizable Multi-Objective Reinforcement Learning Agent
    
    Based on: "Generalizable Pareto-Optimal Offloading with 
    Reinforcement Learning in Mobile Edge Computing"
    """
    
    def __init__(self, env, config: Dict):
        """
        Initialize GMORL agent
        
        Args:
            env: Task offloading environment
            config: Configuration dictionary
        """
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Hyperparameters
        self.learning_rate = config.get('learning_rate', 3e-4)
        self.gamma = config.get('gamma', 0.99)
        self.num_objectives = 2  # delay and energy
        
        # Networks
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        hidden_dims = config.get('hidden_dims', [256, 256])
        
        self.policy = PreferenceConditionedPolicyNetwork(
            obs_dim, action_dim, hidden_dims
        ).to(self.device)
        
        self.value = MultiObjectiveValueNetwork(
            obs_dim, self.num_objectives, hidden_dims
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            list(self.policy.parameters()) + list(self.value.parameters()),
            lr=self.learning_rate
        )
        
        # Storage for Pareto front
        self.pareto_solutions = []
        
    def sample_preference(self) -> np.ndarray:
        """
        Sample a random preference vector
        
        Returns:
            Preference vector [w_delay, w_energy] where sum = 1
        """
        w_delay = np.random.uniform(0, 1)
        w_energy = 1.0 - w_delay
        return np.array([w_delay, w_energy])
    
    def select_action(self, state: np.ndarray, preference: np.ndarray,
                     deterministic: bool = False):
        """
        Select action based on current policy and preference
        
        Args:
            state: Current state
            preference: User preference vector
            deterministic: Whether to select deterministically
            
        Returns:
            action, log_prob, values
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        preference_tensor = torch.FloatTensor(preference).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.policy(state_tensor, preference_tensor)
            action_probs = torch.softmax(logits, dim=-1)
            values = self.value(state_tensor, preference_tensor)
        
        if deterministic:
            action = torch.argmax(action_probs, dim=-1)
        else:
            dist = Categorical(action_probs)
            action = dist.sample()
        
        log_prob = torch.log(action_probs.squeeze(0)[action] + 1e-8)
        
        return action.item(), log_prob.item(), values.squeeze(0).cpu().numpy()
    
    def train_step(self, batch: Dict):
        """
        Perform one training step
        
        Args:
            batch: Dictionary containing training batch
        """
        states = torch.FloatTensor(batch['states']).to(self.device)
        actions = torch.LongTensor(batch['actions']).to(self.device)
        preferences = torch.FloatTensor(batch['preferences']).to(self.device)
        mo_returns = torch.FloatTensor(batch['mo_returns']).to(self.device)
        
        # Forward pass
        logits = self.policy(states, preferences)
        action_probs = torch.softmax(logits, dim=-1)
        dist = Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        
        values = self.value(states, preferences)
        
        # Calculate advantages for each objective
        advantages = mo_returns - values
        
        # Weighted advantage based on preference
        weighted_advantages = (advantages * preferences).sum(dim=-1)
        
        # Policy loss
        policy_loss = -(log_probs * weighted_advantages.detach()).mean()
        
        # Value loss (separate for each objective)
        value_loss = ((mo_returns - values) ** 2).mean()
        
        # Total loss
        loss = policy_loss + 0.5 * value_loss
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.policy.parameters()) + list(self.value.parameters()),
            max_norm=0.5
        )
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'total_loss': loss.item()
        }
    
    def train(self, num_episodes: int):
        """
        Train the agent across diverse preferences
        
        Args:
            num_episodes: Number of training episodes
        """
        for episode in range(num_episodes):
            # Sample random preference for this episode
            preference = self.sample_preference()
            
            state = self.env.reset(preference_vector=preference)
            episode_data = {
                'states': [],
                'actions': [],
                'preferences': [],
                'mo_rewards': [],  # Multi-objective rewards
                'dones': []
            }
            
            done = False
            while not done:
                action, _, _ = self.select_action(state, preference)
                next_state, reward, done, info = self.env.step(action)
                
                # Store multi-objective reward
                mo_reward = np.array([info['delay'], info['energy']])
                
                episode_data['states'].append(state)
                episode_data['actions'].append(action)
                episode_data['preferences'].append(preference)
                episode_data['mo_rewards'].append(mo_reward)
                episode_data['dones'].append(done)
                
                state = next_state
            
            # Calculate multi-objective returns
            mo_returns = self._calculate_mo_returns(
                episode_data['mo_rewards'],
                episode_data['dones']
            )
            
            # Create training batch
            batch = {
                'states': np.array(episode_data['states']),
                'actions': np.array(episode_data['actions']),
                'preferences': np.array(episode_data['preferences']),
                'mo_returns': mo_returns
            }
            
            # Train
            metrics = self.train_step(batch)
            
            if episode % 10 == 0:
                print(f"Episode {episode}, Preference: {preference}, "
                      f"Loss: {metrics['total_loss']:.4f}")
    
    def _calculate_mo_returns(self, mo_rewards: List, dones: List) -> np.ndarray:
        """
        Calculate multi-objective returns
        
        Args:
            mo_rewards: List of multi-objective rewards
            dones: List of done flags
            
        Returns:
            Multi-objective returns
        """
        mo_returns = []
        mo_return = np.zeros(self.num_objectives)
        
        for reward, done in zip(reversed(mo_rewards), reversed(dones)):
            mo_return = reward + self.gamma * mo_return * (1 - done)
            mo_returns.insert(0, mo_return)
        
        return np.array(mo_returns)
    
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


