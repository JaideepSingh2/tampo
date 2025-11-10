import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple
from torch.distributions import Categorical
from collections import deque

class AttentionLayer(nn.Module):
    """Multi-head attention for processing variable-length server information"""
    
    def __init__(self, d_model: int, num_heads: int = 8):
        super(AttentionLayer, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, x, mask=None):
        batch_size = x.size(0)
        
        # Linear projections
        Q = self.W_q(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = torch.softmax(scores, dim=-1)
        context = torch.matmul(attention, V)
        
        # Concatenate heads and project
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(context)
        
        return output

class MetaPolicyNetwork(nn.Module):
    """
    Meta-policy network with preference conditioning and attention
    for TAM-PO framework
    """
    
    def __init__(self, obs_dim: int, action_dim: int, 
                 hidden_dims: List[int] = [256, 256],
                 num_attention_heads: int = 8):
        super(MetaPolicyNetwork, self).__init__()
        
        self.preference_dim = 2  # [w_delay, w_energy]
        
        # Server encoder with attention
        self.server_encoder = nn.Sequential(
            nn.Linear(67, 128),  # Server feature dimension from observation
            nn.ReLU(),
            nn.LayerNorm(128)
        )
        
        self.attention = AttentionLayer(128, num_attention_heads)
        
        # Preference encoder
        self.preference_encoder = nn.Sequential(
            nn.Linear(self.preference_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64)
        )
        
        # Combined processing
        input_dim = obs_dim + 64  # Base obs + encoded preference
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))
            prev_dim = hidden_dim
        
        self.shared_net = nn.Sequential(*layers)
        self.action_head = nn.Linear(prev_dim, action_dim)
        
    def forward(self, obs, preference, server_mask=None):
        """
        Forward pass with preference conditioning and attention
        
        Args:
            obs: State observation including server information
            preference: [w_delay, w_energy] preference vector
            server_mask: Mask for variable number of servers
        
        Returns:
            Action logits
        """
        # Extract server features (assuming they're part of obs)
        # This depends on your observation structure
        
        # Encode preference
        pref_encoded = self.preference_encoder(preference)
        
        # Combine observation and preference
        combined = torch.cat([obs, pref_encoded], dim=-1)
        
        # Process through shared network
        features = self.shared_net(combined)
        
        # Generate action logits
        logits = self.action_head(features)
        
        # Apply mask if provided
        if server_mask is not None:
            logits = logits.masked_fill(server_mask == 0, -1e9)
        
        return logits

class HypervolumeCalculator:
    """Calculate hypervolume indicator for multi-objective optimization"""
    
    def __init__(self, reference_point: np.ndarray):
        self.reference_point = reference_point
    
    def calculate(self, solutions: np.ndarray) -> float:
        """
        Calculate hypervolume for 2D case (delay, energy)
        
        Args:
            solutions: Array of shape (n_solutions, 2)
        
        Returns:
            Hypervolume value
        """
        if len(solutions) == 0:
            return 0.0
        
        # Sort by first objective (delay)
        sorted_sols = solutions[np.argsort(solutions[:, 0])]
        
        hv = 0.0
        for i in range(len(sorted_sols)):
            if i == 0:
                width = self.reference_point[0] - sorted_sols[i, 0]
            else:
                width = sorted_sols[i-1, 0] - sorted_sols[i, 0]
            
            height = self.reference_point[1] - sorted_sols[i, 1]
            
            if width > 0 and height > 0:
                hv += width * height
        
        return hv

class TAMPOAgent:
    """
    Threshold-Adaptive Meta-Reinforcement Learning for 
    Pareto-Optimal Offloading (TAM-PO)
    
    Combines:
    - Meta-learning for fast adaptation
    - Multi-objective optimization with preference vectors
    - Threshold-based communication mechanism
    """
    
    def __init__(self, env, config: Dict):
        """
        Initialize TAM-PO agent
        
        Args:
            env: Task offloading environment
            config: Configuration dictionary
        """
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Hyperparameters
        self.meta_learning_rate = config.get('meta_learning_rate', 1e-3)
        self.inner_learning_rate = config.get('inner_learning_rate', 1e-2)
        self.inner_steps = config.get('inner_steps', 10)
        self.gamma = config.get('gamma', 0.99)
        
        # Threshold-based communication
        self.hypervolume_threshold = config.get('hypervolume_threshold', 0.7)
        self.moving_avg_window = config.get('moving_average_window', 50)
        
        # Networks
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        hidden_dims = config.get('hidden_dims', [256, 256])
        num_heads = config.get('num_attention_heads', 8)
        
        self.meta_policy = MetaPolicyNetwork(
            obs_dim, action_dim, hidden_dims, num_heads
        ).to(self.device)
        
        # Meta-optimizer
        self.meta_optimizer = optim.Adam(
            self.meta_policy.parameters(),
            lr=self.meta_learning_rate
        )
        
        # Hypervolume tracking
        self.reference_point = np.array([10.0, 1.0])  # Max delay, max energy
        self.hv_calculator = HypervolumeCalculator(self.reference_point)
        self.hv_history = deque(maxlen=self.moving_avg_window)
        
        # Performance history for Pareto front
        self.performance_buffer = []
        
    def sample_preference(self) -> np.ndarray:
        """Sample random preference vector"""
        w_delay = np.random.uniform(0, 1)
        w_energy = 1.0 - w_delay
        return np.array([w_delay, w_energy])
    
    def inner_loop_adaptation(self, task_id: int, num_steps: int = None):
        """
        Perform inner loop adaptation for a specific task
        
        Args:
            task_id: Task identifier
            num_steps: Number of gradient steps (uses default if None)
        
        Returns:
            Adapted policy parameters
        """
        if num_steps is None:
            num_steps = self.inner_steps
        
        # Clone meta-policy for local adaptation
        local_policy = type(self.meta_policy)(
            self.env.observation_space.shape[0],
            self.env.action_space.n,
            hidden_dims=[256, 256]
        ).to(self.device)
        
        local_policy.load_state_dict(self.meta_policy.state_dict())
        
        # Create optimizer for local policy
        local_optimizer = optim.SGD(
            local_policy.parameters(),
            lr=self.inner_learning_rate
        )
        
        # Set task
        self.env.set_task(task_id)
        
        # Collect local experience and adapt
        for step in range(num_steps):
            state = self.env.reset()
            preference = self.sample_preference()
            
            done = False
            episode_data = []
            
            while not done:
                # Select action
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                pref_tensor = torch.FloatTensor(preference).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    logits = local_policy(state_tensor, pref_tensor)
                    action_probs = torch.softmax(logits, dim=-1)
                    dist = Categorical(action_probs)
                    action = dist.sample()
                
                # Take step
                next_state, reward, done, info = self.env.step(action.item())
                
                episode_data.append({
                    'state': state,
                    'action': action.item(),
                    'reward': reward,
                    'preference': preference,
                    'delay': info.get('delay', 0),
                    'energy': info.get('energy', 0)
                })
                
                state = next_state
            
            # Compute loss and update local policy
            if len(episode_data) > 0:
                loss = self._compute_policy_loss(episode_data, local_policy)
                
                local_optimizer.zero_grad()
                loss.backward()
                local_optimizer.step()
        
        return local_policy.state_dict()
    
    def _compute_policy_loss(self, episode_data: List[Dict], policy):
        """Compute policy gradient loss"""
        states = torch.FloatTensor([d['state'] for d in episode_data]).to(self.device)
        actions = torch.LongTensor([d['action'] for d in episode_data]).to(self.device)
        preferences = torch.FloatTensor([d['preference'] for d in episode_data]).to(self.device)
        rewards = torch.FloatTensor([d['reward'] for d in episode_data]).to(self.device)
        
        # Compute returns
        returns = []
        G = 0
        for reward in reversed(rewards.tolist()):
            G = reward + self.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Compute policy loss
        logits = policy(states, preferences)
        action_probs = torch.softmax(logits, dim=-1)
        dist = Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        
        loss = -(log_probs * returns).mean()
        
        return loss
    
    def meta_train(self, num_iterations: int, meta_batch_size: int):
        """
        Meta-training loop
        
        Args:
            num_iterations: Number of meta-training iterations
            meta_batch_size: Number of tasks per meta-batch
        """
        for iteration in range(num_iterations):
            # Sample meta-batch of tasks
            task_batch = self.env.sample_tasks(meta_batch_size)
            
            meta_loss = 0
            
            for task_id in task_batch:
                # Inner loop: adapt to task
                adapted_params = self.inner_loop_adaptation(task_id)
                
                # Create adapted policy
                adapted_policy = type(self.meta_policy)(
                    self.env.observation_space.shape[0],
                    self.env.action_space.n
                ).to(self.device)
                adapted_policy.load_state_dict(adapted_params)
                
                # Collect data with adapted policy
                self.env.set_task(task_id)
                state = self.env.reset()
                preference = self.sample_preference()
                
                done = False
                test_data = []
                
                while not done:
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    pref_tensor = torch.FloatTensor(preference).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        logits = adapted_policy(state_tensor, pref_tensor)
                        action_probs = torch.softmax(logits, dim=-1)
                        dist = Categorical(action_probs)
                        action = dist.sample()
                    
                    next_state, reward, done, info = self.env.step(action.item())
                    
                    test_data.append({
                        'state': state,
                        'action': action.item(),
                        'reward': reward,
                        'preference': preference
                    })
                    
                    state = next_state
                
                # Compute meta-loss
                task_loss = self._compute_policy_loss(test_data, adapted_policy)
                meta_loss += task_loss
            
            # Meta-update
            meta_loss = meta_loss / meta_batch_size
            
            self.meta_optimizer.zero_grad()
            meta_loss.backward()
            self.meta_optimizer.step()
            
            if iteration % 10 == 0:
                print(f"Meta-iteration {iteration}, Meta-loss: {meta_loss.item():.4f}")
    
    def check_threshold_communication(self) -> bool:
        """
        Check if threshold condition is met for communication
        
        Returns:
            True if communication should be triggered
        """
        if len(self.hv_history) < self.moving_avg_window:
            return False
        
        moving_avg = np.mean(self.hv_history)
        
        return moving_avg < self.hypervolume_threshold
    
    def update_hypervolume(self, delay: float, energy: float):
        """Update hypervolume history"""
        self.performance_buffer.append([delay, energy])
        
        # Keep only recent solutions
        if len(self.performance_buffer) > 100:
            self.performance_buffer = self.performance_buffer[-100:]
        
        # Calculate hypervolume
        solutions = np.array(self.performance_buffer)
        hv = self.hv_calculator.calculate(solutions)
        self.hv_history.append(hv)
    
    def save(self, path: str):
        """Save meta-policy"""
        torch.save({
            'meta_policy': self.meta_policy.state_dict(),
            'meta_optimizer': self.meta_optimizer.state_dict(),
            'hv_history': list(self.hv_history)
        }, path)
    
    def load(self, path: str):
        """Load meta-policy"""
        checkpoint = torch.load(path)
        self.meta_policy.load_state_dict(checkpoint['meta_policy'])
        self.meta_optimizer.load_state_dict(checkpoint['meta_optimizer'])
        self.hv_history = deque(checkpoint['hv_history'], maxlen=self.moving_avg_window)
