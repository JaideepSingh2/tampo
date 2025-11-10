import numpy as np
from typing import List, Tuple

def calculate_hypervolume(solutions: np.ndarray, reference_point: np.ndarray) -> float:
    """
    Calculate hypervolume indicator for multi-objective optimization
    
    Args:
        solutions: Array of shape (n_solutions, n_objectives)
        reference_point: Reference point for hypervolume calculation
        
    Returns:
        Hypervolume value
    """
    if len(solutions) == 0:
        return 0.0
    
    # For 2D case (delay, energy)
    if solutions.shape[1] == 2:
        # Sort by first objective
        sorted_sols = solutions[np.argsort(solutions[:, 0])]
        
        hv = 0.0
        for i in range(len(sorted_sols)):
            if i == 0:
                width = reference_point[0] - sorted_sols[i, 0]
            else:
                width = sorted_sols[i-1, 0] - sorted_sols[i, 0]
            
            height = reference_point[1] - sorted_sols[i, 1]
            
            if width > 0 and height > 0:
                hv += width * height
        
        return hv
    else:
        raise NotImplementedError("Hypervolume only implemented for 2D")

def normalize_objectives(delay: float, energy: float, 
                         delay_range: Tuple[float, float],
                         energy_range: Tuple[float, float]) -> Tuple[float, float]:
    """
    Normalize delay and energy to [0, 1] range
    
    Args:
        delay: Delay value
        energy: Energy value
        delay_range: (min_delay, max_delay)
        energy_range: (min_energy, max_energy)
        
    Returns:
        (normalized_delay, normalized_energy)
    """
    norm_delay = (delay - delay_range[0]) / (delay_range[1] - delay_range[0] + 1e-10)
    norm_energy = (energy - energy_range[0]) / (energy_range[1] - energy_range[0] + 1e-10)
    
    return np.clip(norm_delay, 0, 1), np.clip(norm_energy, 0, 1)

class MovingAverage:
    """Calculate moving average of a metric"""
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.values = []
        
    def update(self, value: float):
        """Add new value"""
        self.values.append(value)
        if len(self.values) > self.window_size:
            self.values.pop(0)
    
    def get_average(self) -> float:
        """Get current moving average"""
        if len(self.values) == 0:
            return 0.0
        return np.mean(self.values)
    
    def reset(self):
        """Reset the moving average"""
        self.values = []
