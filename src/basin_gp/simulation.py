import torch
import numpy as np
import matplotlib.pyplot as plt

def create_basin_grid(basin_size, resolution):
    """
    Create a grid of points covering the basin.
    
    Args:
        basin_size: Size of the basin in (x, y) kilometers
        resolution: Number of points along each dimension
        
    Returns:
        grid: Grid points [n_points, 2]
        x1_grid: x-coordinates meshgrid
        x2_grid: y-coordinates meshgrid
    """
    # Create grid of points
    x1 = np.linspace(0, basin_size[0], resolution)
    x2 = np.linspace(0, basin_size[1], resolution)
    x1_grid, x2_grid = np.meshgrid(x1, x2)
    grid = np.column_stack([x1_grid.flatten(), x2_grid.flatten()])
    grid_tensor = torch.tensor(grid, dtype=torch.float32)
    
    return grid_tensor, x1_grid, x2_grid

def true_porosity(x, basin_size=(20, 20)):
    """True porosity function with multiple sweet spots."""
    x_tensor = torch.tensor(x, dtype=torch.float32) if not isinstance(x, torch.Tensor) else x
    
    # Major sweet spot
    spot1 = 0.25 * torch.exp(-0.1 * ((x_tensor[:, 0] - 5)**2 + (x_tensor[:, 1] - 15)**2))
    
    # Secondary sweet spot
    spot2 = 0.2 * torch.exp(-0.15 * ((x_tensor[:, 0] - 15)**2 + (x_tensor[:, 1] - 8)**2))
    
    # Background trend (increasing toward the north-east)
    trend = 0.05 + 0.1 * (x_tensor[:, 0] / basin_size[0] + x_tensor[:, 1] / basin_size[1]) / 2
    
    return spot1 + spot2 + trend

def true_permeability(x, basin_size=(20, 20)):
    """Permeability linked to porosity but with its own spatial patterns."""
    x_tensor = torch.tensor(x, dtype=torch.float32) if not isinstance(x, torch.Tensor) else x
    
    # Get porosity (main driver)
    porosity = true_porosity(x_tensor, basin_size)
    
    # Additional spatial variation
    fault_effect = 500 * torch.exp(-0.2 * (x_tensor[:, 0] - 10)**2 / 4)
    
    # Convert from porosity (typical log-linear relationship)
    perm_base = 10**(porosity * 15 - 1)  # Typical transform
    
    return perm_base + fault_effect

def true_thickness(x, basin_size=(20, 20)):
    """Reservoir thickness with structural trends."""
    x_tensor = torch.tensor(x, dtype=torch.float32) if not isinstance(x, torch.Tensor) else x
    
    # Main structural high
    structure = 40 * torch.exp(-0.05 * ((x_tensor[:, 0] - 10)**2 + (x_tensor[:, 1] - 10)**2))
    
    # Basin deepening trend toward the south
    trend = 30 * (1 - x_tensor[:, 1] / basin_size[1])
    
    return structure + trend + 20  # Add baseline thickness

def visualize_true_geology(basin_size=(20, 20), resolution=30):
    """
    Visualize the true geological properties.
    
    Args:
        basin_size: Size of the basin in (x, y) kilometers
        resolution: Number of points along each dimension
    """
    grid_tensor, x1_grid, x2_grid = create_basin_grid(basin_size, resolution)
    
    # Calculate true values for visualization
    true_porosity_grid = true_porosity(grid_tensor, basin_size).reshape(resolution, resolution).numpy()
    true_permeability_grid = true_permeability(grid_tensor, basin_size).reshape(resolution, resolution).numpy()
    true_thickness_grid = true_thickness(grid_tensor, basin_size).reshape(resolution, resolution).numpy()
    
    # Visualize true geology
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    im0 = axes[0].contourf(x1_grid, x2_grid, true_porosity_grid, levels=20, cmap='viridis')
    axes[0].set_title('True Porosity')
    axes[0].set_xlabel('X (km)')
    axes[0].set_ylabel('Y (km)')
    plt.colorbar(im0, ax=axes[0])
    
    im1 = axes[1].contourf(x1_grid, x2_grid, true_permeability_grid, levels=20, cmap='plasma')
    axes[1].set_title('True Permeability (mD)')
    axes[1].set_xlabel('X (km)')
    axes[1].set_ylabel('Y (km)')
    plt.colorbar(im1, ax=axes[1])
    
    im2 = axes[2].contourf(x1_grid, x2_grid, true_thickness_grid, levels=20, cmap='cividis')
    axes[2].set_title('True Thickness (m)')
    axes[2].set_xlabel('X (km)')
    axes[2].set_ylabel('Y (km)')
    plt.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    plt.show()
    
    return grid_tensor, x1_grid, x2_grid, true_porosity_grid, true_permeability_grid, true_thickness_grid

def add_knowledge_driven_wells(basin_gp, n_wells, basin_size, 
                              prior_functions, uncertainty_weight=0.5,
                              seed=42):
    """
    Add initial wells based on prior knowledge and exploration strategy.
    
    Args:
        basin_gp: BasinExplorationGP model
        n_wells: Number of wells to add
        basin_size: Basin size in (x, y) km
        prior_functions: List of prior belief functions for each property
        uncertainty_weight: Weight for balancing high-value vs. high-uncertainty areas
                           (0 = pure value, 1 = pure uncertainty)
        seed: Random seed for sampling
    """
    np.random.seed(seed)
    
    # Create a grid to evaluate our prior beliefs
    resolution = 30
    x1 = np.linspace(0, basin_size[0], resolution)
    x2 = np.linspace(0, basin_size[1], resolution)
    x1_grid, x2_grid = np.meshgrid(x1, x2)
    grid = np.column_stack([x1_grid.flatten(), x2_grid.flatten()])
    grid_tensor = torch.tensor(grid, dtype=torch.float32)
    
    # Evaluate our prior beliefs on the grid
    prior_values = []
    for func in prior_functions:
        values = func(grid_tensor).reshape(resolution, resolution).numpy()
        prior_values.append(values)
    
    # We'll place wells strategically:
    # 1. One well in the location with highest expected value according to our prior
    # 2. One well to test a different area with high uncertainty
    # 3. One well in an area that balances value and uncertainty
    
    locations = []
    
    # Well 1: Highest expected value (use porosity * thickness as proxy for value)
    value_proxy = prior_values[0] * prior_values[2]  # porosity * thickness
    max_idx = np.argmax(value_proxy.flatten())
    locations.append([x1_grid.flatten()[max_idx], x2_grid.flatten()[max_idx]])
    
    # Well 2: Maximize distance from first well (exploration)
    distances = np.sqrt((grid[:, 0] - locations[0][0])**2 + 
                       (grid[:, 1] - locations[0][1])**2)
    max_dist_idx = np.argmax(distances)
    locations.append([grid[max_dist_idx, 0], grid[max_dist_idx, 1]])
    
    # Remaining wells: Balance value and coverage
    for i in range(2, n_wells):
        # Calculate distances to existing wells
        min_distances = np.ones(grid.shape[0]) * float('inf')
        for loc in locations:
            d = np.sqrt((grid[:, 0] - loc[0])**2 + (grid[:, 1] - loc[1])**2)
            min_distances = np.minimum(min_distances, d)
        
        # Normalize distance and value
        norm_dist = min_distances / np.max(min_distances)
        norm_value = value_proxy.flatten() / np.max(value_proxy.flatten())
        
        # Score combines value and distance (uncertainty proxy)
        score = (1 - uncertainty_weight) * norm_value + uncertainty_weight * norm_dist
        next_idx = np.argmax(score)
        locations.append([grid[next_idx, 0], grid[next_idx, 1]])
    
    # Add the wells to our model
    for i, location in enumerate(locations):
        # Convert to tensor properly - fix the reshape error
        location_np = np.array(location).reshape(1, -1)
        location_tensor = torch.tensor(location_np, dtype=torch.float32)
        
        # Measure properties with noise (from true functions)
        porosity = true_porosity(location_tensor, basin_size).item() + np.random.normal(0, 0.01)
        permeability = true_permeability(location_tensor, basin_size).item() + np.random.normal(0, 50)
        thickness = true_thickness(location_tensor, basin_size).item() + np.random.normal(0, 2)
        
        measurements = {
            'porosity': porosity,
            'permeability': permeability,
            'thickness': thickness
        }
        
        basin_gp.add_well(location, measurements, well_name=f"Initial_Well_{i+1}")
    
    return basin_gp

def add_random_wells(basin_gp, n_wells, basin_size, seed=42):
    """
    Add random initial wells to the basin model.
    
    Args:
        basin_gp: BasinExplorationGP model
        n_wells: Number of wells to add
        basin_size: Basin size in (x, y) km
        seed: Random seed
    """
    np.random.seed(seed)
    
    for i in range(n_wells):
        location = np.random.uniform(0, basin_size, size=2)
        
        # Convert to tensor properly
        location_tensor = torch.tensor(location.reshape(1, -1), dtype=torch.float32)
        
        # Measure properties with noise
        porosity = true_porosity(location_tensor, basin_size).item() + np.random.normal(0, 0.01)
        permeability = true_permeability(location_tensor, basin_size).item() + np.random.normal(0, 50)
        thickness = true_thickness(location_tensor, basin_size).item() + np.random.normal(0, 2)
        
        measurements = {
            'porosity': porosity,
            'permeability': permeability,
            'thickness': thickness
        }
        
        basin_gp.add_well(location, measurements, well_name=f"Initial_Well_{i+1}")
        
    return basin_gp

def calculate_resources(mean_predictions, resolution, basin_size):
    """
    Calculate total recoverable resources based on predictions.
    
    Args:
        mean_predictions: Predicted means [n_points, n_properties]
        resolution: Grid resolution
        basin_size: Basin size in (x, y) km
        
    Returns:
        total_barrels: Total recoverable oil in barrels
    """
    # Extract predictions
    porosity = mean_predictions[:, 0]
    permeability = mean_predictions[:, 1] if mean_predictions.shape[1] > 1 else torch.ones_like(porosity) * 100
    thickness = mean_predictions[:, 2] if mean_predictions.shape[1] > 2 else torch.ones_like(porosity) * 50
    
    # Calculate cell area
    cell_area = (basin_size[0] / (resolution-1)) * (basin_size[1] / (resolution-1)) * 1e6  # m²
    
    # Original oil in place
    hydrocarbon_saturation = 1.0 - 0.3  # 1 - water saturation
    formation_volume_factor = 1.1
    ooip = cell_area * thickness * porosity * hydrocarbon_saturation / formation_volume_factor
    
    # Recovery factor
    recovery_factor = 0.1 + 0.2 * torch.log10(torch.clamp(permeability, min=1.0) / 100)
    recovery_factor = torch.clamp(recovery_factor, 0.05, 0.6)
    
    # Recoverable oil
    recoverable_oil = ooip * recovery_factor
    
    # Total recoverable (m³)
    total_recoverable = torch.sum(recoverable_oil).item()
    
    # Convert to barrels
    total_barrels = total_recoverable * 6.29
    
    return total_barrels

def prior_porosity(x, basin_size=(20, 20)):
    """Prior belief about porosity distribution."""
    x_tensor = torch.tensor(x, dtype=torch.float32) if not isinstance(x, torch.Tensor) else x
    
    # We believe there's a sweet spot in the northeast
    spot = 0.2 * torch.exp(-0.1 * ((x_tensor[:, 0] - 15)**2 + (x_tensor[:, 1] - 15)**2))
    
    # General trend of increasing porosity toward the east (due to deepening basin)
    trend = 0.05 + 0.1 * (x_tensor[:, 0] / basin_size[0])
    
    return spot + trend

def prior_permeability(x, basin_size=(20, 20)):
    """Prior belief about permeability, correlated with porosity but with variations."""
    x_tensor = torch.tensor(x, dtype=torch.float32) if not isinstance(x, torch.Tensor) else x
    
    # Permeability is related to our porosity belief
    porosity_belief = prior_porosity(x_tensor, basin_size)
    
    # We believe there's a fault zone enhancing permeability in the central area
    fault_effect = 300 * torch.exp(-0.2 * ((x_tensor[:, 0] - 10)**2) / 4)
    
    # Convert from porosity (typical log-linear relationship)
    perm_base = 10**(porosity_belief * 15)
    
    return perm_base + fault_effect

def prior_thickness(x, basin_size=(20, 20)):
    """Prior belief about reservoir thickness."""
    x_tensor = torch.tensor(x, dtype=torch.float32) if not isinstance(x, torch.Tensor) else x
    
    # We believe there's a structural high in the basin center
    structure = 30 * torch.exp(-0.05 * ((x_tensor[:, 0] - 10)**2 + (x_tensor[:, 1] - 10)**2))
    
    # Basin deepens to the north
    trend = 20 * (x_tensor[:, 1] / basin_size[1])
    
    return structure + trend + 15  # Add baseline thickness