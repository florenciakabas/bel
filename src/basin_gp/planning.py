import torch
import numpy as np
from torch.distributions import Normal

def plan_next_well(model, grid, mean, std, strategy='uncertainty', economic_params=None):
    """
    Plan the next well based on selected strategy.
    
    Args:
        model: The basin exploration model
        grid: Grid of candidate locations
        mean: Mean predictions
        std: Standard deviation predictions
        strategy: Strategy name
        economic_params: Parameters for economic calculations
        
    Returns:
        best_location, best_score, score_grid
    """
    if strategy == 'uncertainty':
        return plan_next_well_uncertainty(grid, mean, std)
    elif strategy == 'ei':
        return plan_next_well_ei(model, grid, mean, std)
    elif strategy == 'economic':
        return plan_next_well_economic(grid, mean, std, economic_params)
    elif strategy == 'balanced':
        return plan_next_well_balanced(grid, mean, std, economic_params)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def plan_next_well_uncertainty(grid, mean, std):
    """
    Plan next well based on maximum uncertainty.
    
    Args:
        grid: Grid of candidate locations [n_points, 2]
        mean: Mean predictions [n_points, n_properties]
        std: Standard deviation predictions [n_points, n_properties]
        
    Returns:
        best_location, best_score, score_grid
    """
    # Total uncertainty across all properties
    total_uncertainty = torch.sum(std, dim=1)
    best_idx = torch.argmax(total_uncertainty)
    best_location = grid[best_idx]
    best_score = total_uncertainty[best_idx]
    score_grid = total_uncertainty
    
    return best_location, best_score, score_grid

def plan_next_well_ei(model, grid, mean, std):
    """
    Plan next well based on expected improvement for property 0.
    
    Args:
        model: The basin exploration model
        grid: Grid of candidate locations [n_points, 2]
        mean: Mean predictions [n_points, n_properties]
        std: Standard deviation predictions [n_points, n_properties]
        
    Returns:
        best_location, best_score, score_grid
    """
    # Expected improvement for first property (e.g., porosity)
    # Get best observed value so far
    observed_values = np.array([well['measurements'][0] for well in model.wells 
                               if well['mask'][0]])
    
    if len(observed_values) == 0:
        # If no observations yet, fall back to uncertainty
        return plan_next_well_uncertainty(grid, mean, std)
        
    best_f = torch.tensor(np.max(observed_values), dtype=torch.float32)
    
    # Calculate improvement
    improvement = mean[:, 0] - best_f
    
    # Calculate z-score
    z = improvement / std[:, 0]
    
    # Expected improvement
    norm_dist = torch.distributions.Normal(0, 1)
    ei = improvement * norm_dist.cdf(z) + std[:, 0] * torch.exp(norm_dist.log_prob(z))
    
    # Find best location
    best_idx = torch.argmax(ei)
    best_location = grid[best_idx]
    best_score = ei[best_idx]
    score_grid = ei
    
    return best_location, best_score, score_grid

def plan_next_well_economic(grid, mean, std, economic_params):
    """
    Plan next well based on expected economic value.
    
    Args:
        grid: Grid of candidate locations [n_points, 2]
        mean: Mean predictions [n_points, n_properties]
        std: Standard deviation predictions [n_points, n_properties]
        economic_params: Parameters for economic calculations
        
    Returns:
        best_location, best_score, score_grid
    """
    if economic_params is None:
        raise ValueError("Economic parameters required for 'economic' strategy")
        
    # Calculate expected monetary value at each location
    emv = calculate_economic_value(grid, mean, std, economic_params)
    
    # Find best location
    best_idx = torch.argmax(emv)
    best_location = grid[best_idx]
    best_score = emv[best_idx]
    score_grid = emv
    
    return best_location, best_score, score_grid

def calculate_economic_value(locations, mean, std, params):
    """
    Calculate expected monetary value at given locations.
    
    Args:
        locations: Locations to evaluate [n_points, 2]
        mean: Predicted means [n_points, n_properties]
        std: Predicted standard deviations [n_points, n_properties]
        params: Economic parameters
            
    Returns:
        emv: Expected monetary value [n_points]
    """
    # For simplicity, assuming:
    # Property 0: Porosity (fraction)
    # Property 1: Permeability (mD)
    # Property 2: Thickness (m)
    
    # Extract predictions
    porosity = mean[:, 0]
    permeability = mean[:, 1] if mean.shape[1] > 1 else torch.ones_like(porosity) * params.get('default_permeability', 100)
    thickness = mean[:, 2] if mean.shape[1] > 2 else torch.ones_like(porosity) * params.get('default_thickness', 50)
    
    # Extract uncertainties
    porosity_std = std[:, 0]
    permeability_std = std[:, 1] if std.shape[1] > 1 else torch.zeros_like(porosity)
    thickness_std = std[:, 2] if std.shape[1] > 2 else torch.zeros_like(porosity)
    
    # Calculate hydrocarbon volume
    area = params.get('area', 1.0e6)  # m²
    hydrocarbon_saturation = 1.0 - params.get('water_saturation', 0.3)
    formation_volume_factor = params.get('formation_volume_factor', 1.1)
    
    # Original oil in place (m³)
    ooip = area * thickness * porosity * hydrocarbon_saturation / formation_volume_factor
    
    # Recovery factor based on permeability
    recovery_factor = 0.1 + 0.2 * torch.log10(torch.clamp(permeability, min=1.0) / 100)
    recovery_factor = torch.clamp(recovery_factor, 0.05, 0.6)
    
    # Recoverable oil (m³)
    recoverable_oil = ooip * recovery_factor
    
    # Convert to barrels
    barrels = recoverable_oil * 6.29
    
    # Revenue
    oil_price = params.get('oil_price', 70)  # $ per barrel
    revenue = barrels * oil_price
    
    # Cost
    base_cost = params.get('drilling_cost', 1e7) + params.get('completion_cost', 5e6)
    
    # Expected monetary value
    emv = revenue - base_cost
    
    # Apply risk adjustment based on uncertainty
    # Higher uncertainty means higher risk, which reduces EMV
    uncertainty_factor = 1.0 - torch.clamp(
        (porosity_std / torch.clamp(porosity, min=1e-10) + 
         permeability_std / torch.clamp(permeability, min=1e-10)) / 2, 
        0, 0.5
    )
    risk_adjusted_emv = emv * uncertainty_factor
    
    return risk_adjusted_emv

def plan_next_well_balanced(grid, mean, std, economic_params, balance_factor=0.5):
    """
    Plan next well based on a balanced approach between uncertainty reduction and economic value.
    This strategy handles the exploration-exploitation trade-off directly by combining
    both objectives with an adjustable balance parameter.

    Args:
        grid: Grid of candidate locations [n_points, 2]
        mean: Mean predictions [n_points, n_properties]
        std: Standard deviation predictions [n_points, n_properties]
        economic_params: Parameters for economic calculations
        balance_factor: Factor between economic value (0) and uncertainty (1)
                Default is 0.5 for an even balance

    Returns:
        best_location, best_score, score_grid
    """
    if economic_params is None:
        raise ValueError("Economic parameters required for 'balanced' strategy")
    
    # Get uncertainty score (exploration)
    _, _, uncertainty_score = plan_next_well_uncertainty(grid, mean, std)
    
    # Get economic score (exploitation)
    _, _, economic_score = plan_next_well_economic(grid, mean, std, economic_params)
    
    # Normalize both scores to [0, 1] range to make them comparable
    norm_uncertainty = (uncertainty_score - uncertainty_score.min()) / (uncertainty_score.max() - uncertainty_score.min() + 1e-10)
    norm_economic = (economic_score - economic_score.min()) / (economic_score.max() - economic_score.min() + 1e-10)
    
    # Weighted combination of normalized scores:
    # balance_factor = 0: pure economic strategy
    # balance_factor = 1: pure uncertainty strategy
    balanced_score = (1 - balance_factor) * norm_economic + balance_factor * norm_uncertainty
    
    # Find best location
    best_idx = torch.argmax(balanced_score)
    best_location = grid[best_idx]
    best_score = balanced_score[best_idx]

    return best_location, best_score, balanced_score