import torch
import numpy as np
from torch.distributions import Normal

def plan_next_well(model, grid, mean, std, strategy='uncertainty', economic_params=None, confidence_target=0.9):
    """
    Plan the next well based on selected strategy.

    Args:
        model: The basin exploration model
        grid: Grid of candidate locations
        mean: Mean predictions
        std: Standard deviation predictions
        strategy: Strategy name
        economic_params: Parameters for economic calculations
        confidence_target: Target confidence level for profitability (used in 'voi' strategy)

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
    elif strategy == 'voi':
        return plan_next_well_voi(model, grid, mean, std, economic_params, confidence_target)
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
    norm_dist = torch.distributions.Normal(torch.tensor(0.0), torch.tensor(1.0))
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

def plan_next_well_voi(model, grid, mean, std, economic_params, confidence_target=0.9):
    """
    Plan next well based on Value of Information (VOI) approach.
    This strategy directly targets maximizing the probability of reaching
    confidence in profitability by selecting locations that will most
    reduce uncertainty in the overall economic value.

    Args:
        model: The basin exploration model
        grid: Grid of candidate locations [n_points, 2]
        mean: Mean predictions [n_points, n_properties]
        std: Standard deviation predictions [n_points, n_properties]
        economic_params: Parameters for economic calculations
        confidence_target: Target confidence level for profitability (0-1)

    Returns:
        best_location, best_score, score_grid
    """
    if economic_params is None:
        raise ValueError("Economic parameters required for 'voi' strategy")

    # Calculate current expected economic value
    emv = calculate_economic_value(grid, mean, std, economic_params)

    # Calculate current variance of economic value
    # This requires propagating uncertainty through the economic model
    emv_variance = calculate_economic_variance(grid, mean, std, economic_params)

    # Get current profitability threshold
    drilling_cost = economic_params.get('drilling_cost', 1e7)
    completion_cost = economic_params.get('completion_cost', 5e6)
    total_cost = drilling_cost + completion_cost

    # Convert to tensor to avoid type mismatch
    total_cost_tensor = torch.tensor(total_cost, dtype=torch.float32)

    # Calculate current probability of profitability at each location
    # Assuming normal distribution of economic value
    prob_profitable = 1 - torch.distributions.Normal(emv, torch.sqrt(emv_variance)).cdf(total_cost_tensor)

    # We want to know how much a new well would reduce uncertainty in profitability
    # Calculate the expected reduction in variance if we drill at each location
    potential_variance_reduction = torch.zeros_like(emv)

    # This is a simplified approach - in a full implementation, we would simulate
    # the potential new well observations and update the model to get exact values
    # Here we use a heuristic that approximates this effect

    # Higher uncertainty locations would lead to more variance reduction
    uncertainty_weight = torch.sum(std, dim=1) / torch.sum(std)

    # Higher economic potential locations have more impact on overall profitability
    economic_weight = torch.clamp((emv - total_cost_tensor) / total_cost_tensor, min=0.0)

    # Distance from current confidence to target confidence
    # Find locations where we're close to but not yet at target confidence
    confidence_target_tensor = torch.tensor(confidence_target, dtype=torch.float32)
    confidence_gap = torch.abs(prob_profitable - confidence_target_tensor)
    # Invert the gap so smaller gaps get higher scores (better potential to reach target)
    confidence_potential = 1.0 / (confidence_gap + 0.1)

    # Locations where we're uncertain AND have high potential economic value
    # AND are close to reaching confidence threshold are most valuable for reducing uncertainty
    potential_variance_reduction = (
        uncertainty_weight *
        economic_weight *
        confidence_potential
    )

    # The value of information is the expected improvement in decision quality
    # due to decreased uncertainty - this is our score
    voi_score = potential_variance_reduction

    # Find best location
    best_idx = torch.argmax(voi_score)
    best_location = grid[best_idx]
    best_score = voi_score[best_idx]

    return best_location, best_score, voi_score

def calculate_economic_variance(grid, mean, std, params):
    """
    Calculate variance of expected monetary value at given locations.
    This estimates uncertainty in the economic outcomes by propagating
    geological property uncertainties.

    Args:
        grid: Grid of candidate locations [n_points, 2]
        mean: Mean predictions [n_points, n_properties]
        std: Standard deviation predictions [n_points, n_properties]
        params: Economic parameters

    Returns:
        emv_variance: Variance of expected monetary value [n_points]
    """
    # Extract predictions
    porosity = mean[:, 0]
    porosity_std = std[:, 0]

    permeability = mean[:, 1] if mean.shape[1] > 1 else torch.ones_like(porosity) * params.get('default_permeability', 100)
    permeability_std = std[:, 1] if std.shape[1] > 1 else torch.zeros_like(porosity)

    thickness = mean[:, 2] if mean.shape[1] > 2 else torch.ones_like(porosity) * params.get('default_thickness', 50)
    thickness_std = std[:, 2] if std.shape[1] > 2 else torch.zeros_like(porosity)

    # Economic parameters
    area = params.get('area', 1.0e6)  # m²
    hydrocarbon_saturation = 1.0 - params.get('water_saturation', 0.3)
    formation_volume_factor = params.get('formation_volume_factor', 1.1)
    oil_price = params.get('oil_price', 70)  # $ per barrel

    # First-order variance approximation formula:
    # Var(f(X)) ≈ (∂f/∂x)² * Var(X) + (∂f/∂y)² * Var(Y) + ...

    # Partial derivatives of the economic value with respect to:
    # 1. Porosity
    d_porosity = (area * thickness * hydrocarbon_saturation / formation_volume_factor *
                  6.29 * oil_price)  # Convert to barrels and then to $

    # 2. Thickness
    d_thickness = (area * porosity * hydrocarbon_saturation / formation_volume_factor *
                   6.29 * oil_price)

    # 3. Permeability (affects recovery factor)
    # Simplified recovery factor model: RF = 0.1 + 0.2 * log10(perm/100)
    # d_RF/d_perm = 0.2 / (perm * ln(10))
    perm_factor = 0.2 / (torch.clamp(permeability, min=1.0) * torch.log(torch.tensor(10.0)))
    d_permeability = (area * porosity * thickness * hydrocarbon_saturation / formation_volume_factor *
                     perm_factor * 6.29 * oil_price)

    # Estimate overall variance using error propagation formula
    emv_variance = (
        (d_porosity * porosity_std)**2 +
        (d_thickness * thickness_std)**2 +
        (d_permeability * permeability_std)**2
    )

    return emv_variance