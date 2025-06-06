"""Economic assessment of exploration and development scenarios."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt
from scipy import stats


class EconomicAssessment:
    """
    Calculates NPV and profitability metrics for exploration scenarios.
    
    This class handles economic evaluations of development scenarios,
    including drilling costs, operating costs, revenue, and NPV calculations.
    """
    
    def __init__(
        self,
        gas_price: float = 4.0,  # $/mcf
        operating_cost: float = 0.5,  # $/mcf
        drilling_cost: float = 10.0,  # $M per well
        discount_rate: float = 0.1,  # annual rate
        development_years: int = 10,
        drilling_cost_std: float = 1.0,  # $M per well (standard deviation)
        target_profit: float = 100.0,  # $M
        target_confidence: float = 0.9  # 90%
    ):
        """
        Initialize the economic assessment model.
        
        Args:
            gas_price: Gas price in dollars per thousand cubic feet ($/mcf).
            operating_cost: Operating cost in dollars per thousand cubic feet ($/mcf).
            drilling_cost: Mean drilling and completion cost in millions of dollars per well.
            discount_rate: Annual discount rate as a fraction.
            development_years: Development horizon in years.
            drilling_cost_std: Standard deviation of drilling costs in millions per well.
            target_profit: Target profit threshold in millions of dollars.
            target_confidence: Target confidence level as a fraction.
        """
        self.gas_price = gas_price
        self.operating_cost = operating_cost
        self.drilling_cost = drilling_cost
        self.drilling_cost_std = drilling_cost_std
        self.discount_rate = discount_rate
        self.development_years = development_years
        self.target_profit = target_profit
        self.target_confidence = target_confidence
    
    def calculate_drilling_costs(
        self,
        num_wells: int,
        random_variation: bool = True,
        random_state: Optional[int] = None
    ) -> np.ndarray:
        """
        Calculate drilling and completion costs.
        
        Args:
            num_wells: Number of wells to drill.
            random_variation: Whether to add random variation to drilling costs.
            random_state: Random seed for reproducibility.
            
        Returns:
            Array of drilling costs in millions of dollars.
        """
        if random_variation:
            if random_state is not None:
                np.random.seed(random_state)
            
            # Generate costs with random variation
            costs = np.random.normal(
                self.drilling_cost, 
                self.drilling_cost_std, 
                num_wells
            )
            
            # Ensure costs are positive
            costs = np.maximum(costs, self.drilling_cost * 0.5)
            
            return costs
        else:
            # Return fixed costs
            return np.ones(num_wells) * self.drilling_cost
    
    def calculate_operating_costs(
        self,
        production: np.ndarray
    ) -> np.ndarray:
        """
        Calculate operating costs based on production volumes.
        
        Args:
            production: Array of production volumes in mcf.
            
        Returns:
            Array of operating costs in millions of dollars.
        """
        # Convert from $/mcf to $M/mcf
        return production * (self.operating_cost / 1e6)
    
    def calculate_revenue(
        self,
        production: np.ndarray
    ) -> np.ndarray:
        """
        Calculate revenue based on production volumes.
        
        Args:
            production: Array of production volumes in mcf.
            
        Returns:
            Array of revenue in millions of dollars.
        """
        # Convert from $/mcf to $M/mcf
        return production * (self.gas_price / 1e6)
    
    def calculate_npv(
        self,
        revenues: np.ndarray,
        operating_costs: np.ndarray,
        drilling_costs: np.ndarray,
        time_points: np.ndarray,
        discount_rate: Optional[float] = None
    ) -> float:
        """
        Calculate Net Present Value (NPV).
        
        Args:
            revenues: Array of revenue values over time.
            operating_costs: Array of operating costs over time.
            drilling_costs: Array of upfront drilling costs.
            time_points: Array of time points in years.
            discount_rate: Annual discount rate. If None, uses the class default.
            
        Returns:
            Net Present Value in millions of dollars.
        """
        if discount_rate is None:
            discount_rate = self.discount_rate
        
        # Calculate cash flow
        cash_flow = revenues - operating_costs
        
        # Apply discount factors
        discount_factors = 1 / (1 + discount_rate) ** time_points
        discounted_cash_flow = cash_flow * discount_factors
        
        # Sum discounted cash flow and subtract drilling costs
        npv = np.sum(discounted_cash_flow) - np.sum(drilling_costs)
        
        return npv
    
    def assess_profitability_distribution(
        self,
        production_profiles: np.ndarray,
        time_points_years: np.ndarray,
        num_wells: int,
        num_simulations: int = 1000,
        random_state: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Assess the profitability distribution through Monte Carlo simulation.
        
        Args:
            production_profiles: Array of production profiles.
            time_points_years: Array of time points in years.
            num_wells: Number of development wells.
            num_simulations: Number of Monte Carlo simulations.
            random_state: Random seed for reproducibility.
            
        Returns:
            Dictionary containing the profitability assessment results.
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        # Initialize arrays for results
        npv_values = np.zeros(num_simulations)
        revenue_values = np.zeros(num_simulations)
        opex_values = np.zeros(num_simulations)
        capex_values = np.zeros(num_simulations)
        
        # Run Monte Carlo simulations
        for i in range(num_simulations):
            # Generate random drilling costs
            drilling_costs = self.calculate_drilling_costs(
                num_wells,
                random_variation=True,
                random_state=None  # Don't reset seed for each iteration
            )
            
            # Calculate total production for each time period
            # Assuming production_profiles is [n_wells, n_time_points]
            if production_profiles.ndim == 2:
                # Multiple wells, each with a production profile
                total_production = np.sum(production_profiles, axis=0)
            else:
                # Single production profile
                total_production = production_profiles
            
            # Calculate time step in years
            time_step = time_points_years[1] - time_points_years[0] if len(time_points_years) > 1 else 1.0
            
            # Convert daily rates to total production per period
            # Assuming production rates are in mcf/day
            days_per_period = time_step * 365.25
            period_production = total_production * days_per_period
            
            # Calculate operating costs and revenue
            operating_costs = self.calculate_operating_costs(period_production)
            revenues = self.calculate_revenue(period_production)
            
            # Calculate NPV
            npv = self.calculate_npv(
                revenues,
                operating_costs,
                drilling_costs,
                time_points_years
            )
            
            # Store results
            npv_values[i] = npv
            revenue_values[i] = np.sum(revenues)
            opex_values[i] = np.sum(operating_costs)
            capex_values[i] = np.sum(drilling_costs)
        
        # Calculate probability of meeting target profit
        prob_target = np.mean(npv_values >= self.target_profit)
        
        # Calculate statistics
        npv_mean = np.mean(npv_values)
        npv_std = np.std(npv_values)
        npv_percentiles = np.percentile(npv_values, [10, 50, 90])
        
        # Assess economic viability
        meets_confidence = prob_target >= self.target_confidence
        
        return {
            'npv_values': npv_values,
            'revenue_values': revenue_values,
            'opex_values': opex_values,
            'capex_values': capex_values,
            'npv_mean': npv_mean,
            'npv_std': npv_std,
            'npv_p10': npv_percentiles[0],
            'npv_p50': npv_percentiles[1],
            'npv_p90': npv_percentiles[2],
            'prob_target': prob_target,
            'meets_confidence': meets_confidence,
            'target_profit': self.target_profit,
            'target_confidence': self.target_confidence
        }
    
    def plot_npv_distribution(
        self,
        npv_values: np.ndarray,
        ax=None,
        title: str = "NPV Distribution",
        show_target: bool = True
    ):
        """
        Plot the NPV distribution.
        
        Args:
            npv_values: Array of NPV values.
            ax: Matplotlib axis to plot on. If None, creates a new figure.
            title: Title for the plot.
            show_target: Whether to show the target profit line.
            
        Returns:
            The matplotlib axis object.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))
        
        # Plot histogram
        ax.hist(npv_values, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        
        # Plot probability density function
        x = np.linspace(np.min(npv_values), np.max(npv_values), 100)
        y = stats.norm.pdf(x, np.mean(npv_values), np.std(npv_values))
        ax.plot(x, y * len(npv_values) * (x[1] - x[0]), color='red', linewidth=2)
        
        # Add vertical line for mean
        ax.axvline(np.mean(npv_values), color='blue', linestyle='--', linewidth=2, 
                 label=f'Mean: ${np.mean(npv_values):.1f}M')
        
        # Add vertical lines for P10, P50, P90
        percentiles = np.percentile(npv_values, [10, 50, 90])
        ax.axvline(percentiles[0], color='green', linestyle=':', linewidth=2,
                 label=f'P10: ${percentiles[0]:.1f}M')
        ax.axvline(percentiles[1], color='purple', linestyle=':', linewidth=2,
                 label=f'P50: ${percentiles[1]:.1f}M')
        ax.axvline(percentiles[2], color='orange', linestyle=':', linewidth=2,
                 label=f'P90: ${percentiles[2]:.1f}M')
        
        # Add target profit line if requested
        if show_target:
            ax.axvline(self.target_profit, color='red', linestyle='-', linewidth=2,
                     label=f'Target: ${self.target_profit:.1f}M')
            
            # Calculate probability of meeting target
            prob_target = np.mean(npv_values >= self.target_profit)
            ax.text(0.05, 0.95, f'P(NPV ≥ ${self.target_profit}M) = {prob_target:.1%}',
                  transform=ax.transAxes, verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add labels and title
        ax.set_xlabel('Net Present Value ($M)')
        ax.set_ylabel('Frequency')
        ax.set_title(title)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='best')
        
        return ax
    
    def plot_tornado_diagram(
        self,
        base_case_npv: float,
        sensitivity_results: Dict[str, Tuple[float, float]],
        ax=None,
        title: str = "Tornado Diagram - NPV Sensitivity",
        sort_by_impact: bool = True
    ):
        """
        Plot a tornado diagram showing sensitivity of NPV to input parameters.
        
        Args:
            base_case_npv: Base case NPV value.
            sensitivity_results: Dictionary mapping parameter names to (low_value, high_value) tuples.
            ax: Matplotlib axis to plot on. If None, creates a new figure.
            title: Title for the plot.
            sort_by_impact: Whether to sort parameters by impact magnitude.
            
        Returns:
            The matplotlib axis object.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 8))
        
        # Calculate impacts relative to base case
        impacts = {}
        for param, (low_val, high_val) in sensitivity_results.items():
            low_impact = low_val - base_case_npv
            high_impact = high_val - base_case_npv
            impacts[param] = (low_impact, high_impact)
        
        # Sort by absolute impact if requested
        if sort_by_impact:
            impact_magnitudes = {param: max(abs(low), abs(high)) 
                               for param, (low, high) in impacts.items()}
            sorted_params = sorted(impact_magnitudes.keys(), 
                                 key=lambda x: impact_magnitudes[x], reverse=True)
        else:
            sorted_params = list(impacts.keys())
        
        # Create arrays for plotting
        y_pos = np.arange(len(sorted_params))
        low_impacts = [impacts[param][0] for param in sorted_params]
        high_impacts = [impacts[param][1] for param in sorted_params]
        
        # Plot horizontal bars
        ax.barh(y_pos, high_impacts, height=0.5, align='center', 
              color='green', alpha=0.7, label='Positive Impact')
        ax.barh(y_pos, low_impacts, height=0.5, align='center', 
              color='red', alpha=0.7, label='Negative Impact')
        
        # Add vertical line for base case
        ax.axvline(0, color='black', linestyle='--', linewidth=1)
        
        # Set y-axis labels and ticks
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_params)
        
        # Add labels and title
        ax.set_xlabel('Change in NPV ($M)')
        ax.set_title(title)
        ax.grid(True, linestyle='--', alpha=0.7, axis='x')
        ax.legend(loc='best')
        
        return ax