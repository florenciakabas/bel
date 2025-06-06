"""
Enhanced Variability Configuration for BEL Simulations

This configuration file contains parameter suggestions to create more realistic
variability and interesting knowledge progression stories in exploration drilling.
"""

# Geological Model Parameters
GEOLOGICAL_PARAMS = {
    # Increased spatial correlation lengths for more uncertainty between wells
    "length_scales": {
        "Thickness": 3.5,      # was ~1.0
        "Porosity": 4.0,       # was ~1.0
        "Permeability": 3.0,   # was ~1.0
        "TOC": 4.5,           # was ~1.0
        "SW": 3.5,            # was ~1.0
        "Depth": 5.0,         # was ~1.0
        "Vclay": 3.0          # was ~1.0
    },
    
    # Wider property ranges for more variability
    "property_ranges": {
        "Thickness": (30, 250),        # was (50, 200)
        "Porosity": (0.01, 0.12),      # was (0.02, 0.08)
        "Permeability": (0.01, 1.0),   # was (0.1, 0.5)
        "TOC": (1, 10),                # was (2, 8)
        "SW": (0.2, 0.8),              # was (0.3, 0.7)
        "Depth": (-12000, -7000),      # was (-11000, -8000)
        "Vclay": (0.2, 0.7)            # was (0.3, 0.6)
    },
    
    # Measurement noise (represents data quality issues)
    "measurement_noise": 0.05,  # was 1e-6
    
    # Kernel type options: "exponential", "squared_exponential", "matern"
    "kernel_type": "matern"  # Matern gives more realistic spatial variability
}

# Production Model Parameters
PRODUCTION_PARAMS = {
    # Wider range of decline parameters
    "b_factor_range": (0.3, 1.5),      # was fixed at 0.8-0.85
    "initial_decline_range": (0.005, 0.3),  # was fixed at 0.01-0.1
    
    # Production uncertainty multiplier
    "production_noise_std": 0.2,  # 20% standard deviation
    
    # Probability of extreme outcomes
    "dry_hole_probability": 0.15,      # 15% chance of minimal production
    "super_well_probability": 0.05,    # 5% chance of exceptional well
    "super_well_multiplier": 3.0,      # Super wells produce 3x expected
    
    # Completion quality factor (represents operational variability)
    "completion_quality_range": (0.6, 1.3),  # Random multiplier on production
    
    # Model complexity
    "model_type": "random_forest",  # More complex than linear
    "rf_params": {
        "n_estimators": 200,  # was 100
        "max_depth": 15,      # was 10
        "min_samples_split": 5
    }
}

# Economic Model Parameters
ECONOMIC_PARAMS = {
    # Base economic parameters
    "gas_price": 4.0,
    "gas_price_volatility": 0.3,      # 30% price volatility
    "operating_cost": 0.5,
    "operating_cost_variability": 0.2,  # 20% cost variability
    
    # Drilling cost uncertainty
    "drilling_cost": 10.0,
    "drilling_cost_std": 3.0,          # was 1.0
    "cost_overrun_probability": 0.1,   # 10% chance of major cost overrun
    "cost_overrun_multiplier": 2.0,    # Overruns double the cost
    
    # Economic targets
    "discount_rate": 0.1,
    "development_years": 10,
    "target_profit": 100.0,
    "target_confidence": 0.9
}

# Value of Information Parameters
VOI_PARAMS = {
    # Reduced maximum uncertainty reduction for more gradual learning
    "max_uncertainty_reduction": 0.5,   # was 0.8
    
    # Distance-based learning decay
    "learning_decay_rate": 0.3,        # Faster decay = less learning from distant wells
    
    # Information surprise factor
    "surprise_probability": 0.2,       # 20% chance well reveals unexpected geology
    "surprise_impact": 1.5,           # Surprises change beliefs by 50% more
    
    # Exploration costs
    "exploration_cost": 10.0,
    "n_realizations": 100,            # was 50
    "n_monte_carlo": 200              # was 100
}

# Simulation Control Parameters
SIMULATION_PARAMS = {
    # Tighter stopping criteria for more exploration
    "max_exploration_wells": 25,       # was 20
    "target_confidence": 0.9,
    "target_profit": 100.0,
    "confidence_threshold": 0.02,      # was 0.05
    "uncertainty_threshold": 0.03,     # was 0.05
    
    # Early stopping prevention
    "min_wells_before_stopping": 5,    # Don't stop before 5 wells
    
    # Adaptive exploration
    "adaptive_voi_threshold": 0.5,     # Switch strategies when VOI drops below this
}

# Knowledge Progression Stages
KNOWLEDGE_STAGES = {
    "exploration_phase": {
        "wells": (1, 3),
        "uncertainty_multiplier": 1.5,   # Higher initial uncertainty
        "surprise_rate": 0.3,           # 30% surprise rate
        "learning_efficiency": 0.6       # Only 60% of expected learning
    },
    "rapid_learning_phase": {
        "wells": (4, 8),
        "uncertainty_multiplier": 1.0,
        "surprise_rate": 0.2,
        "learning_efficiency": 1.2       # 120% learning (exciting discoveries)
    },
    "refinement_phase": {
        "wells": (9, 15),
        "uncertainty_multiplier": 0.8,
        "surprise_rate": 0.1,
        "learning_efficiency": 0.9
    },
    "diminishing_returns_phase": {
        "wells": (16, 25),
        "uncertainty_multiplier": 0.7,
        "surprise_rate": 0.05,
        "learning_efficiency": 0.5       # Only 50% of expected learning
    }
}

# Enhanced Dummy Data Generation Parameters
DUMMYDATA_PARAMS = {
    # Grid parameters
    "grid_size": (200, 200),
    "n_wells": 500,
    
    # Add multiple geological features
    "n_geological_features": 3,        # Combine 3 different patterns
    "feature_scales": [2.0, 1.0, 0.5], # Different scales for each feature
    "feature_weights": [0.5, 0.3, 0.2], # How much each contributes
    
    # Add discontinuities (faults, boundaries)
    "n_discontinuities": 2,
    "discontinuity_impact": 0.3,       # 30% property change across boundary
    
    # Random noise level
    "noise_level": 0.15,              # 15% random variation
    
    # Production correlation adjustments
    "property_weights": {
        "Thickness": 150,     # was 200
        "Porosity": 15000,    # was 10000  
        "Permeability": 2000, # was 1000
        "TOC": 100,          # was 50
        "SW": -200,          # was -100
        "Depth": -0.02,      # was -0.01
        "Vclay": -800        # was -500
    },
    
    # Add non-linear effects
    "include_interactions": True,      # Include property interactions
    "interaction_terms": [
        ("Porosity", "Permeability", 5000),  # Phi*Perm interaction
        ("Thickness", "TOC", 20),            # Thickness*TOC interaction
    ]
}

def get_enhanced_config():
    """Return complete enhanced configuration."""
    return {
        "geological": GEOLOGICAL_PARAMS,
        "production": PRODUCTION_PARAMS,
        "economic": ECONOMIC_PARAMS,
        "voi": VOI_PARAMS,
        "simulation": SIMULATION_PARAMS,
        "knowledge_stages": KNOWLEDGE_STAGES,
        "dummydata": DUMMYDATA_PARAMS
    }