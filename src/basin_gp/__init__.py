from .model import BasinExplorationGP
from .data import prepare_training_data
from .planning import (
    plan_next_well_uncertainty,
    plan_next_well_ei,
    plan_next_well_economic,
    plan_next_well_balanced,
    plan_next_well_voi,
    calculate_economic_value,
    calculate_economic_variance
)
from .simulation import (
    true_porosity,
    true_permeability,
    true_thickness,
    visualize_true_geology,
    analyze_length_scale_sensitivity,
    add_knowledge_driven_wells,
    add_random_wells
)
from .field_data import (
    load_field_data,
    field_thickness,
    field_porosity,
    field_permeability,
    field_toc,
    field_water_saturation,
    field_clay_volume,
    field_depth,
    visualize_field_geology,
    add_field_wells_from_samples
)

# Import visualization functions for length-scale analysis
try:
    # When imported as a package
    from ..plot_length_scale import (
        plot_length_scale_impact,
        visualize_geological_smoothness,
        visualize_length_scale_comparison,
        plot_length_scale_impact_3d
    )
except ImportError:
    # When running scripts directly
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from plot_length_scale import (
        plot_length_scale_impact,
        visualize_geological_smoothness,
        visualize_length_scale_comparison,
        plot_length_scale_impact_3d
    )

__all__ = [
    'BasinExplorationGP',
    'prepare_training_data',
    'plan_next_well_uncertainty',
    'plan_next_well_ei',
    'plan_next_well_economic',
    'plan_next_well_balanced',
    'plan_next_well_voi',
    'calculate_economic_value',
    'calculate_economic_variance',
    # Simulation functions
    'true_porosity',
    'true_permeability',
    'true_thickness',
    'visualize_true_geology',
    'analyze_length_scale_sensitivity',
    'add_knowledge_driven_wells',
    'add_random_wells',
    # Field data functions
    'load_field_data',
    'field_thickness',
    'field_porosity',
    'field_permeability',
    'field_toc',
    'field_water_saturation',
    'field_clay_volume',
    'field_depth',
    'visualize_field_geology',
    'add_field_wells_from_samples',
    # Length-scale visualization functions
    'plot_length_scale_impact',
    'visualize_geological_smoothness',
    'visualize_length_scale_comparison',
    'plot_length_scale_impact_3d',
]