# BEL: Basin Exploration Learning

## Oil & Gas Exploration Optimization: Adaptive Well Placement Strategy

BEL (Basin Exploration Learning) is a framework for optimizing oil and gas exploration strategies using machine learning, Gaussian process modeling, and value of information theory. It helps answer the critical question: **"Will this project enable profit of M millions of dollars with X% certainty?"** with the fewest exploration wells possible.

## Overview

This system uses an adaptive exploration approach to iteratively place exploration wells in locations that maximize learning about subsurface geology and reduce uncertainty in economic assessments. By using Bayesian optimization techniques, the system efficiently converges on a confident decision about whether to proceed with field development.

## Core Workflow

The system follows an adaptive exploration loop:

1. **Initialize** geological understanding using prior distributions modeled as Gaussian processes
2. **Predict** production from geological properties using machine learning models
3. **Estimate** oil/gas recovery for strategic well placement over N-year development horizon
4. **Assess** profitability using economic model to calculate expected NPV with uncertainty
5. **Optimize** next exploration well location using value of information calculations
6. **Simulate** drilling and learning real geological properties at that location
7. **Update** beliefs by updating the Gaussian process model with new data
8. **Iterate** until maximum exploration wells reached OR satisfactory confidence achieved

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/bel.git
cd bel

# Install dependencies
pip install -e .
```

Or install from PyPI:

```bash
pip install bel
```

## Requirements

- Python 3.8+
- numpy>=1.21.0
- scipy>=1.7.0
- matplotlib>=3.4.0
- pandas>=1.3.0
- scikit-learn>=1.0.0
- GPy>=1.10.0
- plotly>=5.0.0
- seaborn>=0.11.0

## Usage

### Basic Example

```python
from bel.geological_model import GaussianProcessGeology
from bel.production_model import ProductionPredictor
from bel.economic_model import EconomicAssessment
from bel.simulation_controller import ExplorationSimulation

# Initialize models
geological_model = GaussianProcessGeology(
    grid_size=(50, 50),
    x_range=(-94.6, -92.9),
    y_range=(31.3, 33.0),
    properties=["Thickness", "Porosity", "Permeability", "TOC", "SW", "Depth", "Vclay"],
    length_scales={"Thickness": 0.5, "Porosity": 0.4, ...},
    property_ranges={"Thickness": (50, 200), "Porosity": (0.02, 0.08), ...}
)

production_model = ProductionPredictor(model_type="linear")
economic_model = EconomicAssessment(
    gas_price=4.0,
    drilling_cost=10.0,
    target_profit=100.0,
    target_confidence=0.9
)

# Run simulation
simulation = ExplorationSimulation(
    geological_model=geological_model,
    production_model=production_model,
    economic_model=economic_model
)

results = simulation.run_exploration_campaign()
```

See the `examples` directory for more detailed examples.

### Running Examples

```bash
# Run basic simulation
python examples/basic_simulation.py --output_dir results --max_wells 10 

# Run parameter sensitivity analysis
python examples/parameter_sensitivity.py --output_dir sensitivity_results --n_repeats 3
```

## Project Structure

```
project_root/
├── src/                  # Source code
│   └── bel/
│       ├── geological_model/     # Geological property modeling
│       ├── production_model/     # Production prediction
│       ├── economic_model/       # Economic assessment
│       ├── optimization/         # Value of information optimization
│       ├── simulation_controller/ # Exploration simulation
│       ├── data_manager/         # Data I/O and validation
│       ├── visualization/        # Results visualization
│       └── utils/                # Utility functions
├── examples/             # Example scripts
│   ├── basic_simulation.py
│   └── parameter_sensitivity.py
├── tests/                # Unit tests
├── data/                 # Data storage
├── requirements.txt      # Dependencies
├── setup.py              # Package setup
└── README.md             # This file
```

## Key Features

- **Gaussian Process Geological Modeling**: Models spatial distributions of properties like porosity, permeability, and thickness
- **Production Forecasting**: Predicts production rates and cumulative production from geological properties
- **Economic Assessment**: Calculates NPV distributions and confidence levels for project profitability
- **Value of Information**: Optimizes exploration well placement to maximize information gain
- **Uncertainty Quantification**: Tracks how uncertainty decreases with each exploration well
- **Visualization Tools**: Creates maps, charts, and dashboards to understand exploration progress

## Key Components

### Geological Model

The `GaussianProcessGeology` class handles spatial modeling of geological properties:

```python
from bel.geological_model import GaussianProcessGeology

# Initialize model
model = GaussianProcessGeology(
    grid_size=(50, 50),
    x_range=(-94.6, -92.9),
    y_range=(31.3, 33.0),
    properties=["Thickness", "Porosity"],
    length_scales={"Thickness": 0.5, "Porosity": 0.4},
    property_ranges={"Thickness": (50, 200), "Porosity": (0.02, 0.08)}
)

# Update with well data
model.update_with_well_data(
    well_locations=np.array([[x1, y1], [x2, y2]]), 
    property_values={"Thickness": [150, 175], "Porosity": [0.05, 0.06]}
)

# Sample realizations
realizations = model.sample_realizations(n_samples=10)

# Calculate uncertainty
uncertainty = model.calculate_uncertainty()
```

### Production Model

The `ProductionPredictor` class predicts production from geological properties:

```python
from bel.production_model import ProductionPredictor

# Initialize and train model
model = ProductionPredictor(model_type="linear")
model.train(geological_properties=df, production_data=df)

# Predict production
initial_rates = model.predict_initial_rate(properties)
b_factors, decline_rates = model.predict_decline_parameters(properties)
production = model.forecast_production_profile(
    initial_rates, b_factors, decline_rates, time_points
)
```

### Economic Model

The `EconomicAssessment` class handles economic calculations:

```python
from bel.economic_model import EconomicAssessment

# Initialize model
model = EconomicAssessment(
    gas_price=4.0,  # $/mcf
    operating_cost=0.5,  # $/mcf
    drilling_cost=10.0,  # $M per well
    target_profit=100.0,  # $M
    target_confidence=0.9  # 90%
)

# Assess profitability
results = model.assess_profitability_distribution(
    production_profiles, time_points, num_wells=100
)

# Check if target met
target_met = results["meets_confidence"]
```

### Simulation Controller

The `ExplorationSimulation` class orchestrates the exploration process:

```python
from bel.simulation_controller import ExplorationSimulation

# Initialize simulation
simulation = ExplorationSimulation(
    geological_model=geological_model,
    production_model=production_model,
    economic_model=economic_model,
    true_model=true_model  # For simulation
)

# Run exploration campaign
results = simulation.run_exploration_campaign()
```

## Mathematical Models

- **Gaussian Process Covariance (Exponential)**:
  ```
  k(x_i, x_j) = σ² * exp(-|x_i - x_j| / l)
  ```
  Where σ² is variance and l is correlation length.

- **Production Decline Model (Arps)**:
  ```
  q(t) = q_i / (1 + b * D_i * t)^(1/b)
  ```
  Where q_i, b, and D_i are functions of geological properties.

- **Net Present Value**:
  ```
  NPV = Σ(t=0 to T) [(Revenue_t - OpCost_t) / (1 + r)^t] - CapCost
  ```

- **Value of Information**:
  ```
  VOI = E[NPV | new_data] - E[NPV | current_data] - drilling_cost
  ```

## Example Results

The framework produces various outputs including:

1. **Geological Property Maps**: Visualizations of property distributions and uncertainty
2. **Economic Distributions**: NPV distributions and confidence metrics
3. **Value of Information Surfaces**: Maps showing optimal drilling locations
4. **Uncertainty Evolution**: Tracking of uncertainty reduction through exploration
5. **Summary Dashboards**: Comprehensive exploration campaign summaries

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

This project builds upon techniques from Bayesian optimization, geostatistics, petroleum engineering, and decision analysis under uncertainty.

## Citation

If you use this framework in your research, please cite:

```
@software{bel2023,
  author = {Your Name},
  title = {BEL: Basin Exploration Learning},
  year = {2023},
  url = {https://github.com/yourusername/bel}
}
```