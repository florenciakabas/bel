# Oil & Gas Exploration Optimization: Adaptive Well Placement Strategy

## Project Overview

Develop an optimal sampling strategy that minimizes the number of exploration wells required to answer the question: **"Will this project enable profit of M millions of dollars with X% certainty?"**

This system uses machine learning, Gaussian process modeling, and value of information theory to iteratively place exploration wells in locations that maximize learning about subsurface geology and reduce uncertainty in economic assessments.

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

## Technical Specifications

### System Parameters

**Geological Properties (7 key parameters):**
- Thickness: 50-200 ft
- Porosity: 0.02-0.08 (fraction)
- Permeability: 0.1-0.5 mD
- Total Organic Carbon (TOC): 2-8%
- Water Saturation: 0.3-0.7 (fraction)
- Depth: 8,000-11,000 ft
- Clay Volume: 0.3-0.6 (fraction)

**Economic Parameters:**
- Drilling + Completion Cost: $10M per well (range: $8-12M)
- Gas Price: $4.00/mcf (fixed initially)
- Operating Cost: $0.50/mcf
- Discount Rate: 10% annually
- Development Horizon: 10 years
- Target Confidence Level: 90%

**Optimization Parameters:**
- Basin Grid Size: 200x200 spatial points
- Maximum Exploration Wells: 20
- Development Wells per Scenario: 100-500
- Spatial Correlation Lengths: 2-5 km (varies by property)

### Software Architecture

The system is designed with modular, loosely-coupled components following single responsibility principles:

#### Core Modules

**1. Geological Model (`geological_model.py`)**
```python
class GaussianProcessGeology:
    """Manages spatial geological property distributions and uncertainty"""
    - initialize_prior_distributions()
    - update_with_well_data()
    - sample_realizations()
    - calculate_uncertainty()
```

**2. Production Model (`production_model.py`)**
```python
class ProductionPredictor:
    """Predicts well production from geological properties"""
    - predict_initial_rate()
    - predict_decline_parameters()
    - forecast_production_profile()
    - calculate_cumulative_production()
```

**3. Economic Model (`economic_model.py`)**
```python
class EconomicAssessment:
    """Calculates NPV and profitability metrics"""
    - calculate_drilling_costs()
    - calculate_operating_costs()
    - calculate_revenue()
    - calculate_npv()
    - assess_profitability_distribution()
```

**4. Optimization Engine (`optimization.py`)**
```python
class ValueOfInformation:
    """Determines optimal exploration well locations"""
    - calculate_voi_surface()
    - select_next_well_location()
    - estimate_uncertainty_reduction()
```

**5. Simulation Controller (`simulation_controller.py`)**
```python
class ExplorationSimulation:
    """Orchestrates the main exploration loop"""
    - run_exploration_campaign()
    - simulate_well_drilling()
    - update_geological_model()
    - check_stopping_criteria()
```

#### Supporting Modules

**6. Data Management (`data_manager.py`)**
```python
class DataManager:
    """Handles data I/O and validation"""
    - load_geological_data()
    - save_simulation_results()
    - validate_input_parameters()
```

**7. Visualization (`visualization.py`)**
```python
class ResultsVisualizer:
    """Creates plots and maps for analysis"""
    - plot_geological_maps()
    - plot_uncertainty_evolution()
    - plot_economic_distributions()
    - create_well_location_maps()
```

**8. Utilities (`utils.py`)**
```python
class MathUtils:
    """Mathematical and statistical utility functions"""
    - spatial_correlation_functions()
    - monte_carlo_sampling()
    - statistical_measures()
```

### Data Flow Architecture

**Input Data:**
- Basin boundary coordinates
- Prior geological parameter distributions
- Economic parameters and constraints
- Optimization settings

**Intermediate Data:**
- Gaussian process realizations
- Production forecasts
- NPV distributions
- VOI maps

**Output Data:**
- Optimal exploration well locations
- Updated geological uncertainty maps
- Profitability assessment with confidence intervals
- Simulation history and metrics

### Implementation Phases

**Phase 1: Core Infrastructure**
- Implement basic Gaussian process geological modeling
- Create simple production prediction model
- Build economic assessment framework
- Establish data management and testing infrastructure

**Phase 2: Optimization Engine**
- Implement value of information calculations
- Create exploration well placement optimization
- Build simulation loop controller
- Add visualization capabilities

**Phase 3: Integration & Validation**
- Integrate all modules into complete workflow
- Implement comprehensive testing suite
- Create example scenarios and validation cases
- Add logging and monitoring capabilities

### Software Engineering Requirements

**Code Quality Standards:**
- All functions must have comprehensive docstrings with parameter descriptions and return values
- Unit tests for all core functions with >80% coverage
- Type hints throughout the codebase
- Consistent naming conventions (snake_case for functions/variables, PascalCase for classes)
- Maximum function length: 50 lines
- Maximum cyclomatic complexity: 10

**Testing Strategy:**
- Unit tests for individual functions
- Integration tests for module interactions
- End-to-end tests for complete workflows
- Property-based testing for mathematical functions
- Mock external dependencies

**Documentation Requirements:**
- README with installation and usage instructions
- API documentation for all public methods
- Example notebooks demonstrating key workflows
- Mathematical formulation documentation

**Performance Considerations:**
- Vectorized operations using NumPy where possible
- Efficient spatial data structures for large grids
- Memory-conscious handling of Monte Carlo realizations
- Parallel processing for independent calculations

### Development Environment

**Required Dependencies:**
```
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.4.0
pandas>=1.3.0
scikit-learn>=1.0.0
GPy>=1.10.0  # For Gaussian processes
plotly>=5.0.0  # For interactive visualization
pytest>=6.0.0  # For testing
```

**File Structure:**
```
project_root/
├── src/
│   ├── geological_model.py
│   ├── production_model.py
│   ├── economic_model.py
│   ├── optimization.py
│   ├── simulation_controller.py
│   ├── data_manager.py
│   ├── visualization.py
│   └── utils.py
├── tests/
│   ├── test_geological_model.py
│   ├── test_production_model.py
│   ├── test_economic_model.py
│   └── test_integration.py
├── examples/
│   ├── basic_simulation.py
│   └── parameter_sensitivity.py
├── data/
│   └── dummy_basin_data.csv
├── requirements.txt
├── setup.py
└── README.md
```

### Key Mathematical Models

**Gaussian Process Covariance (Exponential):**
```
k(x_i, x_j) = σ² * exp(-|x_i - x_j| / l)
```
Where σ² is variance and l is correlation length.

**Production Decline Model (Arps):**
```
q(t) = q_i / (1 + b * D_i * t)^(1/b)
```
Where q_i, b, and D_i are functions of geological properties.

**Net Present Value:**
```
NPV = Σ(t=0 to T) [(Revenue_t - OpCost_t) / (1 + r)^t] - CapCost
```

**Value of Information:**
```
VOI = E[NPV | new_data] - E[NPV | current_data] - drilling_cost
```

### Success Criteria

The system successfully answers the profitability question when:
- Confidence interval on NPV estimate is within ±10% of mean
- Probability of achieving target profit threshold exceeds specified confidence level
- No significant uncertainty reduction expected from additional exploration wells

### Next Steps for Implementation

1. Start with `geological_model.py` - implement basic Gaussian process functionality
2. Create synthetic data generation utilities in `data_manager.py`
3. Build simple production model based on existing decline curve approach
4. Implement basic economic calculations
5. Create unit tests for each module as developed
6. Build integration framework in `simulation_controller.py`
7. Add visualization capabilities
8. Create comprehensive example scenarios