# Field Exploration with Gaussian Processes

This module demonstrates the use of Gaussian Process modeling for efficient petroleum field exploration using Value of Information (VOI) strategy. It generates high-quality visualizations suitable for presentations.

## Key Features

- VOI-based exploration strategy for optimized well placement
- Customizable profitability threshold and confidence target
- Publication-quality visualizations in a consistent visual style
- Economic model visualization and analysis
- Step-by-step exploration progress tracking

## Usage

### Basic Usage

```bash
python src/field_gp_example.py
```

### With Visualizations (Recommended)

```bash
python src/field_gp_example.py --show-plots
```

### Custom Profit Threshold

```bash
python src/field_gp_example.py --show-plots --profit-threshold 50
```

### Custom Confidence Target

```bash
python src/field_gp_example.py --show-plots --confidence-target 0.95
```

### Full Control

```bash
python src/field_gp_example.py --show-plots --profit-threshold 40 --confidence-target 0.85 --length-scale 1.5
```

## Output Visualizations

When run with `--show-plots`, the script generates several visualization files in the `plots/field_exploration/` directory:

1. **field_properties.png** - Visualizes the field geological properties
2. **economic_model.png** - Illustrates the economic model and calculations
3. **confidence_progression.png** - Shows how confidence increases with well count
4. **final_exploration.png** - Final model predictions and uncertainty
5. **exploration_summary.png** - Complete summary of exploration results
6. **porosity_after_well_X.png** - Progress visualizations after each well

## VOI Strategy Explained

The Value of Information (VOI) strategy works by:

1. **START** with initial wells in diverse locations
2. **BUILD** a Gaussian Process model of the field
3. **CALCULATE** Value of Information for all potential locations:
   - VOI = Expected economic gain × Uncertainty reduction
4. **SELECT** location with highest VOI for next well
5. **DRILL** well, collect data, update the model
6. **REPEAT** until confidence in profitability exceeds target

This approach balances exploration (reducing uncertainty) with exploitation (maximizing economic value), typically resulting in more efficient exploration compared to purely uncertainty-based or economic-based strategies.

## Customizing the Economic Model

The economic model parameters can be adjusted by modifying the `economic_params` dictionary in the script:

```python
economic_params = {
    'area': 1.0e6,  # m²
    'water_saturation': 0.5,  # Use average from field data
    'formation_volume_factor': 1.1,
    'oil_price': 80,  # $ per barrel
    'drilling_cost': 8e6,  # $
    'completion_cost': 4e6,  # $
}
```

## Adding to Presentations

The generated visualizations are designed for direct inclusion in presentation slides. They use a consistent color scheme with purples and complementary colors, clear titles, and descriptive annotations.