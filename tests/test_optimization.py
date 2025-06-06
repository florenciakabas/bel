"""Unit tests for the optimization module."""

import unittest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

from bel.geological_model import GaussianProcessGeology
from bel.production_model import ProductionPredictor
from bel.economic_model import EconomicAssessment
from bel.optimization import ValueOfInformation


class TestValueOfInformation(unittest.TestCase):
    """Test cases for the ValueOfInformation class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a small geological model for testing
        grid_size = (10, 10)
        x_range = (0, 1)
        y_range = (0, 1)
        properties = ["Thickness", "Porosity"]
        length_scales = {
            "Thickness": 0.5,
            "Porosity": 0.3
        }
        property_ranges = {
            "Thickness": (50, 200),
            "Porosity": (0.02, 0.08)
        }
        random_state = 42
        
        # Create geological model
        self.geological_model = GaussianProcessGeology(
            grid_size=grid_size,
            x_range=x_range,
            y_range=y_range,
            properties=properties,
            length_scales=length_scales,
            property_ranges=property_ranges,
            kernel_type="exponential",
            random_state=random_state
        )
        
        # Create production model
        self.production_model = ProductionPredictor(
            model_type="linear",
            random_state=random_state
        )
        
        # Since we can't train the model without data, set default weights
        self.production_model.property_weights = {
            "Thickness": 100,
            "Porosity": 10000
        }
        
        # Create economic model
        self.economic_model = EconomicAssessment(
            gas_price=4.0,
            drilling_cost=10.0,
            target_profit=100.0,
            target_confidence=0.9
        )
        
        # Create optimizer
        self.optimizer = ValueOfInformation(
            geological_model=self.geological_model,
            production_model=self.production_model,
            economic_model=self.economic_model,
            exploration_cost=10.0,
            n_realizations=10,
            n_monte_carlo=20,
            random_state=random_state
        )
    
    def test_initialization(self):
        """Test that the optimizer initializes correctly."""
        self.assertEqual(self.optimizer.exploration_cost, 10.0)
        self.assertEqual(self.optimizer.n_realizations, 10)
        self.assertEqual(self.optimizer.n_monte_carlo, 20)
        self.assertEqual(self.optimizer.random_state, 42)
        
        # Check that grid is initialized
        self.assertIsNotNone(self.optimizer.grid_points)
        self.assertIsNotNone(self.optimizer.X)
        self.assertIsNotNone(self.optimizer.Y)
        self.assertEqual(self.optimizer.grid_size, (10, 10))
    
    def test_calculate_voi_surface(self):
        """Test calculation of VOI surface."""
        # Calculate VOI surface
        voi_surface = self.optimizer.calculate_voi_surface(
            development_wells_per_realization=10,
            development_years=10
        )
        
        # Check output
        self.assertEqual(voi_surface.shape, (10, 10))
        
        # VOI should be non-negative
        self.assertTrue(np.all(voi_surface >= 0))
        
        # Check that uncertainty reduction was calculated
        self.assertIsNotNone(self.optimizer.uncertainty_reduction)
        self.assertEqual(self.optimizer.uncertainty_reduction.shape, (10, 10))
    
    def test_select_next_well_location(self):
        """Test selection of next well location."""
        # First calculate VOI surface
        self.optimizer.calculate_voi_surface(
            development_wells_per_realization=10,
            development_years=10
        )
        
        # Select next well location
        location = self.optimizer.select_next_well_location()
        
        # Check output
        self.assertIsInstance(location, tuple)
        self.assertEqual(len(location), 2)
        
        # Location should be within grid bounds
        x, y = location
        self.assertGreaterEqual(x, 0)
        self.assertLessEqual(x, 1)
        self.assertGreaterEqual(y, 0)
        self.assertLessEqual(y, 1)
        
        # Location should correspond to maximum VOI
        max_idx = np.argmax(self.optimizer.voi_surface.flatten())
        i, j = max_idx // 10, max_idx % 10
        max_x, max_y = self.optimizer.X[i, j], self.optimizer.Y[i, j]
        
        self.assertAlmostEqual(x, max_x, places=6)
        self.assertAlmostEqual(y, max_y, places=6)
    
    def test_estimate_uncertainty_reduction(self):
        """Test estimation of uncertainty reduction."""
        # Test with no existing wells
        location = np.array([[0.5, 0.5]])
        reduction = self.optimizer._estimate_uncertainty_reduction(location)
        
        # For first well, should be high
        self.assertGreaterEqual(reduction, 0.5)
        
        # Add a well to the geological model
        well_location = np.array([[0.2, 0.2]])
        property_values = {
            "Thickness": np.array([150]),
            "Porosity": np.array([0.05])
        }
        self.geological_model.update_with_well_data(well_location, property_values)
        
        # Test with one existing well, location far from existing well
        location = np.array([[0.8, 0.8]])
        reduction = self.optimizer._estimate_uncertainty_reduction(location)
        
        # For well far from existing, should be high
        self.assertGreaterEqual(reduction, 0.3)
        
        # Test with one existing well, location close to existing well
        location = np.array([[0.25, 0.25]])
        reduction = self.optimizer._estimate_uncertainty_reduction(location)
        
        # For well close to existing, should be lower
        self.assertLessEqual(reduction, 0.3)
    
    def test_simulate_information_value(self):
        """Test simulation of information value."""
        # Create a "true" model
        true_model = GaussianProcessGeology(
            grid_size=(10, 10),
            x_range=(0, 1),
            y_range=(0, 1),
            properties=["Thickness", "Porosity"],
            length_scales={
                "Thickness": 0.5,
                "Porosity": 0.3
            },
            property_ranges={
                "Thickness": (50, 200),
                "Porosity": (0.02, 0.08)
            },
            kernel_type="exponential",
            random_state=43  # Different seed
        )
        
        # First calculate VOI surface
        self.optimizer.calculate_voi_surface(
            development_wells_per_realization=10,
            development_years=10
        )
        
        # Select a location
        location = (0.5, 0.5)
        
        # Simulate information value
        value_results = self.optimizer.simulate_information_value(
            location=location,
            true_model=true_model,
            development_wells_per_realization=10,
            development_years=10
        )
        
        # Check output structure
        self.assertIn('location', value_results)
        self.assertIn('true_values', value_results)
        self.assertIn('prior_npv_mean', value_results)
        self.assertIn('posterior_npv_mean', value_results)
        self.assertIn('uncertainty_reduction', value_results)
        self.assertIn('confidence_improvement', value_results)
        self.assertIn('net_value', value_results)
        
        # Check location
        self.assertEqual(value_results['location'], location)
        
        # Check that true values were sampled
        self.assertIn('Thickness', value_results['true_values'])
        self.assertIn('Porosity', value_results['true_values'])
        
        # Check that uncertainty was reduced
        self.assertGreaterEqual(value_results['uncertainty_reduction'], 0)
    
    def test_plot_voi_surface(self):
        """Test VOI surface plotting."""
        # First calculate VOI surface
        self.optimizer.calculate_voi_surface(
            development_wells_per_realization=10,
            development_years=10
        )
        
        # Create plot
        fig, ax = plt.subplots()
        ax = self.optimizer.plot_voi_surface(ax=ax)
        
        # Check that the plot was created
        self.assertIsNotNone(ax)
        
        # Check title and labels
        self.assertEqual(ax.get_title(), "Value of Information Surface")
        self.assertEqual(ax.get_xlabel(), "X Coordinate")
        self.assertEqual(ax.get_ylabel(), "Y Coordinate")
    
    def test_plot_uncertainty_reduction(self):
        """Test uncertainty reduction plotting."""
        # First calculate VOI surface
        self.optimizer.calculate_voi_surface(
            development_wells_per_realization=10,
            development_years=10
        )
        
        # Create plot
        fig, ax = plt.subplots()
        ax = self.optimizer.plot_uncertainty_reduction(ax=ax)
        
        # Check that the plot was created
        self.assertIsNotNone(ax)
        
        # Check title and labels
        self.assertEqual(ax.get_title(), "Uncertainty Reduction Surface")
        self.assertEqual(ax.get_xlabel(), "X Coordinate")
        self.assertEqual(ax.get_ylabel(), "Y Coordinate")


if __name__ == "__main__":
    unittest.main()