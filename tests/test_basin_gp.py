import unittest
import torch
import numpy as np
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from basin_gp.model import BasinExplorationGP
from basin_gp.data import prepare_training_data
from basin_gp.planning import (
    plan_next_well_uncertainty, 
    plan_next_well_ei,
    plan_next_well_economic,
    calculate_economic_value
)
from basin_gp.simulation import (
    true_porosity, 
    true_permeability, 
    true_thickness,
    create_basin_grid
)

class TestBasinExplorationGP(unittest.TestCase):
    
    def setUp(self):
        """Set up a small test model instance before each test"""
        # Create a small basin for faster tests
        self.model = BasinExplorationGP(
            basin_size=(10, 10),  # Small basin
            properties=['porosity', 'permeability', 'thickness']
        )
        
        # Add a few test wells
        self.model.add_well([2, 3], {'porosity': 0.15, 'permeability': 100, 'thickness': 30})
        self.model.add_well([7, 8], {'porosity': 0.25, 'permeability': 200, 'thickness': 40})
        
    def test_initialization(self):
        """Test that model is initialized correctly"""
        # Check if properties are correctly set
        self.assertEqual(self.model.basin_size, (10, 10))
        self.assertEqual(self.model.properties, ['porosity', 'permeability', 'thickness'])
        self.assertEqual(self.model.n_properties, 3)
        
        # Check initial state
        self.assertIsNone(self.model.model)
        self.assertIsNone(self.model.likelihood)
        self.assertEqual(len(self.model.wells), 2)  # From setUp
        
    def test_add_well(self):
        """Test adding a well works correctly"""
        # Initial count
        initial_count = len(self.model.wells)
        
        # Add a new well
        well_data = self.model.add_well(
            [5, 5], 
            {'porosity': 0.2, 'permeability': 150}
        )
        
        # Check the well was added
        self.assertEqual(len(self.model.wells), initial_count + 1)
        
        # Check well properties
        self.assertEqual(well_data['name'], f"Well_{initial_count + 1}")
        self.assertTrue(np.array_equal(well_data['location'], np.array([5, 5])))
        
        # Check measurements
        np.testing.assert_almost_equal(well_data['measurements'][0], 0.2)  # porosity
        np.testing.assert_almost_equal(well_data['measurements'][1], 150)  # permeability
        self.assertTrue(np.isnan(well_data['measurements'][2]))  # thickness (not provided)
        
        # Check mask
        self.assertTrue(well_data['mask'][0])   # porosity provided
        self.assertTrue(well_data['mask'][1])   # permeability provided
        self.assertFalse(well_data['mask'][2])  # thickness not provided
        
    def test_prepare_training_data(self):
        """Test the data preparation logic"""
        # Get training data
        X, Y, mask = prepare_training_data(self.model.wells, self.model.n_properties)
        
        # Check shapes
        self.assertEqual(X.shape, (2, 2))         # 2 wells, 2D locations
        self.assertEqual(Y.shape, (2, 3))         # 2 wells, 3 properties
        self.assertEqual(mask.shape, (2, 3))      # Mask for 2 wells, 3 properties
        
        # Check values
        np.testing.assert_almost_equal(X[0].numpy(), np.array([2, 3]))  # First well location
        np.testing.assert_almost_equal(X[1].numpy(), np.array([7, 8]))  # Second well location
        
        # Check Y values for first well
        np.testing.assert_almost_equal(Y[0, 0].item(), 0.15)  # Porosity
        np.testing.assert_almost_equal(Y[0, 1].item(), 100)   # Permeability
        np.testing.assert_almost_equal(Y[0, 2].item(), 30)    # Thickness
        
        # Check mask values
        self.assertTrue(mask[0, 0])  # First well has porosity
        self.assertTrue(mask[0, 1])  # First well has permeability
        self.assertTrue(mask[0, 2])  # First well has thickness
        
    def test_fit_model(self):
        """Test that model fitting works"""
        # Before fitting, model should be None
        self.assertIsNone(self.model.model)
        
        # Fit the model
        loss = self.model.fit(iterations=10, verbose=False)
        
        # After fitting, model should exist
        self.assertIsNotNone(self.model.model)
        self.assertIsNotNone(self.model.likelihood)
        
        # Loss should be a number
        self.assertTrue(isinstance(loss, float))
        
    def test_predict(self):
        """Test prediction capability"""
        # Fit model first
        self.model.fit(iterations=10, verbose=False)
        
        # Create test grid
        grid = torch.tensor([[1.0, 1.0], [5.0, 5.0], [9.0, 9.0]], dtype=torch.float32)
        
        # Make predictions
        mean, std = self.model.predict(grid)
        
        # Check shapes
        self.assertEqual(mean.shape, (3, 3))  # 3 locations, 3 properties
        self.assertEqual(std.shape, (3, 3))   # 3 locations, 3 properties
        
        # Standard deviations should be positive
        self.assertTrue(torch.all(std > 0))
        
    def test_plan_next_well_uncertainty(self):
        """Test uncertainty-based well planning"""
        # Create test grid and predictions
        grid = torch.tensor([[1.0, 1.0], [5.0, 5.0], [9.0, 9.0]], dtype=torch.float32)
        mean = torch.tensor([[0.1, 100, 20], [0.2, 150, 30], [0.15, 120, 25]], dtype=torch.float32)
        std = torch.tensor([[0.05, 50, 5], [0.1, 80, 10], [0.02, 30, 2]], dtype=torch.float32)
        
        # Location 2 has highest total uncertainty, so should be selected
        location, score, score_grid = plan_next_well_uncertainty(grid, mean, std)
        
        # Check that it selected the highest uncertainty location
        self.assertTrue(torch.allclose(location, grid[1]))  # Should be [5.0, 5.0]
        
    def test_economic_calculation(self):
        """Test economic value calculation"""
        # Create test data
        grid = torch.tensor([[1.0, 1.0], [5.0, 5.0]], dtype=torch.float32)
        mean = torch.tensor([[0.1, 100, 20], [0.2, 200, 30]], dtype=torch.float32) 
        std = torch.tensor([[0.01, 10, 2], [0.02, 20, 3]], dtype=torch.float32)
        
        economic_params = {
            'area': 1.0e6,  # m²
            'water_saturation': 0.3,
            'formation_volume_factor': 1.1,
            'oil_price': 80,
            'drilling_cost': 1e7,
            'completion_cost': 5e6
        }
        
        # Calculate EMV
        emv = calculate_economic_value(grid, mean, std, economic_params)
        
        # Check shape
        self.assertEqual(emv.shape, (2,))
        
        # Location 2 should have higher EMV due to better properties
        self.assertGreater(emv[1].item(), emv[0].item())
        
    def test_true_functions(self):
        """Test the simulation functions"""
        # Create test locations
        locations = torch.tensor([[5.0, 15.0], [15.0, 8.0], [0.0, 0.0]], dtype=torch.float32)
        
        # Test porosity function
        porosity = true_porosity(locations)
        self.assertEqual(porosity.shape, (3,))
        self.assertTrue(torch.all(porosity >= 0) and torch.all(porosity <= 1))
        
        # Location 1 should have high porosity due to sweet spot
        self.assertGreater(porosity[0].item(), porosity[2].item())
        
        # Test permeability function
        permeability = true_permeability(locations)
        self.assertEqual(permeability.shape, (3,))
        self.assertTrue(torch.all(permeability > 0))
        
        # Test thickness function
        thickness = true_thickness(locations)
        self.assertEqual(thickness.shape, (3,))
        self.assertTrue(torch.all(thickness > 0))
        
    def test_sequential_exploration(self):
        """Test sequential exploration"""
        # Create a small grid
        grid_tensor, _, _ = create_basin_grid((10, 10), 10)
        
        # Run sequential exploration for a single well
        history = self.model.sequential_exploration(
            grid_tensor,
            n_wells=1,
            true_functions=[true_porosity, true_permeability, true_thickness],
            noise_std=0.01,
            strategy='uncertainty',
            plot=False
        )
        
        # Check that history was recorded
        self.assertEqual(len(history), 1)
        
        # Check that the well was added
        self.assertEqual(len(self.model.wells), 3)  # 2 initial + 1 new
        
        # Check history format
        step = history[0]
        self.assertIn('well_location', step)
        self.assertIn('measurements', step)
        self.assertIn('score', step)
        self.assertEqual(step['strategy'], 'uncertainty')
        
if __name__ == '__main__':
    unittest.main()