import unittest
import numpy as np
import sys
import os
# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from bel import (
    BayesianEvidentialLearning, 
    simulate_well_measurements,
    create_realistic_kernel
)

class TestBayesianEvidentialLearning(unittest.TestCase):
    
    def setUp(self):
        """Set up a small test BEL instance before each test"""
        # Create a small basin for faster tests
        self.bel = BayesianEvidentialLearning(
            basin_size=(1000, 1000),  # Very small basin
            grid_resolution=250,       # Coarse grid
            param_names=['porosity', 'thickness']  # Only two parameters for simplicity
        )
    
    def test_initialization(self):
        """Test that BEL initializes correctly"""
        # Check if the grid is created with correct dimensions
        expected_grid_shape = (5, 5)  # For a 1000x1000 basin with 250m resolution
        self.assertEqual(self.bel.X_grid.shape, expected_grid_shape)
        self.assertEqual(self.bel.Y_grid.shape, expected_grid_shape)
        
        # Check if parameters are correctly set
        self.assertEqual(len(self.bel.param_names), 2)
        self.assertIn('porosity', self.bel.param_names)
        self.assertIn('thickness', self.bel.param_names)
        
        # Check if prior means exist for each parameter
        for param in self.bel.param_names:
            self.assertIn(param, self.bel.prior_means)
    
    def test_drill_well(self):
        """Test that drilling a well works and updates the model"""
        # Initial state
        initial_well_count = len(self.bel.well_locations)
        
        # Drill a well in the center of the basin
        location = (500, 500)
        measurements = self.bel.drill_well(location, with_logs=False)
        
        # Check that the well location was recorded
        self.assertEqual(len(self.bel.well_locations), initial_well_count + 1)
        self.assertEqual(self.bel.well_locations[-1], location)
        
        # Check that measurements were recorded
        self.assertEqual(len(self.bel.well_measurements), initial_well_count + 1)
        
        # Check that we got measurements for each parameter
        for param in self.bel.param_names:
            self.assertIn(param, measurements)
            self.assertTrue(isinstance(measurements[param], (int, float)))
    
    def test_well_measurements(self):
        """Test that well measurements have realistic values with appropriate noise"""
        # Create a set of true values
        true_values = {
            'porosity': 0.15,
            'thickness': 20
        }
        
        # Generate measurements multiple times to check statistical properties
        n_trials = 100
        porosity_measurements = []
        thickness_measurements = []
        
        for _ in range(n_trials):
            measurements = simulate_well_measurements(true_values)
            porosity_measurements.append(measurements['porosity'])
            thickness_measurements.append(measurements['thickness'])
        
        # Check that measurements are within reasonable bounds
        mean_porosity = np.mean(porosity_measurements)
        std_porosity = np.std(porosity_measurements)
        
        # Mean should be close to true value (within 3 standard errors)
        std_error = std_porosity / np.sqrt(n_trials)
        self.assertTrue(abs(mean_porosity - true_values['porosity']) < 3 * std_error)
        
        # Check that porosity measurements are constrained between 0 and 1
        self.assertTrue(all(0 <= p <= 1 for p in porosity_measurements))
        
        # Similar checks for thickness
        mean_thickness = np.mean(thickness_measurements)
        std_thickness = np.std(thickness_measurements)
        std_error = std_thickness / np.sqrt(n_trials)
        self.assertTrue(abs(mean_thickness - true_values['thickness']) < 3 * std_error)
        
        # Thickness should be positive
        self.assertTrue(all(t > 0 for t in thickness_measurements))
    
    def test_calculate_npv(self):
        """Test that NPV calculation produces sensible results"""
        # Test case 1: Good reservoir properties should give positive NPV
        good_params = {
            'porosity': 0.25,
            'permeability': 100,
            'thickness': 30,
            'saturation': 0.8,
            'depth': 2000
        }
        good_npv = self.bel.calculate_npv(good_params)
        self.assertGreater(good_npv, 0)
        
        # Test case 2: Poor reservoir properties should give negative NPV
        poor_params = {
            'porosity': 0.05,
            'permeability': 1,
            'thickness': 5,
            'saturation': 0.2,
            'depth': 4000
        }
        poor_npv = self.bel.calculate_npv(poor_params)
        self.assertLess(poor_npv, 0)
        
        # Test case 3: Better properties should give higher NPV
        better_params = {
            'porosity': 0.3,
            'permeability': 200,
            'thickness': 40,
            'saturation': 0.9,
            'depth': 1800
        }
        better_npv = self.bel.calculate_npv(better_params)
        self.assertGreater(better_npv, good_npv)


class TestKernelCreation(unittest.TestCase):
    
    def test_kernel_creation(self):
        """Test that appropriate kernels are created for different parameters"""
        # Test depth kernel
        depth_kernel = create_realistic_kernel('depth', 3000)
        
        # Test porosity kernel
        porosity_kernel = create_realistic_kernel('porosity', 1500)
        
        # Test permeability kernel
        perm_kernel = create_realistic_kernel('permeability', 800)
        
        # Check that kernels are different for different parameters
        # This is a simple test that just checks if the string representations differ
        self.assertNotEqual(str(depth_kernel), str(porosity_kernel))
        self.assertNotEqual(str(porosity_kernel), str(perm_kernel))
        self.assertNotEqual(str(depth_kernel), str(perm_kernel))


if __name__ == '__main__':
    unittest.main()