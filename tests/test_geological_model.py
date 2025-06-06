"""Unit tests for the geological model module."""

import unittest
import numpy as np
from bel.geological_model import GaussianProcessGeology


class TestGaussianProcessGeology(unittest.TestCase):
    """Test cases for the GaussianProcessGeology class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a small geological model for testing
        self.grid_size = (10, 10)
        self.x_range = (0, 1)
        self.y_range = (0, 1)
        self.properties = ["Thickness", "Porosity"]
        self.length_scales = {
            "Thickness": 0.5,
            "Porosity": 0.3
        }
        self.property_ranges = {
            "Thickness": (50, 200),
            "Porosity": (0.02, 0.08)
        }
        self.random_state = 42
        
        self.model = GaussianProcessGeology(
            grid_size=self.grid_size,
            x_range=self.x_range,
            y_range=self.y_range,
            properties=self.properties,
            length_scales=self.length_scales,
            property_ranges=self.property_ranges,
            kernel_type="exponential",
            random_state=self.random_state
        )
    
    def test_initialization(self):
        """Test that the model initializes correctly."""
        # Check grid dimensions
        self.assertEqual(self.model.grid_size, self.grid_size)
        self.assertEqual(self.model.X.shape, (self.grid_size[1], self.grid_size[0]))
        self.assertEqual(self.model.Y.shape, (self.grid_size[1], self.grid_size[0]))
        
        # Check properties
        self.assertEqual(self.model.properties, self.properties)
        self.assertEqual(self.model.length_scales, self.length_scales)
        self.assertEqual(self.model.property_ranges, self.property_ranges)
        
        # Check distributions initialized
        for prop in self.properties:
            self.assertIn(prop, self.model.property_distributions)
            self.assertIn("mean", self.model.property_distributions[prop])
            self.assertIn("cov", self.model.property_distributions[prop])
            self.assertIn("L", self.model.property_distributions[prop])
            
            # Check mean dimension
            self.assertEqual(
                len(self.model.property_distributions[prop]["mean"]),
                self.grid_size[0] * self.grid_size[1]
            )
    
    def test_sample_realizations(self):
        """Test sampling realizations from the model."""
        n_samples = 5
        realizations = self.model.sample_realizations(n_samples)
        
        # Check structure
        self.assertEqual(len(realizations), len(self.properties))
        for prop in self.properties:
            self.assertIn(prop, realizations)
            self.assertEqual(realizations[prop].shape, (n_samples, self.grid_size[1], self.grid_size[0]))
            
            # Check value ranges
            min_val, max_val = self.property_ranges[prop]
            self.assertGreaterEqual(np.min(realizations[prop]), min_val)
            self.assertLessEqual(np.max(realizations[prop]), max_val)
    
    def test_calculate_uncertainty(self):
        """Test uncertainty calculation."""
        uncertainty = self.model.calculate_uncertainty()
        
        # Check structure
        self.assertEqual(len(uncertainty), len(self.properties))
        for prop in self.properties:
            self.assertIn(prop, uncertainty)
            self.assertEqual(uncertainty[prop].shape, (self.grid_size[1], self.grid_size[0]))
            
            # Uncertainty should be positive
            self.assertGreater(np.min(uncertainty[prop]), 0)
    
    def test_get_property_mean(self):
        """Test retrieving property mean maps."""
        for prop in self.properties:
            mean_map = self.model.get_property_mean(prop)
            
            # Check shape
            self.assertEqual(mean_map.shape, (self.grid_size[1], self.grid_size[0]))
            
            # Check value ranges
            min_val, max_val = self.property_ranges[prop]
            self.assertGreaterEqual(np.min(mean_map), min_val)
            self.assertLessEqual(np.max(mean_map), max_val)
    
    def test_update_with_well_data(self):
        """Test updating the model with well data."""
        # Initial state
        initial_uncertainty = self.model.calculate_uncertainty()
        
        # Create some well data
        well_locations = np.array([[0.5, 0.5], [0.2, 0.8]])
        property_values = {
            "Thickness": np.array([150, 120]),
            "Porosity": np.array([0.05, 0.03])
        }
        
        # Update model
        self.model.update_with_well_data(well_locations, property_values)
        
        # Check that well data was stored
        for prop in self.properties:
            self.assertEqual(len(self.model.well_data[prop]), 2)
            for i, (loc, val) in enumerate(self.model.well_data[prop]):
                np.testing.assert_array_equal(loc, well_locations[i])
                self.assertEqual(val, property_values[prop][i])
        
        # Check that uncertainty decreased
        new_uncertainty = self.model.calculate_uncertainty()
        for prop in self.properties:
            total_old_uncertainty = np.sum(initial_uncertainty[prop])
            total_new_uncertainty = np.sum(new_uncertainty[prop])
            self.assertLess(total_new_uncertainty, total_old_uncertainty)
            
            # Check that uncertainty is lowest near well locations
            for well_loc in well_locations:
                # Find grid indices closest to well location
                x_idx = int((well_loc[0] - self.x_range[0]) / (self.x_range[1] - self.x_range[0]) * (self.grid_size[0] - 1))
                y_idx = int((well_loc[1] - self.y_range[0]) / (self.y_range[1] - self.y_range[0]) * (self.grid_size[1] - 1))
                
                # Check surrounding points (3x3 grid around well)
                x_min, x_max = max(0, x_idx - 1), min(self.grid_size[0] - 1, x_idx + 1)
                y_min, y_max = max(0, y_idx - 1), min(self.grid_size[1] - 1, y_idx + 1)
                
                # Calculate average uncertainty in this region
                region_uncertainty = np.mean(new_uncertainty[prop][y_min:y_max+1, x_min:x_max+1])
                # Calculate average uncertainty in the whole grid
                global_uncertainty = np.mean(new_uncertainty[prop])
                
                # Local uncertainty should be lower than global
                self.assertLess(region_uncertainty, global_uncertainty)
    
    def test_multiple_updates(self):
        """Test multiple updates with well data."""
        # Create and update with first well
        well1_location = np.array([[0.3, 0.3]])
        property_values1 = {
            "Thickness": np.array([180]),
            "Porosity": np.array([0.06])
        }
        self.model.update_with_well_data(well1_location, property_values1)
        
        # Check mean maps after first update
        thickness_map1 = self.model.get_property_mean("Thickness")
        porosity_map1 = self.model.get_property_mean("Porosity")
        
        # Create and update with second well with different values
        well2_location = np.array([[0.7, 0.7]])
        property_values2 = {
            "Thickness": np.array([100]),
            "Porosity": np.array([0.03])
        }
        self.model.update_with_well_data(well2_location, property_values2)
        
        # Check mean maps after second update
        thickness_map2 = self.model.get_property_mean("Thickness")
        porosity_map2 = self.model.get_property_mean("Porosity")
        
        # Maps should be different after updates
        self.assertFalse(np.array_equal(thickness_map1, thickness_map2))
        self.assertFalse(np.array_equal(porosity_map1, porosity_map2))
        
        # Check that values near wells are closer to the observed values
        # Find grid indices closest to well locations
        x1_idx = int((well1_location[0, 0] - self.x_range[0]) / (self.x_range[1] - self.x_range[0]) * (self.grid_size[0] - 1))
        y1_idx = int((well1_location[0, 1] - self.y_range[0]) / (self.y_range[1] - self.y_range[0]) * (self.grid_size[1] - 1))
        
        x2_idx = int((well2_location[0, 0] - self.x_range[0]) / (self.x_range[1] - self.x_range[0]) * (self.grid_size[0] - 1))
        y2_idx = int((well2_location[0, 1] - self.y_range[0]) / (self.y_range[1] - self.y_range[0]) * (self.grid_size[1] - 1))
        
        # Values near first well should be closer to first well's values
        self.assertGreater(thickness_map2[y1_idx, x1_idx], thickness_map2[y2_idx, x2_idx])
        self.assertGreater(porosity_map2[y1_idx, x1_idx], porosity_map2[y2_idx, x2_idx])


if __name__ == "__main__":
    unittest.main()