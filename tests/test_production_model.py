"""Unit tests for the production model module."""

import unittest
import numpy as np
import pandas as pd
from bel.production_model import ProductionPredictor


class TestProductionPredictor(unittest.TestCase):
    """Test cases for the ProductionPredictor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a simple production model
        self.model = ProductionPredictor(model_type="linear", random_state=42)
        
        # Create synthetic data for training
        np.random.seed(42)
        n_samples = 100
        
        # Generate geological properties
        thickness = np.random.uniform(50, 200, n_samples)
        porosity = np.random.uniform(0.02, 0.08, n_samples)
        permeability = np.random.uniform(0.1, 0.5, n_samples)
        
        # Generate initial rates using a simple linear model
        qi = 100 * thickness + 10000 * porosity + 1000 * permeability + np.random.normal(0, 100, n_samples)
        qi = np.maximum(qi, 0)  # Ensure non-negative
        
        # Generate decline parameters
        b = 0.8 * np.ones(n_samples)
        di = 0.1 * np.ones(n_samples)
        
        # Create DataFrame
        self.training_data = pd.DataFrame({
            'Thickness': thickness,
            'Porosity': porosity,
            'Permeability': permeability,
            'qi': qi,
            'b': b,
            'di': di
        })
        
        # Train the model
        self.model.train(
            geological_properties=self.training_data,
            production_data=self.training_data,
            property_columns=['Thickness', 'Porosity', 'Permeability'],
            initial_rate_column='qi',
            train_decline_params=True
        )
    
    def test_initialization(self):
        """Test that the model initializes correctly."""
        model = ProductionPredictor(model_type="linear", random_state=42)
        self.assertEqual(model.model_type, "linear")
        self.assertEqual(model.random_state, 42)
        self.assertIsNone(model.initial_rate_model)
        
        # Test invalid model type
        with self.assertRaises(ValueError):
            ProductionPredictor(model_type="invalid_model")
    
    def test_training(self):
        """Test model training."""
        # Check that models were created
        self.assertIsNotNone(self.model.initial_rate_model)
        self.assertIsNotNone(self.model.b_factor_model)
        self.assertIsNotNone(self.model.decline_rate_model)
        
        # Check that weights were extracted (for linear model)
        self.assertIsNotNone(self.model.property_weights)
        self.assertEqual(len(self.model.property_weights), 3)
        for prop in ['Thickness', 'Porosity', 'Permeability']:
            self.assertIn(prop, self.model.property_weights)
    
    def test_predict_initial_rate(self):
        """Test initial rate prediction."""
        # Create test data
        test_data = pd.DataFrame({
            'Thickness': [150, 100],
            'Porosity': [0.05, 0.03],
            'Permeability': [0.3, 0.2]
        })
        
        # Predict initial rates
        predicted_rates = self.model.predict_initial_rate(test_data)
        
        # Check output
        self.assertEqual(len(predicted_rates), 2)
        self.assertGreater(predicted_rates[0], 0)
        
        # Higher property values should generally lead to higher rates
        self.assertGreater(predicted_rates[0], predicted_rates[1])
        
        # Test with numpy array
        array_input = test_data.values
        array_predicted = self.model.predict_initial_rate(array_input)
        self.assertEqual(len(array_predicted), 2)
        
        # Test with dictionary
        dict_input = {
            'Thickness': np.array([150, 100]),
            'Porosity': np.array([0.05, 0.03]),
            'Permeability': np.array([0.3, 0.2])
        }
        dict_predicted = self.model.predict_initial_rate(dict_input)
        self.assertEqual(len(dict_predicted), 2)
    
    def test_predict_decline_parameters(self):
        """Test decline parameter prediction."""
        # Create test data
        test_data = pd.DataFrame({
            'Thickness': [150, 100],
            'Porosity': [0.05, 0.03],
            'Permeability': [0.3, 0.2]
        })
        
        # Predict decline parameters
        b_factors, decline_rates = self.model.predict_decline_parameters(test_data)
        
        # Check output
        self.assertEqual(len(b_factors), 2)
        self.assertEqual(len(decline_rates), 2)
        
        # Values should be positive
        self.assertGreater(np.min(b_factors), 0)
        self.assertGreater(np.min(decline_rates), 0)
    
    def test_forecast_production_profile(self):
        """Test production profile forecasting."""
        # Create time points
        time_points = np.linspace(0, 365 * 10, 121)  # 10 years with monthly points
        
        # Forecast with scalar values
        initial_rate = 1000
        b_factor = 0.8
        decline_rate = 0.1
        
        production = self.model.forecast_production_profile(
            initial_rate, b_factor, decline_rate, time_points
        )
        
        # Check output
        self.assertEqual(len(production), len(time_points))
        self.assertEqual(production[0], initial_rate)  # First value should be initial rate
        self.assertLess(production[-1], initial_rate)  # Production should decline
        
        # Check that production is monotonically decreasing
        self.assertTrue(np.all(np.diff(production) <= 0))
        
        # Forecast with array values
        initial_rates = np.array([1000, 2000])
        b_factors = np.array([0.8, 0.9])
        decline_rates = np.array([0.1, 0.05])
        
        production_array = self.model.forecast_production_profile(
            initial_rates, b_factors, decline_rates, time_points
        )
        
        # Check output
        self.assertEqual(production_array.shape, (2, len(time_points)))
        np.testing.assert_array_equal(production_array[:, 0], initial_rates)
        
        # Higher initial rate should lead to higher production
        self.assertTrue(np.all(production_array[1, :] > production_array[0, :]))
    
    def test_calculate_cumulative_production(self):
        """Test cumulative production calculation."""
        # Calculate with scalar values
        initial_rate = 1000
        b_factor = 0.8
        decline_rate = 0.1
        time_span = 365 * 10  # 10 years
        
        cumulative = self.model.calculate_cumulative_production(
            initial_rate, b_factor, decline_rate, time_span
        )
        
        # Check output
        self.assertGreater(cumulative, 0)
        
        # Calculate with array values
        initial_rates = np.array([1000, 2000])
        b_factors = np.array([0.8, 0.9])
        decline_rates = np.array([0.1, 0.05])
        
        cumulative_array = self.model.calculate_cumulative_production(
            initial_rates, b_factors, decline_rates, time_span
        )
        
        # Check output
        self.assertEqual(len(cumulative_array), 2)
        self.assertGreater(np.min(cumulative_array), 0)
        
        # Higher initial rate should lead to higher cumulative production
        self.assertGreater(cumulative_array[1], cumulative_array[0])
    
    def test_predict_full_well_performance(self):
        """Test full well performance prediction."""
        # Create test data
        test_data = pd.DataFrame({
            'Thickness': [150, 100],
            'Porosity': [0.05, 0.03],
            'Permeability': [0.3, 0.2]
        })
        
        # Predict performance
        results = self.model.predict_full_well_performance(
            test_data,
            development_years=10
        )
        
        # Check output structure
        self.assertIn('initial_rates', results)
        self.assertIn('b_factors', results)
        self.assertIn('decline_rates', results)
        self.assertIn('time_points_years', results)
        self.assertIn('production_profiles', results)
        self.assertIn('cumulative_production', results)
        
        # Check dimensions
        self.assertEqual(len(results['initial_rates']), 2)
        self.assertEqual(len(results['b_factors']), 2)
        self.assertEqual(len(results['decline_rates']), 2)
        self.assertEqual(results['production_profiles'].shape[0], 2)
        self.assertEqual(len(results['cumulative_production']), 2)
        
        # First point in profile should match initial rate
        np.testing.assert_array_almost_equal(
            results['production_profiles'][:, 0],
            results['initial_rates']
        )


if __name__ == "__main__":
    unittest.main()