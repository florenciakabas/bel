"""Unit tests for the economic model module."""

import unittest
import numpy as np
from bel.economic_model import EconomicAssessment


class TestEconomicAssessment(unittest.TestCase):
    """Test cases for the EconomicAssessment class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a simple economic model
        self.model = EconomicAssessment(
            gas_price=4.0,  # $/mcf
            operating_cost=0.5,  # $/mcf
            drilling_cost=10.0,  # $M per well
            discount_rate=0.1,  # annual rate
            development_years=10,
            drilling_cost_std=1.0,  # $M per well (standard deviation)
            target_profit=100.0,  # $M
            target_confidence=0.9  # 90%
        )
        
        # Create test production data
        self.time_points = np.linspace(0, 10, 121)  # 10 years with monthly points
        self.initial_rate = 1000  # mcf/day
        self.b_factor = 0.8
        self.decline_rate = 0.1
        
        # Generate a production profile
        self.production_profile = self.initial_rate / (1 + self.b_factor * self.decline_rate * self.time_points * 365.25) ** (1 / self.b_factor)
        
        # Create multiple production profiles for Monte Carlo testing
        n_wells = 5
        self.multi_production = np.zeros((n_wells, len(self.time_points)))
        for i in range(n_wells):
            self.multi_production[i, :] = (self.initial_rate * (i + 1)) / (1 + self.b_factor * self.decline_rate * self.time_points * 365.25) ** (1 / self.b_factor)
    
    def test_initialization(self):
        """Test that the model initializes correctly."""
        self.assertEqual(self.model.gas_price, 4.0)
        self.assertEqual(self.model.operating_cost, 0.5)
        self.assertEqual(self.model.drilling_cost, 10.0)
        self.assertEqual(self.model.discount_rate, 0.1)
        self.assertEqual(self.model.development_years, 10)
        self.assertEqual(self.model.drilling_cost_std, 1.0)
        self.assertEqual(self.model.target_profit, 100.0)
        self.assertEqual(self.model.target_confidence, 0.9)
    
    def test_calculate_drilling_costs(self):
        """Test drilling cost calculation."""
        n_wells = 10
        
        # Test with fixed costs
        fixed_costs = self.model.calculate_drilling_costs(
            num_wells=n_wells,
            random_variation=False
        )
        
        # Check output
        self.assertEqual(len(fixed_costs), n_wells)
        np.testing.assert_array_equal(fixed_costs, np.ones(n_wells) * self.model.drilling_cost)
        
        # Test with random variation
        random_costs = self.model.calculate_drilling_costs(
            num_wells=n_wells,
            random_variation=True,
            random_state=42
        )
        
        # Check output
        self.assertEqual(len(random_costs), n_wells)
        self.assertFalse(np.array_equal(random_costs, fixed_costs))
        
        # Costs should be around the mean drilling cost
        self.assertAlmostEqual(np.mean(random_costs), self.model.drilling_cost, delta=2.0)
        
        # Check that setting random state produces consistent results
        random_costs2 = self.model.calculate_drilling_costs(
            num_wells=n_wells,
            random_variation=True,
            random_state=42
        )
        np.testing.assert_array_equal(random_costs, random_costs2)
    
    def test_calculate_operating_costs(self):
        """Test operating cost calculation."""
        # Test with scalar production
        production = 1000000  # mcf
        opex = self.model.calculate_operating_costs(production)
        expected_opex = production * (self.model.operating_cost / 1e6)
        self.assertAlmostEqual(opex, expected_opex)
        
        # Test with array production
        production_array = np.array([1000000, 2000000, 3000000])
        opex_array = self.model.calculate_operating_costs(production_array)
        expected_opex_array = production_array * (self.model.operating_cost / 1e6)
        np.testing.assert_array_almost_equal(opex_array, expected_opex_array)
    
    def test_calculate_revenue(self):
        """Test revenue calculation."""
        # Test with scalar production
        production = 1000000  # mcf
        revenue = self.model.calculate_revenue(production)
        expected_revenue = production * (self.model.gas_price / 1e6)
        self.assertAlmostEqual(revenue, expected_revenue)
        
        # Test with array production
        production_array = np.array([1000000, 2000000, 3000000])
        revenue_array = self.model.calculate_revenue(production_array)
        expected_revenue_array = production_array * (self.model.gas_price / 1e6)
        np.testing.assert_array_almost_equal(revenue_array, expected_revenue_array)
    
    def test_calculate_npv(self):
        """Test NPV calculation."""
        # Generate test data
        time_steps = 10
        revenues = np.ones(time_steps) * 20.0  # $M per year
        operating_costs = np.ones(time_steps) * 5.0  # $M per year
        drilling_costs = np.array([10.0, 10.0])  # $M per well
        time_points = np.arange(time_steps)
        
        # Calculate NPV
        npv = self.model.calculate_npv(
            revenues=revenues,
            operating_costs=operating_costs,
            drilling_costs=drilling_costs,
            time_points=time_points,
            discount_rate=0.1
        )
        
        # Check output
        self.assertIsInstance(npv, float)
        
        # NPV should be positive when revenues > costs
        self.assertGreater(npv, 0)
        
        # Calculate expected NPV manually
        cash_flow = revenues - operating_costs
        discount_factors = 1 / (1 + 0.1) ** time_points
        discounted_cash_flow = cash_flow * discount_factors
        expected_npv = np.sum(discounted_cash_flow) - np.sum(drilling_costs)
        
        self.assertAlmostEqual(npv, expected_npv)
        
        # Test with different discount rate
        npv2 = self.model.calculate_npv(
            revenues=revenues,
            operating_costs=operating_costs,
            drilling_costs=drilling_costs,
            time_points=time_points,
            discount_rate=0.2
        )
        
        # Higher discount rate should lead to lower NPV
        self.assertLess(npv2, npv)
    
    def test_assess_profitability_distribution(self):
        """Test profitability distribution assessment."""
        # Assess profitability
        results = self.model.assess_profitability_distribution(
            production_profiles=self.multi_production,
            time_points_years=self.time_points,
            num_wells=5,
            num_simulations=100,
            random_state=42
        )
        
        # Check output structure
        self.assertIn('npv_values', results)
        self.assertIn('revenue_values', results)
        self.assertIn('opex_values', results)
        self.assertIn('capex_values', results)
        self.assertIn('npv_mean', results)
        self.assertIn('npv_std', results)
        self.assertIn('npv_p10', results)
        self.assertIn('npv_p50', results)
        self.assertIn('npv_p90', results)
        self.assertIn('prob_target', results)
        self.assertIn('meets_confidence', results)
        
        # Check dimensions
        self.assertEqual(len(results['npv_values']), 100)
        
        # Check statistics
        self.assertAlmostEqual(np.mean(results['npv_values']), results['npv_mean'])
        self.assertAlmostEqual(np.std(results['npv_values']), results['npv_std'])
        
        # Check percentiles
        p10 = np.percentile(results['npv_values'], 10)
        p50 = np.percentile(results['npv_values'], 50)
        p90 = np.percentile(results['npv_values'], 90)
        self.assertAlmostEqual(p10, results['npv_p10'])
        self.assertAlmostEqual(p50, results['npv_p50'])
        self.assertAlmostEqual(p90, results['npv_p90'])
        
        # Check probability calculation
        prob_target = np.mean(results['npv_values'] >= self.model.target_profit)
        self.assertAlmostEqual(prob_target, results['prob_target'])
        
        # Check confidence decision
        meets_confidence = prob_target >= self.model.target_confidence
        self.assertEqual(meets_confidence, results['meets_confidence'])
    
    def test_plot_npv_distribution(self):
        """Test NPV distribution plotting."""
        # Generate some NPV values
        np.random.seed(42)
        npv_values = np.random.normal(150, 50, 1000)
        
        # Create plot
        fig, ax = plt.subplots()
        ax = self.model.plot_npv_distribution(npv_values, ax=ax)
        
        # Check that the plot was created
        self.assertIsNotNone(ax)
        
        # Check title and labels
        self.assertEqual(ax.get_title(), "NPV Distribution")
        self.assertEqual(ax.get_xlabel(), "Net Present Value ($M)")
        self.assertEqual(ax.get_ylabel(), "Frequency")
        
        # Check for target line
        lines = ax.get_lines()
        self.assertGreaterEqual(len(lines), 1)


# Mock matplotlib for testing
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


if __name__ == "__main__":
    unittest.main()