"""Production prediction from geological properties."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import datetime


class ProductionPredictor:
    """
    Predicts well production from geological properties.
    
    This class handles the prediction of initial production rates,
    decline curve parameters, and production profiles based on geological properties.
    """
    
    def __init__(
        self,
        model_type: str = "linear",
        random_state: Optional[int] = None
    ):
        """
        Initialize the production predictor.
        
        Args:
            model_type: Type of model to use: "linear" or "random_forest".
            random_state: Random seed for reproducibility.
        """
        self.model_type = model_type
        self.random_state = random_state
        self.initial_rate_model = None
        self.b_factor_model = None
        self.decline_rate_model = None
        self.property_weights = None
        self.scaler = None
        
        # Default decline curve parameters if not trained
        self.default_b = 0.8
        self.default_di = 0.1
    
    def _create_model(self):
        """Create the appropriate model based on model_type."""
        if self.model_type == "linear":
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', LinearRegression())
            ])
        elif self.model_type == "random_forest":
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', RandomForestRegressor(
                    n_estimators=100, 
                    max_depth=10,
                    random_state=self.random_state
                ))
            ])
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
            
        return model
    
    def train(
        self,
        geological_properties: pd.DataFrame,
        production_data: pd.DataFrame,
        property_columns: Optional[List[str]] = None,
        initial_rate_column: str = "qi",
        train_decline_params: bool = False
    ):
        """
        Train the production prediction model.
        
        Args:
            geological_properties: DataFrame containing geological property data.
            production_data: DataFrame containing production data.
            property_columns: List of geological property column names to use.
            initial_rate_column: Column name for initial production rate.
            train_decline_params: Whether to train models for decline curve parameters.
        """
        # Merge geological properties with production data if needed
        if "well_id" in geological_properties.columns and "well_id" in production_data.columns:
            data = pd.merge(geological_properties, production_data, on="well_id")
        else:
            data = geological_properties.copy()
            
        # Determine which property columns to use
        if property_columns is None:
            # Use all columns that might be geological properties
            exclude_cols = ["well_id", "x", "y", initial_rate_column, "b", "di"]
            property_columns = [col for col in data.columns if col not in exclude_cols]
        
        # Prepare features and target
        X = data[property_columns].values
        y_qi = data[initial_rate_column].values
        
        # Train initial rate model
        self.initial_rate_model = self._create_model()
        self.initial_rate_model.fit(X, y_qi)
        
        # Extract and store weights if using linear regression
        if self.model_type == "linear":
            self.property_weights = {
                prop: coef for prop, coef in zip(
                    property_columns, 
                    self.initial_rate_model.named_steps['regressor'].coef_
                )
            }
        
        # Train decline parameter models if requested
        if train_decline_params and "b" in data.columns and "di" in data.columns:
            y_b = data["b"].values
            y_di = data["di"].values
            
            self.b_factor_model = self._create_model()
            self.b_factor_model.fit(X, y_b)
            
            self.decline_rate_model = self._create_model()
            self.decline_rate_model.fit(X, y_di)
    
    def predict_initial_rate(
        self,
        geological_properties: Union[pd.DataFrame, np.ndarray, Dict[str, np.ndarray]]
    ) -> np.ndarray:
        """
        Predict initial production rates from geological properties.
        
        Args:
            geological_properties: DataFrame, array, or dictionary of geological properties.
            
        Returns:
            Array of predicted initial production rates.
        """
        # Convert input to numpy array
        if isinstance(geological_properties, pd.DataFrame):
            # If property_weights exists, use only those columns in the same order
            if self.property_weights is not None:
                props = list(self.property_weights.keys())
                X = geological_properties[props].values
            else:
                # Otherwise, just use all columns (risky if columns don't match training data)
                X = geological_properties.values
        elif isinstance(geological_properties, dict):
            # If it's a dictionary, convert to array
            if self.property_weights is not None:
                props = list(self.property_weights.keys())
                X = np.column_stack([geological_properties[prop] for prop in props])
            else:
                # Otherwise, just stack all values (risky)
                X = np.column_stack(list(geological_properties.values()))
        else:
            # Assume it's already a numpy array
            X = geological_properties
        
        # If no model is trained, use a simple weighted sum method
        if self.initial_rate_model is None:
            # Default weights if not trained
            default_weights = np.ones(X.shape[1]) / X.shape[1]
            return np.sum(X * default_weights, axis=1) * 1000
        
        # Use the trained model
        return self.initial_rate_model.predict(X)
    
    def predict_decline_parameters(
        self,
        geological_properties: Union[pd.DataFrame, np.ndarray, Dict[str, np.ndarray]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict decline curve parameters from geological properties.
        
        Args:
            geological_properties: DataFrame, array, or dictionary of geological properties.
            
        Returns:
            Tuple of (b_factor, decline_rate) arrays.
        """
        # Convert input to numpy array
        if isinstance(geological_properties, pd.DataFrame):
            if self.property_weights is not None:
                props = list(self.property_weights.keys())
                X = geological_properties[props].values
            else:
                X = geological_properties.values
        elif isinstance(geological_properties, dict):
            if self.property_weights is not None:
                props = list(self.property_weights.keys())
                X = np.column_stack([geological_properties[prop] for prop in props])
            else:
                X = np.column_stack(list(geological_properties.values()))
        else:
            X = geological_properties
        
        # If models are not trained, use default values
        if self.b_factor_model is None or self.decline_rate_model is None:
            b_factor = np.ones(len(X)) * self.default_b
            decline_rate = np.ones(len(X)) * self.default_di
            return b_factor, decline_rate
        
        # Use the trained models
        b_factor = self.b_factor_model.predict(X)
        decline_rate = self.decline_rate_model.predict(X)
        
        return b_factor, decline_rate
    
    def forecast_production_profile(
        self,
        initial_rate: Union[float, np.ndarray],
        b_factor: Union[float, np.ndarray],
        decline_rate: Union[float, np.ndarray],
        time_points: np.ndarray
    ) -> np.ndarray:
        """
        Forecast production profile using Arps decline curve.
        
        Args:
            initial_rate: Initial production rate.
            b_factor: b factor in Arps equation.
            decline_rate: Initial decline rate.
            time_points: Array of time points to forecast.
            
        Returns:
            Array of production rates at the specified time points.
        """
        # Arps decline curve: q(t) = q_i / (1 + b * D_i * t)^(1/b)
        if np.isscalar(initial_rate):
            return initial_rate / (1 + b_factor * decline_rate * time_points) ** (1 / b_factor)
        else:
            # Vectorized calculation for multiple wells
            production = np.zeros((len(initial_rate), len(time_points)))
            for i in range(len(initial_rate)):
                production[i, :] = initial_rate[i] / (1 + b_factor[i] * decline_rate[i] * time_points) ** (1 / b_factor[i])
            return production
    
    def calculate_cumulative_production(
        self,
        initial_rate: Union[float, np.ndarray],
        b_factor: Union[float, np.ndarray],
        decline_rate: Union[float, np.ndarray],
        time_span: float
    ) -> np.ndarray:
        """
        Calculate cumulative production over a time span.
        
        Args:
            initial_rate: Initial production rate.
            b_factor: b factor in Arps equation.
            decline_rate: Initial decline rate.
            time_span: Time span in days.
            
        Returns:
            Cumulative production.
        """
        # For b ≠ 1: Q(t) = (q_i / ((1-b) * D_i)) * (1 - (1 + b * D_i * t)^(1-1/b))
        # For b = 1: Q(t) = (q_i / D_i) * ln(1 + D_i * t)
        
        if np.isscalar(initial_rate):
            if abs(b_factor - 1.0) < 1e-6:  # b ≈ 1
                return (initial_rate / decline_rate) * np.log(1 + decline_rate * time_span)
            else:  # b ≠ 1
                return (initial_rate / ((1 - b_factor) * decline_rate)) * (
                    1 - (1 + b_factor * decline_rate * time_span) ** (1 - 1/b_factor)
                )
        else:
            # Vectorized calculation for multiple wells
            cum_production = np.zeros(len(initial_rate))
            for i in range(len(initial_rate)):
                if abs(b_factor[i] - 1.0) < 1e-6:  # b ≈ 1
                    cum_production[i] = (initial_rate[i] / decline_rate[i]) * np.log(1 + decline_rate[i] * time_span)
                else:  # b ≠ 1
                    cum_production[i] = (initial_rate[i] / ((1 - b_factor[i]) * decline_rate[i])) * (
                        1 - (1 + b_factor[i] * decline_rate[i] * time_span) ** (1 - 1/b_factor[i])
                    )
            return cum_production
    
    def predict_full_well_performance(
        self,
        geological_properties: Union[pd.DataFrame, np.ndarray, Dict[str, np.ndarray]],
        time_points_years: np.ndarray = None,
        development_years: int = 10
    ) -> Dict[str, np.ndarray]:
        """
        Predict complete well performance from geological properties.
        
        Args:
            geological_properties: DataFrame, array, or dictionary of geological properties.
            time_points_years: Array of time points in years for the forecast.
            development_years: Total development horizon in years.
            
        Returns:
            Dictionary with production forecast results.
        """
        # Set default time points if not provided
        if time_points_years is None:
            time_points_years = np.linspace(0, development_years, development_years * 12 + 1)
        
        # Convert time to days
        time_points_days = time_points_years * 365.25
        
        # Predict initial rates
        initial_rates = self.predict_initial_rate(geological_properties)
        
        # Predict decline parameters
        b_factors, decline_rates = self.predict_decline_parameters(geological_properties)
        
        # Forecast production profiles
        production_profiles = self.forecast_production_profile(
            initial_rates, b_factors, decline_rates, time_points_days
        )
        
        # Calculate cumulative production
        cumulative_production = self.calculate_cumulative_production(
            initial_rates, b_factors, decline_rates, time_points_days[-1]
        )
        
        return {
            'initial_rates': initial_rates,
            'b_factors': b_factors,
            'decline_rates': decline_rates,
            'time_points_years': time_points_years,
            'production_profiles': production_profiles,
            'cumulative_production': cumulative_production
        }
    
    def plot_production_profile(
        self,
        production_profiles: np.ndarray,
        time_points_years: np.ndarray,
        ax=None,
        title: str = "Production Forecast",
        ylabel: str = "Production Rate (mcf/day)",
        log_y: bool = True,
        well_ids: Optional[List[str]] = None
    ):
        """
        Plot production profiles.
        
        Args:
            production_profiles: Array of production profiles.
            time_points_years: Array of time points in years.
            ax: Matplotlib axis to plot on. If None, creates a new figure.
            title: Title for the plot.
            ylabel: Label for the y-axis.
            log_y: Whether to use a logarithmic y-axis.
            well_ids: List of well identifiers for the legend.
            
        Returns:
            The matplotlib axis object.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))
        
        # Handle different input shapes
        if production_profiles.ndim == 1:
            # Single well profile
            ax.plot(time_points_years, production_profiles, linewidth=2)
        else:
            # Multiple well profiles
            for i in range(len(production_profiles)):
                label = f"Well {i+1}" if well_ids is None else well_ids[i]
                ax.plot(time_points_years, production_profiles[i], label=label)
            
            if well_ids is not None:
                ax.legend(loc='best')
        
        # Set y-axis scale
        if log_y:
            ax.set_yscale('log')
        
        # Set labels and title
        ax.set_xlabel("Time (years)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, which='both', linestyle='--', alpha=0.6)
        
        return ax