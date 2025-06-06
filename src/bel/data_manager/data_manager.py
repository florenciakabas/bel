"""Data management for geological modeling and exploration simulation."""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt
from pathlib import Path


class DataManager:
    """
    Handles data I/O and validation for the exploration simulation.
    
    This class is responsible for loading and saving geological data,
    simulation results, and validating input parameters.
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the data manager.
        
        Args:
            data_dir: Directory to use for data storage and retrieval.
        """
        self.data_dir = Path(data_dir)
        os.makedirs(self.data_dir, exist_ok=True)
    
    def load_geological_data(
        self, 
        filename: str, 
        required_properties: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Load geological data from a CSV or JSON file.
        
        Args:
            filename: Name of the file to load.
            required_properties: List of required property columns.
            
        Returns:
            Dictionary containing the loaded data.
        """
        file_path = self.data_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} not found")
        
        # Load data based on file extension
        if file_path.suffix.lower() == '.csv':
            data = pd.read_csv(file_path)
            
            # Check if required properties are present
            if required_properties:
                missing = [prop for prop in required_properties if prop not in data.columns]
                if missing:
                    raise ValueError(f"Missing required properties in data file: {missing}")
            
            # Convert to dictionary
            return {
                'data': data,
                'properties': list(data.columns),
                'source': str(file_path)
            }
            
        elif file_path.suffix.lower() == '.json':
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Check if required properties are present
            if required_properties:
                if 'properties' not in data:
                    raise ValueError("Missing 'properties' key in JSON data")
                
                missing = [prop for prop in required_properties if prop not in data['properties']]
                if missing:
                    raise ValueError(f"Missing required properties in data file: {missing}")
            
            return data
        
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    def save_simulation_results(
        self, 
        results: Dict[str, Any], 
        filename: str
    ) -> str:
        """
        Save simulation results to a file.
        
        Args:
            results: Dictionary containing the simulation results.
            filename: Name of the file to save.
            
        Returns:
            Path to the saved file.
        """
        file_path = self.data_dir / filename
        
        # Create parent directories if they don't exist
        os.makedirs(file_path.parent, exist_ok=True)
        
        # Save data based on file extension
        if file_path.suffix.lower() == '.csv':
            # Convert results to DataFrame if not already
            if not isinstance(results, pd.DataFrame):
                # Try to convert nested dict to DataFrame
                try:
                    df = pd.DataFrame(results)
                except ValueError:
                    # Handle nested dictionaries
                    flat_results = self._flatten_dict(results)
                    df = pd.DataFrame([flat_results])
                    
            else:
                df = results
                
            df.to_csv(file_path, index=False)
            
        elif file_path.suffix.lower() == '.json':
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = self._make_json_serializable(results)
            
            with open(file_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
                
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        return str(file_path)
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert numpy arrays and other non-serializable objects to serializable types."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '') -> Dict[str, Any]:
        """Flatten a nested dictionary structure."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def validate_input_parameters(
        self, 
        parameters: Dict[str, Any], 
        schema: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """
        Validate input parameters against a schema.
        
        Args:
            parameters: Dictionary of input parameters to validate.
            schema: Dictionary defining the expected schema.
            
        Returns:
            Tuple of (is_valid, error_messages).
        """
        errors = []
        
        # Check for required parameters
        if 'required' in schema:
            for param in schema['required']:
                if param not in parameters:
                    errors.append(f"Missing required parameter: {param}")
        
        # Check parameter types and ranges
        if 'properties' in schema:
            for param, value in parameters.items():
                if param in schema['properties']:
                    param_schema = schema['properties'][param]
                    
                    # Check type
                    if 'type' in param_schema:
                        expected_type = param_schema['type']
                        if expected_type == 'number':
                            if not isinstance(value, (int, float)):
                                errors.append(f"Parameter {param} should be a number")
                        elif expected_type == 'integer':
                            if not isinstance(value, int):
                                errors.append(f"Parameter {param} should be an integer")
                        elif expected_type == 'string':
                            if not isinstance(value, str):
                                errors.append(f"Parameter {param} should be a string")
                        elif expected_type == 'array':
                            if not isinstance(value, (list, tuple, np.ndarray)):
                                errors.append(f"Parameter {param} should be an array")
                        elif expected_type == 'object':
                            if not isinstance(value, dict):
                                errors.append(f"Parameter {param} should be an object")
                        elif expected_type == 'boolean':
                            if not isinstance(value, bool):
                                errors.append(f"Parameter {param} should be a boolean")
                    
                    # Check minimum value
                    if 'minimum' in param_schema and isinstance(value, (int, float)):
                        if value < param_schema['minimum']:
                            errors.append(f"Parameter {param} should be >= {param_schema['minimum']}")
                    
                    # Check maximum value
                    if 'maximum' in param_schema and isinstance(value, (int, float)):
                        if value > param_schema['maximum']:
                            errors.append(f"Parameter {param} should be <= {param_schema['maximum']}")
                    
                    # Check enum values
                    if 'enum' in param_schema:
                        if value not in param_schema['enum']:
                            errors.append(f"Parameter {param} should be one of {param_schema['enum']}")
                
                else:
                    # Unknown parameter
                    if not schema.get('additionalProperties', True):
                        errors.append(f"Unknown parameter: {param}")
        
        return len(errors) == 0, errors
    
    def generate_synthetic_data(
        self,
        n_wells: int = 500,
        grid_size: Tuple[int, int] = (200, 200),
        x_range: Tuple[float, float] = (-94.6, -92.9),
        y_range: Tuple[float, float] = (31.3, 33.0),
        seed: Optional[int] = 1,
        save_to_file: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Generate synthetic geological data similar to the dummydatagen approach.
        
        Args:
            n_wells: Number of synthetic wells to generate.
            grid_size: Tuple (nx, ny) specifying the grid dimensions.
            x_range: Tuple (x_min, x_max) specifying the x-coordinate range.
            y_range: Tuple (y_min, y_max) specifying the y-coordinate range.
            seed: Random seed for reproducibility.
            save_to_file: If provided, save the generated data to this file.
            
        Returns:
            DataFrame containing the synthetic data.
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Create the grid
        nx, ny = grid_size
        x = np.linspace(x_range[0], x_range[1], nx)
        y = np.linspace(y_range[0], y_range[1], ny)
        X, Y = np.meshgrid(x, y)
        
        # Define the peaks function for generating synthetic data
        def peaks(x, y):
            return (3 * (1 - x)**2 * np.exp(-(x**2) - (y + 1)**2)
                    - 10 * (x / 5 - x**3 - y**5) * np.exp(-x**2 - y**2)
                    - 1/3 * np.exp(-(x + 1)**2 - y**2))
        
        def peaks_linspace(x0, x1, n):
            x = np.linspace(x0, x1, n)
            y = np.linspace(x0, x1, n)
            X, Y = np.meshgrid(x, y)
            res = peaks(X, Y)
            return res
        
        def peaks_num(n):
            x = np.linspace(-3, 3, n)
            y = np.linspace(-3, 3, n)
            X, Y = np.meshgrid(x, y)
            Z = peaks(X, Y)
            return Z
        
        # Normalization function 
        def min_max_norm(x):
            return (x - np.min(x)) / (np.max(x) - np.min(x))
        
        # Generate synthetic properties
        # Thickness (50-200 ft)
        gthk = peaks_linspace(-2, 2, nx)
        gthkn = min_max_norm(gthk)
        gthk = gthkn * 150 + 50
        
        # Porosity (0.02-0.08)
        gphi = peaks_linspace(-1, 1, nx)
        gphin = min_max_norm(gphi)
        gphi = gphin * 0.06 + 0.02
        
        # Permeability (0.1-0.5 mD)
        gpern = gphin.T
        gper = gpern * 0.4 + 0.1
        
        # TOC (2-8%)
        gtoc = peaks_linspace(-0.5, 0.5, nx)
        gtocn = min_max_norm(gtoc)
        gtoc = gtocn * 6 + 2
        
        # Water Saturation (0.3-0.7)
        gswn = np.fliplr(gtocn)
        gsw = gswn * 0.4 + 0.3
        
        # Depth (8000-11000 ft)
        gdep = peaks_linspace(-0.2, 0.2, nx)
        gdepn = min_max_norm(gdep)
        gdep = gdepn * (-3000) + (-8000)
        
        # Clay Volume (0.3-0.6)
        tmp = peaks_num(2 * nx)
        gvly = tmp[50:(nx + 50), 50:(nx + 50)]
        gvlyn = min_max_norm(gvly)
        gvly = gvlyn * 0.3 + 0.3
        
        # Random Sampling for wells
        xind = np.random.randint(0, nx, n_wells)
        yind = np.random.randint(0, nx, n_wells)
        
        # Create samples dictionary
        samples = {
            'x': X[xind, yind],
            'y': Y[xind, yind],
            'Depth': gdep[xind, yind],
            'Depth_norm': gdepn[xind, yind],
            'TOC': gtoc[xind, yind],
            'TOC_norm': gtocn[xind, yind],
            'Vclay': gvly[xind, yind],
            'Vclay_norm': gvlyn[xind, yind],
            'Thickness': gthk[xind, yind],
            'Thickness_norm': gthkn[xind, yind],
            'SW': gsw[xind, yind],
            'SW_norm': gswn[xind, yind],
            'Permeability': gper[xind, yind],
            'Permeability_norm': gpern[xind, yind],
            'Porosity': gphi[xind, yind],
            'Porosity_norm': gphin[xind, yind],
        }
        
        # Calculate synthetic initial production rate
        wt = [200, 250, 400, 50, 25, 5, 1]
        df_samples = pd.DataFrame(samples)
        
        syn = (
            df_samples['Vclay_norm'] * wt[0] + 
            df_samples['Thickness_norm'] * wt[1] + 
            df_samples['Depth_norm'] * wt[2] +
            df_samples['SW_norm'] * wt[3] + 
            df_samples['Porosity_norm'] * wt[4] + 
            df_samples['TOC_norm'] * wt[5] + 
            df_samples['Permeability_norm'] * wt[6]
        )
        
        # Initial production rate
        qi = syn * 1e3
        df_samples['qi'] = qi
        
        # Add well IDs
        df_samples['well_id'] = np.arange(1, n_wells + 1)
        
        # Save to file if requested
        if save_to_file:
            file_path = self.data_dir / save_to_file
            os.makedirs(file_path.parent, exist_ok=True)
            
            if file_path.suffix.lower() == '.csv':
                df_samples.to_csv(file_path, index=False)
            elif file_path.suffix.lower() == '.json':
                df_samples.to_json(file_path, orient='records', indent=2)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        return df_samples
    
    def plot_property_map(
        self, 
        property_data: Dict[str, np.ndarray],
        property_name: str,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        well_locs: Optional[np.ndarray] = None,
        well_values: Optional[np.ndarray] = None,
        ax=None,
        title: Optional[str] = None,
        colorbar_label: Optional[str] = None,
        cmap: str = 'viridis'
    ):
        """
        Plot a property map from the provided data.
        
        Args:
            property_data: Dictionary mapping property names to 2D arrays.
            property_name: Name of the property to plot.
            x_coords: 2D array of x coordinates.
            y_coords: 2D array of y coordinates.
            well_locs: Optional array of well locations.
            well_values: Optional array of well property values.
            ax: Matplotlib axis to plot on. If None, creates a new figure.
            title: Title for the plot. If None, uses the property name.
            colorbar_label: Label for the colorbar. If None, uses the property name.
            cmap: Colormap to use for the plot.
            
        Returns:
            The matplotlib axis object.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 8))
        
        # Get the property map
        property_map = property_data[property_name]
        
        # Plot the property map
        im = ax.contourf(x_coords, y_coords, property_map, cmap=cmap)
        
        # Add well locations if provided
        if well_locs is not None:
            if well_values is not None:
                scatter = ax.scatter(
                    well_locs[:, 0], 
                    well_locs[:, 1], 
                    c=well_values, 
                    cmap=cmap,
                    edgecolor='black', 
                    s=100, 
                    zorder=10
                )
            else:
                ax.scatter(
                    well_locs[:, 0], 
                    well_locs[:, 1], 
                    c='white', 
                    edgecolor='black', 
                    s=100, 
                    zorder=10
                )
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label=colorbar_label or property_name)
        
        # Add title and labels
        ax.set_title(title or f"{property_name} Map")
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        
        return ax