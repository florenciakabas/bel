"""
Field data module for Basin Exploration GP.

This module provides functions to work with field data, including the 
geological property functions from Dummy_Field_Testing.py and utilities 
to load and manipulate field data.
"""

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

# Store global variables for field properties
field_props = {
    'gthk': None,  # Thickness
    'gphi': None,  # Porosity
    'gper': None,  # Permeability
    'gtoc': None,  # TOC
    'gsw': None,   # Water Saturation
    'gvly': None,  # Clay Volume
    'gdep': None,  # Depth
    'X': None,     # X coordinates meshgrid
    'Y': None,     # Y coordinates meshgrid
    'x': None,     # X coordinates 1D
    'y': None,     # Y coordinates 1D
    'n': None,     # Grid size
    'df_full': None, # Full dataframe with samples
}

def field_peaks(x, y):
    """
    Python implementation of the Matlab peaks function used in field data generation.
    
    Args:
        x: X coordinates
        y: Y coordinates
    
    Returns:
        Peaks function values
    """
    return (3 * (1 - x)**2 * np.exp(-(x**2) - (y + 1)**2)
            - 10 * (x / 5 - x**3 - y**5) * np.exp(-x**2 - y**2)
            - 1/3 * np.exp(-(x + 1)**2 - y**2))

def min_max_norm(x):
    """
    Min-max normalization function.
    
    Args:
        x: Array to normalize
    
    Returns:
        Normalized array in [0,1] range
    """
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def load_field_data(show_plots=False):
    """
    Generate and load field properties data.
    Similar to Dummy_Field_Testing.py but structured as a reusable function.

    Args:
        show_plots: Whether to display visualization plots
    
    Returns:
        A pandas DataFrame with sampled field data
    """
    global field_props
    
    # Initialize with same settings as in Dummy_Field_Testing.py
    np.random.seed(1)
    
    n = 200
    nw = 500
    x = np.linspace(-94.6, -92.9, n)
    y = np.linspace(31.3, 33, n)
    X, Y = np.meshgrid(x, y)
    
    # Store coordinates for later interpolation
    field_props['x'] = x
    field_props['y'] = y
    field_props['X'] = X
    field_props['Y'] = Y
    field_props['n'] = n
    
    # Function to generate peaks with different ranges
    def peaks_linspace(x0, x1, n):
        x = np.linspace(x0, x1, n)
        y = np.linspace(x0, x1, n)
        X, Y = np.meshgrid(x, y)
        res = field_peaks(X, Y)
        return res
    
    def peaks_num(n):
        x = np.linspace(-3, 3, n)
        y = np.linspace(-3, 3, n)
        X, Y = np.meshgrid(x, y)
        Z = field_peaks(X, Y)
        return Z
    
    # Generate property surfaces
    
    # Thickness
    gthk = peaks_linspace(-2, 2, n)
    gthkn = min_max_norm(gthk)
    gthk = gthkn * 150 + 50
    field_props['gthk'] = gthk
    
    # Porosity
    gphi = peaks_linspace(-1, 1, n)
    gphin = min_max_norm(gphi)
    gphi = gphin * 0.06 + 0.02
    field_props['gphi'] = gphi
    
    # Permeability
    gpern = gphin.T
    gper = gpern * 0.4 + 0.1
    field_props['gper'] = gper
    
    # TOC
    gtoc = peaks_linspace(-0.5, 0.5, n)
    gtocn = min_max_norm(gtoc)
    gtoc = gtocn * 6 + 2
    field_props['gtoc'] = gtoc
    
    # Water Saturation
    gswn = np.fliplr(gtocn)
    gsw = gswn * 0.4 + 0.3
    field_props['gsw'] = gsw
    
    # Depth
    gdep = peaks_linspace(-0.2, 0.2, n)
    gdepn = min_max_norm(gdep)
    gdep = gdepn * (-3000) + (-8000)
    field_props['gdep'] = gdep
    
    # Clay volume
    tmp = peaks_num(2 * n)
    gvly = tmp[50:(n + 50), 50:(n + 50)]
    gvlyn = min_max_norm(gvly)
    gvly = gvlyn * 0.3 + 0.3
    field_props['gvly'] = gvly
    
    # Plot if requested
    if show_plots:
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        titles = ['Thickness', 'Porosity', 'Perm', 'TOC', 'SW', 'Vclay', 'Depth']
        data_list = [gthk, gphi, gper, gtoc, gsw, gvly, gdep]
        for i, ax in enumerate(axes.flat[:7]):
            cs = ax.contourf(X, Y, data_list[i], cmap='plasma')
            ax.set_title(titles[i])
            plt.colorbar(cs, ax=ax)
        plt.tight_layout()
        plt.show()
    
    # Random Sampling for wells
    xind = np.random.randint(0, n, nw)
    yind = np.random.randint(0, n, nw)
    
    samples = {
        'xx': X[xind, yind],
        'yy': Y[xind, yind],
        'wdep': gdepn[xind, yind],
        'wtoc': gtocn[xind, yind],
        'wvly': gvlyn[xind, yind],
        'wthk': gthkn[xind, yind],
        'wsw': gswn[xind, yind],
        'wper': gpern[xind, yind],
        'wphi': gphin[xind, yind],
    }
    
    # Create full dataframe
    df_full = pd.DataFrame({
        'x': samples['xx'],
        'y': samples['yy'],
        'Depth': gdep[xind, yind],  # Use actual values, not normalized
        'TOC': gtoc[xind, yind],
        'Vclay': gvly[xind, yind],
        'Thickness': gthk[xind, yind],
        'SW': gsw[xind, yind],
        'Perm': gper[xind, yind],
        'Porosity': gphi[xind, yind],
    })
    
    # Store the dataframe for later use
    field_props['df_full'] = df_full
    
    return df_full

# Functions to access field properties (similar to true_* functions in simulation.py)
def field_thickness(coordinates, basin_size=None):
    """
    Get thickness values at specified coordinates using interpolation from field data.
    
    Args:
        coordinates: Tensor or array of [x, y] coordinates
        basin_size: Optional basin size, ignored (for compatibility with other functions)
    
    Returns:
        Thickness values at given coordinates as tensor
    """
    if field_props['gthk'] is None:
        load_field_data()
    
    # Convert tensor to numpy if needed
    coords_np = coordinates.numpy() if isinstance(coordinates, torch.Tensor) else coordinates
    
    # Scale coordinates to match field data coordinates
    x_field = field_props['x']
    y_field = field_props['y']
    x_min, x_max = x_field[0], x_field[-1]
    y_min, y_max = y_field[0], y_field[-1]
    
    # Map [0, basin_size] to field coordinate range
    scaled_x = x_min + (x_max - x_min) * coords_np[:, 0] / 20  # Assuming 20 km basin size
    scaled_y = y_min + (y_max - y_min) * coords_np[:, 1] / 20  # Adjust as needed
    
    # Interpolate values
    from scipy.interpolate import griddata
    points = np.column_stack((field_props['X'].flatten(), field_props['Y'].flatten()))
    values = field_props['gthk'].flatten()
    
    interpolated = griddata(points, values, (scaled_x, scaled_y), method='linear')
    
    # Convert back to tensor
    return torch.tensor(interpolated, dtype=torch.float32)

def field_porosity(coordinates, basin_size=None):
    """
    Get porosity values at specified coordinates using interpolation from field data.
    
    Args:
        coordinates: Tensor or array of [x, y] coordinates
        basin_size: Optional basin size, ignored (for compatibility with other functions)
    
    Returns:
        Porosity values at given coordinates as tensor
    """
    if field_props['gphi'] is None:
        load_field_data()
    
    # Convert tensor to numpy if needed
    coords_np = coordinates.numpy() if isinstance(coordinates, torch.Tensor) else coordinates
    
    # Scale coordinates to match field data coordinates
    x_field = field_props['x']
    y_field = field_props['y']
    x_min, x_max = x_field[0], x_field[-1]
    y_min, y_max = y_field[0], y_field[-1]
    
    # Map [0, basin_size] to field coordinate range
    scaled_x = x_min + (x_max - x_min) * coords_np[:, 0] / 20  # Assuming 20 km basin size
    scaled_y = y_min + (y_max - y_min) * coords_np[:, 1] / 20  # Adjust as needed
    
    # Interpolate values
    from scipy.interpolate import griddata
    points = np.column_stack((field_props['X'].flatten(), field_props['Y'].flatten()))
    values = field_props['gphi'].flatten()
    
    interpolated = griddata(points, values, (scaled_x, scaled_y), method='linear')
    
    # Convert back to tensor
    return torch.tensor(interpolated, dtype=torch.float32)

def field_permeability(coordinates, basin_size=None):
    """
    Get permeability values at specified coordinates using interpolation from field data.
    
    Args:
        coordinates: Tensor or array of [x, y] coordinates
        basin_size: Optional basin size, ignored (for compatibility with other functions)
    
    Returns:
        Permeability values at given coordinates as tensor
    """
    if field_props['gper'] is None:
        load_field_data()
    
    # Convert tensor to numpy if needed
    coords_np = coordinates.numpy() if isinstance(coordinates, torch.Tensor) else coordinates
    
    # Scale coordinates to match field data coordinates
    x_field = field_props['x']
    y_field = field_props['y']
    x_min, x_max = x_field[0], x_field[-1]
    y_min, y_max = y_field[0], y_field[-1]
    
    # Map [0, basin_size] to field coordinate range
    scaled_x = x_min + (x_max - x_min) * coords_np[:, 0] / 20  # Assuming 20 km basin size
    scaled_y = y_min + (y_max - y_min) * coords_np[:, 1] / 20  # Adjust as needed
    
    # Interpolate values
    from scipy.interpolate import griddata
    points = np.column_stack((field_props['X'].flatten(), field_props['Y'].flatten()))
    values = field_props['gper'].flatten()
    
    interpolated = griddata(points, values, (scaled_x, scaled_y), method='linear')
    
    # Convert back to tensor
    return torch.tensor(interpolated, dtype=torch.float32)

def field_toc(coordinates, basin_size=None):
    """
    Get TOC values at specified coordinates using interpolation from field data.
    
    Args:
        coordinates: Tensor or array of [x, y] coordinates
        basin_size: Optional basin size, ignored (for compatibility with other functions)
    
    Returns:
        TOC values at given coordinates as tensor
    """
    if field_props['gtoc'] is None:
        load_field_data()
    
    # Convert tensor to numpy if needed
    coords_np = coordinates.numpy() if isinstance(coordinates, torch.Tensor) else coordinates
    
    # Scale coordinates to match field data coordinates
    x_field = field_props['x']
    y_field = field_props['y']
    x_min, x_max = x_field[0], x_field[-1]
    y_min, y_max = y_field[0], y_field[-1]
    
    # Map [0, basin_size] to field coordinate range
    scaled_x = x_min + (x_max - x_min) * coords_np[:, 0] / 20
    scaled_y = y_min + (y_max - y_min) * coords_np[:, 1] / 20
    
    # Interpolate values
    from scipy.interpolate import griddata
    points = np.column_stack((field_props['X'].flatten(), field_props['Y'].flatten()))
    values = field_props['gtoc'].flatten()
    
    interpolated = griddata(points, values, (scaled_x, scaled_y), method='linear')
    
    # Convert back to tensor
    return torch.tensor(interpolated, dtype=torch.float32)

def field_water_saturation(coordinates, basin_size=None):
    """
    Get water saturation values at specified coordinates using interpolation from field data.
    
    Args:
        coordinates: Tensor or array of [x, y] coordinates
        basin_size: Optional basin size, ignored (for compatibility with other functions)
    
    Returns:
        Water saturation values at given coordinates as tensor
    """
    if field_props['gsw'] is None:
        load_field_data()
    
    # Convert tensor to numpy if needed
    coords_np = coordinates.numpy() if isinstance(coordinates, torch.Tensor) else coordinates
    
    # Scale coordinates to match field data coordinates
    x_field = field_props['x']
    y_field = field_props['y']
    x_min, x_max = x_field[0], x_field[-1]
    y_min, y_max = y_field[0], y_field[-1]
    
    # Map [0, basin_size] to field coordinate range
    scaled_x = x_min + (x_max - x_min) * coords_np[:, 0] / 20
    scaled_y = y_min + (y_max - y_min) * coords_np[:, 1] / 20
    
    # Interpolate values
    from scipy.interpolate import griddata
    points = np.column_stack((field_props['X'].flatten(), field_props['Y'].flatten()))
    values = field_props['gsw'].flatten()
    
    interpolated = griddata(points, values, (scaled_x, scaled_y), method='linear')
    
    # Convert back to tensor
    return torch.tensor(interpolated, dtype=torch.float32)

def field_clay_volume(coordinates, basin_size=None):
    """
    Get clay volume values at specified coordinates using interpolation from field data.
    
    Args:
        coordinates: Tensor or array of [x, y] coordinates
        basin_size: Optional basin size, ignored (for compatibility with other functions)
    
    Returns:
        Clay volume values at given coordinates as tensor
    """
    if field_props['gvly'] is None:
        load_field_data()
    
    # Convert tensor to numpy if needed
    coords_np = coordinates.numpy() if isinstance(coordinates, torch.Tensor) else coordinates
    
    # Scale coordinates to match field data coordinates
    x_field = field_props['x']
    y_field = field_props['y']
    x_min, x_max = x_field[0], x_field[-1]
    y_min, y_max = y_field[0], y_field[-1]
    
    # Map [0, basin_size] to field coordinate range
    scaled_x = x_min + (x_max - x_min) * coords_np[:, 0] / 20
    scaled_y = y_min + (y_max - y_min) * coords_np[:, 1] / 20
    
    # Interpolate values
    from scipy.interpolate import griddata
    points = np.column_stack((field_props['X'].flatten(), field_props['Y'].flatten()))
    values = field_props['gvly'].flatten()
    
    interpolated = griddata(points, values, (scaled_x, scaled_y), method='linear')
    
    # Convert back to tensor
    return torch.tensor(interpolated, dtype=torch.float32)

def field_depth(coordinates, basin_size=None):
    """
    Get depth values at specified coordinates using interpolation from field data.
    
    Args:
        coordinates: Tensor or array of [x, y] coordinates
        basin_size: Optional basin size, ignored (for compatibility with other functions)
    
    Returns:
        Depth values at given coordinates as tensor
    """
    if field_props['gdep'] is None:
        load_field_data()
    
    # Convert tensor to numpy if needed
    coords_np = coordinates.numpy() if isinstance(coordinates, torch.Tensor) else coordinates
    
    # Scale coordinates to match field data coordinates
    x_field = field_props['x']
    y_field = field_props['y']
    x_min, x_max = x_field[0], x_field[-1]
    y_min, y_max = y_field[0], y_field[-1]
    
    # Map [0, basin_size] to field coordinate range
    scaled_x = x_min + (x_max - x_min) * coords_np[:, 0] / 20
    scaled_y = y_min + (y_max - y_min) * coords_np[:, 1] / 20
    
    # Interpolate values
    from scipy.interpolate import griddata
    points = np.column_stack((field_props['X'].flatten(), field_props['Y'].flatten()))
    values = field_props['gdep'].flatten()
    
    interpolated = griddata(points, values, (scaled_x, scaled_y), method='linear')
    
    # Convert back to tensor
    return torch.tensor(interpolated, dtype=torch.float32)

def visualize_field_geology(basin_size=(20, 20), resolution=30, show_plots=True):
    """
    Create a visualization of the field geology surfaces.
    
    Args:
        basin_size: Size of the basin in (x, y) km
        resolution: Grid resolution for visualization
        show_plots: Whether to display the plots
    
    Returns:
        grid_tensor, x1_grid, x2_grid, thickness, porosity, permeability, toc, water_saturation, clay_volume, depth
    """
    if field_props['gthk'] is None:
        load_field_data()
    
    # Create a grid for visualization
    grid_tensor, x1_grid, x2_grid, _ = create_basin_grid(basin_size, resolution)
    
    # Get property values on the grid
    thickness = field_thickness(grid_tensor).reshape(resolution, resolution).numpy()
    porosity = field_porosity(grid_tensor).reshape(resolution, resolution).numpy()
    permeability = field_permeability(grid_tensor).reshape(resolution, resolution).numpy()
    toc = field_toc(grid_tensor).reshape(resolution, resolution).numpy()
    water_saturation = field_water_saturation(grid_tensor).reshape(resolution, resolution).numpy()
    clay_volume = field_clay_volume(grid_tensor).reshape(resolution, resolution).numpy()
    depth = field_depth(grid_tensor).reshape(resolution, resolution).numpy()
    
    if show_plots:
        # Create 2x4 subplot grid for visualization
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('Field Geology Properties', fontsize=16)
        
        # Plot properties
        im0 = axes[0, 0].contourf(x1_grid, x2_grid, thickness, levels=20, cmap='viridis')
        axes[0, 0].set_title('Thickness (ft)')
        plt.colorbar(im0, ax=axes[0, 0])
        
        im1 = axes[0, 1].contourf(x1_grid, x2_grid, porosity, levels=20, cmap='plasma')
        axes[0, 1].set_title('Porosity (fraction)')
        plt.colorbar(im1, ax=axes[0, 1])
        
        im2 = axes[0, 2].contourf(x1_grid, x2_grid, permeability, levels=20, cmap='inferno')
        axes[0, 2].set_title('Permeability (mD)')
        plt.colorbar(im2, ax=axes[0, 2])
        
        im3 = axes[0, 3].contourf(x1_grid, x2_grid, toc, levels=20, cmap='cividis')
        axes[0, 3].set_title('Total Organic Carbon (wt%)')
        plt.colorbar(im3, ax=axes[0, 3])
        
        im4 = axes[1, 0].contourf(x1_grid, x2_grid, water_saturation, levels=20, cmap='Blues')
        axes[1, 0].set_title('Water Saturation (fraction)')
        plt.colorbar(im4, ax=axes[1, 0])
        
        im5 = axes[1, 1].contourf(x1_grid, x2_grid, clay_volume, levels=20, cmap='Greens')
        axes[1, 1].set_title('Clay Volume (fraction)')
        plt.colorbar(im5, ax=axes[1, 1])
        
        im6 = axes[1, 2].contourf(x1_grid, x2_grid, depth, levels=20, cmap='terrain')
        axes[1, 2].set_title('Depth (ft)')
        plt.colorbar(im6, ax=axes[1, 2])
        
        # Add well locations if samples are available
        if field_props['df_full'] is not None:
            samples = field_props['df_full']
            for ax in axes.flatten():
                ax.scatter(samples['x'], samples['y'], s=10, c='black', alpha=0.5)
        
        plt.tight_layout()
        plt.show()
    
    return (grid_tensor, x1_grid, x2_grid, thickness, porosity, permeability, 
            toc, water_saturation, clay_volume, depth)

def create_basin_grid(basin_size, resolution, geojson_file=None):
    """
    Create a grid of points covering the basin.
    Duplicate of the function in simulation.py for convenience.
    
    Args:
        basin_size: Size of the basin in (x, y) kilometers
        resolution: Number of points along each dimension
        geojson_file: Optional path to a GeoJSON file defining the basin shape
        
    Returns:
        grid: Grid points [n_points, 2]
        x1_grid: x-coordinates meshgrid
        x2_grid: y-coordinates meshgrid
        mask: Optional binary mask for non-rectangular regions (None if not using geojson)
    """
    # Create grid of points
    x1 = np.linspace(0, basin_size[0], resolution)
    x2 = np.linspace(0, basin_size[1], resolution)
    x1_grid, x2_grid = np.meshgrid(x1, x2)
    grid = np.column_stack([x1_grid.flatten(), x2_grid.flatten()])
    grid_tensor = torch.tensor(grid, dtype=torch.float32)
    
    # Create mask from GeoJSON if provided
    mask = None
    if geojson_file is not None:
        try:
            import json
            from shapely.geometry import shape, Point
            
            # Load GeoJSON file
            with open(geojson_file, 'r') as f:
                geojson = json.load(f)
            
            # Get the geometry from the GeoJSON
            geometry = None
            if 'features' in geojson and len(geojson['features']) > 0:
                geometry = shape(geojson['features'][0]['geometry'])
            elif 'geometry' in geojson:
                geometry = shape(geojson['geometry'])
            else:
                geometry = shape(geojson)
            
            # Create mask based on points inside the geometry
            mask = np.zeros((resolution, resolution), dtype=bool)
            for i in range(resolution):
                for j in range(resolution):
                    point = Point(x1_grid[i, j], x2_grid[i, j])
                    mask[i, j] = geometry.contains(point)
        except ImportError:
            print("Warning: shapely package not installed. Cannot process GeoJSON file.")
            print("Install with: pip install shapely")
        except Exception as e:
            print(f"Error processing GeoJSON file: {e}")
    
    return grid_tensor, x1_grid, x2_grid, mask

def add_field_wells_from_samples(basin_gp, n_wells=10, seed=42, use_existing_coordinates=True):
    """
    Add wells to the basin model using sample data from the field.
    
    Args:
        basin_gp: BasinExplorationGP model
        n_wells: Number of wells to add (max 500, the size of the df_full dataset)
        seed: Random seed for well selection
        use_existing_coordinates: Whether to use the coordinates from df_full 
                                  or map them to the basin grid
    
    Returns:
        basin_gp: Updated BasinExplorationGP model with wells added
    """
    if field_props['df_full'] is None:
        load_field_data()
    
    np.random.seed(seed)
    df = field_props['df_full']
    
    # Select n_wells from the dataframe
    n_wells = min(n_wells, len(df))
    indices = np.random.choice(len(df), n_wells, replace=False)
    
    for i, idx in enumerate(indices):
        if use_existing_coordinates:
            # Use the actual field coordinates
            x = df.iloc[idx]['x']
            y = df.iloc[idx]['y']
            location = np.array([x, y])
        else:
            # Map field coordinates to basin grid (0 to basin_size)
            x_field = field_props['x']
            y_field = field_props['y']
            x_min, x_max = x_field[0], x_field[-1]
            y_min, y_max = y_field[0], y_field[-1]
            
            x_rel = (df.iloc[idx]['x'] - x_min) / (x_max - x_min)
            y_rel = (df.iloc[idx]['y'] - y_min) / (y_max - y_min)
            
            # Map to basin size
            x = x_rel * basin_gp.basin_size[0]
            y = y_rel * basin_gp.basin_size[1]
            location = np.array([x, y])
        
        # Get properties from the sample
        measurements = {
            'porosity': df.iloc[idx]['Porosity'],
            'permeability': df.iloc[idx]['Perm'],
            'thickness': df.iloc[idx]['Thickness'],
            'toc': df.iloc[idx]['TOC'],
            'water_saturation': df.iloc[idx]['SW'],
            'clay_volume': df.iloc[idx]['Vclay'],
            'depth': df.iloc[idx]['Depth']
        }
        
        basin_gp.add_well(location, measurements, well_name=f"Well_{i+1}")
    
    return basin_gp