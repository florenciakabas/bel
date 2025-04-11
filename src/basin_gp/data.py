import numpy as np
import torch

def prepare_training_data(wells, n_properties):
    """
    Prepare training data from wells.
    
    Args:
        wells: List of well data dictionaries
        n_properties: Number of properties
        
    Returns:
        X: Well locations [n_wells, 2]
        Y: Property measurements [n_wells, n_properties]
        mask: Boolean mask for valid measurements [n_wells, n_properties]
    """
    n_wells = len(wells)
    
    X = np.zeros((n_wells, 2))
    Y = np.zeros((n_wells, n_properties))
    mask = np.zeros((n_wells, n_properties), dtype=bool)
    
    for i, well in enumerate(wells):
        X[i] = well['location']
        Y[i] = well['measurements']
        mask[i] = well['mask']
        
    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.float32)
    
    # Handle missing values (replace NaNs with dummy values that will be masked)
    Y_tensor[torch.isnan(Y_tensor)] = 0.0
        
    return X_tensor, Y_tensor, mask