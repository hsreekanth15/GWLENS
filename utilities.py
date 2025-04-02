import numpy as np
from tqdm import tqdm
from scipy.interpolate import RegularGridInterpolator
import h5py
import os



def compute_and_store_results_2D(hdf5_file, w_values, y_values,ampf):
    with h5py.File(hdf5_file, 'w') as file:
        
        #The values of the dimensionless frequency  and y_values will be stored here
        file.create_dataset("w_values", data=w_values)
        file.create_dataset("y_values", data=y_values)

        #Result of the interpolation will be stored here
        dset = file.create_dataset("results", (len(w_values), len(y_values),2), maxshape=(None, None, None), dtype='float64')
        print(f"Storting results in {hdf5_file}")

        total_iterations = len(w_values) * len(y_values)
        progress = tqdm(total=total_iterations, desc="Storing Results", unit="calc")
        
        for i, w in enumerate(w_values):
            for j, y in enumerate(y_values):
                # Assume ampf computes a complex number
                result = ampf(w, y)
                # Store real and imaginary parts separately
                dset[i, j, 0] = result.real
                dset[i, j, 1] = result.imag
                progress.update(1)
        progress.close()        


def compute_and_store_results_3D(hdf5_file, w_values, y_values, rs_values,ampf):
    with h5py.File(hdf5_file, 'w') as file:
        #The values of the dimensionless frequency  and y_values will be stored here
        print("Running Interpolator")
        file.create_dataset("w_values", data=w_values)
        file.create_dataset("y_values", data=y_values)
        file.create_dataset("rs_values", data=rs_values)

        dset = file.create_dataset("results", (len(w_values), len(y_values),len(rs_values), 2), maxshape=(None, None, None, None), dtype='float64')
        total_iterations = len(w_values) * len(y_values) * len(rs_values)
        progress = tqdm(total=total_iterations, desc="Storing Results", unit="calc")
        for i, w in enumerate(w_values):
            for j, y in enumerate(y_values):
                for k, rs in enumerate(rs_values):
                    # Compute the result using the ampf function
                    result = ampf(w, y, rs)
                    # Store the real and imaginary parts of the result separately in the dataset
                    dset[i, j, k, 0] = result.real
                    dset[i, j, k, 1] = result.imag
                    progress.update(1)
        progress.close() 



def load_and_interpolate_2D(interpolation_file, w_query, y_query):
    with h5py.File(interpolation_file, 'r') as file:
        w_values = file['w_values'][:]
        y_values = file['y_values'][:]
        data = file['results'][:]  # shape: (len(w), len(y), 2)

    # Separate real and imaginary components
    real_data = data[:, :, 0]
    imag_data = data[:, :, 1]

    # Create interpolators
    real_interp = RegularGridInterpolator((w_values, y_values), real_data)
    imag_interp = RegularGridInterpolator((w_values, y_values), imag_data)

    # Evaluate interpolation at the query point(s)
    point = np.array([[w_query, y_query]])  # shape (1, 2)
    result = real_interp(point)[0] + 1j * imag_interp(point)[0]
    return result

def load_and_interpolate_3D(interpolation_file, w_values, y_values, rs_values, w_query, y_query, rs_query):
    with h5py.File(interpolation_file, 'r') as file:
        w_values = file['w_values'][:]
        y_values = file['y_values'][:]
        rs_values = file['rs_values'][:]
        data = file['results'][:]  # shape: (len(w), len(y), len(rs), 2)

    real_data = data[:, :, :, 0]
    imag_data = data[:, :, :, 1]

    # Create interpolators
    real_interp = RegularGridInterpolator((w_values, y_values, rs_values), real_data)
    imag_interp = RegularGridInterpolator((w_values, y_values, rs_values), imag_data)

    # Evaluate at point
    point = np.array([[w_query, y_query, rs_query]])
    result = real_interp(point)[0] + 1j * imag_interp(point)[0]
    return result

def ensure_1d_array(w):
    """
    Ensures that the input 'w' is a 1D numpy array.

    Args:
    w: Input variable which could be a scalar or any numpy ndarray.

    Returns:
    np.ndarray: A 1D numpy array, whether originally an array or converted from a scalar.
    """
    # Check if 'w' is an ndarray and has one dimension
    if isinstance(w, np.ndarray):
        if w.ndim == 0:
            # Convert 0D array (scalar in array form) to 1D
            return np.array([w.item()])
        elif w.ndim == 1:
            # Already a 1D array
            return w
        else:
            # It's an array but not 1D, handle as needed or raise an error
            raise ValueError("Input 'w' is neither a scalar nor a 1D array. It has multiple dimensions.")
    else:
        # If 'w' is a scalar (not an array), convert it to a 1D array
        return np.array([w])
    


def grid_interpolate_2D(hdf5_file, w_array, y_array):
    with h5py.File(hdf5_file, 'r') as file:
        w_values = file['w_values'][:]
        y_values = file['y_values'][:]
        data = file['results'][:]  # shape: (len(w), len(y), 2)

    # Build interpolators
    real_interp = RegularGridInterpolator((w_values, y_values), data[:, :, 0])
    imag_interp = RegularGridInterpolator((w_values, y_values), data[:, :, 1])

    # Create a 2D meshgrid of w and y
    W, Y = np.meshgrid(w_array, y_array, indexing='ij')  # shape (len(w), len(y))

    # Flatten the meshgrid into (N, 2) query points
    points = np.column_stack([W.ravel(), Y.ravel()])  # shape (N, 2)

    # Interpolate real and imaginary parts
    real_vals = real_interp(points)
    imag_vals = imag_interp(points)

    # Reshape result back to (len(w), len(y)) and return as complex array
    result = real_vals.reshape(W.shape) + 1j * imag_vals.reshape(W.shape)
    return result