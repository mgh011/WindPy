import numpy as np

def dynamicTiltCorr(u, v, w, roll, pitch, yaw):
    """
    Applies a rotation based on roll, pitch, and yaw angles to the wind velocity components (u, v, w)
    for each timestamp. The components u, v, and w are defined in the instrument's local coordinate system.

    Args:
        u (numpy.ndarray): 1-D array of wind components along the instrument's x-axis.
        v (numpy.ndarray): 1-D array of wind components along the instrument's y-axis.
        w (numpy.ndarray): 1-D array of wind components along the instrument's z-axis.
        roll (numpy.ndarray): 1-D array of roll angles in degrees (rotation around x-axis).
        pitch (numpy.ndarray): 1-D array of pitch angles in degrees (rotation around y-axis).  
        yaw (numpy.ndarray): 1-D array of yaw angles in degrees (rotation around z-axis).
    
    Returns:
       U_rotated, V_rotated, W_rotated (numpy.ndarray): Arrays of rotated wind components in the global coordinate system.

    Author: M. Ghirardelli - Last modified: 13-08-2024
    """
    
    # Convert angles to radians
    roll_rad = np.radians(roll)
    pitch_rad = np.radians(pitch)
    yaw_rad = np.radians(yaw)
    
    # Precompute sin and cos for all angles
    cos_roll = np.cos(roll_rad)
    sin_roll = np.sin(roll_rad)
    cos_pitch = np.cos(pitch_rad)
    sin_pitch = np.sin(pitch_rad)
    cos_yaw = np.cos(yaw_rad)
    sin_yaw = np.sin(yaw_rad)
    
    # Compute the rotation matrix components
    # Apply yaw rotation around z-axis
    R11 = cos_yaw * cos_pitch
    R12 = cos_yaw * sin_pitch * sin_roll - sin_yaw * cos_roll
    R13 = cos_yaw * sin_pitch * cos_roll + sin_yaw * sin_roll

    R21 = sin_yaw * cos_pitch
    R22 = sin_yaw * sin_pitch * sin_roll + cos_yaw * cos_roll
    R23 = sin_yaw * sin_pitch * cos_roll - cos_yaw * sin_roll

    R31 = -sin_pitch
    R32 = cos_pitch * sin_roll
    R33 = cos_pitch * cos_roll
    
    # Apply the rotation matrix to the wind components
    U = R11 * u + R12 * v + R13 * w
    V = R21 * u + R22 * v + R23 * w
    W = R31 * u + R32 * v + R33 * w
    
    return U, V, W
