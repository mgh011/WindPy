import numpy as np

def static_corr(u, v, w=None, method='rot2'):
    """
    Corrects the wind velocity components for non-zero tilt angles and computes them in the wind-based coordinate system.

    Args:
        u (numpy.ndarray): 1-D array. First wind component (first horizontal component).
        v (numpy.ndarray): 1-D array. Second wind component (second horizontal component).
        w (numpy.ndarray, optional): 1-D array. Third wind component (vertical component). If not provided, only horizontal rotation is applied.
        method (str, optional): Method to use for tilt correction ('rot1', 'rot2', 'rot3'). Default is 'rot2'.
    
    Returns:
        U1 (numpy.ndarray): 1-D array. Along wind component.
        V1 (numpy.ndarray): 1-D array. Across wind component.
        W1 (numpy.ndarray or None): 1-D array or None. Vertical wind component. None if not provided.

    Author: M. Ghirardelli - Last modified: 13-08-2024 
    """
    
    # If w is not provided, only horizontal plane rotation (rot1)
    if method.lower() == 'rot1' or w is None:
        A0 = np.array([u, v])
        R = np.arctan2(np.mean(A0[1]), np.mean(A0[0]))  # Calculate mean wind direction angle
        R1 = np.array([[np.cos(R), -np.sin(R)], [np.sin(R), np.cos(R)]])  # Rotation matrix
        A1 = np.dot(R1, A0)  # Apply rotation
        U1 = A1[0]  # Along-wind component
        V1 = A1[1]  # Across-wind component
        W1 = None
        return U1, V1, W1

    # Double rotation method (rot2)
    elif method.lower() == 'rot2':
        # First rotation (around z-axis)
        A0 = np.array([u, v])
        R1_angle = np.arctan2(np.mean(A0[1]), np.mean(A0[0]))  # Mean wind direction in horizontal plane
        R1 = np.array([[np.cos(R1_angle), np.sin(R1_angle)], [-np.sin(R1_angle), np.cos(R1_angle)]])  # Rotation matrix
        A1 = np.dot(R1, A0)
        u1 = A1[0]
        V1 = A1[1]

        # Second rotation (around y-axis)
        A2 = np.array([u1, w])
        R2_angle = np.arctan2(np.mean(A2[1]), np.mean(A2[0]))  # Tilt correction in vertical plane
        R2 = np.array([[np.cos(R2_angle), np.sin(R2_angle)], [-np.sin(R2_angle), np.cos(R2_angle)]])  # Rotation matrix
        A3 = np.dot(R2, A2)
        U1 = A3[0]  # Along-wind component after second rotation
        W1 = A3[1]  # Vertical wind component after second rotation

        return U1, V1, W1

    # Triple rotation method (rot3)
    elif method.lower() == 'rot3':
        # First rotation (around z-axis)
        A0 = np.array([u, v])
        R1_angle = np.arctan2(np.mean(A0[1]), np.mean(A0[0]))  # Mean wind direction in horizontal plane
        R1 = np.array([[np.cos(R1_angle), np.sin(R1_angle)], [-np.sin(R1_angle), np.cos(R1_angle)]])  # Rotation matrix
        A1 = np.dot(R1, A0)
        u1 = A1[0]
        v1 = A1[1]

        # Second rotation (around y-axis)
        A2 = np.array([u1, w])
        R2_angle = np.arctan2(np.mean(A2[1]), np.mean(A2[0]))  # Tilt correction in vertical plane
        R2 = np.array([[np.cos(R2_angle), np.sin(R2_angle)], [-np.sin(R2_angle), np.cos(R2_angle)]])  # Rotation matrix
        A3 = np.dot(R2, A2)
        U1 = A3[0]  # Along-wind component after second rotation
        w1 = A3[1]  # Intermediate vertical wind component

        # Third rotation (around x-axis)
        A4 = np.array([v1, w1])
        covVW = np.mean(v1 * w1)
        diffVW = np.var(v1) - np.var(w1)
        R3_angle = 0.5 * np.arctan2(2 * covVW, diffVW)  # Angle for third rotation
        R3 = np.array([[np.cos(R3_angle), np.sin(R3_angle)], [-np.sin(R3_angle), np.cos(R3_angle)]])  # Rotation matrix
        A5 = np.dot(R3, A4)
        V1 = A5[0]  # Across-wind component after third rotation
        W1 = A5[1]  # Final vertical wind component after third rotation

        return U1, V1, W1

    else:
        raise ValueError("Invalid method specified. Use 'rot1', 'rot2', or 'rot3'.")
        

def dynamic_corr(u, v, w, roll, pitch, yaw):
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

def planar_fit():
    return
