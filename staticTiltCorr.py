import numpy as np

def staticTiltCorr(u, v, w=None, method='rot2'):
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
