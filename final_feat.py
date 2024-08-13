import time
import matplotlib.pyplot as plt
import numpy as np
import itertools

def generate_points(num_points, num_lanes=4, lane_width=0.25):
    """
    Generates random points within specified lanes.

    Parameters:
    num_points: Number of points to generate.
    num_lanes: Number of lanes.
    lane_width: Width of each lane.

    Returns: Array of generated points.
    """
    points = []
    for _ in range(num_points):
        lane = np.random.choice(num_lanes)
        x = np.random.uniform(0, 1)
        y = np.random.uniform(lane * lane_width, (lane + 1) * lane_width)
        points.append([x, y])
    return np.array(points)

def calculate_features(points):
    """
    Calculates features for given points using triangles formed by the points.

    Parameters:
    points: Array of points.

    Returns: Array of features.
    List of triangle indices.
    """

    indices = range(len(points))
    triangles = list(itertools.combinations(indices, 3))
    features = []
    for triangle in triangles:
        pts = np.array([points[i] for i in triangle])
        edge_lengths = np.linalg.norm(pts - np.roll(pts, -1, axis=0), axis=1)
        area = np.linalg.norm(np.cross(pts[1] - pts[0], pts[2] - pts[0])) / 2
        angles = np.arccos(np.clip((np.sum(edge_lengths**2) - 2 * np.prod(edge_lengths)) / (2 * np.prod(edge_lengths)), -1.0, 1.0))
        features.append(np.hstack([edge_lengths, angles, area]))
    return np.array(features), triangles

def centroid(points):
    """
    Calculates the centroid of given points.

    Parameters:
    points: Array of points.

    Returns: Centroid of the points.
    """
    return np.mean(points, axis=0)

def optimal_rigid_transform(source_points, target_points):
    """
    Calculates the optimal rigid transformation between source and target points.

    Parameters:
    source_points: Source points (Translated + Noisey points)
    target_points: Target points (Original points)

    Returns:
    Rotation matrix.
    Translation vector.
    """
    source_center = centroid(source_points)
    target_center = centroid(target_points)
    source_centered = source_points - source_center
    target_centered = target_points - target_center
    u, _, vt = np.linalg.svd(np.dot(target_centered.T, source_centered))
    R = np.dot(vt.T, u.T)
    if np.linalg.det(R) < 0:
        vt[-1, :] *= -1
        R = np.dot(vt.T, u.T)
    t = target_center - np.dot(source_center, R)
    return R, t

def match_features(source_features, target_features, tolerance=0.05):
    """
    Matches features between source and target based on a tolerance.

    Parameters:
    source_features: Source features.
    target_features: Target features.
    tolerance: Matching tolerance.

    Returns: List of matched feature indices.
    """
    matches = []
    for i, source_feature in enumerate(source_features):
        for j, target_feature in enumerate(target_features):
            distance = np.linalg.norm(source_feature - target_feature)
            if distance < tolerance:
                matches.append((i, j))
    return matches

def icp(source, target, features_func, match_func, iterations=100, tolerance=1e-10):
    """
    ICP algorithm for aligning source points to target points.

    Parameters:
    source: Source points.
    target: Target points.
    features_func (function): Function to calculate features of points.
    match_func (function): Function to match features between source and target.
    iterations: Number of iterations.
    tolerance: Convergence tolerance.

    Returns: Aligned source points.
    List of errors per iteration.
    """
    source_features, source_triangles = features_func(source)
    target_features, target_triangles = features_func(target)
    errors = []
    prev_error = np.inf
    for i in range(iterations):
        matches = match_func(source_features, target_features)
        if not matches:
            print("No matches found.")
            return source, errors
        
        source_matched = []
        target_matched = []
        for source_idx, target_idx in matches:
            for tri_index in source_triangles[source_idx]:
                source_matched.append(source[tri_index])
            for tri_index in target_triangles[target_idx]:
                target_matched.append(target[tri_index])
        source_matched = np.array(source_matched)
        target_matched = np.array(target_matched)
        R, t = optimal_rigid_transform(source_matched, target_matched)
        source = np.dot(source, R) + t
        current_error = np.linalg.norm(source_matched - target_matched)
        errors.append(current_error)
        
        if abs(prev_error - current_error) < tolerance:
            print(f"Converged after {i+1} iterations.")
            break
        prev_error = current_error
    return source, errors

def add_noise(points, noise_points=1, noise_magnitude=0.2):
    """
    Adds noise to given points.

    Parameters:
    points: Original points.
    noise_points: Number of noise points to add.
    noise_magnitude: Magnitude of noise.

    Returns: Points with added noise.
    Indices of noise points.
    """
    noise = np.random.uniform(-noise_magnitude, noise_magnitude, (noise_points, 2))
    noise_indices = np.random.choice(np.arange(len(points)), noise_points, replace=True)
    noisy_points = points.copy()
    noisy_points = np.vstack((noisy_points, points[noise_indices] + noise))
    return noisy_points, np.arange(len(points), len(noisy_points))

def delete_random_points(points, delete_points):
    """
    Deletes random points from the given points.

    Parameters: 
    points: Original points.
    delete_points: Number of points to delete.

    Returns: Points after deletion.
    """
    if delete_points >= len(points):
        raise ValueError("Number of points to delete is greater than or equal to the number of points available.")
    delete_indices = np.random.choice(len(points), delete_points, replace=False)
    remaining_points = np.delete(points, delete_indices, axis=0)
    return remaining_points

def plot_points(num_points, noise_points=0, translated_points=0):
    """
    Plots original, translated, and ICP aligned points with noise.

    Parameters:
    num_points: Number of original points.
    noise_points: Number of noise points.
    translated_points: Number of translated points.
    """

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    original_points = generate_points(num_points)

    for point in original_points:
        axs[0].plot(point[0], point[1], 'ro', label='Original Points' if len(axs[0].lines) == 0 else "")

    # Draw lane dividers
    for i in range(1, 4):
        axs[0].plot([0, 1], [i * 0.25, i * 0.25], 'k--')

    axs[0].set_xlim(0, 1)
    axs[0].set_ylim(0, 1)
    axs[0].set_aspect('equal', adjustable='box')
    axs[0].set_title('Original Points')
    axs[0].legend()

    # Generate a single random translation
    translation_x = np.random.uniform(-0.2, 0.2)
    translation_y = np.random.uniform(-0.2, 0.2)

    translated_points_array = original_points + np.array([translation_x, translation_y])

    # Calculate the number of points to delete
    delete_points = max(0, num_points - translated_points)

    # Delete random points from the translated points
    translated_points_array = delete_random_points(translated_points_array, delete_points)

    for point in original_points:
        axs[1].plot(point[0], point[1], 'ro', label='Original Points' if len(axs[1].lines) == 0 else "")
    for point in translated_points_array:
        axs[1].plot(point[0], point[1], 'bo', label='Translated Points' if len(axs[1].lines) == (num_points - delete_points) else "")

    # Draw lane dividers
    for i in range(1, 4):
        axs[1].plot([0, 1], [i * 0.25, i * 0.25], 'k--')

    # Add noise to the translated points
    noisy_translated_points, noise_indices = add_noise(translated_points_array, noise_points)

    axs[1].scatter(noisy_translated_points[noise_indices, 0], noisy_translated_points[noise_indices, 1], color='purple', label='Noise Points')

    axs[1].set_xlim(0, 1)
    axs[1].set_ylim(0, 1)
    axs[1].set_aspect('equal', adjustable='box')
    axs[1].set_title('Original and Translated Points with Noise')
    axs[1].legend()

    start_time = time.time()
    aligned_points, _ = icp(noisy_translated_points, original_points, calculate_features, match_features)
    end_time = time.time()
    
    print(f"ICP execution time: {end_time - start_time} seconds")
    for point in original_points:
        axs[2].plot(point[0], point[1], 'ro', label='Original Points' if len(axs[2].lines) == 0 else "")

    # Plot aligned points with their original color
    for i, aligned_point in enumerate(aligned_points):
        if i in noise_indices:
            axs[2].scatter(aligned_point[0], aligned_point[1], color='purple', label='Aligned Noise Points' if len(axs[2].lines) == 0 else "")
        else:
            axs[2].scatter(aligned_point[0], aligned_point[1], color='blue', label='Aligned Translated Points' if len(axs[2].lines) == 0 else "")

    # Draw lane dividers
    for i in range(1, 4):
        axs[2].plot([0, 1], [i * 0.25, i * 0.25], 'k--')

    axs[2].set_xlim(0, 1)
    axs[2].set_ylim(0, 1)
    axs[2].set_aspect('equal', adjustable='box')
    axs[2].set_title('ICP Aligned Points with Noise')
    axs[2].legend()

    plt.tight_layout()
    plt.show()

def main():
    num_points = int(input("Enter the number of points: "))
    translated_points = int(input("How many translated points do you want? "))
    noise_points = int(input("Enter the number of noise points: "))
    plot_points(num_points, noise_points, translated_points)

if __name__ == "__main__":
    main()
