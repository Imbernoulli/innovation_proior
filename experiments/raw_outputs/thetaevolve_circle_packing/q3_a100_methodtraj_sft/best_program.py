# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=N_CIRCLES circles"""
import numpy as np


def construct_packing():
    """
    Construct a specific arrangement of N_CIRCLES circles in a unit square
    that attempts to maximize the sum of their radii.

    Returns:
        Tuple of (centers, radii, sum_of_radii)
        centers: np.array of shape (N_CIRCLES, 2) with (x, y) coordinates
        radii: np.array of shape (N_CIRCLES) with radius of each circle
        sum_of_radii: Sum of all radii
    """
    # Initialize arrays for N_CIRCLES circles
    n = N_CIRCLES
    centers = np.zeros((n, 2))

    # Create a grid-based pattern with controlled spacing
    # This pattern allows for better spacing and more efficient use of space
    
    # Define spacing parameters
    spacing = 0.2
    row_count = 5
    col_count = 5
    
    # Place the large central circle
    centers[0] = [0.5, 0.5]
    
    # Place circles in a grid pattern
    for i in range(row_count):
        for j in range(col_count):
            if i == 2 and j == 2:
                continue  # Skip the center circle
            
            # Calculate grid position
            x = 0.5 + (j - 2) * spacing
            y = 0.5 + (i - 2) * spacing
            centers[1 + i * col_count + j] = [x, y]

    # Add a small random perturbation to the initial layout
    centers += np.random.uniform(-0.05, 0.05, centers.shape)

    # Additional positioning adjustment to make sure all circles
    # are inside the square and don't overlap
    # First perform a more sophisticated local optimization
    centers = local_optimization(centers, max_iterations=200, learning_rate=0.02)

    # Clip to ensure everything is inside the unit square
    centers = np.clip(centers, 0.01, 0.99)

    # Compute maximum valid radii for this configuration
    radii = compute_max_radii(centers)

    # Calculate the sum of radii
    sum_radii = np.sum(radii)

    return centers, radii, sum_radii


def compute_max_radii(centers):
    """
    Compute the maximum possible radii for each circle position
    such that they don't overlap and stay within the unit square.

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates

    Returns:
        np.array of shape (n) with radius of each circle
    """
    n = centers.shape[0]
    radii = np.ones(n)

    # First, limit by distance to square borders
    for i in range(n):
        x, y = centers[i]
        # Distance to borders
        radii[i] = min(x, y, 1 - x, 1 - y)

    # Then, limit by distance to other circles
    # Each pair of circles with centers at distance d can have
    # sum of radii at most d to avoid overlap
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))

            # If current radii would cause overlap
            if radii[i] + radii[j] > dist:
                # Scale both radii proportionally with a stability factor
                scale = dist / (radii[i] + radii[j]) * 0.95
                radii[i] *= scale
                radii[j] *= scale

    # Apply a small numerical stability adjustment
    radii = np.clip(radii, 0, 1.0)
    return radii


# EVOLVE-BLOCK-END

def local_optimization(centers, max_iterations=200, learning_rate=0.02):
    """
    Perform a more sophisticated local optimization of circle positions to
    improve packing density. This algorithm better balances circle spacing
    and edge effects while maintaining non-overlapping constraints.
    """
    n = centers.shape[0]
    centers_opt = centers.copy()
    
    # Optimization parameters
    edge_penalty = 1.2
    overlap_penalty = 1.5
    
    for _ in range(max_iterations):
        # Initialize gradients
        grads = np.zeros_like(centers_opt)
        
        # Compute gradients from border constraints
        for i in range(n):
            x, y = centers_opt[i]
            # Distance to borders
            border_dist = np.array([x, y, 1 - x, 1 - y])
            min_border = np.min(border_dist)
            
            # Gradient points toward the border with penalty for edge proximity
            border_dir = centers_opt[i] - np.array([0.5, 0.5])
            grads[i] = -np.sign(border_dir) * 0.1 * np.exp(-min_border * 5)
            
            # If we're touching a border, push the circle away with edge penalty
            if min_border < 1e-6:
                grads[i] = np.sign(border_dir) * 0.1 * edge_penalty
                
        # Compute gradients from circle-circle constraints
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers_opt[i] - centers_opt[j]) ** 2))
                # If circles are overlapping, push them apart with overlap penalty
                if dist < 1e-6:
                    overlap_dir = centers_opt[i] - centers_opt[j]
                    grads[i] += overlap_dir / np.sqrt(np.sum(overlap_dir ** 2)) * 0.1 * overlap_penalty
                    grads[j] -= overlap_dir / np.sqrt(np.sum(overlap_dir ** 2)) * 0.1 * overlap_penalty
                
        # Update positions with adaptive learning rate
        centers_opt += grads * learning_rate * np.clip(1 - (1 - dist / 0.3), 0, 1)
        
        # Clip to ensure everything is inside the unit square
        centers_opt = np.clip(centers_opt, 0.01, 0.99)
        
    return centers_opt


def run_circle_packing():
    centers, radii, sum_radii = construct_packing()
    current_solution = {'data': (centers.tolist(), radii.tolist())}
    save_search_results(best_perfect_solution=None, current_solution=current_solution,
                       n_circles=N_CIRCLES, target_value=TARGET_VALUE)

    return centers, radii, sum_radii



if __name__ == "__main__":

    ######## get parameters from config ########
    from openevolve.modular_utils.file_io_controller import save_search_results
    from openevolve.modular_utils.evaluation_controller import get_current_problem_config
    PROBLEM_CONFIG = get_current_problem_config()
    N_CIRCLES = PROBLEM_CONFIG['core_parameters']['n_circles']
    TARGET_VALUE = PROBLEM_CONFIG['target_value']
    PROBLEM_TYPE = PROBLEM_CONFIG['problem_type']
    ###############################################

    centers, radii, sum_radii = run_circle_packing()
    print(f"\\nGenerated {PROBLEM_TYPE} packing (constructor approach):")
    print(f"Sum of radii: {sum_radii:.10f}")
    print(f"Target: {TARGET_VALUE} ({100*sum_radii/TARGET_VALUE:.1f}% of target)")
    
    # Optional: Visualize (requires matplotlib)
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.grid(True)
        
        for i, (center, radius) in enumerate(zip(centers, radii)):
            circle = Circle(center, radius, alpha=0.5)
            ax.add_patch(circle)
            ax.text(center[0], center[1], str(i), ha="center", va="center", fontsize=8)
        
        plt.title(f"Circle Packing Constructor (n={N_CIRCLES}, sum={sum_radii:.6f})")
        plt.savefig("circle_packing_constructor.png")
        print("Visualization saved as circle_packing_constructor.png")
        
    except ImportError:
        print("Matplotlib not available - skipping visualization")