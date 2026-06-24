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

    # Place circles in a structured pattern
    # This is a simple pattern - evolution will improve this

    # First, place a large circle in the center
    centers[0] = [0.5, 0.5]

    # Place circles in a randomized asymmetric layout
    centers = np.random.rand(n, 2)
    # Ensure all centers are inside the unit square
    centers = np.clip(centers, 0.01, 0.99)
    
    # Apply a small perturbation to avoid symmetry
    np.random.seed(42)
    centers += np.random.normal(0, 0.05, centers.shape)
    centers = np.clip(centers, 0.01, 0.99)

    # Additional positioning adjustment to make sure all circles
    # are inside the square and don't overlap
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

    # Then, limit by distance to other circles and square borders
    for i in range(n):
        x, y = centers[i]
        # Distance to borders
        radii[i] = min(x, y, 1 - x, 1 - y)

    # Apply local optimization to refine radii
    for _ in range(100):
        for i in range(n):
            x, y = centers[i]
            # Distance to square borders
            max_radius = min(x, y, 1 - x, 1 - y)
            # Distance to other circles
            for j in range(n):
                if i != j:
                    dx, dy = centers[j] - centers[i]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < 1e-6:
                        continue
                    if radii[i] + radii[j] > dist:
                        # Adjust radii to avoid overlap
                        scale = dist / (radii[i] + radii[j])
                        radii[i] *= scale
                        radii[j] *= scale
            # Ensure radii do not exceed the square boundary
            radii[i] = min(radii[i], max_radius)

    return radii


# EVOLVE-BLOCK-END


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