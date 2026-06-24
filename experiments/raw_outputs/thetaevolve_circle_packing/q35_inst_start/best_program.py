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

    # Use a hybrid pattern: central cluster + outer ring for better packing
    # This is more suitable for 26 circles than pure concentric rings
    
    # Phase 1: Place central large circle
    centers[0] = [0.5, 0.5]
    
    # Phase 2: Place 5 circles in a tight inner cluster around center
    # This creates a more efficient packing than 8 in a ring
    inner_spacing = 0.28  # Reduced from 0.3 for tighter packing
    for i in range(5):
        angle = 2 * np.pi * i / 5
        centers[i + 1] = [0.5 + inner_spacing * np.cos(angle), 0.5 + inner_spacing * np.sin(angle)]
    
    # Phase 3: Place remaining 20 circles in outer rings
    # Use asymmetric spacing to avoid overlaps
    outer_spacing = 0.55
    
    # First outer ring: 10 circles
    for i in range(10):
        angle = 2 * np.pi * i / 10 + 0.1  # Offset to avoid alignment with inner ring
        centers[i + 6] = [0.5 + outer_spacing * np.cos(angle), 0.5 + outer_spacing * np.sin(angle)]
    
    # Second outer ring: 10 circles with different offset
    for i in range(10):
        angle = 2 * np.pi * i / 10 + 0.3  # Different offset
        centers[i + 16] = [0.5 + 0.62 * np.cos(angle), 0.5 + 0.62 * np.sin(angle)]
    
    # Ensure all centers are within bounds with small margin
    centers = np.clip(centers, 0.02, 0.98)
    
    # Add small random jitter to break symmetry and prevent duplicate positions
    # This is important for avoiding the exact overlap issue
    np.random.seed(42)  # For reproducibility
    jitter = 0.002 * np.random.randn(n, 2)
    centers = centers + jitter
    centers = np.clip(centers, 0.02, 0.98)

    # Compute maximum valid radii for this configuration
    radii = compute_max_radii(centers)

    # Calculate the sum of radii
    sum_radii = np.sum(radii)

    return centers, radii, sum_radii


def compute_max_radii(centers):
    """
    Compute the maximum possible radii for each circle position
    such that they don't overlap and stay within the unit square.
    Uses iterative refinement for better accuracy.

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates

    Returns:
        np.array of shape (n) with radius of each circle
    """
    n = centers.shape[0]
    radii = np.ones(n)

    # First, limit by distance to square borders (vectorized)
    x, y = centers[:, 0], centers[:, 1]
    radii = np.minimum(radii, np.minimum(np.minimum(x, y), np.minimum(1 - x, 1 - y)))

    # Iteratively refine radii based on pairwise constraints
    for _ in range(3):  # Multiple passes for better accuracy
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                
                # If current radii would cause overlap
                if radii[i] + radii[j] > dist:
                    # Scale both radii proportionally
                    scale = dist / (radii[i] + radii[j])
                    radii[i] *= scale
                    radii[j] *= scale
    
    # Ensure radii are non-negative
    radii = np.maximum(radii, 1e-6)

    return radii


# EVOLVE-BLOCK-END


def run_circle_packing():
    centers, radii, sum_radii = construct_packing()
    
    # Apply local optimization to improve the initial packing
    # This is critical for resolving overlaps in the initial configuration
    centers, radii = local_optimize(centers, radii, n=N_CIRCLES)
    
    # Validate solution
    is_valid, validation_msg = validate_solution(centers, radii)
    
    current_solution = {'data': (centers.tolist(), radii.tolist())}
    save_search_results(best_perfect_solution=None, current_solution=current_solution,
                       n_circles=N_CIRCLES, target_value=TARGET_VALUE)

    return centers, radii, sum_radii, is_valid



def local_optimize(centers, radii, n):
    """
    Apply local optimization to refine positions and radii.
    Uses greedy local search with multiple strategies to improve packing.
    
    Args:
        centers: Current center positions
        radii: Current radii
        n: Number of circles
        
    Returns:
        Optimized centers and radii
    """
    centers = centers.copy()
    radii = radii.copy()
    
    # Strategy 1: Resolve overlapping circles by moving them apart
    for iteration in range(20):
        improved = False
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                min_dist = radii[i] + radii[j]
                
                if dist <= min_dist:
                    # Circles are overlapping or touching - need to separate
                    direction = centers[i] - centers[j]
                    direction = direction / np.linalg.norm(direction)
                    
                    # Move both circles apart
                    move_amount = 0.003  # Small movement
                    centers[i] += move_amount * direction
                    centers[j] -= move_amount * direction
                    
                    # Ensure bounds
                    centers[i] = np.clip(centers[i], 0.02, 0.98)
                    centers[j] = np.clip(centers[j], 0.02, 0.98)
                    
                    improved = True
            
            # Check for duplicate positions (distance < tolerance)
            for j in range(n):
                if i == j:
                    continue
                dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                if dist < 1e-8:
                    # Move circle i away from j
                    direction = centers[i] - centers[j]
                    direction = direction / np.linalg.norm(direction)
                    centers[i] += 0.005 * direction
                    centers[i] = np.clip(centers[i], 0.02, 0.98)
                    improved = True
        
        if not improved:
            break
    
    # Strategy 2: Greedy radius optimization - expand radii where possible
    for iteration in range(15):
        improved = False
        for i in range(n):
            # Calculate current limiting factors for circle i
            min_radius = min(centers[i][0], centers[i][1], 1 - centers[i][0], 1 - centers[i][1])
            
            # Check constraints from other circles
            for j in range(n):
                if i == j:
                    continue
                dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                max_radius = dist - radii[j]
                min_radius = min(min_radius, max_radius)
            
            # Try to increase radius
            if min_radius > radii[i] * 0.99:  # Allow 1% growth
                radii[i] = min_radius
                improved = True
        
        if not improved:
            break
    
    # Recompute radii after optimization (final refinement)
    radii = compute_max_radii(centers)
    
    return centers, radii

def validate_solution(centers, radii):
    """
    Validate that all circles are within bounds and don't overlap.
    
    Returns:
        Tuple of (is_valid, validation_message)
    """
    n = len(centers)
    tolerance = 1e-9  # Very small tolerance for numerical precision
    
    # Check bounds
    for i, (x, y) in enumerate(centers):
        if x < 0.0 or x > 1.0 or y < 0.0 or y > 1.0:
            return False, f"Circle {i} outside bounds: ({x:.6f}, {y:.6f})"
        if radii[i] <= 0.0 or radii[i] > 0.5:
            return False, f"Circle {i} has invalid radius: {radii[i]}"
    
    # Check overlaps with tighter tolerance
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            # Use smaller tolerance for overlap detection
            if dist < tolerance:
                return False, f"Circles {i} and {j} at same position: dist={dist:.10f}"
            if radii[i] + radii[j] > dist + tolerance:
                return False, f"Circles {i} and {j} overlap: dist={dist:.10f}, r1+r2={radii[i]+radii[j]:.10f}"
    
    return True, "Valid solution"

if __name__ == "__main__":

    ######## get parameters from config ########
    from openevolve.modular_utils.file_io_controller import save_search_results
    from openevolve.modular_utils.evaluation_controller import get_current_problem_config
    PROBLEM_CONFIG = get_current_problem_config()
    N_CIRCLES = PROBLEM_CONFIG['core_parameters']['n_circles']
    TARGET_VALUE = PROBLEM_CONFIG['target_value']
    PROBLEM_TYPE = PROBLEM_CONFIG['problem_type']
    ###############################################

    centers, radii, sum_radii, is_valid = run_circle_packing()
    print(f"\\nGenerated {PROBLEM_TYPE} packing (constructor approach):")
    print(f"Sum of radii: {sum_radii:.10f}")
    print(f"Target: {TARGET_VALUE} ({100*sum_radii/TARGET_VALUE:.1f}% of target)")
    print(f"Solution valid: {is_valid}")
    
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