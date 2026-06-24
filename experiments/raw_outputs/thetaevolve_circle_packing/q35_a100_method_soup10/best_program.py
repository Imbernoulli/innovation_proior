# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=N_CIRCLES circles"""
import numpy as np


def construct_packing():
    """
    Construct a specific arrangement of N_CIRCLES circles in a unit square
    that attempts to maximize the sum of their radii.

    Uses hexagonal-like packing with corner optimization for better space utilization.

    Returns:
        Tuple of (centers, radii, sum_of_radii)
        centers: np.array of shape (N_CIRCLES, 2) with (x, y) coordinates
        radii: np.array of shape (N_CIRCLES) with radius of each circle
        sum_of_radii: Sum of all radii
    """
    # Initialize arrays for N_CIRCLES circles
    n = N_CIRCLES
    centers = np.zeros((n, 2))
    
    # Use hexagonal-like packing with optimized parameters for n=26
    # This pattern places circles in staggered rows for better space utilization
    
    # Row 0: 4 circles in corners (optimized for larger radii at boundaries)
    row_spacing = 0.25  # Optimized spacing for hexagonal-like pattern
    
    # Place 4 corner circles first (these can be larger due to boundary constraints)
    # Place corner circles with better positioning for larger radii
    # Moving slightly inward to allow for larger radii while maintaining packing efficiency
    corners = [(0.12, 0.12), (0.88, 0.12), (0.12, 0.88), (0.88, 0.88)]
    for i, corner in enumerate(corners):
        centers[i] = corner
    
    # Place center circle
    centers[4] = [0.5, 0.5]
    
    # Fill remaining positions with optimized hexagonal staggered pattern
    # This creates more efficient packing with better boundary utilization
    remaining_count = n - 5
    
    # Calculate hexagonal packing parameters for better density
    # Using tighter spacing to maximize circle count while maintaining valid radii
    row_spacing = 0.22  # Tighter horizontal spacing
    row_height = np.sqrt(3) / 2 * row_spacing  # Optimal vertical spacing for hexagonal packing
    
    # Fill staggered rows with adaptive column placement
    row_idx = 0
    col_idx = 0
    for i in range(remaining_count):
        # Stagger every other row for hexagonal packing
        if row_idx % 2 == 0:
            x = 0.12 + col_idx * row_spacing
        else:
            x = 0.12 + row_spacing / 2 + col_idx * row_spacing  # Offset by half spacing
        
        y = 0.12 + row_idx * row_height
        
        # Clip to unit square with safety margin
        x = np.clip(x, 0.01, 0.99)
        y = np.clip(y, 0.01, 0.99)
        
        centers[4 + i] = [x, y]
        
        col_idx += 1
        if col_idx >= 4:  # Max 4 columns per row
            col_idx = 0
            row_idx += 1

    # Compute maximum valid radii for this configuration
    radii = compute_max_radii(centers)

    # Calculate the sum of radii
    sum_radii = np.sum(radii)

    return centers, radii, sum_radii


def compute_max_radii(centers):
    """
    Compute the maximum possible radii for each circle position
    such that they don't overlap and stay within the unit square.

    Uses iterative constraint resolution for more robust results.

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates

    Returns:
        np.array of shape (n) with radius of each circle
    """
    n = centers.shape[0]
    
    # Initialize radii based on boundary constraints only
    radii = np.zeros(n)
    for i in range(n):
        x, y = centers[i]
        # Distance to borders (must be non-negative)
        radii[i] = max(0.0, min(x, y, 1 - x, 1 - y))

    # Iteratively resolve pairwise constraints
    # Multiple passes to converge on valid solution
    for iteration in range(10):  # Increased iterations for better convergence
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                
                # If current radii would cause overlap
                if radii[i] + radii[j] > dist:
                    changed = True
                    # Scale down proportionally to maintain optimal ratio
                    scale = dist / (radii[i] + radii[j])
                    radii[i] *= scale
                    radii[j] *= scale
                    # Ensure radii don't exceed boundary constraints
                    radii[i] = min(radii[i], max(0.0, min(centers[i][0], centers[i][1], 1 - centers[i][0], 1 - centers[i][1])))
                    radii[j] = min(radii[j], max(0.0, min(centers[j][0], centers[j][1], 1 - centers[j][0], 1 - centers[j][1])))
        
        # Early exit if no changes
        if not changed:
            break

    # Final pass: ensure all constraints are satisfied with boundary as hard limit
    for i in range(n):
        x, y = centers[i]
        border_limit = max(0.0, min(x, y, 1 - x, 1 - y))
        radii[i] = min(radii[i], border_limit)
        
        # Check all pairwise constraints one final time
        for j in range(n):
            if i == j:
                continue
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if radii[i] + radii[j] > dist:
                # Adjust the larger radius to fit
                if radii[i] > radii[j]:
                    radii[i] = dist - radii[j]
                else:
                    radii[j] = dist - radii[i]
                # Re-apply boundary constraint
                radii[i] = min(radii[i], border_limit)
                radii[j] = min(radii[j], max(0.0, min(centers[j][0], centers[j][1], 1 - centers[j][0], 1 - centers[j][1])))

    return radii


def optimize_positions(centers, radii, n):
    """
    Local optimization to improve circle positions after initial placement.
    Uses gradient-free optimization with boundary-aware updates.
    """
    best_sum = np.sum(radii)
    best_centers = centers.copy()
    best_radii = radii.copy()
    
    # Adaptive perturbation sizes based on current radius magnitudes
    base_perturbation = 0.005
    
    # More iterations for better optimization
    for iteration in range(30):
        for i in range(n):
            # Adaptive perturbation based on iteration number
            current_perturbation = base_perturbation * (0.8 ** (iteration // 5))
            
            # Try perturbations in multiple directions including diagonals
            perturbations = [
                [-current_perturbation, 0], [current_perturbation, 0],
                [0, -current_perturbation], [0, current_perturbation],
                [-current_perturbation, -current_perturbation],
                [current_perturbation, -current_perturbation],
                [-current_perturbation, current_perturbation],
                [current_perturbation, current_perturbation]
            ]
            
            for dx, dy in perturbations:
                new_center = centers[i].copy()
                new_center[0] = np.clip(new_center[0] + dx, 0.001, 0.999)
                new_center[1] = np.clip(new_center[1] + dy, 0.001, 0.999)
                
                # Compute radii for new position
                temp_radii = compute_max_radii_single(centers.copy(), radii.copy(), i, new_center)
                new_sum = np.sum(temp_radii)
                
                if new_sum > best_sum:
                    best_sum = new_sum
                    best_centers[i] = new_center
                    best_radii = temp_radii
                    # Only keep best configuration
                    centers = best_centers.copy()
                    radii = best_radii.copy()
    
    centers = best_centers
    radii = best_radii
    return centers, radii

def compute_max_radii_single(centers, radii, idx, new_center):
    """Helper to compute radii with one circle moved"""
    n = centers.shape[0]
    temp_centers = centers.copy()
    temp_radii = radii.copy()
    
    # Update the moved circle's position
    temp_centers[idx] = new_center
    
    # Update the moved circle's border constraint
    x, y = new_center
    temp_radii[idx] = max(0.0, min(x, y, 1 - x, 1 - y))
    
    # Quick pairwise check - optimized for speed
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((temp_centers[i] - temp_centers[j]) ** 2))
            if temp_radii[i] + temp_radii[j] > dist:
                # Scale down proportionally
                scale = dist / (temp_radii[i] + temp_radii[j])
                temp_radii[i] *= scale
                temp_radii[j] *= scale
                # Apply boundary constraints
                temp_radii[i] = min(temp_radii[i], max(0.0, min(temp_centers[i][0], temp_centers[i][1], 1 - temp_centers[i][0], 1 - temp_centers[i][1])))
                temp_radii[j] = min(temp_radii[j], max(0.0, min(temp_centers[j][0], temp_centers[j][1], 1 - temp_centers[j][0], 1 - temp_centers[j][1])))
    
    return temp_radii

# EVOLVE-BLOCK-END


def run_circle_packing():
    centers, radii, sum_radii = construct_packing()
    
    # Apply local optimization to improve the layout
    centers, radii = optimize_positions(centers, radii, N_CIRCLES)
    
    sum_radii = np.sum(radii)
    
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