import numpy as np
from shapely.geometry import MultiPolygon, Polygon, Point # Enable precise geometric operations
from shapely.prepared import prep # Enable efficient geometric operations
from skimage import measure
import random
from time import time
from typing import Tuple, Optional, List, Union

from .utils import (
    log
)

from .operations import (
    add_triangle,
)

from .predicates import (
    is_valid,
)


# ---------- Fractal Functions ---------- #

def generate_mandelbrot_boundary(
    resolution: int = 500,
    max_iterations: int = 100,
    x_range: Tuple[float, float] = (-2.0, 1.0),
    y_range: Tuple[float, float] = (-1.5, 1.5),
    verbose: int = 1
) -> np.ndarray:
    """
    This function computes the boundary of the Mandelbrot set by iterating over a grid of points in the complex plane.
    Each point is checked to determine whether it escapes to infinity within a maximum number of iterations. Points that
    do not escape are considered part of the Mandelbrot set, and their boundary is calculated.

    Parameters:
    resolution : int, optional
        Determines the resolution of the grid. Higher resolution results in a finer grid and more detailed boundary.
    max_iterations : int, optional
        Maximum number of iterations to determine if a point is in the Mandelbrot set.
    x_range : tuple of float, optional
        The range of the real axis (x-axis) in the complex plane.
    y_range : tuple of float, optional
        The range of the imaginary axis (y-axis) in the complex plane.
    verbose : int, optional
        Integer indicating the verbosity level (0: silent, 1: basic, 2: detailed, 3: highly detailed).

    Returns:
    boundary_polygon : np.ndarray
        An Nx2 array of (x, y) coordinates defining the boundary of the Mandelbrot set.
    """

    
    # Validate input parameters
    if not isinstance(resolution, int) or resolution <= 0:
        raise ValueError("Parameter 'resolution' must be a positive integer.")
    if not isinstance(max_iterations, int) or max_iterations <= 0:
        raise ValueError("Parameter 'max_iterations' must be a positive integer.")
    if not (isinstance(x_range, tuple) and len(x_range) == 2):
        raise ValueError("Parameter 'x_range' must be a tuple of two floats.")
    if not (isinstance(y_range, tuple) and len(y_range) == 2):
        raise ValueError("Parameter 'y_range' must be a tuple of two floats.")
    if not isinstance(verbose, int) or verbose < 0:
        raise ValueError("Parameter 'verbose' must be a non-negative integer.")
    
    start_time = time()
    log(f"Starting Mandelbrot boundary generation with resolution={resolution}x{resolution}, "
        f"max_iterations={max_iterations}, x_range={x_range}, y_range={y_range}", verbose, level=1)
    
    # Create linearly spaced arrays for x and y
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y
    Z = np.zeros_like(C, dtype=np.complex128)
    iteration_array = np.full(C.shape, max_iterations, dtype=int)

    # Minimum distance between points in the complex plane
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    min_distance = min(dx, dy)

    log(f"Generated meshgrid and initialized iteration arrays.", verbose, level=2)
    
    # Start Mandelbrot iteration
    iteration_start_time = time()
    for i in range(max_iterations):
        mask = np.abs(Z) <= 2
        Z[mask] = Z[mask]**2 + C[mask]
        diverged = (np.abs(Z) > 2) & (iteration_array == max_iterations)
        iteration_array[diverged] = i
        
        if verbose >= 3:
            num_diverged = np.sum(diverged)
            log(f"Iteration {i+1}/{max_iterations}: {num_diverged} points diverged.", verbose, level=3)
    
    iteration_time = time() - iteration_start_time
    log(f"Completed Mandelbrot iterations in {iteration_time:.4f} seconds.", verbose, level=1)
    if verbose >= 2:
        total_diverged = np.sum(iteration_array < max_iterations)
        log(f"Total diverged points: {total_diverged} out of {resolution**2}", verbose, level=2)
    
    # Extract contours at the level max_iterations - 1
    contour_start_time = time()
    contours = measure.find_contours(iteration_array, level=max_iterations - 1) # Allows us to find all the components of the boundary formed by the mandelbrot set
    contour_time = time() - contour_start_time
    log(f"Extracted contours in {contour_time:.4f} seconds.", verbose, level=1)
    
    if not contours:
        raise ValueError("No contours found. Try increasing 'max_iterations' or adjusting the fractal parameters.")
    
    # Select the longest contour assuming it's the main boundary
    boundary_contour = max(contours, key=lambda x: len(x)) # Allows us to select the longest boundary from all the components of the mandelbrot set
    log(f"Selected the longest contour with {boundary_contour.shape[0]} points.", verbose, level=2)
    
    # Map contour coordinates to the complex plane
    boundary_polygon = np.zeros((boundary_contour.shape[0], 2))
    boundary_polygon[:, 0] = x[np.clip(boundary_contour[:, 1].astype(int), 0, resolution - 1)]
    boundary_polygon[:, 1] = y[np.clip(boundary_contour[:, 0].astype(int), 0, resolution - 1)]
    
    log(f"Mapped contour coordinates to complex plane to form boundary polygon.", verbose, level=2)
    
    total_time = time() - start_time
    log(f"Generated Mandelbrot boundary polygon with {boundary_polygon.shape[0]} points in {total_time:.4f} seconds.", verbose, level=1)
    
    # Filter out duplicate points
    boundary_polygon = remove_duplicate_points(boundary_polygon, tol=min_distance*1e-1)

    return boundary_polygon, min_distance

# ---------- Generate Generic Non Convex Polygon ---------- #

def generate_non_convex_boundary(
    num_points: int = 20, 
    radius: float = 1.0, 
    noise_factor: float = 0.3
) -> List[Tuple[float, float]]:
    """
    Generates boundary points of a non-convex shape in counter-clockwise (CCW) order.

    Description:
    This function generates a set of points representing the boundary of a non-convex shape. 
    The points are generated in polar coordinates and then converted to Cartesian coordinates. 
    The shape is made non-convex by introducing random noise to the radii. The resulting points 
    are sorted in counter-clockwise order based on their angles.

    Parameters:
    - num_points (int): Number of boundary points to generate (default is 20).
    - radius (float): Approximate radius of the shape (default is 1.0).
    - noise_factor (float): Amount of noise to add to create non-convexity, ranging from 0 to 1 
      (default is 0.3).

    Returns:
    - List[Tuple[float, float]]: A list of points (x, y) representing the boundary in CCW order.
    """
    # Create angles evenly spaced around the circle
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    
    # Generate radii with some noise to create a non-convex shape
    radii = radius * (1 + noise_factor * (np.random.rand(num_points) - 0.5))
    
    # Generate boundary points in polar coordinates, then convert to Cartesian coordinates
    boundary_node_coords = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
    
    # Sort points in CCW order based on their angle
    angles = np.arctan2(boundary_node_coords[:, 1], boundary_node_coords[:, 0])
    boundary_node_coords = boundary_node_coords[np.argsort(angles)]
    
    return boundary_node_coords.tolist()

# ------------------------------------ Initialization Functions ------------------------------------ #

def initialize_node_elems(num_delaunay_node_coords: int) -> List[List[int]]:
    """
    Initializes the delaunay_node_elems adjacency list.

    Parameters:
    - num_delaunay_node_coords (int): Total number of node_coords in the mesh.

    Returns:
    - List[List[int]]: A list of lists, where each sublist contains triangle indices associated with each node.
    """
    return [[] for _ in range(num_delaunay_node_coords)]

def initialize_node_nodes(num_delaunay_node_coords: int) -> List[List[int]]:
    """
    Initializes the delaunay_node_nodes adjacency list.

    Parameters:
    - num_delaunay_node_coords (int): Total number of node_coords in the mesh.

    Returns:
    - List[List[int]]: A list of lists, where each sublist contains adjacent node indices.
    """
    return [[] for _ in range(num_delaunay_node_coords)]

def initialize_triangulation(
    cloud_node_coords: List[Tuple[float, float]], 
    elem_nodes: List[Optional[Tuple[int, int, int]]], 
    node_elems: List[List[int]], 
    node_nodes: List[List[int]]
) -> np.ndarray:
    """
    Initializes the triangulation with four affinely independent node_coords.
    
    Parameters:
    - cloud_node_coords (List[Tuple[float, float]]): List of points (tuples of (x, y)) to triangulate.
    - elem_nodes (List[Optional[Tuple[int, int, int]]]): List of elem_nodes representing the triangulated mesh.
    - node_elems (List[List[int]]): List of lists, where each sublist contains triangle indices associated with each node.
    - node_nodes (List[List[int]]): List of lists, where each sublist contains adjacent node indices.
    
    Returns:
    - np.ndarray: Array of initial triangle coordinates forming the super-triangle.
    
    Raises:
    - ValueError: If less than four node_coords are provided.
    """
    if len(cloud_node_coords) < 4:
        raise ValueError("At least four node_coords are required to initialize the triangulation.")
    
    # Compute bounding box for the given node coordinates
    xs = [v[0] for v in cloud_node_coords]
    ys = [v[1] for v in cloud_node_coords]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    dx = max_x - min_x
    dy = max_y - min_y
    delta_max = max(dx, dy) * 5  # Make it large enough to encompass the points
    
    # Create three additional points forming a super-triangle
    super_v0 = (min_x - delta_max, min_y - delta_max)
    super_v1 = (min_x + 2 * delta_max, min_y - delta_max)
    super_v2 = (min_x - delta_max, min_y + 2 * delta_max)
    
    # Add the super-triangle to the triangulation
    add_triangle(0, 1, 2, elem_nodes, node_elems, node_nodes)

    # Initialize the node_coords with super-triangle coordinates
    return np.array([super_v0, super_v1, super_v2])

def remove_duplicate_points(
    points: np.ndarray,
    tol: float = 1e-6
) -> np.ndarray:
    """
    This function takes an array of points and removes duplicates while maintaining the original 
    order of appearance. It uses a set to track seen points for efficient duplicate detection.

    Parameters:
    points : np.ndarray
    An array of points with shape (n, 2), where each row represents a point (x, y).
    tol : float, optional 
    Tolerance for considering two points equal. Default is 1e-6.

    Returns:
    np.ndarray
    An array of unique points with shape (m, 2), preserving the original order.
    """
    seen = set()  # Set to track seen points
    unique_points = []  # List to store unique points

    for point in points:
    # Convert the point to a tuple for hashable comparison in the set
        pt_tuple = tuple((np.round(point[0] / tol) * tol,np.round(point[1] / tol) * tol))
        if pt_tuple not in seen:
            seen.add(pt_tuple)  # Mark this point as seen
            unique_points.append(point)  # Add the unique point to the list

    # Convert the list of unique points back to a numpy array and return it
    return np.array(unique_points)

# ---------- Generate Boundary Functions ---------- #

def generate_boundary_node_coords(
    polygon: List[Tuple[float, float]], 
    min_distance: float, 
    verbose: int = 1
) -> np.ndarray:
    """
    This function generates points evenly spaced along the edges of a polygon such that the distance between 
    adjacent points is at least `min_distance`. The points are generated using linear interpolation along each 
    polygon edge.

    Parameters:
    polygon : list of tuples or numpy.ndarray
        A list of (x, y) tuples representing the polygon's vertices in counterclockwise order.
    min_distance : float
        Minimum distance between points on the boundary.
    verbose : int, optional
        Integer indicating the verbosity level (0: silent, 1: basic, 2: detailed, 3: highly detailed).

    Returns:
    numpy.ndarray
        An array of boundary points distributed along the polygon's edges.
    """

    boundary_node_coords = []
    perimeter = 0
    num_delaunay_node_coords = len(polygon)
    
    # Calculate the total perimeter of the polygon
    for i in range(num_delaunay_node_coords):
        start = np.array(polygon[i])
        end = np.array(polygon[(i + 1) % num_delaunay_node_coords])
        edge_length = np.linalg.norm(end - start)
        perimeter += edge_length
        log(f"Edge {i}: Length = {edge_length:.4f}", verbose, level=3)
    
    log(f"Total perimeter of polygon: {perimeter:.4f}", verbose, level=2)
    
    # Distribute points along each edge of the polygon
    for i in range(num_delaunay_node_coords):
        start = np.array(polygon[i])
        end = np.array(polygon[(i + 1) % num_delaunay_node_coords])
        edge_vector = end - start
        edge_length = np.linalg.norm(edge_vector)
        num_edge_points = max(int(edge_length / min_distance), 1)  # Ensure at least one point per edge
        
        log(f"Processing edge {i}: {start} to {end}, Length = {edge_length:.4f}, "
            f"Number of points to generate = {num_edge_points}", verbose, level=3)
        
        # Generate points evenly spaced along the edge
        for j in range(num_edge_points):
            t = j / num_edge_points  # Parameter from 0 to 1
            point = (1 - t) * start + t * end  # Linear interpolation
            boundary_node_coords.append(point)
            log(f"Added boundary point {len(boundary_node_coords)}: {point}", verbose, level=3)
    
    # Convert the list of points to a NumPy array for consistency
    boundary_node_coords = np.array(boundary_node_coords)
    
    return boundary_node_coords

# ---------- Poisson Disk Sampling ---------- #

def generate_point_around(
        point: Tuple[float, float], 
        r: float
) -> Tuple[float, float]:
    """
    This function generates a random point around a given point within a specified range. The distance 
    from the given point to the generated point is randomly chosen between `r` and `2 * r`, and the angle 
    is chosen randomly between 0 and 2π.

    Parameters:
    - point (Tuple[float, float]): The original point (x, y) around which the new point will be generated.
    - r (float): The minimum distance from the original point to the generated point.

    Returns:
    - Tuple[float, float]: A tuple representing the coordinates of the generated point (x, y).
    """
    # Generate a random angle between 0 and 2π
    theta = np.random.uniform(0, 2 * np.pi)
    
    # Generate a random distance between r and 2 * r
    distance = np.random.uniform(r, 2 * r)
    
    # Calculate the new point's coordinates using polar coordinates
    new_x = point[0] + distance * np.cos(theta)
    new_y = point[1] + distance * np.sin(theta)
    
    return new_x, new_y

def offset_polygon_shapely(
    node_coords: List[Tuple[float, float]], 
    d: float
) -> List[List[Tuple[float, float]]]:
    """
    This function creates a new polygon by offsetting the given polygon (defined by `node_coords`) 
    inward or outward by a specified distance `d`. A positive `d` value offsets the polygon outward, and a 
    negative `d` value offsets it inward. It uses Shapely's `buffer` method to perform the offset.

    Parameters:
    - node_coords (List[Tuple[float, float]]): List of vertex coordinates (x, y) defining the polygon.
    - d (float): The distance by which to offset the polygon. Positive for outward, negative for inward.

    Returns:
    - buffered_poly (Polygon or MultiPolygon): The offset polygon created using Shapely.
    
    Raises:
    - ValueError: If the input polygon is not valid.
    """
    # Create a Shapely Polygon from the provided coordinates
    poly = Polygon(node_coords)

    # Offset the polygon using the buffer method
    buffered_poly = poly.buffer(d)

    return buffered_poly

def insert_point(grid_values, min_x, min_y, cell_size, point):
    grid_x = int((point[0] - min_x) / cell_size)
    grid_y = int((point[1] - min_y) / cell_size)
    grid_values[grid_x, grid_y] = point

def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def is_valid(
        grid_values,
        grid_width,
        grid_height,
        min_x,
        min_y,
        cell_size,
        radius,
        new_sample
):
    grid_x = int((new_sample[0] - min_x) / cell_size)
    grid_y = int((new_sample[1] - min_y) / cell_size)
    if not (0 <= grid_x < grid_width and 0 <= grid_y < grid_height):
        return False
        
    if grid_values[grid_x, grid_y] != 0:
        return False
    
    # Check neighboring cells for minimum distance
    for dx in [-2, -1, 0, 1, 2]:
        for dy in [-2, -1, 0, 1, 2]:
            nx, ny = grid_x + dx, grid_y + dy
            if 0 <= nx < grid_width and 0 <= ny < grid_height:
                if isinstance(grid_values[nx, ny], tuple):
                    if distance(grid_values[nx, ny], new_sample) < radius:
                        return False
    return True

def visualize_grid_and_polygon(
    shell_coords: List[Tuple[float, float]],
    holes_coords: List[List[Tuple[float, float]]],
    grid_values: np.ndarray,
    min_x: float,
    min_y: float,
    cell_size: float,
    samples: Optional[List[Tuple[float, float]]] = None
) -> None:
    """
    Visualize the grid cells, polygon, and sample points.
    
    Args:
        shell_coords: Coordinates of the polygon's outer shell
        holes_coords: List of coordinates for holes in the polygon
        grid_values: The grid array containing None, 0, or point coordinates
        min_x: Minimum x coordinate of the grid
        min_y: Minimum y coordinate of the grid
        cell_size: Size of each grid cell
        samples: Optional list of generated sample points
    """
    import matplotlib.pyplot as plt
    from shapely.geometry import Polygon
    import numpy as np
    from typing import List, Tuple, Optional, Union
    from matplotlib.patches import Rectangle


    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Plot the polygon
    polygon = Polygon(shell=shell_coords, holes=holes_coords)
    x, y = polygon.exterior.xy
    ax.plot(x, y, 'k-', linewidth=2, label='Polygon Boundary')
    
    # Plot holes if any
    for hole in holes_coords:
        hole_x = [p[0] for p in hole + [hole[0]]]
        hole_y = [p[1] for p in hole + [hole[0]]]
        ax.plot(hole_x, hole_y, 'k-', linewidth=2)
    
    # Plot grid cells with different colors based on their values
    grid_width, grid_height = grid_values.shape

    for i in range(grid_width):
        for j in range(grid_height):
            cell_x = min_x + i * cell_size
            cell_y = min_y + j * cell_size

            if grid_values[i, j] is None:
                color = 'lightgray'
                alpha = 0.3
            elif grid_values[i, j] == 0:
                color = 'lightblue'
                alpha = 0.5
            else:
                color = 'green'
                alpha = 0.3
                
            rect = Rectangle(
                (cell_x, cell_y),
                cell_size,
                cell_size,
                facecolor=color,
                alpha=alpha,
                edgecolor='gray',
                linewidth=0.5
            )
            ax.add_patch(rect)
    
    # Plot sample points if provided
    if samples:
        sample_x = [p[0] for p in samples]
        sample_y = [p[1] for p in samples]
        ax.scatter(sample_x, sample_y, c='red', s=20, label='Sample Points')
    
    # Add legend
    custom_lines = [
        Rectangle((0, 0), 1, 1, facecolor='lightgray', alpha=0.3),
        Rectangle((0, 0), 1, 1, facecolor='lightblue', alpha=0.5),
        Rectangle((0, 0), 1, 1, facecolor='green', alpha=0.3)
    ]
    ax.legend(custom_lines + ([plt.Line2D([0], [0], marker='o', color='red', linestyle='None')] if samples else []),
             ['Outside Grid Cells', 'Valid Grid Cells', 'Occupied Cells'] + (['Sample Points'] if samples else []))
    
    # Set equal aspect ratio and display
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.title('Polygon Grid Visualization')
    plt.show()

    


def poisson_disk_sampling_complex_geometry(
    polygon_outer: List[Tuple[float, float]], 
    polygons_holes: List[List[Tuple[float, float]]], 
    radius: float, 
    k: int = 100, 
    verbose: int = 1
) -> List[Tuple[float, float]]:
    """
    This function generates points within the provided polygon using Poisson Disk Sampling, ensuring that 
    each point is at least a specified minimum distance from others. The function handles polygons with 
    holes and allows adjusting the verbosity level for logging.

    Parameters:
    polygon_outer : List[Tuple[float, float]]
        Outer boundary of the polygon as a list of (x, y) tuples.
    polygons_holes : List[List[Tuple[float, float]]]
        List of hole polygons, each as a list of (x, y) tuples.
    radius : float
        Minimum distance between points.
    k : int, optional
        Number of attempts before removing a point from the active list. Default is 50.
    verbose : int, optional
        Integer indicating the verbosity level (0: silent, 1: basic, 2: detailed, 3: highly detailed).

    Returns:
    List[Tuple[float, float]]
        Filtered list of points generated by Poisson Disk Sampling.
    """
    # Create the Shapely polygon from the outer boundary and holes

    # Offset the outer polygon by the negative radius
    polygon_outer_shapely = offset_polygon_shapely(polygon_outer, -radius)

    # Initialize lists to hold the final polygons and their holes
    polygons_with_holes = []

    # Check if the outer polygon is a MultiPolygon or a simple Polygon
    if isinstance(polygon_outer_shapely, MultiPolygon):
        # Process each individual polygon in the MultiPolygon
        for individual_polygon in polygon_outer_shapely.geoms:
            shell_coords = list(individual_polygon.exterior.coords)

            # Find the holes that are contained within this individual polygon
            individual_holes_coords = []
            if polygons_holes:
                for polygon_hole in polygons_holes:
                    # Offset the hole by a positive radius
                    offset_hole = offset_polygon_shapely(polygon_hole, +radius)

                    # Check if the hole is within the individual polygon
                    if offset_hole.within(individual_polygon):
                        individual_holes_coords.append(list(offset_hole.exterior.coords))

            # Add the polygon and its associated holes
            polygons_with_holes.append(Polygon(shell=shell_coords, holes=individual_holes_coords))

    else:
        # If it's a simple Polygon, process it directly
        shell_coords = list(polygon_outer_shapely.exterior.coords)

        # Offset and assign holes
        if polygons_holes:
            polygons_holes_shapely = [
                offset_polygon_shapely(polygon_hole, +radius) for polygon_hole in polygons_holes
            ]
            holes_coords = [list(hole.exterior.coords) for hole in polygons_holes_shapely if hole.within(polygon_outer_shapely)]
        else:
            holes_coords = []

        polygons_with_holes.append(Polygon(shell=shell_coords, holes=holes_coords))

    global_samples = []

    for polygon in polygons_with_holes:
        samples = poisson_disk_sampling(polygon, radius, k, verbose)
        global_samples.extend(samples)

    return global_samples

def poisson_disk_sampling(
        polygon: Polygon,
        radius: float, 
        k: int = 50, 
        verbose: int = 1
) -> List[Tuple[float, float]]:

    prepared_polygon = prep(polygon)

    # Calculate bounding box and create grid
    min_x, min_y, max_x, max_y = polygon.bounds

    log(f"Calculated bounding box: x=({min_x}, {max_x}), y=({min_y}, {max_y})", verbose, level=2)

    cell_size = radius / np.sqrt(2)
    grid_width = int(np.ceil((max_x - min_x) / cell_size) + 1)
    grid_height = int(np.ceil((max_y - min_y) / cell_size) + 1)

    # Initialize grid values to None
    grid_values = np.full((grid_width, grid_height), None)

    start_time = time()
    nb_grid_polygon_values = 0

    # Iterate through each cell and assign values
    for i in range(grid_width):
        for j in range(grid_height):
            # Define the cell as a polygon
            cell = Polygon([
                (min_x + i * cell_size, min_y + j * cell_size),
                (min_x + i * cell_size, min_y + (j + 1) * cell_size),
                (min_x + (i + 1) * cell_size, min_y + (j + 1) * cell_size),
                (min_x + (i + 1) * cell_size, min_y + j * cell_size)
            ])
            # Check if the cell intersects with the polygon
            if prepared_polygon.intersects(cell):
                grid_values[i, j] = 0
                nb_grid_polygon_values += 1

    grid_generated_time = time()
    log(f"Initialized in {grid_generated_time - start_time:.4f} seconds grid with cell size {cell_size:.4f}, dimensions {grid_width}x{grid_height}. with {nb_grid_polygon_values} cells intersecting the polygon", verbose, level=1)
    
    # List of samples (points)
    samples: List[Tuple[float, float]] = []

    # Active list of points
    active_list: List[Tuple[float, float]] = []

    start_time = time()

    # Generate the first sample to be inside the polygon ie the cell it belongs to is not None
    # We take n attemps such that given the number of filled cells in the grid, we have 99.9% chance of finding a valid point
    n = int(np.ceil(np.log(0.999) / np.log( nb_grid_polygon_values / (grid_width * grid_height) ))) + 10
    for _ in range(n):
        x = np.random.uniform(min_x, max_x)
        y = np.random.uniform(min_y, max_y)
        point = (x, y)
        grid_x = int(np.ceil((point[0] - min_x) / cell_size))
        grid_y = int(np.ceil((point[1] - min_y) / cell_size))
        if grid_values[grid_x, grid_y] is not None:
            insert_point(grid_values, min_x, min_y, cell_size, point)
            samples.append(point)
            active_list.append(point)
            break
        
        


    # Generate the rest of the samples
    while active_list:
        random_idx = random.randint(0, len(active_list)-1)
        sample = active_list[random_idx]
        found = False

        for _ in range(k):
            new_sample = generate_point_around(sample, radius)
            if is_valid(grid_values, grid_width, grid_height, min_x, min_y, cell_size, radius, new_sample):
                insert_point(grid_values, min_x, min_y, cell_size, new_sample)
                samples.append(new_sample)
                active_list.append(new_sample)
                found = True
                break

        if not found:
            active_list.pop(random_idx)

    generated_time = time()
    log(f"Time taken to generate {len(samples)} points: {generated_time - start_time:.4f} seconds.", verbose, level=1)
    
    return samples

def heterogen_disk_sampling():
    pass

# ---------- Main Function ---------- #

def generate_cloud(
    polygon_outer: Union[List[Tuple[float, float]], np.ndarray], 
    polygons_holes: Union[List[Tuple[float, float]], np.ndarray], 
    min_distance_outer: float, 
    verbose: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    This function generates a set of points within the provided polygon while ensuring a specified 
    minimum distance between them. It handles both the boundary and the interior of the polygon 
    using Poisson Disk Sampling for interior points and distance-based spacing for boundary points.

    Parameters:
    polygon_outer : list of tuples or numpy.ndarray
        A list of (x, y) tuples representing the outer boundary of the polygon in counterclockwise order.
    polygons_holes : list of lists of tuples or numpy.ndarray
        A list of lists of (x, y) tuples representing the holes in the polygon, if any.
    min_distance_outer : float
        Minimum distance between points in the polygon outer boundary.
    verbose : int, optional
        Integer indicating the verbosity level (0: silent, 1: basic, 2: detailed, 3: highly detailed).

    Returns:
    tuple
        A tuple containing:
            - numpy.ndarray: Combined array of boundary and interior points.
            - numpy.ndarray: Array of boundary points.
            - numpy.ndarray: Array of interior points.
    """
    log("Starting point cloud generation.", verbose, level=1)
    
    # Step 1: Remove duplicate points from the polygon
    polygon_outer = remove_duplicate_points(polygon_outer, tol=min_distance_outer*1e-2)
    log(f"Removed duplicate points. Number of polygon vertices: {len(polygon_outer)}", verbose, level=2)
    
    # Step 2: Calculate the minimum distance between points on the inner holes
    if polygons_holes:  # Check if there are holes to avoid errors
        min_distance_polygon_holes = min(
            min(
                np.linalg.norm(np.array(polygon_hole[i]) - np.array(polygon_hole[(i + 1) % len(polygon_hole)]))
                for i in range(len(polygon_hole))
            )
            for polygon_hole in polygons_holes
        )
    else:
        min_distance_polygon_holes = float('inf')  # No holes, set to a large value

    min_distance = min(min_distance_outer, min_distance_polygon_holes)
    log(f"Minimum distance between points: {min_distance:.4f}", verbose, level=2)

    # Step 3: Generate boundary points along the polygon edges
    outer_boundary_node_coords = generate_boundary_node_coords(polygon_outer, min_distance, verbose)
    outer_boundary_node_coords = remove_duplicate_points(outer_boundary_node_coords)
    outer_boundary_node_coords = np.array(outer_boundary_node_coords)
    log(f"Generated {len(outer_boundary_node_coords)} outer boundary points.", verbose, level=1)

    hole_boundaries_node_coords = []
    if polygons_holes:
        for polygon_hole in polygons_holes:
            hole_boundary_node_coords = generate_boundary_node_coords(polygon_hole, min_distance, verbose)
            hole_boundary_node_coords = remove_duplicate_points(hole_boundary_node_coords)
            hole_boundary_node_coords = np.array(hole_boundary_node_coords)
            hole_boundaries_node_coords.append(hole_boundary_node_coords)

    # Flatten hole boundaries into a single array
    if hole_boundaries_node_coords:
        hole_boundaries_node_coords_flattened = np.concatenate(hole_boundaries_node_coords)
    else:
        hole_boundaries_node_coords_flattened = np.array([])
    log(f"Generated {len(hole_boundaries_node_coords_flattened)} hole boundary points.", verbose, level=1)


    log(f"Total boundary points: {len(outer_boundary_node_coords) + len(hole_boundaries_node_coords_flattened)}", verbose, level=1)
    
    # Step 5: Generate interior points using Poisson Disk Sampling
    log("Generating interior points using Poisson Disk Sampling.", verbose, level=1)
    interior_node_coords = poisson_disk_sampling_complex_geometry(
        polygon_outer=polygon_outer,
        polygons_holes=polygons_holes,
        radius=min_distance,
        k=50,
        verbose=verbose
    )
    interior_node_coords = remove_duplicate_points(interior_node_coords)
    interior_node_coords = np.array(interior_node_coords)
    log(f"Generated {len(interior_node_coords)} interior points.", verbose, level=1)
    
    # Step 6: Combine boundary and interior points into a single array
    if hole_boundaries_node_coords_flattened.size == 0:
        boundary_node_coords = outer_boundary_node_coords
    else:
        boundary_node_coords = np.concatenate([outer_boundary_node_coords, hole_boundaries_node_coords_flattened])

    combined_points = np.concatenate([boundary_node_coords, interior_node_coords])
    log(f"Total points in cloud: {combined_points.shape[0]}", verbose, level=1)
    
    return combined_points, outer_boundary_node_coords, hole_boundaries_node_coords, interior_node_coords