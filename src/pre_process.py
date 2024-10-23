import numpy as np
import math
import random
from shapely.geometry import Polygon, Point, MultiPolygon # Enable precise geometric operations
from skimage import measure
from scipy.spatial import cKDTree
from time import time

from src.utils import log

from src.operations import (
    add_triangle,
    )

from src.predicates import (
    is_valid,
    is_point_inside,
    distance,
    )


# ---------- Fractal Functions ---------- #

def generate_mandelbrot_boundary(
    resolution=500,
    max_iterations=100,
    x_range=(-2.0, 1.0),
    y_range=(-1.5, 1.5),
    verbose=1
) -> np.ndarray:
    """
    Generates the Mandelbrot set boundary as a polygon based on the specified depth.

    Parameters:
    ----------
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
    -------
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
    contours = measure.find_contours(iteration_array, level=max_iterations - 1)
    contour_time = time() - contour_start_time
    log(f"Extracted contours in {contour_time:.4f} seconds.", verbose, level=1)
    
    if not contours:
        raise ValueError("No contours found. Try increasing 'max_iterations' or adjusting the fractal parameters.")
    
    # Select the longest contour assuming it's the main boundary
    boundary_contour = max(contours, key=lambda x: len(x))
    log(f"Selected the longest contour with {boundary_contour.shape[0]} points.", verbose, level=2)
    
    # Map contour coordinates to the complex plane
    boundary_polygon = np.zeros((boundary_contour.shape[0], 2))
    boundary_polygon[:, 0] = x[np.clip(boundary_contour[:, 1].astype(int), 0, resolution - 1)]
    boundary_polygon[:, 1] = y[np.clip(boundary_contour[:, 0].astype(int), 0, resolution - 1)]
    
    log(f"Mapped contour coordinates to complex plane to form boundary polygon.", verbose, level=2)
    
    total_time = time() - start_time
    log(f"Generated Mandelbrot boundary polygon with {boundary_polygon.shape[0]} points in {total_time:.4f} seconds.", verbose, level=1)
    
    return boundary_polygon


# ---------- Generate Generic Non Convex Polygon ---------- #

def generate_non_convex_boundary(num_points=20, radius=1.0, noise_factor=0.3):
    """
    Generates boundary points of a non-convex shape in CCW order.

    Parameters:
    - num_points: Number of boundary points to generate.
    - radius: Approximate radius of the shape.
    - noise_factor: Amount of noise to add to create non-convexity (0 to 1).

    Returns:
    - boundary_node_coords: List of points (x, y) in CCW order.
    """
    # Create angles evenly spaced around the circle
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    
    # Generate radii with some noise to create a non-convex shape
    radii = radius * (1 + noise_factor * (np.random.rand(num_points) - 0.5))
    
    # Generate boundary points in polar coordinates, then convert to Cartesian
    boundary_node_coords = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
    
    # Sort points in CCW order based on their angle (ensuring CCW order)
    angles = np.arctan2(boundary_node_coords[:, 1], boundary_node_coords[:, 0])
    boundary_node_coords = boundary_node_coords[np.argsort(angles)]
    
    return boundary_node_coords


# ------------------------------------ Initialization Functions ------------------------------------ #

def initialize_delaunay_node_elems(num_delaunay_node_coords):
    """
    Initializes the delaunay_node_elems adjacency list.

    Parameters:
    - num_delaunay_node_coords: Total number of delaunay_node_coords in the mesh.

    Returns:
    - delaunay_node_elems: List of lists, where each sublist contains triangle indices.
    """
    return [[] for _ in range(num_delaunay_node_coords)]

def initialize_triangulation(cloud_node_coords, triangles, delaunay_dic_edge_triangle, delaunay_node_elems):
    """
    Initializes the triangulation with four affinely independent delaunay_node_coords.
    
    Parameters:
    - cloud_node_coords: List of points (tuples of (x, y)) to triangulate.
    - triangles: List to store triangles.
    - delaunay_dic_edge_triangle: Dictionary mapping edges to triangle indices.
    - delaunay_node_elems: List of lists, where each sublist contains triangle indices.
    
    Returns:
    - List of initial triangle indices.
    """
    if len(cloud_node_coords) < 4:
        raise ValueError("At least four delaunay_node_coords are required to initialize the triangulation.")
    

    # For 2D Delaunay triangulation, initialize with a super-triangle or similar.

    # Compute bounding box
    xs = [v[0] for v in cloud_node_coords]
    ys = [v[1] for v in cloud_node_coords]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    dx = max_x - min_x
    dy = max_y - min_y
    delta_max = max(dx, dy) * 5 # Make it large enough
    
    # Create three additional points forming a super-triangle
    super_v0 = (min_x - delta_max, min_y - delta_max)
    super_v1 = (min_x + 2 * delta_max, min_y - delta_max)
    super_v2 = (min_x - delta_max, min_y + 2 * delta_max)
    

    # Add the super-triangle 
    add_triangle(0, 1, 2, triangles, delaunay_dic_edge_triangle, delaunay_node_elems)

    # Initialize the delaunay_node_coords with super-triangle delaunay_node_coords
    return np.array([super_v0, super_v1, super_v2])



# ---------- Main Functions ---------- #

def generate_boundary_node_coords(polygon, min_distance, verbose=1):
    """
    Generate points along the boundary of a polygon based on a specified density.

    The density determines how closely spaced the boundary points are. Higher density results in more points.

    Parameters:
    ----------
    polygon : list or numpy.ndarray
        A list of (x, y) tuples representing the polygon's delaunay_node_coords.
    min_distance : float
        Minimum distance between points on the boundary.
    verbose : int, optional
        Integer indicating the verbosity level (0: silent, 1: basic, 2: detailed, 3: highly detailed).

    Returns:
    -------
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
    
    log(f"Generated {boundary_node_coords.shape[0]} boundary points.", verbose, level=1)
    
    return boundary_node_coords

# ---------- Poisson Disk Sampling ---------- #

def generate_point_around(point, r):
    theta = np.random.uniform(0, 2 * math.pi)
    r = np.random.uniform(r, 2 * r)
    return (point[0] + r * math.cos(theta), point[1] + r * math.sin(theta))

def compute_centroid(delaunay_node_coords):
    """
    Compute the centroid of a polygon.
    delaunay_node_coords: list of (x, y) tuples
    """
    signed_area = 0
    Cx = 0
    Cy = 0
    n = len(delaunay_node_coords)
    for i in range(n):
        x0, y0 = delaunay_node_coords[i]
        x1, y1 = delaunay_node_coords[(i + 1) % n]
        A = x0 * y1 - x1 * y0
        signed_area += A
        Cx += (x0 + x1) * A
        Cy += (y0 + y1) * A
    signed_area *= 0.5
    if signed_area == 0:
        raise ValueError("Polygon area is zero, invalid polygon.")
    Cx /= (6 * signed_area)
    Cy /= (6 * signed_area)
    return (Cx, Cy)

def scale_polygon(delaunay_node_coords, d):
    """
    Scale a polygon by moving each vertex outward by distance d.
    delaunay_node_coords: list of (x, y) tuples in CCW order
    Sadly this only work on convex polygons.
    d: offset distance
    Returns a new list of scaled delaunay_node_coords.
    """
    centroid = compute_centroid(delaunay_node_coords)
    scaled_delaunay_node_coords = []
    for (x, y) in delaunay_node_coords:
        # Direction from centroid to vertex
        dx = x - centroid[0]
        dy = y - centroid[1]
        length = math.hypot(dx, dy)
        if length == 0:
            raise ValueError("A vertex coincides with the centroid.")
        # Unit direction vector
        ux = dx / length
        uy = dy / length
        # Offset vertex
        new_x = x + ux * d
        new_y = y + uy * d
        scaled_delaunay_node_coords.append((new_x, new_y))
    return scaled_delaunay_node_coords

def offset_polygon_shapely(delaunay_node_coords, d):
    """
    Offset a polygon using Shapely's buffer method.
    Positive d offsets outward, negative d offsets inward.
    Returns a list of polygons, each represented as a list of (x, y) tuples.
    """
    poly = Polygon(delaunay_node_coords)
    if not poly.is_valid:
        raise ValueError("Invalid polygon.")

    # Buffer the polygon by distance d
    buffered_poly = poly.buffer(d)

    return buffered_poly

def poisson_disk_sampling(polygon_outer, polygons_holes, radius, k=50, verbose=1):
    """
    Perform Poisson Disk Sampling within a polygon with optional holes.

    Parameters:
    ----------
    polygon_outer : list or numpy.ndarray
        Outer boundary of the polygon as a list of (x, y) tuples.
    polygons_holes : list of lists or numpy.ndarray
        List of hole polygons, each as a list of (x, y) tuples.
    radius : float
        Minimum distance between points.
    k : int, optional
        Number of attempts before removing a point from the active list. Default is 50.
    verbose : int, optional
        Integer indicating the verbosity level (0: silent, 1: basic, 2: detailed, 3: highly detailed).

    Returns:
    -------
    list of tuples
        Filtered list of points generated by Poisson Disk Sampling.
    """
        
    # Calculate bounding box
    min_x = min(p[0] for p in polygon_outer)
    max_x = max(p[0] for p in polygon_outer)
    min_y = min(p[1] for p in polygon_outer)
    max_y = max(p[1] for p in polygon_outer)
    
    log(f"Calculated bounding box: x=({min_x}, {max_x}), y=({min_y}, {max_y})", verbose, level=2)
    
    # Initialize the grid for the bounding box
    cell_size = radius / math.sqrt(2)
    grid_width = int((max_x - min_x) / cell_size) + 1
    grid_height = int((max_y - min_y) / cell_size) + 1
    grid = [[None for _ in range(grid_height)] for _ in range(grid_width)]
    
    log(f"Initialized grid with cell size {cell_size:.4f}, dimensions {grid_width}x{grid_height}.", verbose, level=2)
    
    start_time = time()

    # Generate the first point
    while True:
        x = np.random.uniform(min_x, max_x)
        y = np.random.uniform(min_y, max_y)
        first_point = (x, y)
        if is_valid(first_point, min_x, min_y, cell_size, grid, grid_width, grid_height, radius):
            break
        log(f"First point {first_point} is invalid. Retrying...", verbose, level=3)
    
    points = [first_point]
    active = [first_point]
    grid_x = int((first_point[0] - min_x) / cell_size)
    grid_y = int((first_point[1] - min_y) / cell_size)
    grid[grid_x][grid_y] = first_point

    while active:
        idx = np.random.randint(0, len(active))
        point = active[idx]
        found = False
    
        log(f"Processing active point {idx + 1}/{len(active)}: {point}", verbose, level=3)
    
        for attempt in range(k):
            new_point = generate_point_around(point, radius)
            if (min_x <= new_point[0] < max_x) and (min_y <= new_point[1] < max_y):
                if is_valid(new_point, min_x, min_y, cell_size, grid, grid_width, grid_height, radius):
                    points.append(new_point)
                    active.append(new_point)
                    grid_x = int((new_point[0] - min_x) / cell_size)
                    grid_y = int((new_point[1] - min_y) / cell_size)
                    grid[grid_x][grid_y] = new_point
                    found = True
                    log(f"Added new point: {new_point}", verbose, level=4)
                    break
                else:
                    log(f"Attempt {attempt + 1}: Point {new_point} is invalid.", verbose, level=4)
            else:
                log(f"Attempt {attempt + 1}: Point {new_point} is out of bounds.", verbose, level=3)
    
        if not found:
            active.pop(idx)
            log(f"No valid point found after {k} attempts. Removing point {point} from active list.", verbose, level=3)
    
    generated_time = time()
    log(f"Time taken to generate the points: {generated_time - start_time:.4f} seconds.", verbose, level=1)
    
    # Offset polygons for filtering
    log("Offsetting polygons for filtering.", verbose, level=2)
    polygon_outer_shapely = offset_polygon_shapely(polygon_outer, -radius / 2)
    polygons_holes_shapely = [offset_polygon_shapely(polygon_hole, +radius / 2) for polygon_hole in polygons_holes]
    
    # Filter points to keep only those inside the polygon and not in holes
    log("Filtering points inside the polygon and outside the holes.", verbose, level=2)
    filtered_points = []
    for idx, p in enumerate(points):
        point_shapely = Point(p)
        if polygon_outer_shapely.contains(point_shapely) and all(not hole.contains(point_shapely) for hole in polygons_holes_shapely):
            filtered_points.append(p)
            log(f"Point {idx + 1} ({p}) is inside the polygon and kept.", verbose, level=3)
        else:
            log(f"Point {idx + 1} ({p}) is outside the polygon or inside a hole and discarded.", verbose, level=3)
    
    filter_time = time()
    log(f"Time taken to filter the points: {filter_time - generated_time:.4f} seconds.", verbose, level=1)
    
    log(f"Generated {len(filtered_points)} points after filtering.", verbose, level=1)
    
    return filtered_points

def generate_cloud(polygon, min_distance, verbose=1):
    """
    Generate a point cloud within a polygon with a specified density and margin.

    Parameters:
    ----------
    polygon : list or numpy.ndarray
        A list of (x, y) tuples representing the polygon's delaunay_node_coords.
    min_distance : float
        Minimum distance between points in the generated cloud.
    verbose : int, optional
        Integer indicating the verbosity level (0: silent, 1: basic, 2: detailed, 3: highly detailed).

    Returns:
    -------
    tuple
        A tuple containing:
            - numpy.ndarray: Combined array of boundary and interior points.
            - numpy.ndarray: Array of boundary points.
            - numpy.ndarray: Array of interior points.
    """
    log("Starting point cloud generation.", verbose, level=1)
    
    # Remove duplicate points from the polygon
    polygon = remove_duplicate_points(polygon)
    log(f"Removed duplicate points. Number of delaunay_node_coords: {len(polygon)}", verbose, level=2)
    
    # Get the minimum distance between points in the input polygon edges
    min_distance_polygon = min(
        np.linalg.norm(np.array(polygon[i]) - np.array(polygon[(i + 1) % len(polygon)]))
        for i in range(len(polygon))
    )
    log(f"Minimum edge length in polygon: {min_distance_polygon:.4f}", verbose, level=2)
    
    # Determine the effective minimum distance
    effective_min_distance = min(min_distance, min_distance_polygon)
    log(f"Effective minimum distance between points: {effective_min_distance:.4f}", verbose, level=2)
    
    # Generate boundary points
    boundary_node_coords = generate_boundary_node_coords(polygon, effective_min_distance, verbose)
    boundary_node_coords = remove_duplicate_points(boundary_node_coords)
    boundary_node_coords = np.array(boundary_node_coords)
    
    # Generate interior points using Poisson Disk Sampling
    log("Generating interior points using Poisson Disk Sampling.", verbose, level=1)
    interior_node_coords = poisson_disk_sampling(
        polygon_outer=polygon,
        polygons_holes=[],  # Assuming no holes; modify as needed
        radius=effective_min_distance,
        k=50,
        verbose=verbose
    )
    interior_node_coords = remove_duplicate_points(interior_node_coords)
    interior_node_coords = np.array(interior_node_coords)
    
    # Combine boundary and interior points
    combined_points = np.concatenate([boundary_node_coords, interior_node_coords])
    log(f"Total points in cloud: {combined_points.shape[0]}", verbose, level=1)
    
    return combined_points, boundary_node_coords, interior_node_coords


def remove_duplicate_points(points):
    """
    Remove duplicate points from an array of points while preserving the original order.

    Parameters:
    - points (np.ndarray): An array of points with shape (n, 2).

    Returns:
    - unique_points (np.ndarray): An array of unique points with shape (m, 2), preserving the original order.
    """
    seen = set()
    unique_points = []
    for point in points:
        # Convert the point to a tuple so it can be added to a set
        pt_tuple = tuple(point)
        if pt_tuple not in seen:
            seen.add(pt_tuple)
            unique_points.append(point)
    return np.array(unique_points)

