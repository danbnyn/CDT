from random import seed
from time import time
import numpy as np
import matplotlib.pyplot as plt

from src.visualize import (
    plot_triangulation, 
    plot_triangulation_with_points_ordered,
    plot_triangulation_with_points,
    plot_points,
    plot_adjancy_matrix,
    plot_points_ordered
    )

from src.pre_process import (
    generate_cloud, 
    generate_mandelbrot_boundary, 
    remove_duplicate_points,
    initialize_triangulation,
    offset_polygon_shapely
    )

from src.dt import delaunay_triangulation
 
from src.cdt import (
    constrained_delaunay_triangulation, 
    CavityCDT, 
    convex_dt
    )

from src.utils import (
    generate_edges_from_points, 
    convert_edges_to_ids, 
    convert_points_to_ids
    )

from src.post_process import (
    clean_mesh,
    convert_to_mesh_format,
    apply_rcm,
    build_adjacency_list
    )

import numpy as np
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import unary_union
import math
from time import time
from typing import Callable, List, Tuple, Optional

def compute_kernel(polygon: Polygon) -> Polygon:
    """
    Compute the kernel of a star-shaped polygon.
    The kernel is the set of points from which the entire polygon is visible.
    
    Parameters:
    ----------
    polygon : shapely.geometry.Polygon
        The input polygon
        
    Returns:
    -------
    shapely.geometry.Polygon
        The kernel of the polygon
    """
    vertices = list(polygon.exterior.coords)[:-1]
    n = len(vertices)
    
    # For each vertex, create the visibility cone
    visibility_regions = []
    for i in range(n):
        prev = vertices[(i-1) % n]
        curr = vertices[i]
        next_ = vertices[(i+1) % n]
        
        # Create vectors
        v1 = np.array([prev[0] - curr[0], prev[1] - curr[1]])
        v2 = np.array([next_[0] - curr[0], next_[1] - curr[1]])
        
        # Normalize vectors
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        
        # Create the visibility cone as a triangle
        far_point1 = curr + v1 * polygon.bounds[2] * 2  # Use twice the polygon's width
        far_point2 = curr + v2 * polygon.bounds[2] * 2
        
        cone = Polygon([curr, far_point1, far_point2])
        visibility_regions.append(cone)
    
    # The kernel is the intersection of all visibility regions
    kernel = visibility_regions[0]
    for region in visibility_regions[1:]:
        kernel = kernel.intersection(region)
    
    return kernel.intersection(polygon)

def get_density_function(method: str, max_radius: float, min_radius: float, 
                        decay_rate: float = 1.0) -> Callable[[float, float], float]:
    """
    Create a density function based on the specified method.
    
    Parameters:
    ----------
    method : str
        The type of density function ('linear', 'exponential', 'logarithmic')
    max_radius : float
        Maximum radius for points far from the border
    min_radius : float
        Minimum radius for points near the border
    decay_rate : float
        Rate at which the radius changes (affects exponential and logarithmic functions)
        
    Returns:
    -------
    Callable[[float, float], float]
        A function that takes (distance, max_distance) and returns the radius
    """
    if method == 'linear':
        return lambda d, max_d: min_radius + (max_radius - min_radius) * (d / max_d)
    
    elif method == 'exponential':
        return lambda d, max_d: min_radius + (max_radius - min_radius) * (1 - math.exp(-decay_rate * d / max_d))
    
    elif method == 'logarithmic':
        return lambda d, max_d: min_radius + (max_radius - min_radius) * (math.log(1 + d) / math.log(1 + max_d))
    
    else:
        raise ValueError(f"Unknown density method: {method}")

def variable_density_poisson_disk_sampling(
    polygon_outer: List[Tuple[float, float]],
    polygons_holes: List[List[Tuple[float, float]]],
    max_radius: float,
    min_radius: float,
    density_method: str = 'exponential',
    decay_rate: float = 1.0,
    k: int = 50,
    verbose: int = 1
) -> List[Tuple[float, float]]:
    """
    Perform Variable Density Poisson Disk Sampling within a polygon with optional holes.
    
    Parameters:
    ----------
    polygon_outer : List[Tuple[float, float]]
        Outer boundary of the polygon as a list of (x, y) tuples
    polygons_holes : List[List[Tuple[float, float]]]
        List of hole polygons, each as a list of (x, y) tuples
    max_radius : float
        Maximum distance between points (for interior)
    min_radius : float
        Minimum distance between points (near border)
    density_method : str
        Method for density variation ('linear', 'exponential', 'logarithmic')
    decay_rate : float
        Rate at which the radius changes
    k : int
        Number of attempts before removing a point from the active list
    verbose : int
        Verbosity level (0: silent, 1: basic, 2: detailed)
        
    Returns:
    -------
    List[Tuple[float, float]]
        List of generated points
    """
    # Convert to shapely objects
    poly_outer = Polygon(polygon_outer)
    poly_holes = [Polygon(hole) for hole in polygons_holes]
    
    # Compute the kernel and its centroid
    kernel = compute_kernel(poly_outer)
    center_point = kernel.centroid
    
    # Calculate maximum possible distance from center to any point in polygon
    max_distance = max(Point(center_point).distance(Point(p)) 
                      for p in polygon_outer)
    
    # Get density function
    density_func = get_density_function(density_method, max_radius, min_radius, decay_rate)
    
    def get_local_radius(point: Tuple[float, float]) -> float:
        """Calculate local radius based on distance from center"""
        dist = Point(point).distance(Point(center_point))
        return density_func(dist, max_distance)
    
    # Calculate bounding box
    min_x = min(p[0] for p in polygon_outer)
    max_x = max(p[0] for p in polygon_outer)
    min_y = min(p[1] for p in polygon_outer)
    max_y = max(p[1] for p in polygon_outer)
    
    # Initialize grid with minimum radius (most conservative)
    cell_size = min_radius / math.sqrt(2)
    grid_width = int((max_x - min_x) / cell_size) + 1
    grid_height = int((max_y - min_y) / cell_size) + 1
    grid = [[None for _ in range(grid_height)] for _ in range(grid_width)]
    
    def is_valid(point: Tuple[float, float], existing_points: List[Tuple[float, float]]) -> bool:
        """Check if a point is valid based on local radius"""
        point_radius = get_local_radius(point)
        
        # Check distance to existing points
        for p in existing_points:
            p_radius = get_local_radius(p)
            # Use minimum of both radii for conservative check
            min_required_dist = min(point_radius, p_radius)
            if Point(point).distance(Point(p)) < min_required_dist:
                return False
        return True
    
    # Generate points
    points = []
    active = []
    
    # Generate first point near the center
    first_point = (center_point.x, center_point.y)
    if poly_outer.contains(Point(first_point)):
        points.append(first_point)
        active.append(first_point)
    
    while active:
        idx = np.random.randint(0, len(active))
        point = active[idx]
        found = False
        
        local_radius = get_local_radius(point)
        
        for _ in range(k):
            # Generate point in annulus
            angle = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(local_radius, 2 * local_radius)
            new_point = (
                point[0] + r * math.cos(angle),
                point[1] + r * math.sin(angle)
            )
            
            # Check if point is within polygon and not in holes
            point_shapely = Point(new_point)
            if (poly_outer.contains(point_shapely) and 
                all(not hole.contains(point_shapely) for hole in poly_holes) and
                is_valid(new_point, points)):
                
                points.append(new_point)
                active.append(new_point)
                found = True
                break
        
        if not found:
            active.pop(idx)
    
    return points

def visualize_points(points: List[Tuple[float, float]], 
                    polygon_outer: List[Tuple[float, float]],
                    polygons_holes: List[List[Tuple[float, float]]]) -> None:
    """
    Visualize the generated points and the polygon.
    
    Parameters:
    ----------
    points : List[Tuple[float, float]]
        List of generated points
    polygon_outer : List[Tuple[float, float]]
        Outer boundary of the polygon
    polygons_holes : List[List[Tuple[float, float]]]
        List of hole polygons
    """
    import matplotlib.pyplot as plt
    
    # Plot polygon
    poly = plt.Polygon(polygon_outer, fill=False, color='black')
    plt.gca().add_patch(poly)
    
    # Plot holes
    for hole in polygons_holes:
        hole_poly = plt.Polygon(hole, fill=True, color='gray', alpha=0.3)
        plt.gca().add_patch(hole_poly)
    
    # Plot points
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    plt.scatter(x, y, c='blue', s=10)
    
    # Plot kernel centroid
    kernel = compute_kernel(Polygon(polygon_outer))
    centroid = kernel.centroid
    plt.scatter([centroid.x], [centroid.y], c='red', s=100, marker='*')
    
    plt.axis('equal')
    plt.show()


def main():

    seed(18)

    resolution = 200
    max_iteration = 100
    min_distance = 0.02

    verbose = 2

    from shapely.geometry import Polygon
    from shapely import make_valid

    # Generate a cloud of points and a boundary
    boundary = generate_mandelbrot_boundary(resolution, max_iteration, verbose=verbose)

    cloud_points, boundary_points, interior_points = generate_cloud(boundary, min_distance, verbose = verbose)


    buffered_poly = offset_polygon_shapely(boundary_points, -0.01)

    # Convert the buffered polygon to a list of points
    boundary = np.array(buffered_poly.exterior.coords)

    plot_points(boundary)

if __name__ == "__main__":
    main()