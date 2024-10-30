from shapely.geometry import Polygon, LineString, box, Point
from shapely.affinity import translate
import numpy as np

from .pre_process import generate_cloud, generate_mandelbrot_boundary, offset_polygon_shapely

import math
import numpy as np
from shapely.geometry import Polygon, Point
from shapely.prepared import prep
from time import time

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

    def log(message, verbose_level, level_required):
        if verbose_level >= level_required:
            print(message)

    def is_valid(point, grid, grid_width, grid_height, radius, prepared_polygon, prepared_holes):
        """
        Check if the point is valid:
        - Within the polygon and not in any hole.
        - At least `radius` away from existing points.
        """
        p = Point(point)
        if not prepared_polygon.contains(p):
            return False
        for hole in prepared_holes:
            if hole.contains(p):
                return False

        grid_x = int(point[0] / cell_size)
        grid_y = int(point[1] / cell_size)

        # Define the range to check neighboring cells
        x_min = max(grid_x - 2, 0)
        y_min = max(grid_y - 2, 0)
        x_max = min(grid_x + 3, grid_width)
        y_max = min(grid_y + 3, grid_height)

        for x in range(x_min, x_max):
            for y in range(y_min, y_max):
                neighbor_idx = grid[x][y]
                if neighbor_idx is not None:
                    neighbor_point = points[neighbor_idx]
                    dx = neighbor_point[0] - point[0]
                    dy = neighbor_point[1] - point[1]
                    distance_sq = dx * dx + dy * dy
                    if distance_sq < radius * radius:
                        return False
        return True

    def generate_point_around(point, radius):
        """
        Generate a random point around the given point within the annulus [radius, 2*radius].
        """
        r1 = np.random.uniform(radius, 2 * radius)
        r2 = np.random.uniform(0, 2 * math.pi)
        new_x = point[0] + r1 * math.cos(r2)
        new_y = point[1] + r1 * math.sin(r2)
        return (new_x, new_y)

    # Create shapely polygons for inclusion checks
    polygon = Polygon(polygon_outer, polygons_holes)
    prepared_polygon = prep(polygon)
    prepared_holes = [prep(Polygon(hole)) for hole in polygons_holes]

    # Calculate bounding box
    min_x, min_y, max_x, max_y = polygon.bounds

    log(f"Calculated bounding box: x=({min_x}, {max_x}), y=({min_y}, {max_y})", verbose, level_required=2)

    # Initialize the grid based on the polygon's bounding box
    cell_size = radius / math.sqrt(2)
    grid_width = int(math.ceil((max_x - min_x) / cell_size))
    grid_height = int(math.ceil((max_y - min_y) / cell_size))
    grid = [[None for _ in range(grid_height)] for _ in range(grid_width)]

    log(f"Initialized grid with cell size {cell_size:.4f}, dimensions {grid_width}x{grid_height}.", verbose, level_required=2)

    start_time = time()

    # Generate a list of all possible grid cell centers within the polygon
    valid_grid_cells = []
    for i in range(grid_width):
        for j in range(grid_height):
            cell_center_x = min_x + (i + 0.5) * cell_size
            cell_center_y = min_y + (j + 0.5) * cell_size
            p = Point(cell_center_x, cell_center_y)
            if prepared_polygon.contains(p) and all(not hole.contains(p) for hole in prepared_holes):
                valid_grid_cells.append((i, j))

    log(f"Number of valid grid cells within the polygon: {len(valid_grid_cells)}", verbose, level_required=2)

    if not valid_grid_cells:
        log("No valid grid cells found within the polygon. Exiting.", verbose, level_required=1)
        return []

    # Select the initial sample from the valid grid cells
    initial_cell = valid_grid_cells[np.random.randint(len(valid_grid_cells))]
    grid_x, grid_y = initial_cell
    first_point_x = min_x + (grid_x + 0.5) * cell_size
    first_point_y = min_y + (grid_y + 0.5) * cell_size
    first_point = (first_point_x, first_point_y)

    points = [first_point]
    active = [0]  # Store indices of points
    grid[grid_x][grid_y] = 0  # Store the index of the point in the grid

    log(f"Added initial point: {first_point} at grid cell ({grid_x}, {grid_y})", verbose, level_required=2)

    while active:
        idx = np.random.choice(active)
        point = points[idx]
        found = False

        log(f"Processing active point {idx + 1}/{len(active)}: {point}", verbose, level_required=3)

        for attempt in range(k):
            new_point = generate_point_around(point, radius)
            if not prepared_polygon.contains(Point(new_point)):
                log(f"Attempt {attempt + 1}: Point {new_point} is outside the polygon.", verbose, level_required=3)
                continue
            if any(hole.contains(Point(new_point)) for hole in prepared_holes):
                log(f"Attempt {attempt + 1}: Point {new_point} is inside a hole.", verbose, level_required=3)
                continue

            # Convert point to grid coordinates
            grid_x_new = int((new_point[0] - min_x) / cell_size)
            grid_y_new = int((new_point[1] - min_y) / cell_size)

            if grid_x_new < 0 or grid_x_new >= grid_width or grid_y_new < 0 or grid_y_new >= grid_height:
                log(f"Attempt {attempt + 1}: Point {new_point} is out of grid bounds.", verbose, level_required=3)
                continue

            if grid[grid_x_new][grid_y_new] is not None:
                log(f"Attempt {attempt + 1}: Grid cell ({grid_x_new}, {grid_y_new}) is already occupied.", verbose, level_required=4)
                continue

            if is_valid(new_point, grid, grid_width, grid_height, radius, prepared_polygon, prepared_holes):
                points.append(new_point)
                active.append(len(points) - 1)
                grid[grid_x_new][grid_y_new] = len(points) - 1
                found = True
                log(f"Added new point: {new_point} at grid cell ({grid_x_new}, {grid_y_new})", verbose, level_required=4)
                break
            else:
                log(f"Attempt {attempt + 1}: Point {new_point} is too close to existing points.", verbose, level_required=4)

        if not found:
            active.remove(idx)
            log(f"No valid point found after {k} attempts. Removing point {idx + 1} from active list.", verbose, level_required=3)

    generated_time = time()
    log(f"Time taken to generate the points: {generated_time - start_time:.4f} seconds.", verbose, level_required=1)

    log(f"Generated {len(points)} points within the polygon.", verbose, level_required=1)

    return points


# Example usage
if __name__ == "__main__":
    # Define your star-shaped polygon

    resolution = 200
    max_iteration = 100
    min_distance = 0.02

    verbose = 2

    from shapely.geometry import Polygon
    from shapely import make_valid

    # Generate a cloud of points and a boundary
    boundary = generate_mandelbrot_boundary(resolution, max_iteration, verbose=verbose)

    cloud_points, boundary_points, interior_points = generate_cloud(boundary, min_distance, verbose = verbose)

    polygon = Polygon(boundary_points)

    #Get the centroid of the polygon
    centroid = polygon.centroid

    # Offset the polygon
    offset = -0.01
    polygon_offset = polygon.buffer(offset)

    # Get the centroid of the offset polygon
    centroid_offset = polygon_offset.centroid

    # Plot the original and offset polygons
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    x,y = polygon.exterior.xy
    ax.plot(x, y, color='blue')

    if polygon_offset.geom_type == 'Polygon':
        x,y = polygon_offset.exterior.xy
        ax.plot(x, y, color='red')
    else:
        for geom in polygon_offset.geoms:
            x,y = geom.exterior.xy
            ax.plot(x, y, color='red')

    ax.plot(centroid.x, centroid.y, 'o', color='blue')
    ax.plot(centroid_offset.x, centroid_offset.y, 'o', color='red')

    plt.show()

