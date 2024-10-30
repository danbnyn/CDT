import numpy as np
from typing import List, Tuple

def orient(
    a_idx: int, 
    b_idx: int, 
    c_idx: int, 
    delaunay_node_coords: List[Tuple[float, float]]
) -> float:
    """
    This function determines the orientation of three points `a`, `b`, and `c` in a 2D plane. 
    The points are represented by their indices in the `delaunay_node_coords` list, which contains 
    the x and y coordinates of each point. The orientation is determined using the cross-product of the vectors 
    `ab` and `ac`, defined as:

        orientation = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)

    This cross-product helps determine if the points are arranged in a counter-clockwise, clockwise, 
    or collinear manner.

    Parameters:
    - a_idx (int): Index of the first point `a` in the `delaunay_node_coords` list.
    - b_idx (int): Index of the second point `b` in the `delaunay_node_coords` list.
    - c_idx (int): Index of the third point `c` in the `delaunay_node_coords` list.
    - delaunay_node_coords (List[Tuple[float, float]]): List of 2D coordinates of points, 
      where each element is a tuple `(x, y)` representing a point's x and y coordinates.

    Returns:
    - float: A positive value if the points are in a counter-clockwise order, a negative value if clockwise, 
      and zero if collinear.
    """
    ax, ay = delaunay_node_coords[a_idx]
    bx, by = delaunay_node_coords[b_idx]
    cx, cy = delaunay_node_coords[c_idx]
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)

def in_circle(
    u_idx: int, 
    v_idx: int, 
    w_idx: int, 
    x_idx: int, 
    delaunay_node_coords: List[Tuple[float, float]]
) -> float:
    """
    This function checks if a point `x` lies inside the circumcircle formed by the points `u`, `v`, and `w`. 
    It constructs a matrix based on the coordinates of these points and calculates its determinant. The 
    determinant indicates the relative position of point `x` with respect to the circumcircle:

    - If the determinant is positive, point `x` lies inside the circumcircle.
    - If the determinant is zero, point `x` lies exactly on the circumcircle.
    - If the determinant is negative, point `x` lies outside the circumcircle.

    The circumcircle of a triangle is the unique circle that passes through all three vertices of the triangle.

    Parameters:
    - u_idx (int): Index of the first point `u` in the `delaunay_node_coords` list.
    - v_idx (int): Index of the second point `v` in the `delaunay_node_coords` list.
    - w_idx (int): Index of the third point `w` in the `delaunay_node_coords` list.
    - x_idx (int): Index of the point `x` in the `delaunay_node_coords` list to be tested.
    - delaunay_node_coords (List[Tuple[float, float]]): List of 2D coordinates of points, 
      where each element is a tuple `(x, y)` representing a point's x and y coordinates.

    Returns:
    - float: The determinant of the matrix, which indicates the relative position of point `x` with respect to 
      the circumcircle. A positive value indicates that `x` lies inside, zero indicates it lies on, and negative 
      indicates it lies outside the circumcircle.
    """
    # Extract coordinates of the points
    ux, uy = delaunay_node_coords[u_idx]
    vx, vy = delaunay_node_coords[v_idx]
    wx, wy = delaunay_node_coords[w_idx]
    xx, xy = delaunay_node_coords[x_idx]

    # Construct the matrix for the in-circle test
    mat = [
        [ux - xx, uy - xy, (ux - xx)**2 + (uy - xy)**2],
        [vx - xx, vy - xy, (vx - xx)**2 + (vy - xy)**2],
        [wx - xx, wy - xy, (wx - xx)**2 + (wy - xy)**2],
    ]

    # Compute and return the determinant of the matrix
    return np.linalg.det(mat)

def get_circumcircle(
    a: Tuple[float, float], 
    b: Tuple[float, float], 
    c: Tuple[float, float]
) -> Tuple[Tuple[float, float], float]:
    """
    Computes the circumcircle of the triangle defined by points a, b, and c.

    Parameters:
    - a (Tuple[float, float]): Coordinates of the first vertex of the triangle.
    - b (Tuple[float, float]): Coordinates of the second vertex of the triangle.
    - c (Tuple[float, float]): Coordinates of the third vertex of the triangle.

    Returns:
    - Tuple[Tuple[float, float], float]: A tuple containing the center coordinates of the circumcircle 
      and its radius.
    """
    
    # Calculate the determinant d used to find the circumcenter
    d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
    
    # Compute the x-coordinate of the circumcenter
    ux = (
        ((a[0]**2 + a[1]**2) * (b[1] - c[1]) +
         (b[0]**2 + b[1]**2) * (c[1] - a[1]) +
         (c[0]**2 + c[1]**2) * (a[1] - b[1])) / d
    )
    
    # Compute the y-coordinate of the circumcenter
    uy = (
        ((a[0]**2 + a[1]**2) * (c[0] - b[0]) +
         (b[0]**2 + b[1]**2) * (a[0] - c[0]) +
         (c[0]**2 + c[1]**2) * (b[0] - a[0])) / d
    )
    
    # The circumcenter coordinates
    center = (ux, uy)
    
    # Calculate the radius as the distance from the circumcenter to one of the triangle's vertices (e.g., point a)
    radius = np.sqrt((center[0] - a[0])**2 + (center[1] - a[1])**2)
    
    return center, radius

def in_triangle(
    u_idx: int, 
    v_idx: int, 
    w_idx: int, 
    x_idx: int, 
    delaunay_node_coords: List[Tuple[float, float]]
) -> bool:
    """
    Determines if a point x is inside the triangle formed by points u, v, and w using orientation checks.

    Description:
    This function checks if a point `x` lies inside or on the edges of a triangle formed by the points `u`, `v`, and `w`. 
    The algorithm uses the orientation of the triplets `(u, v, w)`, `(u, v, x)`, `(v, w, x)`, and `(w, u, x)`. 
    If all the orientation results have the same sign (either all non-negative or all non-positive), 
    the point `x` is considered to be inside or on the triangle.

    Parameters:
    - u_idx (int): Index of the first vertex `u` of the triangle in the `delaunay_node_coords` list.
    - v_idx (int): Index of the second vertex `v` of the triangle in the `delaunay_node_coords` list.
    - w_idx (int): Index of the third vertex `w` of the triangle in the `delaunay_node_coords` list.
    - x_idx (int): Index of the point `x` to check in the `delaunay_node_coords` list.
    - delaunay_node_coords (List[Tuple[float, float]]): List of 2D coordinates of points, where each point is 
      represented as a tuple `(x, y)`.

    Returns:
    - bool: True if the point `x` is inside or on the edges of the triangle formed by points `u`, `v`, and `w`, 
      False otherwise.
    """
    orient_uvw = orient(u_idx, v_idx, w_idx, delaunay_node_coords)
    orient_uvx = orient(u_idx, v_idx, x_idx, delaunay_node_coords)
    orient_vwx = orient(v_idx, w_idx, x_idx, delaunay_node_coords)
    orient_wux = orient(w_idx, u_idx, x_idx, delaunay_node_coords)

    # Check if all orientations have the same sign
    return (orient_uvw >= 0 and orient_uvx >= 0 and orient_vwx >= 0 and orient_wux >= 0) or \
           (orient_uvw <= 0 and orient_uvx <= 0 and orient_vwx <= 0 and orient_wux <= 0)

def is_point_on_edge(
    u_idx: int, 
    edge_idx: Tuple[int, int], 
    delaunay_node_coords: List[Tuple[float, float]], 
    epsilon: float = 1e-12
) -> bool:
    """
    This function uses the orientation test to determine if the point defined by `u_idx` lies on the edge 
    formed by the vertices indexed by `edge_idx`. It utilizes a small tolerance (epsilon) for floating-point 
    comparisons to account for numerical inaccuracies.

    Parameters:
    - u_idx (int): Index of the point to check.
    - edge_idx (Tuple[int, int]): Tuple of two vertex indices defining the edge.
    - delaunay_node_coords (List[Tuple[float, float]]): List of vertex coordinates (x, y).
    - epsilon (float): Tolerance for floating-point comparisons (default is 1e-12).

    Returns:
    - bool: True if the point lies on the edge, False otherwise.
    """
    v, w = edge_idx

    # Check if the point is collinear with the edge
    if abs(orient(v, w, u_idx, delaunay_node_coords)) > epsilon:
        return False  # The point is not on the edge
    else:
        return True  # The point is on the edge

def distance(
        p1: Tuple[float, float], 
        p2: Tuple[float, float]
) -> float:
    """
    Calculates the Euclidean distance between two points.

    Parameters:
    - p1 (Tuple[float, float]): Coordinates of the first point (x1, y1).
    - p2 (Tuple[float, float]): Coordinates of the second point (x2, y2).

    Returns:
    - float: The Euclidean distance between points p1 and p2.
    """
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def is_valid(
    point: Tuple[float, float], 
    min_x: float, 
    min_y: float, 
    cell_size: float, 
    grid: List[List[Tuple[float, float]]], 
    grid_width: int, 
    grid_height: int, 
    radius: float
) -> bool:
    """
    This function determines if the specified `point` is valid by checking whether it is within a specified 
    radius of any points in a grid. It calculates the grid cell corresponding to the point and checks 
    nearby cells (up to a 2-cell radius) for existing points.

    Parameters:
    - point (Tuple[float, float]): Coordinates of the point to check (x, y).
    - min_x (float): Minimum x-coordinate of the grid.
    - min_y (float): Minimum y-coordinate of the grid.
    - cell_size (float): Size of each cell in the grid.
    - grid (List[List[Tuple[float, float]]]): 2D list representing the grid, where each cell can hold a point.
    - grid_width (int): Width of the grid (number of cells in the x-direction).
    - grid_height (int): Height of the grid (number of cells in the y-direction).
    - radius (float): The radius within which the point should not fall to be considered valid.

    Returns:
    - bool: True if the point is valid (not within radius of existing points), False otherwise.
    """
    # Determine the grid cell coordinates for the given point
    cell_x = int((point[0] - min_x) / cell_size)
    cell_y = int((point[1] - min_y) / cell_size)

    # Check surrounding cells for existing points
    for i in range(max(0, cell_x - 2), min(grid_width, cell_x + 3)):
        for j in range(max(0, cell_y - 2), min(grid_height, cell_y + 3)):
            if grid[i][j] is not None:
                # Check if the distance to the existing point is less than the radius
                if distance(point, grid[i][j]) < radius:
                    return False  # The point is invalid if it falls within the radius

    return True  # The point is valid if no nearby points were found