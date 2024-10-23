import numpy as np

def orient(a_idx, b_idx, c_idx, delaunay_node_coords):
    """
    Computes the orientation of the triplet (a, b, c).
    Returns:
    - >0 if counter-clockwise
    - <0 if clockwise
    - 0 if colinear
    """
    ax, ay = delaunay_node_coords[a_idx]
    bx, by = delaunay_node_coords[b_idx]
    cx, cy = delaunay_node_coords[c_idx]
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)

def in_circle(u_idx, v_idx, w_idx, x_idx, delaunay_node_coords):
    """
    Determines if the point x lies inside the circumcircle of the triangle (u, v, w).
    Returns:
    
    """

    ux, uy = delaunay_node_coords[u_idx]
    vx, vy = delaunay_node_coords[v_idx]
    wx, wy = delaunay_node_coords[w_idx]
    xx, xy = delaunay_node_coords[x_idx]

    mat = [
        [ux - xx, uy - xy, (ux - xx)**2 + (uy - xy)**2],
        [vx - xx, vy - xy, (vx - xx)**2 + (vy - xy)**2],
        [wx - xx, wy - xy, (wx - xx)**2 + (wy - xy)**2],
    ]
    # Compute the determinant of the matrix.
    return np.linalg.det(mat)

def in_triangle(u_idx, v_idx, w_idx, x_idx, delaunay_node_coords):
    orient_uvw = orient(u_idx, v_idx, w_idx, delaunay_node_coords)
    orient_uvx = orient(u_idx, v_idx, x_idx, delaunay_node_coords)
    orient_vwx = orient(v_idx, w_idx, x_idx, delaunay_node_coords)
    orient_wux = orient(w_idx, u_idx, x_idx, delaunay_node_coords)
    
    return (orient_uvw >= 0 and orient_uvx >= 0 and orient_vwx >= 0 and orient_wux >= 0) or \
           (orient_uvw <= 0 and orient_uvx <= 0 and orient_vwx <= 0 and orient_wux <= 0)

def ray_casting_pip(point, polygon):
    """
    Determines if a point is inside a polygon using the Ray Casting algorithm.
    The algorithm casts a ray from the point to the right and counts how many edges of the polygon it intersects.
    
    Parameters:
    - point: Tuple (x, y) representing the point to test.
    - polygon: List of (x, y) tuples representing the polygon's delaunay_node_coords.
    
    Returns:
    - True if the point is inside the polygon, False otherwise.
    """
    x, y = point
    n = len(polygon)
    inside = False

    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[(i + 1) % n]

        # Check if point is on the same y-level as the edge
        if ((yi > y) != (yj > y)):
            # Compute the x-coordinate of the intersection of the edge with the horizontal line y = point.y
            x_intersect = (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi  # Avoid division by zero

            if x < x_intersect:
                inside = not inside

    return inside

def is_point_inside(polygon_outer, polygons_holes, point):
    """
    Determines if a point is inside a polygon with optional holes.
    
    Parameters:
    - polygon_outer: List of (x, y) tuples representing the outer boundary.
    - polygons_holes: List of lists, where each sublist is a hole defined by (x, y) tuples.
    - point: Tuple (x, y) representing the point to test.
    
    Returns:
    - True if the point is inside the outer boundary and not inside any hole.
    - False otherwise.
    """
    if not ray_casting_pip(point, polygon_outer):
        return False

    for hole in polygons_holes:
        if ray_casting_pip(point, hole):
            return False

    return True

def compute_centroid(triangle, delaunay_node_coords):
    """
    Computes the centroid of a triangle.
    
    Parameters:
    - triangle: Tuple of three vertex indices (v0, v1, v2).
    - delaunay_node_coords: NumPy array of vertex coordinates, shape (n_delaunay_node_coords, 2).
    
    Returns:
    - Tuple (x, y) representing the centroid.
    """
    v0, v1, v2 = triangle
    centroid = (delaunay_node_coords[v0] + delaunay_node_coords[v1] + delaunay_node_coords[v2]) / 3.0
    return tuple(centroid)

def compute_angle(v, common_vertex, delaunay_node_coords):
    """
    Computes the angle of vertex 'v' relative to 'common_vertex'.
    
    Parameters:
    - v: Vertex index for which the angle is computed.
    - common_vertex: Vertex index serving as the origin for angle computation.
    - delaunay_node_coords: List of vertex coordinates.
    
    Returns:
    - Angle in radians.
    """
    x, y = delaunay_node_coords[v]
    cx, cy = delaunay_node_coords[common_vertex]
    return np.atan2(y - cy, x - cx)

def get_circumcircle(a, b, c):
    # Calculate circumcenter and radius
    d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
    ux = ((a[0]**2 + a[1]**2) * (b[1] - c[1]) + (b[0]**2 + b[1]**2) * (c[1] - a[1]) + (c[0]**2 + c[1]**2) * (a[1] - b[1])) / d
    uy = ((a[0]**2 + a[1]**2) * (c[0] - b[0]) + (b[0]**2 + b[1]**2) * (a[0] - c[0]) + (c[0]**2 + c[1]**2) * (b[0] - a[0])) / d
    center = (ux, uy)
    radius = np.sqrt((center[0] - a[0])**2 + (center[1] - a[1])**2)
    return center, radius

def polygon_area(delaunay_node_coords):
    """
    Calculate the area of a polygon using the shoelace formula.

    Parameters:
    delaunay_node_coords (numpy.ndarray): An array of shape (n, 2) representing the (x, y) coordinates of the polygon's delaunay_node_coords.

    Returns:
    float: The absolute area of the polygon.
    """
    # Extract x and y coordinates from the delaunay_node_coords
    x = delaunay_node_coords[:, 0]
    y = delaunay_node_coords[:, 1]
    
    # Apply the shoelace formula
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    
    return area

def distance_point_to_segment(point, seg_start, seg_end):
    """
    Compute the minimum distance from a point to a line segment.

    This function calculates the shortest distance between a given point and a line segment defined by two endpoints.

    Parameters:
    point (tuple or list): The (x, y) coordinates of the point.
    seg_start (tuple or list): The (x, y) coordinates of the segment's start point.
    seg_end (tuple or list): The (x, y) coordinates of the segment's end point.

    Returns:
    float: The minimum distance from the point to the segment.
    """
    px, py = point
    x1, y1 = seg_start
    x2, y2 = seg_end
    dx = x2 - x1
    dy = y2 - y1
    
    if dx == 0 and dy == 0:
        # The segment is a single point
        return np.hypot(px - x1, py - y1)
    
    # Parameter t determines the projection of the point onto the segment
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    
    # Find the closest point on the segment
    nearest_x = x1 + t * dx
    nearest_y = y1 + t * dy
    
    # Calculate the Euclidean distance between the point and the closest point on the segment
    distance = np.hypot(px - nearest_x, py - nearest_y)
    
    return distance

def is_point_on_edge(u_idx, edge_idx, delaunay_node_coords, epsilon=1e-12):
    """
    Checks if the point with index v_idx lies on the edge defined by the tuple of vertex indices `edge`.
    
    Parameters:
    - v_idx: Index of the point to check.
    - edge_idx: Tuple of two vertex indices defining the edge.
    - delaunay_node_coords: List of vertex coordinates.
    - epsilon: Tolerance for floating-point comparisons.
    
    Returns:
    - True if the point lies on the edge, False otherwise.
    """
    v, w = edge_idx

    if orient(v, w, u_idx, delaunay_node_coords) != 0:
        return False
    else:
        return True
    
def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def is_valid(point, min_x, min_y, cell_size, grid, grid_width, grid_height, radius):
    cell_x, cell_y = int((point[0] - min_x) / cell_size), int((point[1] - min_y) / cell_size)
    for i in range(max(0, cell_x - 2), min(grid_width, cell_x + 3)):
        for j in range(max(0, cell_y - 2), min(grid_height, cell_y + 3)):
            if grid[i][j] is not None:
                if distance(point, grid[i][j]) < radius:
                    return False
    return True