import numpy as np
from time import time

from src.predicates import is_point_inside, compute_centroid
from src.operations import add_triangle, delete_triangle

def generate_edges_from_points(points):
    """
    Generates a list of edges from a list of points, assuming the points form a closed polygon. 
    """
    edges = []
    for i in range(len(points)):
        edges.append((points[i], points[(i + 1) % len(points)]))
    return edges


def convert_points_to_ids(points, delaunay_node_coords):
    """
    Converts a list of points to a list of vertex indices.
    """
    return [np.where((delaunay_node_coords == point).all(axis=1))[0][0] for point in points]


def convert_idx_to_points(delaunay_node_coords, idx_list):
    """
    Converts a list of vertex indices to a list of vertex coordinates.
    """
    return [delaunay_node_coords[i] for i in idx_list]

def convert_edges_by_ids_to_positions(edges_by_ids, delaunay_node_coords):
    """
    Converts a list of edges defined by vertex IDs to a list of edges defined by vertex positions.
    
    Parameters:
    - edges_by_ids: List of edges, where each edge is (id0, id1).
    - delaunay_node_coords: NumPy array of shape (n_delaunay_node_coords, 2) containing vertex positions.
    
    Returns:
    - edges_by_pos: List of edges, where each edge is ((x0, y0), (x1, y1)).
    """
    return [(delaunay_node_coords[id0], delaunay_node_coords[id1]) for id0, id1 in edges_by_ids]

def convert_edges_to_ids(edges, delaunay_node_coords):
    def find_vertex_id(position):
        return np.where((delaunay_node_coords == position).all(axis=1))[0][0]
    
    return [(int(find_vertex_id(edge[0])), int(find_vertex_id(edge[1]))) for edge in edges]

def convert_delaunay_node_coords_to_ids(delaunay_node_coords, boundary_node_coords):
    """
    Converts a list of boundary points to a list of vertex indices.
    """
    return [np.where((delaunay_node_coords == point).all(axis=1))[0][0] for point in boundary_node_coords]

def convert_triangle_delaunay_node_coords_idx_to_triangle_idx(triangle_delaunay_node_coords_idx, elem_nodes, delaunay_dic_edge_triangle):
    """
    Converts a list of triangle vertex indices to the corresponding triangle index.

    Parameters:
    - triangle_delaunay_node_coords_idx: List of three vertex indices defining a triangle.
    - elem_nodes: List of elem_nodes, where each triangle is a tuple of three vertex indices (v0, v1, v2).
    - delaunay_dic_edge_triangle: Dictionary mapping edges to triangle indices.

    Returns:
    - The index of the triangle in the elem_nodes list.
    """
    edge1 = tuple(sorted([triangle_delaunay_node_coords_idx[0], triangle_delaunay_node_coords_idx[1]]))
    edge2 = tuple(sorted([triangle_delaunay_node_coords_idx[1], triangle_delaunay_node_coords_idx[2]]))
    edge3 = tuple(sorted([triangle_delaunay_node_coords_idx[2], triangle_delaunay_node_coords_idx[0]]))

    def to_set(value):
        return {value} if isinstance(value, int) else set(value)

    triangle_edge_1 = to_set(delaunay_dic_edge_triangle[edge1])
    triangle_edge_2 = to_set(delaunay_dic_edge_triangle[edge2])
    triangle_edge_3 = to_set(delaunay_dic_edge_triangle[edge3])

    # Find the intersection of the sets to get the common triangle index
    triangle_idx = triangle_edge_1.intersection(triangle_edge_2).intersection(triangle_edge_3)

    if not triangle_idx:
        raise ValueError("No common triangle found for the given vertex indices.")

    return list(triangle_idx)[0]

    
def log(message, verbose, level=1):
    """Helper function to print messages based on verbosity level."""
    if verbose >= level:
        print(message)