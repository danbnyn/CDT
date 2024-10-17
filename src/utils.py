from src.operations import add_triangle, delete_triangle
import numpy as np
from src.predicates import is_point_inside, compute_centroid

from time import time

def generate_edges_from_points(points):
    """
    Generates a list of edges from a list of points, assuming the points form a closed polygon. 
    """
    edges = []
    for i in range(len(points)):
        edges.append((points[i], points[(i + 1) % len(points)]))
    return edges


def convert_points_to_ids(points, vertices):
    """
    Converts a list of points to a list of vertex indices.
    """
    return [np.where((vertices == point).all(axis=1))[0][0] for point in points]


def convert_idx_to_points(vertices, idx_list):
    """
    Converts a list of vertex indices to a list of vertex coordinates.
    """
    return [vertices[i] for i in idx_list]

def convert_edges_by_ids_to_positions(edges_by_ids, vertices):
    """
    Converts a list of edges defined by vertex IDs to a list of edges defined by vertex positions.
    
    Parameters:
    - edges_by_ids: List of edges, where each edge is (id0, id1).
    - vertices: NumPy array of shape (n_vertices, 2) containing vertex positions.
    
    Returns:
    - edges_by_pos: List of edges, where each edge is ((x0, y0), (x1, y1)).
    """
    return [(vertices[id0], vertices[id1]) for id0, id1 in edges_by_ids]

def convert_edges_to_ids(edges, vertices):
    def find_vertex_id(position):
        return np.where((vertices == position).all(axis=1))[0][0]
    
    return [(int(find_vertex_id(edge[0])), int(find_vertex_id(edge[1]))) for edge in edges]

def convert_vertices_to_ids(vertices, boundary_points):
    """
    Converts a list of boundary points to a list of vertex indices.
    """
    return [np.where((vertices == point).all(axis=1))[0][0] for point in boundary_points]

def convert_triangle_vertices_idx_to_triangle_idx(triangle_vertices_idx, triangles, edge_to_triangle):
    """
    Converts a list of triangle vertex indices to the corresponding triangle index.

    Parameters:
    - triangle_vertices_idx: List of three vertex indices defining a triangle.
    - triangles: List of triangles, where each triangle is a tuple of three vertex indices (v0, v1, v2).
    - edge_to_triangle: Dictionary mapping edges to triangle indices.

    Returns:
    - The index of the triangle in the triangles list.
    """
    edge1 = tuple(sorted([triangle_vertices_idx[0], triangle_vertices_idx[1]]))
    edge2 = tuple(sorted([triangle_vertices_idx[1], triangle_vertices_idx[2]]))
    edge3 = tuple(sorted([triangle_vertices_idx[2], triangle_vertices_idx[0]]))

    def to_set(value):
        return {value} if isinstance(value, int) else set(value)

    triangle_edge_1 = to_set(edge_to_triangle[edge1])
    triangle_edge_2 = to_set(edge_to_triangle[edge2])
    triangle_edge_3 = to_set(edge_to_triangle[edge3])

    # Find the intersection of the sets to get the common triangle index
    triangle_idx = triangle_edge_1.intersection(triangle_edge_2).intersection(triangle_edge_3)

    if not triangle_idx:
        raise ValueError("No common triangle found for the given vertex indices.")

    return list(triangle_idx)[0]

    
def log(message, verbose, level=1):
    """Helper function to print messages based on verbosity level."""
    if verbose >= level:
        print(message)