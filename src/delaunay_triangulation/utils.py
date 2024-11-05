import numpy as np
from typing import List, Tuple

def generate_edges_from_points(
        points: List[Tuple[float, float]]
) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """
    This function takes a list of points representing the vertices of a polygon and generates 
    a list of edges connecting consecutive points in the list. The polygon is assumed to be closed, 
    meaning the last point is connected back to the first point.

    Parameters:
    - points (List[Tuple[float, float]]): List of points (x, y) representing the vertices of the polygon.

    Returns:
    - List[Tuple[Tuple[float, float], Tuple[float, float]]]: A list of edges, where each edge is represented as a tuple 
      containing two points.
    """
    edges = []
    for i in range(len(points)):
        edges.append((points[i], points[(i + 1) % len(points)]))
    return edges

def convert_edges_to_ids(
    edges: List[Tuple[Tuple[float, float], Tuple[float, float]]], 
    node_coords: np.ndarray
) -> List[Tuple[int, int]]:
    """
    This function takes a list of edges represented by point coordinates and converts them to a list 
    of edges represented by vertex indices based on their positions in the `node_coords` array.

    Parameters:
    - edges (List[Tuple[Tuple[float, float], Tuple[float, float]]]): List of edges defined by point coordinates.
    - node_coords (np.ndarray): Array of vertex coordinates with shape `(N, 2)`.

    Returns:
    - List[Tuple[int, int]]: List of edges defined by vertex indices.
    """
    def find_vertex_id(position: Tuple[float, float]) -> int:
        return np.where((node_coords == position).all(axis=1))[0][0]

    return [(int(find_vertex_id(edge[0])), int(find_vertex_id(edge[1]))) for edge in edges]

def nodes_to_triangle_idx(
    triangle_delaunay_node_coords_idx: Tuple[int, int, int], 
    elem_nodes: List[Tuple[int, int, int]], 
    node_elems: List[List[int]]
) -> int:
    """
    This function identifies the index of a triangle in `elem_nodes` that matches the given vertex indices 
    `(u, v, w)`. It checks the common triangles connected to each vertex and returns the triangle index 
    if a match is found.

    Parameters:
    - triangle_delaunay_node_coords_idx (Tuple[int, int, int]): Tuple of three vertex indices `(u, v, w)`.
    - elem_nodes (List[Tuple[int, int, int]]): List of existing triangles, each represented as a tuple of 3 vertex indices.
    - node_elems (List[List[int]]): List of lists, where each sublist contains triangle indices connected to a vertex.

    Returns:
    - int: The index of the matching triangle in `elem_nodes` if found, or -1 if not found.
    """
    u, v, w = triangle_delaunay_node_coords_idx

    # Get the triangles associated with each vertex
    triangles_u = set(node_elems[u])
    triangles_v = set(node_elems[v])
    triangles_w = set(node_elems[w])

    # Find the common triangles that include all three vertices (u, v, and w)
    common_triangles = triangles_u.intersection(triangles_v).intersection(triangles_w)

    for triangle_idx in common_triangles:
        # Retrieve the actual triangle from elem_nodes
        tri = elem_nodes[triangle_idx]
        if tri is None:
            continue  # Skip if the triangle has been deleted

        # Check if the vertices match (order doesn't matter)
        if set(tri) == set(triangle_delaunay_node_coords_idx):
            return triangle_idx

    return -1  # No matching triangle found

def log(
        message: str, 
        verbose: int, 
        level: int = 1
) -> None:
    """
    This function prints a message if the specified verbosity level meets or exceeds the required level.

    Parameters:
    - message (str): The message to print.
    - verbose (int): The current verbosity level (higher values indicate more detailed output).
    - level (int, optional): The minimum verbosity level required to print the message. Default is 1.

    Returns:
    - None.
    """
    if verbose >= level:
        print(message)