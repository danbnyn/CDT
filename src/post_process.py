from time import time
from collections import deque  
import copy
import numpy as np
from collections import defaultdict


from src.operations import (
    add_triangle,
    delete_triangle
    )

from src.predicates import (
    is_point_inside, 
    compute_centroid
    )

from src.visualize import (
    plot_triangulation_with_points,
    plot_triangulation,
    plot_adjancy_matrix,
    )

from src.utils import (
    log
    )

import numpy as np
from time import time

import numpy as np
from time import time

def clean_mesh(vertices, triangles, edge_to_triangle, super_vertices, vertex_to_triangles, polygon_outer_edges, verbose=1):
    """
    Cleans the mesh by removing the super triangle, reindexing the vertices, and updating the vertex_to_triangles list.

    Parameters:
    - vertices: List of vertices including the original point cloud and super-triangle vertices.
    - triangles: List of triangles representing the triangulated mesh.
    - edge_to_triangle: Dictionary mapping edges to triangle indices.
    - super_vertices: List of indices of the super-triangle vertices.
    - vertex_to_triangles: List of lists, where each sublist contains triangle indices.
    - polygon_outer_edges: List of edges defining the outer boundary of the polygon.
    - verbose: Integer indicating the verbosity level (0: silent, 1: basic, 2: detailed).

    Returns:
    - vertices: Updated list of vertices after cleaning.
    - new_triangles: Updated list of triangles after cleaning.
    - edge_to_triangle: Updated edge-to-triangle mapping.
    - vertex_to_triangles: Updated vertex-to-triangles list.
    """

    start_time = time()
    log(f"Starting mesh cleaning at {time():.4f}", verbose, level=2)

    # Step 1: Remove super-triangle triangles
    step_start = time()
    super_tri_idx = []
    for vertex in super_vertices:
        super_tri_idx.extend(vertex_to_triangles[vertex])

    unique_super_tri_idx = list(set(super_tri_idx))  # Remove duplicates
    if verbose >=1:
        log(f"Identified {len(unique_super_tri_idx)} super-triangle triangles to remove.", verbose, level=1)
    for tri_idx in unique_super_tri_idx:
        delete_triangle(tri_idx, triangles, edge_to_triangle, vertex_to_triangles)
    
    step_time = time() - step_start
    log(f"Step 1: Removed super-triangle triangles in {step_time:.4f} seconds.", verbose, level=1)
    if verbose >=2:
        log(f"Total triangles after removal: {len(triangles)}", verbose, level=2)

    plot_triangulation(vertices, triangles, title="Triangulation after removing super-triangle triangles")

    # Step 2: Remove the triangles exterior to the boundary
    step_start = time()
    exterior_removed = filter_exterior_triangles(vertices, triangles, vertex_to_triangles, edge_to_triangle, polygon_outer_edges)
    step_time = time() - step_start
    log(f"Step 2: Removed exterior triangles in {step_time:.4f} seconds.", verbose, level=1)
    if verbose >=2:
        log(f"Total triangles after removing exterior: {len(triangles)}", verbose, level=2)

    # Step 3: Remove the super-triangle vertices
    step_start = time()
    vertices = np.delete(vertices, super_vertices, axis=0)
    step_time = time() - step_start
    log(f"Step 3: Removed super-triangle vertices in {step_time:.4f} seconds.", verbose, level=1)
    if verbose >=2:
        log(f"Total vertices after removal: {len(vertices)}", verbose, level=2)

    # Step 4: Reindex the vertices in the triangles lists and remove the None values
    step_start = time()
    num_removed = len(super_vertices)
    new_triangles = []
    for tri in triangles:
        if tri is not None:
            # Reindex vertex IDs by subtracting the number of removed super-triangle vertices
            new_tri = tuple([v - num_removed for v in tri])
            new_triangles.append(new_tri)
    
    step_time = time() - step_start
    log(f"Step 4: Reindexed vertices in triangles in {step_time:.4f} seconds.", verbose, level=1)
    if verbose >=2:
        log(f"Total triangles after reindexing: {len(new_triangles)}", verbose, level=2)
    
    # Step 5: Rebuild the vertex_to_triangles list
    step_start = time()
    new_vertex_to_triangles = [[] for _ in range(len(vertices))]
    for tri_idx, tri in enumerate(new_triangles):
        for vertex in tri:
            new_vertex_to_triangles[vertex].append(tri_idx)
    
    vertex_to_triangles.clear()
    vertex_to_triangles.extend(new_vertex_to_triangles)
    step_time = time() - step_start
    log(f"Step 5: Updated vertex_to_triangles in {step_time:.4f} seconds.", verbose, level=1)
    if verbose >=2:
        log(f"Total vertex-to-triangle mappings: {len(vertex_to_triangles)}", verbose, level=2)
    
    # Step 6: Rebuild the edge_to_triangle mapping
    step_start = time()
    new_edge_to_triangle = {}
    for tri_idx, tri in enumerate(new_triangles):
        edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]
        
        for edge in edges:
            # Sort the edge tuple so that edge (a, b) and (b, a) are treated the same
            edge = tuple(sorted(edge))
            
            if edge not in new_edge_to_triangle:
                # First triangle associated with this edge
                new_edge_to_triangle[edge] = tri_idx
            else:
                # Edge already has one triangle, convert to tuple if necessary
                if isinstance(new_edge_to_triangle[edge], int):
                    # Convert from single int to tuple of two triangles
                    new_edge_to_triangle[edge] = (new_edge_to_triangle[edge], tri_idx)
                else:
                    # It's already a tuple, so this case shouldn't normally happen
                    log(f"Warning: Edge {edge} already associated with multiple triangles.", verbose, level=2)
    
    edge_to_triangle.clear()
    edge_to_triangle.update(new_edge_to_triangle)
    step_time = time() - step_start
    log(f"Step 6: Rebuilt edge_to_triangle mapping in {step_time:.4f} seconds.", verbose, level=1)
    if verbose >=2:
        log(f"Total unique edges: {len(edge_to_triangle)}", verbose, level=2)
    
    total_time = time() - start_time
    log(f"Mesh cleaning completed in {total_time:.4f} seconds.", verbose, level=1)
    
    return vertices, new_triangles, edge_to_triangle, vertex_to_triangles


def filter_exterior_triangles(vertices, triangles, vertex_to_triangles, edge_to_triangle, polygon_outer_edges):
    """
    Filters out exterior triangles such that the resulting boundary matches polygon_outer_edges.

    Parameters:
    - vertices: List of vertices including the original point cloud and super-triangle vertices.
    - triangles: List of triangles, where each triangle is a tuple of three vertex indices (v0, v1, v2).
    - vertex_to_triangles: List of lists, where each sublist contains triangle indices.
    - edge_to_triangle: Dictionary mapping edges to triangle indices or tuples of two triangle indices.
    - polygon_outer_edges: List of edges defining the outer boundary of the polygon. 

    Returns:
    - List of EXTERIOR triangle indices.
    """
    # Helper function to standardize edge representation (sorted tuple)
    def get_edge(v1, v2):
        return tuple(sorted((v1, v2)))

    # Create deep copies to avoid modifying the original data structures
    triangles_copy = copy.deepcopy(triangles)
    edge_to_triangle_copy = copy.deepcopy(edge_to_triangle)
    vertex_to_triangles_copy = copy.deepcopy(vertex_to_triangles)

    # Initialize a set for polygon_outer_edges for faster lookup
    polygon_outer_edges_set = set(get_edge(*edge) for edge in polygon_outer_edges)

    # Initialize current_outer_edges with edges that are currently boundaries
    current_outer_edges = set()
    for edge, tri in edge_to_triangle_copy.items():
        standardized_edge = get_edge(*edge)
        if isinstance(tri, int):
            current_outer_edges.add(standardized_edge)

    # Initialize queue with edges that are boundary and not part of polygon_outer_edges
    edge_queue = deque(edge for edge in current_outer_edges if edge not in polygon_outer_edges_set)

    # Set to keep track of exterior triangle indices
    exterior_triangle_indices = set()

    while edge_queue:
        edge = edge_queue.popleft()

        # If the edge is part of the desired polygon_outer_edges, skip processing
        if edge in polygon_outer_edges_set:
            continue

        # Get the triangle associated with this edge
        tri_index = edge_to_triangle_copy.get(edge)
        if tri_index is None:
            continue  # Edge might have been processed already
        if isinstance(tri_index, tuple):
            # This should not happen for boundary edges, but handle gracefully
            # Skip processing in this case
            continue

        if triangles_copy[tri_index] is None:
            continue  # Triangle already deleted

        # Add the triangle to exterior triangles
        exterior_triangle_indices.add(tri_index)

        # Get the triangle's edges before deletion
        triangle = triangles_copy[tri_index]
        tri_edges = [get_edge(triangle[0], triangle[1]),
                     get_edge(triangle[1], triangle[2]),
                     get_edge(triangle[2], triangle[0])]

        # Delete the triangle using the provided delete_triangle function
        delete_triangle(tri_index, triangles_copy, edge_to_triangle_copy, vertex_to_triangles_copy)

        # After deletion, check each edge to see if it has become a boundary edge
        for te in tri_edges:
            # If the edge is now a boundary edge and not part of polygon_outer_edges, enqueue it
            associated = edge_to_triangle_copy.get(te)
            if associated is not None:
                if isinstance(associated, int):
                    # Edge is now a boundary edge
                    if te not in polygon_outer_edges_set:
                        if te not in current_outer_edges:
                            current_outer_edges.add(te)
                            edge_queue.append(te)
                elif isinstance(associated, tuple):
                    # Edge is shared by two triangles; not a boundary
                    continue
            else:
                # Edge has been completely removed; no action needed
                continue

    for tri_idx in exterior_triangle_indices:
        delete_triangle(tri_idx, triangles, edge_to_triangle, vertex_to_triangles)


def convert_to_mesh_format(vertices, triangles):
    """
    Convert the data structures to a more efficient format for ulterior usage.

    Parameters:
    - vertices: List of vertices coordinates.
    - triangles: List of triangles, where each triangle is a tuple of three vertex indices (v0, v1, v2).

    Returns:
    - node_coord: List of vertices coordinates.
    - numb_elem: Number of elements in the mesh.
    - elem2node: List of vertex indices
    - p_elem2node: Pointer for the elem2node list, indicating the start of each element of the mesh.

    """

    node_coord = [ [vertices[i][0], vertices[i][1]] for i in range(len(vertices))]
    elem2node = []
    p_elem2node = [0]

    for tri in triangles:
        elem2node.extend(tri)
        p_elem2node.append(p_elem2node[-1] + 3)


    numb_elem = len(triangles)


    return node_coord, numb_elem, elem2node, p_elem2node

# ----------------- Optimization of the data structure ----------------- #

# ----------------------- Reverse Cuthill–McKee Algorithm -----------------------

def build_adjacency_list(elem2node, p_elem2node):
    """Build adjacency list from elem2node and p_elem2node."""
    adjacency = defaultdict(set)
    for elem_idx in range(len(p_elem2node) - 1):
        start = p_elem2node[elem_idx]
        end = p_elem2node[elem_idx + 1]
        triangle = elem2node[start:end]
        for i in range(3):
            for j in range(i + 1, 3):
                adjacency[triangle[i]].add(triangle[j])
                adjacency[triangle[j]].add(triangle[i])
    return adjacency

def reverse_cuthill_mckee(adjacency, n_nodes):
    """Implement the Reverse Cuthill–McKee algorithm."""
    visited = [False] * n_nodes
    permutation = []

    # Find the node with the minimum degree to start
    degrees = [len(adjacency[i]) for i in range(n_nodes)]
    start_node = degrees.index(min(degrees))

    queue = deque()
    queue.append(start_node)
    visited[start_node] = True

    while queue:
        node = queue.popleft()
        permutation.append(node)
        neighbors = sorted(adjacency[node], key=lambda x: len(adjacency[x]))
        for neighbor in neighbors:
            if not visited[neighbor]:
                visited[neighbor] = True
                queue.append(neighbor)

    # Check for disconnected graph
    for node in range(n_nodes):
        if not visited[node]:
            queue.append(node)
            visited[node] = True
            while queue:
                current = queue.popleft()
                permutation.append(current)
                neighbors = sorted(adjacency[current], key=lambda x: len(adjacency[x]))
                for neighbor in neighbors:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)

    # Reverse the permutation for RCM
    permutation = permutation[::-1]
    # Create a mapping from old indices to new indices
    mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(permutation)}
    return mapping

def apply_rcm(node_coord, elem2node, p_elem2node):
    """Apply the Reverse Cuthill–McKee algorithm to reorder nodes."""

    n_nodes = len(node_coord)

    adjacency = build_adjacency_list(elem2node, p_elem2node)
    mapping = reverse_cuthill_mckee(adjacency, n_nodes)

    # Create new node_coord array
    new_node_coord = np.zeros_like(node_coord)
    for old_idx, new_idx in mapping.items():
        new_node_coord[new_idx] = node_coord[old_idx]

    # Create a list to map old to new indices
    old_to_new = mapping
    # Update elem2node
    new_elem2node = np.array([old_to_new[old_idx] for old_idx in elem2node], dtype='int64')

    return new_node_coord, new_elem2node

