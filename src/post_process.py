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

def clean_mesh(delaunay_node_coords, elem_nodes, delaunay_dic_edge_triangle, super_delaunay_node_coords, delaunay_node_elems, polygon_outer_edges, verbose=1):
    """
    Cleans the mesh by removing the super triangle, reindexing the delaunay_node_coords, and updating the delaunay_node_elems list.

    Parameters:
    - delaunay_node_coords: List of delaunay_node_coords including the original point cloud and super-triangle delaunay_node_coords.
    - elem_nodes: List of elem_nodes representing the triangulated mesh.
    - delaunay_dic_edge_triangle: Dictionary mapping edges to triangle indices.
    - super_delaunay_node_coords: List of indices of the super-triangle delaunay_node_coords.
    - delaunay_node_elems: List of lists, where each sublist contains triangle indices.
    - polygon_outer_edges: List of edges defining the outer boundary of the polygon.
    - verbose: Integer indicating the verbosity level (0: silent, 1: basic, 2: detailed).

    Returns:
    - delaunay_node_coords: Updated list of delaunay_node_coords after cleaning.
    - new_elem_nodes: Updated list of elem_nodes after cleaning.
    - delaunay_dic_edge_triangle: Updated edge-to-triangle mapping.
    - delaunay_node_elems: Updated vertex-to-elem_nodes list.
    """

    start_time = time()
    log(f"Starting mesh cleaning at {time():.4f}", verbose, level=2)

    # Step 1: Remove super-triangle elem_nodes
    step_start = time()
    super_tri_idx = []
    for vertex in super_delaunay_node_coords:
        super_tri_idx.extend(delaunay_node_elems[vertex])

    unique_super_tri_idx = list(set(super_tri_idx))  # Remove duplicates
    if verbose >=1:
        log(f"Identified {len(unique_super_tri_idx)} super-triangle elem_nodes to remove.", verbose, level=1)
    for tri_idx in unique_super_tri_idx:
        delete_triangle(tri_idx, elem_nodes, delaunay_dic_edge_triangle, delaunay_node_elems)
    
    step_time = time() - step_start
    log(f"Step 1: Removed super-triangle elem_nodes in {step_time:.4f} seconds.", verbose, level=1)
    if verbose >=2:
        log(f"Total elem_nodes after removal: {len(elem_nodes)}", verbose, level=2)

    plot_triangulation(delaunay_node_coords, elem_nodes, title="Triangulation after removing super-triangle elem_nodes")

    # Step 2: Remove the elem_nodes exterior to the boundary
    step_start = time()
    exterior_removed = filter_exterior_elem_nodes(delaunay_node_coords, elem_nodes, delaunay_node_elems, delaunay_dic_edge_triangle, polygon_outer_edges)
    step_time = time() - step_start
    log(f"Step 2: Removed exterior elem_nodes in {step_time:.4f} seconds.", verbose, level=1)
    if verbose >=2:
        log(f"Total elem_nodes after removing exterior: {len(elem_nodes)}", verbose, level=2)

    # Step 3: Remove the super-triangle delaunay_node_coords
    step_start = time()
    delaunay_node_coords = np.delete(delaunay_node_coords, super_delaunay_node_coords, axis=0)
    step_time = time() - step_start
    log(f"Step 3: Removed super-triangle delaunay_node_coords in {step_time:.4f} seconds.", verbose, level=1)
    if verbose >=2:
        log(f"Total delaunay_node_coords after removal: {len(delaunay_node_coords)}", verbose, level=2)

    # Step 4: Reindex the delaunay_node_coords in the elem_nodes lists and remove the None values
    step_start = time()
    num_removed = len(super_delaunay_node_coords)
    new_elem_nodes = []
    for tri in elem_nodes:
        if tri is not None:
            # Reindex vertex IDs by subtracting the number of removed super-triangle delaunay_node_coords
            new_tri = tuple([v - num_removed for v in tri])
            new_elem_nodes.append(new_tri)
    
    step_time = time() - step_start
    log(f"Step 4: Reindexed delaunay_node_coords in elem_nodes in {step_time:.4f} seconds.", verbose, level=1)
    if verbose >=2:
        log(f"Total elem_nodes after reindexing: {len(new_elem_nodes)}", verbose, level=2)
    
    # Step 5: Rebuild the delaunay_node_elems list
    step_start = time()
    new_delaunay_node_elems = [[] for _ in range(len(delaunay_node_coords))]
    for tri_idx, tri in enumerate(new_elem_nodes):
        for vertex in tri:
            new_delaunay_node_elems[vertex].append(tri_idx)
    
    delaunay_node_elems.clear()
    delaunay_node_elems.extend(new_delaunay_node_elems)
    step_time = time() - step_start
    log(f"Step 5: Updated delaunay_node_elems in {step_time:.4f} seconds.", verbose, level=1)
    if verbose >=2:
        log(f"Total vertex-to-triangle mappings: {len(delaunay_node_elems)}", verbose, level=2)
    
    # Step 6: Rebuild the delaunay_dic_edge_triangle mapping
    step_start = time()
    new_delaunay_dic_edge_triangle = {}
    for tri_idx, tri in enumerate(new_elem_nodes):
        edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]
        
        for edge in edges:
            # Sort the edge tuple so that edge (a, b) and (b, a) are treated the same
            edge = tuple(sorted(edge))
            
            if edge not in new_delaunay_dic_edge_triangle:
                # First triangle associated with this edge
                new_delaunay_dic_edge_triangle[edge] = tri_idx
            else:
                # Edge already has one triangle, convert to tuple if necessary
                if isinstance(new_delaunay_dic_edge_triangle[edge], int):
                    # Convert from single int to tuple of two elem_nodes
                    new_delaunay_dic_edge_triangle[edge] = (new_delaunay_dic_edge_triangle[edge], tri_idx)
                else:
                    # It's already a tuple, so this case shouldn't normally happen
                    log(f"Warning: Edge {edge} already associated with multiple elem_nodes.", verbose, level=2)
    
    delaunay_dic_edge_triangle.clear()
    delaunay_dic_edge_triangle.update(new_delaunay_dic_edge_triangle)
    step_time = time() - step_start
    log(f"Step 6: Rebuilt delaunay_dic_edge_triangle mapping in {step_time:.4f} seconds.", verbose, level=1)
    if verbose >=2:
        log(f"Total unique edges: {len(delaunay_dic_edge_triangle)}", verbose, level=2)
    
    total_time = time() - start_time
    log(f"Mesh cleaning completed in {total_time:.4f} seconds.", verbose, level=1)
    
    return delaunay_node_coords, new_elem_nodes, delaunay_dic_edge_triangle, delaunay_node_elems


def filter_exterior_elem_nodes(delaunay_node_coords, elem_nodes, delaunay_node_elems, delaunay_dic_edge_triangle, polygon_outer_edges):
    """
    Filters out exterior elem_nodes such that the resulting boundary matches polygon_outer_edges.

    Parameters:
    - delaunay_node_coords: List of delaunay_node_coords including the original point cloud and super-triangle delaunay_node_coords.
    - elem_nodes: List of elem_nodes, where each triangle is a tuple of three vertex indices (v0, v1, v2).
    - delaunay_node_elems: List of lists, where each sublist contains triangle indices.
    - delaunay_dic_edge_triangle: Dictionary mapping edges to triangle indices or tuples of two triangle indices.
    - polygon_outer_edges: List of edges defining the outer boundary of the polygon. 

    Returns:
    - List of EXTERIOR triangle indices.
    """
    # Helper function to standardize edge representation (sorted tuple)
    def get_edge(v1, v2):
        return tuple(sorted((v1, v2)))

    # Create deep copies to avoid modifying the original data structures
    elem_nodes_copy = copy.deepcopy(elem_nodes)
    delaunay_dic_edge_triangle_copy = copy.deepcopy(delaunay_dic_edge_triangle)
    delaunay_node_elems_copy = copy.deepcopy(delaunay_node_elems)

    # Initialize a set for polygon_outer_edges for faster lookup
    polygon_outer_edges_set = set(get_edge(*edge) for edge in polygon_outer_edges)

    # Initialize current_outer_edges with edges that are currently boundaries
    current_outer_edges = set()
    for edge, tri in delaunay_dic_edge_triangle_copy.items():
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
        tri_index = delaunay_dic_edge_triangle_copy.get(edge)
        if tri_index is None:
            continue  # Edge might have been processed already
        if isinstance(tri_index, tuple):
            # This should not happen for boundary edges, but handle gracefully
            # Skip processing in this case
            continue

        if elem_nodes_copy[tri_index] is None:
            continue  # Triangle already deleted

        # Add the triangle to exterior elem_nodes
        exterior_triangle_indices.add(tri_index)

        # Get the triangle's edges before deletion
        triangle = elem_nodes_copy[tri_index]
        tri_edges = [get_edge(triangle[0], triangle[1]),
                     get_edge(triangle[1], triangle[2]),
                     get_edge(triangle[2], triangle[0])]

        # Delete the triangle using the provided delete_triangle function
        delete_triangle(tri_index, elem_nodes_copy, delaunay_dic_edge_triangle_copy, delaunay_node_elems_copy)

        # After deletion, check each edge to see if it has become a boundary edge
        for te in tri_edges:
            # If the edge is now a boundary edge and not part of polygon_outer_edges, enqueue it
            associated = delaunay_dic_edge_triangle_copy.get(te)
            if associated is not None:
                if isinstance(associated, int):
                    # Edge is now a boundary edge
                    if te not in polygon_outer_edges_set:
                        if te not in current_outer_edges:
                            current_outer_edges.add(te)
                            edge_queue.append(te)
                elif isinstance(associated, tuple):
                    # Edge is shared by two elem_nodes; not a boundary
                    continue
            else:
                # Edge has been completely removed; no action needed
                continue

    for tri_idx in exterior_triangle_indices:
        delete_triangle(tri_idx, elem_nodes, delaunay_dic_edge_triangle, delaunay_node_elems)


def convert_to_mesh_format(delaunay_node_coords, elem_nodes):
    """
    Convert the data structures to a more efficient format for ulterior usage.

    Parameters:
    - delaunay_node_coords: List of delaunay_node_coords coordinates.
    - elem_nodes: List of elem_nodes, where each triangle is a tuple of three vertex indices (v0, v1, v2).

    Returns:
    - node_coord: List of delaunay_node_coords coordinates.
    - numb_elem: Number of elements in the mesh.
    - elem2node: List of vertex indices
    - p_elem2node: Pointer for the elem2node list, indicating the start of each element of the mesh.

    """

    node_coord = [ [delaunay_node_coords[i][0], delaunay_node_coords[i][1]] for i in range(len(delaunay_node_coords))]
    elem2node = []
    p_elem2node = [0]

    for tri in elem_nodes:
        elem2node.extend(tri)
        p_elem2node.append(p_elem2node[-1] + 3)


    numb_elem = len(elem_nodes)


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

