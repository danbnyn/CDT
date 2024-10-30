from time import time
from collections import deque  
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Optional

from .operations import (
    delete_triangle
    )

from .visualize import (
    plot_triangulation_with_points,
    plot_triangulation,
    plot_adjancy_matrix,
    )

from .utils import (
    log
    )

def clean_mesh(
    node_coords: np.ndarray,
    elem_nodes: List[Optional[Tuple[int, int, int]]],
    node_elems: List[List[int]],
    node_nodes: List[List[int]],
    super_node_coords: List[int],
    polygon_outer_edges: List[Tuple[int, int]],
    verbose: int = 1
) -> Tuple[np.ndarray, List[Tuple[int, int, int]], List[List[int]]]:
    """
    This function cleans the mesh by performing the following steps:
    1. **Remove Super-Triangle Elements**: Identifies and removes elements (triangles) that are connected to the super-triangle nodes.
    2. **Filter Exterior Elements**: Removes elements outside the specified polygon outer boundary using the `filter_exterior_elem_nodes` function.
    3. **Remove Super-Triangle Nodes**: Deletes the super-triangle nodes from the node coordinates.
    4. **Reindex Elements**: Reindexes the vertex IDs in the elements list to reflect the removal of super-triangle nodes.
    5. **Rebuild Node-to-Elements Mapping**: Updates the `node_elems` list to reflect the updated elements and node coordinates.

    Parameters:
    - node_coords (np.ndarray): Array of node coordinates including the original point cloud and super-triangle node coordinates.
    - elem_nodes (List[Optional[Tuple[int, int, int]]]): List of elements (triangles), where each element is a tuple of three vertex indices.
    - node_elems (List[List[int]]): List of lists, where each sublist contains the triangle indices connected to a node.
    - node_nodes (List[List[int]]): List of lists, where each sublist contains the nodes connected to a node.
    - super_node_coords (List[int]): List of indices representing the super-triangle nodes in `node_coords`.
    - polygon_outer_edges (List[Tuple[int, int]]): List of edges defining the outer boundary of the polygon, where each edge is a tuple of two vertex indices.
    - verbose (int): Verbosity level for logging (0: silent, 1: basic, 2: detailed). Default is 1.

    Returns:
    - node_coords (np.ndarray): Updated array of node coordinates after removing super-triangle nodes.
    - new_elem_nodes (List[Tuple[int, int, int]]): Updated list of elements (triangles) after cleaning, with reindexed vertex IDs.
    - node_elems (List[List[int]]): Updated list of lists, where each sublist contains the triangle indices connected to a node.
    """

    start_time = time()
    log(f"Starting mesh cleaning at {time():.4f}", verbose, level=2)

    # Step 1: Remove super-triangle elem_nodes
    step_start = time()
    super_tri_idx = []
    for vertex in super_node_coords:
        super_tri_idx.extend(node_elems[vertex])

    unique_super_tri_idx = list(set(super_tri_idx))  # Remove duplicates
    if verbose >=1:
        log(f"Identified {len(unique_super_tri_idx)} super-triangle elem_nodes to remove.", verbose, level=1)
    for tri_idx in unique_super_tri_idx:
        delete_triangle(tri_idx, elem_nodes, node_elems, node_nodes)
    
    step_time = time() - step_start
    log(f"Step 1: Removed super-triangle elem_nodes in {step_time:.4f} seconds.", verbose, level=1)
    if verbose >=2:
        log(f"Total elem_nodes after removal: {len(elem_nodes)}", verbose, level=2)

    plot_triangulation(node_coords, elem_nodes, title="Triangulation after removing super-triangle elem_nodes")

    # Step 2: Remove the elem_nodes exterior to the boundary
    step_start = time()
    exterior_removed = filter_exterior_elem_nodes(elem_nodes, node_elems, node_nodes, polygon_outer_edges)
    step_time = time() - step_start
    log(f"Step 2: Removed exterior elem_nodes in {step_time:.4f} seconds.", verbose, level=1)
    if verbose >=2:
        log(f"Total elem_nodes after removing exterior: {len(elem_nodes)}", verbose, level=2)

    # Step 3: Remove the super-triangle node_coords
    step_start = time()
    node_coords = np.delete(node_coords, super_node_coords, axis=0)
    step_time = time() - step_start
    log(f"Step 3: Removed super-triangle node_coords in {step_time:.4f} seconds.", verbose, level=1)
    if verbose >=2:
        log(f"Total node_coords after removal: {len(node_coords)}", verbose, level=2)

    # Step 4: Reindex the node_coords in the elem_nodes lists and remove the None values
    step_start = time()
    num_removed = len(super_node_coords)
    new_elem_nodes = []
    for tri in elem_nodes:
        if tri is not None:
            # Reindex vertex IDs by subtracting the number of removed super-triangle node_coords
            new_tri = tuple([v - num_removed for v in tri])
            new_elem_nodes.append(new_tri)
    
    step_time = time() - step_start
    log(f"Step 4: Reindexed node_coords in elem_nodes in {step_time:.4f} seconds.", verbose, level=1)
    if verbose >=2:
        log(f"Total elem_nodes after reindexing: {len(new_elem_nodes)}", verbose, level=2)
    
    # Step 5: Rebuild the node_elems list
    step_start = time()
    new_node_elems = [[] for _ in range(len(node_coords))]
    for tri_idx, tri in enumerate(new_elem_nodes):
        for vertex in tri:
            new_node_elems[vertex].append(tri_idx)
    
    node_elems.clear()
    node_elems.extend(new_node_elems)
    step_time = time() - step_start
    log(f"Step 5: Updated node_elems in {step_time:.4f} seconds.", verbose, level=1)
    if verbose >=2:
        log(f"Total vertex-to-triangle mappings: {len(node_elems)}", verbose, level=2)
    


    total_time = time() - start_time
    log(f"Mesh cleaning completed in {total_time:.4f} seconds.", verbose, level=1)
    
    return node_coords, new_elem_nodes, node_elems

def get_edge(
        v1: int,
        v2: int
) -> Tuple[int, int]:
    """
    Helper function to standardize edge representation as a sorted tuple.
    
    Parameters:
    - v1 (int): First vertex index.
    - v2 (int): Second vertex index.

    Returns:
    - Tuple[int, int]: Edge representation as a sorted tuple.
    """
    return tuple(sorted((v1, v2)))

def filter_exterior_elem_nodes(
    elem_nodes: List[Optional[Tuple[int, int, int]]],
    node_elems: List[List[int]],
    node_nodes: List[List[int]],
    polygon_outer_edges: List[Tuple[int, int]]
) -> List[int]:
    """
    This function identifies and removes triangles that are exterior to the polygon defined by `polygon_outer_edges`. 
    It does so by iteratively removing triangles connected to boundary edges that are not part of the specified polygon 
    outer edges. The function updates the `elem_nodes` list to reflect the removal of these exterior triangles.

    Parameters:
    - elem_nodes (List[Optional[Tuple[int, int, int]]]): List of existing triangles, where each triangle is represented 
      as a tuple of three vertex indices (v0, v1, v2). A value of `None` indicates a deleted triangle.
    - node_elems (List[List[int]]): List of lists, where each sublist contains the indices of triangles connected to a node.
    - node_nodes (List[List[int]]): List of lists, where each sublist contains the indices of nodes connected to a node.
    - polygon_outer_edges (List[Tuple[int, int]]): List of edges defining the outer boundary of the polygon, represented 
      as tuples of two vertex indices (v0, v1).

    Returns:
    - List[int]: A list of indices of the triangles that were identified as exterior and removed.
    """

    # Step 1: Initialize polygon_outer_edges_set for faster lookup
    polygon_outer_edges_set = set(get_edge(*edge) for edge in polygon_outer_edges)

    # Step 2: Initialize a set to keep track of boundary edges (shared by exactly one triangle)
    boundary_edges = set()

    # Step 3: Identify all boundary edges
    for tri_idx, tri in enumerate(elem_nodes):
        if tri is None:
            continue  # Skip deleted triangles
        u, v, w = tri
        edges = [get_edge(u, v), get_edge(v, w), get_edge(w, u)]
        for edge in edges:
            # Find all triangles sharing this edge by intersecting node_elems of both vertices
            triangles_u = set(node_elems[edge[0]])
            triangles_v = set(node_elems[edge[1]])
            common_triangles = triangles_u.intersection(triangles_v)
            if len(common_triangles) == 1:
                # This is a boundary edge
                boundary_edges.add(edge)

    # Step 4: Initialize queue with boundary edges not in polygon_outer_edges_set
    unwanted_boundary_edges = boundary_edges - polygon_outer_edges_set
    edge_queue = deque(unwanted_boundary_edges)

    # Step 5: Initialize set to keep track of exterior triangle indices
    exterior_triangle_indices = set()

    while edge_queue:
        edge = edge_queue.popleft()

        # Find the triangle that owns this boundary edge
        triangles_u = set(node_elems[edge[0]])
        triangles_v = set(node_elems[edge[1]])
        owning_triangles = triangles_u.intersection(triangles_v)

        if not owning_triangles:
            continue  # Edge might have been processed already

        tri_idx = owning_triangles.pop()

        if elem_nodes[tri_idx] is None:
            continue  # Triangle already deleted

        # Add the triangle to exterior_triangle_indices
        exterior_triangle_indices.add(tri_idx)

        # Retrieve the triangle's vertices and edges
        tri = elem_nodes[tri_idx]
        u, v, w = tri
        tri_edges = [get_edge(u, v), get_edge(v, w), get_edge(w, u)]

        # Delete the triangle using the provided delete_triangle function
        delete_triangle(tri_idx, elem_nodes, node_elems, node_nodes)

        # After deletion, check each edge to see if it has become a boundary edge
        for te in tri_edges:
            # Re-identify if the edge is now a boundary edge
            triangles_u_te = set(node_elems[te[0]])
            triangles_v_te = set(node_elems[te[1]])
            common_triangles_te = triangles_u_te.intersection(triangles_v_te)

            if len(common_triangles_te) == 1:
                # This edge is now a boundary edge
                if te not in polygon_outer_edges_set and te not in boundary_edges:
                    boundary_edges.add(te)
                    edge_queue.append(te)


############################################### To be continued ########################################################


def convert_to_mesh_format(node_coords, elem_nodes):
    """
    Convert the data structures to a more efficient format for ulterior usage.

    Parameters:
    - node_coords: List of node_coords coordinates.
    - elem_nodes: List of elem_nodes, where each triangle is a tuple of three vertex indices (v0, v1, v2).

    Returns:
    - node_coord: List of node_coords coordinates.
    - numb_elem: Number of elements in the mesh.
    - elem2node: List of vertex indices
    - p_elem2node: Pointer for the elem2node list, indicating the start of each element of the mesh.

    """

    node_coord = [ [node_coords[i][0], node_coords[i][1]] for i in range(len(node_coords))]
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

