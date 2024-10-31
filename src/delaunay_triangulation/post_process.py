from time import time
from collections import deque  
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Optional, Dict, Set

from .operations import (
    delete_triangle
    )

from .visualize import (
    plot_triangulation_with_points,
    plot_triangulation,
    plot_adjancy_matrix,
    plot_triangulation_with_elem
    )

from .utils import (
    log
    )

# ----------------------- Main Mesh Cleaning Function ----------------------- #

def clean_mesh(
    node_coords: np.ndarray,
    elem_nodes: List[Optional[Tuple[int, int, int]]],
    node_elems: List[List[int]],
    node_nodes: List[List[int]],
    super_node_coords: List[int],
    polygon_outer_edges: List[Tuple[int, int]],
    polygon_holes_edges: List[List[Tuple[int, int]]],
    verbose: int = 1
) -> Tuple[np.ndarray, List[Tuple[int, int, int]], List[List[int]]]:
    """
    Cleans the mesh by performing the following steps:
    1. Remove Super-Triangle Elements
    2. Filter Exterior Elements
    3. Filter Elements in Holes
    3. Remove Super-Triangle Nodes
    4. Reindex Elements
    5. Rebuild Node-to-Elements Mapping
    6. Rebuild Node-to-Nodes Mapping

    Parameters:
    - node_coords (np.ndarray): Array of node coordinates including the original point cloud and super-triangle node coordinates.
    - elem_nodes (List[Optional[Tuple[int, int, int]]]): List of elements (triangles), where each element is a tuple of three vertex indices.
    - node_elems (List[List[int]]): List of lists, where each sublist contains the triangle indices connected to a node.
    - node_nodes (List[List[int]]): List of lists, where each sublist contains the nodes connected to a node.
    - super_node_coords (List[int]): List of indices representing the super-triangle nodes in `node_coords`.
    - polygon_outer_edges (List[Tuple[int, int]]): List of edges defining the outer boundary of the polygon, where each edge is a tuple of two vertex indices.
    - polygon_holes_edges (List[List[Tuple[int, int]]]): List of lists, where each sublist contains the edges defining a hole in the polygon.
    - verbose (int): Verbosity level for logging (0: silent, 1: basic, 2: detailed). Default is 1.

    Returns:
    - Tuple containing:
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
        log(f"Total elem_nodes after removal: {len([tri for tri in elem_nodes if tri is not None])}", verbose, level=2)

    if verbose >=2:
        plot_triangulation(node_coords, elem_nodes, title="Triangulation after removing super-triangle elem_nodes")
    
    # Step 2a: Remove the elem_nodes exterior to the boundary
    step_start = time()
    exterior_removed = filter_exterior_elem_nodes(elem_nodes, node_elems, node_nodes, polygon_outer_edges)
    step_time = time() - step_start
    log(f"Step 2: Removed exterior elem_nodes in {step_time:.4f} seconds.", verbose, level=1)
    if verbose >=2:
        log(f"Total elem_nodes after removing exterior: {len([tri for tri in elem_nodes if tri is not None])}", verbose, level=2)

    # Step 2b: Remove the elem_nodes in the holes
    step_start = time()
    holes_removed = filter_holes_elem_nodes(elem_nodes, node_elems, node_nodes, polygon_holes_edges)
    step_time = time() - step_start
    log(f"Step 3: Removed elem_nodes in holes in {step_time:.4f} seconds.", verbose, level=1)
    if verbose >=2:
        log(f"Total elem_nodes after removing holes: {len([tri for tri in elem_nodes if tri is not None])}", verbose, level=2)

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
            # This assumes that super_node_coords were at the end and removed in order
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
    
    # Step 6: Rebuild the node_nodes list
    step_start = time()
    # Initialize new_node_nodes as list of sets to avoid duplicates
    new_node_nodes = [set() for _ in range(len(node_coords))]
    
    for tri in new_elem_nodes:
        u, v, w = tri
        new_node_nodes[u].update([v, w])
        new_node_nodes[v].update([u, w])
        new_node_nodes[w].update([u, v])
    
    # Convert sets to sorted lists
    new_node_nodes_sorted = [sorted(list(neighbors)) for neighbors in new_node_nodes]
    
    # Clear and update node_nodes
    node_nodes.clear()
    node_nodes.extend(new_node_nodes_sorted)
    
    step_time = time() - step_start
    log(f"Step 6: Rebuilt node_nodes in {step_time:.4f} seconds.", verbose, level=1)
    if verbose >=2:
        log(f"Total node_nodes after rebuilding: {len(node_nodes)}", verbose, level=2)
    
    total_time = time() - start_time
    log(f"Mesh cleaning completed in {total_time:.4f} seconds.", verbose, level=1)
    
    return node_coords, new_elem_nodes, node_elems, node_nodes

# ----------------------- Mesh Cleaning Helper Functions ----------------------- #

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

def find_adjacent_triangles(
        edge: Tuple[int, int],
        node_elems: List[List[int]]
) -> Set[int]:
    """
    Finds all triangles adjacent to a given edge.

    Parameters:
    - edge (Tuple[int, int]): Tuple representing the edge as two vertex indices.
    - node_elems (List[List[int]]): List of lists, where each sublist contains the indices of triangles connected to a node.

    Returns:
    - Set[int]: A set of triangle indices adjacent to the given edge.
    
    """
    triangles_u = set(node_elems[edge[0]])
    triangles_v = set(node_elems[edge[1]])
    return triangles_u.intersection(triangles_v)

def flood_fill_inside_hole(
    start_tri_idx: int,
    elem_nodes: List[Optional[Tuple[int, int, int]]],
    node_elems: List[List[int]],
    hole_edges_set: Set[Tuple[int, int]]
) -> Set[int]:
    """
    Performs flood-fill traversal to find all triangles inside a hole.

    Parameters:
    - start_tri_idx (int): Index of the starting triangle inside the hole.
    - elem_nodes (List[Optional[Tuple[int, int, int]]]): List of existing triangles, where each triangle is represented
        as a tuple of three vertex indices (v0, v1, v2). A value of `None` indicates a deleted triangle.
    - node_elems (List[List[int]]): List of lists, where each sublist contains the indices of triangles connected to a node.
    - hole_edges_set (Set[Tuple[int, int]]): Set of edges defining the hole boundary.

    Returns:
    - Set[int]: A set of triangle indices inside the hole.

    """
    visited = set()
    queue = deque([start_tri_idx])

    while queue:
        tri_idx = queue.popleft()
        if tri_idx in visited:
            continue
        visited.add(tri_idx)
        tri = elem_nodes[tri_idx]
        if tri is None:
            continue  # Skip deleted triangles
        u, v, w = tri
        edges = [get_edge(u, v), get_edge(v, w), get_edge(w, u)]
        for edge in edges:
            if edge in hole_edges_set:
                continue  # Do not cross hole boundary edges
            neighbor_triangles = find_adjacent_triangles(edge, node_elems)
            for neighbor_tri_idx in neighbor_triangles:
                if neighbor_tri_idx not in visited:
                    queue.append(neighbor_tri_idx)
    return visited

def filter_holes_elem_nodes(
    elem_nodes: List[Optional[Tuple[int, int, int]]],
    node_elems: List[List[int]],
    node_nodes: List[List[int]],
    polygon_holes_edges: List[List[Tuple[int, int]]],
) -> List[int]:
    """
    This function identifies and removes triangles that are interior to each of the polygon holes defined by `polygon_holes_edges`. 

    Parameters:
    - elem_nodes (List[Optional[Tuple[int, int, int]]]): List of existing triangles, where each triangle is represented 
      as a tuple of three vertex indices (v0, v1, v2). A value of `None` indicates a deleted triangle.
    - node_elems (List[List[int]]): List of lists, where each sublist contains the indices of triangles connected to a node.
    - node_nodes (List[List[int]]): List of lists, where each sublist contains the indices of nodes connected to a node.
    - polygon_holes_edges (List[List[Tuple[int, int]]]): List of lists, where each sublist contains the edges defining a hole in the polygon.

    Returns:
    - List[int]: A list of indices of the triangles that were identified as interior to holes and removed.
    """
    triangles_to_remove = set()

    for hole_idx, hole_edges in enumerate(polygon_holes_edges):
        # Step 1: Collect all edges of the current hole into a set
        hole_edges_set = set(get_edge(*edge) for edge in hole_edges)
        hole_nodes_set = set([edge[0] for edge in hole_edges] + [edge[1] for edge in hole_edges])

        # Step 2: Identify an initial triangle inside the hole
        inside_tri_idx = None
        for edge in hole_edges:
            adjacent_triangles = find_adjacent_triangles(get_edge(*edge), node_elems)

            tri_indices = list(adjacent_triangles)
            tri1 = elem_nodes[tri_indices[0]]
            tri2 = elem_nodes[tri_indices[1]]
            tri1_extra_vertex = (set(tri1) - set(edge)).pop()
            tri2_extra_vertex = (set(tri2) - set(edge)).pop()

            # Check if one of the triangles has a vertex that is not on the boundary, because triangles in the hole are formed exclusively by boundary nodes
            if tri1_extra_vertex in hole_nodes_set:
                pass 
            else:
                inside_tri_idx = tri_indices[1]
                break
            if tri2_extra_vertex in hole_nodes_set:
                pass
            else:
                inside_tri_idx = tri_indices[0]
                break

        if inside_tri_idx is None:
            # If no initial triangle found, possibly due to mesh issues
            print(f"Warning: Could not find an initial triangle inside hole {hole_idx}.")
            continue

        # Step 3: Perform flood-fill to find all triangles inside the hole
        triangles_in_hole = flood_fill_inside_hole(
            inside_tri_idx, elem_nodes, node_elems, hole_edges_set
        )
        triangles_to_remove.update(triangles_in_hole)

    # Step 4: Delete all identified triangles inside holes
    for tri_idx in triangles_to_remove:
        delete_triangle(tri_idx, elem_nodes, node_elems, node_nodes)

    return list(triangles_to_remove)
        
# -----------------------  Convertion Functions ----------------------- #

def convert_to_mesh_format(
    node_coords: List[Tuple[float, float]], 
    elem_nodes: List[Tuple[int, int, int]], 
    node_elems: List[List[int]], 
    node_nodes: List[List[int]]
) -> Tuple[
    List[List[float]], int, List[int], List[int], List[int], List[int], List[int], List[int]
]:
    """
    This function takes the mesh data structures and converts them into a format consisting of flat lists and pointers,
    making them more efficient for traversal, querying, and memory usage in subsequent operations.

    Parameters:
    - node_coords (List[Tuple[float, float]]): List of node coordinates as (x, y) tuples.
    - elem_nodes (List[Tuple[int, int, int]]): List of triangles, each represented as a tuple of three vertex indices (v0, v1, v2).
    - node_elems (List[List[int]]): List of lists, where each sublist contains the indices of triangles connected to a node.
    - node_nodes (List[List[int]]): List of lists, where each sublist contains the indices of nodes connected to a node.

    Returns:
    - Tuple containing:
        - node_coords: List of node coordinates.
        - numb_elems: Number of elements in the mesh.
        - elem2nodes: Flat list of vertex indices for each triangle.
        - p_elem2nodes: Pointer list indicating the start index of each triangle in `elem2nodes`.
        - node2elems: Flat list of triangle indices connected to each node.
        - p_node2elems: Pointer list indicating the start index of each node's elements in `node2elems`.
        - node2nodes: Flat list of node indices connected to each node.
        - p_node2nodes: Pointer list indicating the start index of each node's neighbors in `node2nodes`.
    """
    # 1. Copy the coordinates
    node_coords = [[x, y] for x, y in node_coords]

    # 2. Flatten the elem_nodes and create the pointer list p_elem2nodes
    elem2nodes = []
    p_elem2nodes = [0]
    current_p_elem2node = 0
    for tri in elem_nodes:
        elem2nodes.extend(tri)
        p_elem2nodes.append(current_p_elem2node + len(tri))
        current_p_elem2node += len(tri)

    numb_elems = len(elem_nodes)

    # 3. Flatten the node_elems and create the pointer list p_node2elems
    node2elems = []
    p_node2elems = [0]
    current_p_node2elems = 0
    for elems in node_elems:
        node2elems.extend(elems)
        p_node2elems.append(current_p_node2elems + len(elems))
        current_p_node2elems += len(elems)

    # 4. Flatten the node_nodes and create the pointer list p_node2nodes
    node2nodes = []
    p_node2nodes = [0]
    current_p_node2nodes = 0
    for nodes in node_nodes:
        node2nodes.extend(nodes)
        p_node2nodes.append(current_p_node2nodes + len(nodes))
        current_p_node2nodes += len(nodes)

    return node_coords, numb_elems, elem2nodes, p_elem2nodes, node2elems, p_node2elems, node2nodes, p_node2nodes


# ----------------------- Reverse Cuthill–McKee Algorithm ----------------------- #

def build_adjacency_map(
    elem2nodes: List[int], 
    p_elem2nodes: List[int]
) -> Dict[int, Set[int]]:
    """
    Builds an adjacency map from the flat `elem2nodes` list and its pointer list `p_elem2nodes`.
    
    Description:
    This function constructs an adjacency dictionary where each node is mapped to a set of its adjacent nodes 
    by iterating through each triangle defined in the mesh. The adjacency map is essential for graph-based 
    algorithms like the Reverse Cuthill–McKee (RCM) algorithm.
    
    Parameters:
    - elem2nodes (List[int]): Flat list of vertex indices for each triangle.
    - p_elem2nodes (List[int]): Pointer list indicating the start index of each triangle in `elem2nodes`.
    
    Returns:
    - Dict[int, Set[int]]: A dictionary mapping each vertex index to a set of its adjacent vertex indices.
    """
    adjacency: Dict[int, Set[int]] = defaultdict(set)
    numb_elems = len(p_elem2nodes) - 1  # Number of triangles

    for elem_idx in range(numb_elems):
        start = p_elem2nodes[elem_idx]
        end = p_elem2nodes[elem_idx + 1]
        triangle = elem2nodes[start:end]

        if len(triangle) != 3:
            continue  # Skip invalid triangles

        u, v, w = triangle
        # Add edges to the adjacency map
        adjacency[u].update([v, w])
        adjacency[v].update([u, w])
        adjacency[w].update([u, v])

    return adjacency

def reverse_cuthill_mckee(
    adjacency: Dict[int, Set[int]], 
    n_nodes: int
) -> List[int]:
    """
    Implements the Reverse Cuthill–McKee (RCM) algorithm to reorder nodes for reduced bandwidth.
    
    Description:
    The RCM algorithm reorders the nodes of a graph to reduce the bandwidth of the adjacency matrix. 
    This is particularly useful for optimizing memory access patterns and improving the performance 
    of sparse matrix operations.
    
    Parameters:
    - adjacency (Dict[int, Set[int]]): Adjacency map where each key is a node index and the value is a set of adjacent node indices.
    - n_nodes (int): Total number of nodes in the graph.
    
    Returns:
    - List[int]: A list where the index is the old node index and the value is the new node index.
    """
    visited: List[bool] = [False] * n_nodes
    permutation: List[int] = []
    
    # Find the node with the minimum degree to start
    degrees: List[int] = [len(adjacency.get(i, [])) for i in range(n_nodes)]
    try:
        start_node: int = degrees.index(min(degrees))
    except ValueError:
        start_node = 0  # Default to node 0 if no nodes exist
    
    queue: deque = deque()
    queue.append(start_node)
    visited[start_node] = True
    
    while queue:
        node = queue.popleft()
        permutation.append(node)
        neighbors_sorted = sorted(adjacency.get(node, []), key=lambda x: len(adjacency.get(x, [])))
        for neighbor in neighbors_sorted:
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
                neighbors_sorted = sorted(adjacency.get(current, []), key=lambda x: len(adjacency.get(x, [])))
                for neighbor in neighbors_sorted:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
    
    # Reverse the permutation for RCM
    permutation = permutation[::-1]
    
    # Create a mapping from old indices to new indices
    mapping = [0] * n_nodes
    for new_idx, old_idx in enumerate(permutation):
        mapping[old_idx] = new_idx

    return mapping

def apply_rcm(
    node_coords: List[List[float]], 
    elem2nodes: List[int], 
    p_elem2nodes: List[int],
    node2elems: List[int],
    p_node2elems: List[int],
    node2nodes: List[int],
    p_node2nodes: List[int]
) -> Tuple[
    List[List[float]], 
    List[int], 
    List[int], 
    List[int], 
    List[int], 
    List[int], 
    List[int]
]:
    """
    Applies the Reverse Cuthill–McKee (RCM) algorithm to reorder nodes and updates all related data structures.
    
    Description:
    This function reorders the nodes of the mesh using the RCM algorithm to reduce the bandwidth of the adjacency matrix.
    It updates the node coordinates, element-to-node mappings, and node-to-elements/nodes mappings based on the new ordering.
    
    Parameters:
    - node_coords (List[List[float]]): List of node coordinates `[x, y]`.
    - elem2nodes (List[int]): Flat list of vertex indices for each triangle.
    - p_elem2nodes (List[int]): Pointer list indicating the start index of each triangle in `elem2nodes`.
    - node2elems (List[int]): Flat list of triangle indices connected to each node.
    - p_node2elems (List[int]): Pointer list indicating the start index of each node's triangles in `node2elems`.
    - node2nodes (List[int]): Flat list of node indices connected to each node.
    - p_node2nodes (List[int]): Pointer list indicating the start index of each node's neighbors in `node2nodes`.
    
    Returns:
    - Tuple containing:
        - new_node_coords (List[List[float]]): Reordered list of node coordinates.
        - new_elem2nodes (List[int]): Reordered flat list of vertex indices for each triangle.
        - new_node2elems (List[int]): Reordered flat list of triangle indices connected to each node.
        - new_p_node2elems (List[int]): Reordered pointer list for `node2elems`.
        - new_node2nodes (List[int]): Reordered flat list of node indices connected to each node.
        - new_p_node2nodes (List[int]): Reordered pointer list for `node2nodes`.
    """
    n_nodes: int = len(node_coords)
    
    # Step 1: Build adjacency map from elem2nodes and p_elem2nodes
    adjacency: Dict[int, Set[int]] = build_adjacency_map(elem2nodes, p_elem2nodes)
    
    # Step 2: Get RCM mapping (old_idx -> new_idx)
    mapping: List[int] = reverse_cuthill_mckee(adjacency, n_nodes)
    
    # Step 3: Create inverse mapping: new_idx -> old_idx
    inverse_mapping: List[int] = [0] * n_nodes
    for old_idx, new_idx in enumerate(mapping):
        inverse_mapping[new_idx] = old_idx
    
    # Step 4: Reorder node_coords based on mapping
    new_node_coords: List[List[float]] = [ [0.0, 0.0] for _ in range(n_nodes) ]
    for new_idx, old_idx in enumerate(inverse_mapping):
        new_node_coords[new_idx] = node_coords[old_idx]
    
    # Step 5: Reorder elem2nodes based on mapping
    new_elem2nodes: List[int] = [ mapping[old_idx] for old_idx in elem2nodes ]
    
    # Step 6: Reorder node2elems and p_node2elems based on mapping
    new_node2elems: List[int] = []
    new_p_node2elems: List[int] = [0]
    for new_node_idx in range(n_nodes):
        old_node_idx = inverse_mapping[new_node_idx]
        old_start = p_node2elems[old_node_idx]
        old_end = p_node2elems[old_node_idx + 1]
        triangles = node2elems[old_start:old_end]
        new_node2elems.extend(triangles)
        new_p_node2elems.append(len(new_node2elems))
    
    # Step 7: Reorder node2nodes and p_node2nodes using inverse mapping
    new_node2nodes: List[int] = []
    new_p_node2nodes: List[int] = [0]
    for new_node_idx in range(n_nodes):
        old_node_idx = inverse_mapping[new_node_idx]
        old_start = p_node2nodes[old_node_idx]
        old_end = p_node2nodes[old_node_idx + 1]
        neighbors = node2nodes[old_start:old_end]
        new_neighbors = [ mapping[old_idx] for old_idx in neighbors ]
        new_node2nodes.extend(new_neighbors)
        new_p_node2nodes.append(len(new_node2nodes))
    
    return (
        new_node_coords, 
        new_elem2nodes, 
        new_node2elems, 
        new_p_node2elems, 
        new_node2nodes, 
        new_p_node2nodes
    )