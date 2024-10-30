from typing import List, Tuple, Optional, Set

from .predicates import (
    orient
)

# ----- Function that modify the triangulation data structures ----- #

def add_triangle(
    u: int, 
    v: int, 
    w: int, 
    elem_nodes: List[Optional[Tuple[int, int, int]]], 
    node_elems: List[List[int]], 
    node_nodes: List[List[int]]
) -> int:
    """
    This function inserts a new triangle defined by the vertices `u`, `v`, and `w` into the `elem_nodes` list. 
    It updates the `node_elems` structure to reflect the new triangle's membership for each vertex, and updates 
    the `node_nodes` list to record the new connections between the nodes. The function also verifies that each 
    edge is shared by at most two triangles, raising an error if an edge is shared by more than two triangles.

    Parameters:
    - u (int): Index of the first vertex of the new triangle.
    - v (int): Index of the second vertex of the new triangle.
    - w (int): Index of the third vertex of the new triangle.
    - elem_nodes (List[Tuple[int, int, int]]): List where each element is a tuple of three vertex indices representing a triangle.
    - node_elems (List[List[int]]): List of lists, where each sublist contains the indices of triangles connected to a node.
    - node_nodes (List[List[int]]): List of lists, where each sublist contains the indices of nodes connected to a node.

    Returns:
    - int: The index of the newly added triangle in the `elem_nodes` list.

    Raises:
    - ValueError: If an edge is already shared by more than two triangles.
    """
    # Step 1: Add the new triangle to elem_nodes
    new_triangle_idx = len(elem_nodes)
    elem_nodes.append((u, v, w))

    # Step 2: Update node_elems for each vertex
    for vertex in (u, v, w):
        node_elems[vertex].append(new_triangle_idx)

    # Step 3: Define edges of the triangle
    edges = [
        tuple(sorted((u, v))),
        tuple(sorted((v, w))),
        tuple(sorted((w, u)))
    ]

    # Step 4: Update node_nodes and verify edge sharing
    for edge in edges:
        node1, node2 = edge

        # Update node-to-node connectivity
        if node2 not in node_nodes[node1]:
            node_nodes[node1].append(node2)
        if node1 not in node_nodes[node2]:
            node_nodes[node2].append(node1)

        # Check edge sharing by finding common triangles between node1 and node2
        common_triangles = set(node_elems[node1]).intersection(node_elems[node2])
        if len(common_triangles) > 2:
            raise ValueError(f"Edge {edge} is already shared by more than two triangles.")

    return new_triangle_idx

def delete_triangle(
    t_idx: int, 
    elem_nodes: List[Optional[Tuple[int, int, int]]], 
    node_elems: List[List[int]], 
    node_nodes: List[List[int]]
) -> None:
    """
    This function deletes a triangle identified by its index `t_idx` from the triangulation. It updates 
    the `node_elems` structure to remove references to the deleted triangle from each of its vertices, and 
    updates `node_nodes` to remove connections between nodes if no other triangle shares an edge between them.

    Parameters:
    - t_idx (int): Index of the triangle to delete.
    - elem_nodes (List[Optional[Tuple[int, int, int]]]): List of existing triangles, where each triangle is represented 
      as a tuple of three vertex indices. A value of `None` indicates a deleted triangle.
    - node_elems (List[List[int]]): List of lists, where each sublist contains the indices of triangles connected to a node.
    - node_nodes (List[List[int]]): List of lists, where each sublist contains the indices of nodes connected to a node.

    Returns:
    - None. Updates the structures in place.

    Raises:
    - IndexError: If the triangle index `t_idx` is out of bounds.
    - ValueError: If the triangle is not found in the `node_elems` list or if an inconsistency is detected.
    """
    # Step 0: Validate the triangle index
    if t_idx < 0 or t_idx >= len(elem_nodes):
        raise IndexError(f"Triangle index {t_idx} is out of bounds.")

    tri = elem_nodes[t_idx]
    if tri is None:
        return  # Triangle already deleted

    # Step 1: Define the edges of the triangle (sorted)
    edges = [
        tuple(sorted((tri[0], tri[1]))),
        tuple(sorted((tri[1], tri[2]))),
        tuple(sorted((tri[2], tri[0])))
    ]

    # Step 2: For each edge, update node_nodes accordingly
    for edge in edges:
        node1, node2 = edge

        # Find triangles sharing this edge
        triangles_node1 = set(node_elems[node1])
        triangles_node2 = set(node_elems[node2])

        common_triangles = triangles_node1.intersection(triangles_node2)

        # Remove the current triangle from the set
        common_triangles.discard(t_idx)

        if not common_triangles:
            # No other triangle shares this edge, remove the connection between node1 and node2
            if node2 in node_nodes[node1]:
                node_nodes[node1].remove(node2)
            if node1 in node_nodes[node2]:
                node_nodes[node2].remove(node1)

    # Step 3: Remove the triangle from node_elems for each vertex
    for vertex in tri:
        if t_idx in node_elems[vertex]:
            node_elems[vertex].remove(t_idx)
        else:
            raise ValueError(f"Triangle index {t_idx} not found in node_elems[{vertex}].")

    # Step 4: Remove the triangle by setting it to None
    elem_nodes[t_idx] = None

# ----- Function that query the triangulation data structures ----- #

def get_triangle_neighbors(
    triangle_idx: int, 
    elem_nodes: List[Optional[Tuple[int, int, int]]], 
    node_elems: List[List[int]]
) -> List[int]:
    """
    This function finds the neighboring triangles of a specified triangle by examining the triangles that share 
    each of its edges. A neighbor is defined as a triangle that shares an edge with the specified triangle. 
    If no neighbor exists across a particular edge, the corresponding entry in the returned list is set to -1.

    Parameters:
    - triangle_idx (int): Index of the triangle whose neighbors are to be found.
    - elem_nodes (List[Optional[Tuple[int, int, int]]]): List of existing triangles, where each triangle is represented 
      as a tuple of three vertex indices. A value of `None` indicates a deleted triangle.
    - node_elems (List[List[int]]): List of lists, where each sublist contains the indices of triangles connected 
      to a node at the corresponding index.

    Returns:
    - List[int]: A list of neighboring triangle indices, with a length of 3. A neighbor index is -1 if there is 
      no neighbor across the corresponding edge. The order of the neighbors corresponds to the edges of the triangle as follows:
        - Edge 0: (u, v)
        - Edge 1: (v, w)
        - Edge 2: (w, u)

    Raises:
    - IndexError: If the triangle index is out of bounds.
    - ValueError: If the triangle has been deleted.
    """
    # Validate the triangle index
    if triangle_idx < 0 or triangle_idx >= len(elem_nodes):
        raise IndexError(f"Triangle index {triangle_idx} is out of bounds.")

    tri = elem_nodes[triangle_idx]
    if tri is None:
        raise ValueError(f"Triangle index {triangle_idx} has been deleted.")

    # Unpack vertices of the triangle
    u, v, w = tri
    edges = [
        (u, v),  # Edge 0
        (v, w),  # Edge 1
        (w, u)   # Edge 2
    ]

    neighbors = []
    for edge in edges:
        node1, node2 = edge

        # Retrieve the list of triangles connected to each node of the edge
        node1_triangles = node_elems[node1]
        node2_triangles = node_elems[node2]

        # Find the common triangles between the two nodes, excluding the current triangle
        common_triangles = set(node1_triangles).intersection(node2_triangles)
        common_triangles.discard(triangle_idx)

        # Determine the neighbor triangle across the edge
        if common_triangles:
            neighbors.append(next(iter(common_triangles)))
        else:
            neighbors.append(-1)

    return neighbors

def get_triangle_neighbors_constrained(
    triangle_idx: int, 
    elem_nodes: List[Optional[Tuple[int, int, int]]], 
    node_elems: List[List[int]], 
    constrained_edges_set: Set[Tuple[int, int]]
) -> List[int]:
    """
    This function finds the neighboring triangles of a specified triangle by examining the triangles that share 
    each of its edges, while avoiding edges that are marked as constrained. A constrained edge is an edge that 
    should not be crossed due to specific conditions or restrictions in the triangulation.

    Parameters:
    - triangle_idx (int): Index of the triangle whose neighbors are to be found.
    - elem_nodes (List[Optional[Tuple[int, int, int]]]): List of existing triangles, where each triangle is represented 
      as a tuple of three vertex indices. A value of `None` indicates a deleted triangle.
    - node_elems (List[List[int]]): List of lists, where each sublist contains the indices of triangles connected 
      to a node at the corresponding index.
    - constrained_edges_set (Set[Tuple[int, int]]): Set of edges (as sorted tuples) that are marked as constrained.

    Returns:
    - List[int]: A list of neighboring triangle indices, with a length of 3. A neighbor index is -1 if there is 
      no neighbor across the corresponding edge or if the edge is constrained. The order of the neighbors corresponds 
      to the edges of the triangle as follows:
        - Edge 0: (u, v)
        - Edge 1: (v, w)
        - Edge 2: (w, u)

    Raises:
    - IndexError: If the triangle index is out of bounds.
    - ValueError: If the triangle has been deleted.
    """
    # Validate the triangle index
    if triangle_idx < 0 or triangle_idx >= len(elem_nodes):
        raise IndexError(f"Triangle index {triangle_idx} is out of bounds.")

    tri = elem_nodes[triangle_idx]
    if tri is None:
        raise ValueError(f"Triangle index {triangle_idx} has been deleted.")

    # Unpack vertices of the triangle
    u, v, w = tri
    edges = [
        (u, v),  # Edge 0
        (v, w),  # Edge 1
        (w, u)   # Edge 2
    ]

    neighbors = []
    for edge in edges:
        # Sort edge to check against the constrained edges set
        sorted_edge = tuple(sorted(edge))
        if sorted_edge in constrained_edges_set:
            neighbors.append(-1)
            continue

        node1, node2 = edge

        # Retrieve the list of triangles connected to each node of the edge
        node1_triangles = node_elems[node1]
        node2_triangles = node_elems[node2]

        # Find the common triangles between the two nodes, excluding the current triangle
        common_triangles = set(node1_triangles).intersection(node2_triangles)
        common_triangles.discard(triangle_idx)

        # Determine the neighbor triangle across the edge, considering constraints
        if common_triangles:
            neighbors.append(next(iter(common_triangles)))
        else:
            neighbors.append(-1)

    return neighbors

def adjacent(
    v: int, 
    w: int, 
    elem_nodes: List[Optional[Tuple[int, int, int]]], 
    node_elems: List[List[int]]
) -> Optional[int]:
    """
    This function identifies the vertex that is opposite to a given edge (v, w) in an adjacent triangle. 
    It searches for triangles that share the edge (v, w), and then returns the third vertex of one of those triangles.

    Parameters:
    - v (int): Index of the first vertex of the edge.
    - w (int): Index of the second vertex of the edge.
    - elem_nodes (List[Optional[Tuple[int, int, int]]]): List of existing triangles, where each triangle is represented 
      as a tuple of three vertex indices. A value of `None` indicates a deleted triangle.
    - node_elems (List[List[int]]): List of lists, where each sublist contains the indices of triangles connected 
      to a node at the corresponding index.

    Returns:
    - Optional[int]: The index of the vertex opposite to the edge (v, w) in the adjacent triangle, or `None` 
      if no adjacent triangle exists.
    """
    # Retrieve the triangles connected to both nodes v and w
    v_triangles = node_elems[v]
    w_triangles = node_elems[w]

    # Find the common triangles that contain both v and w
    common_triangles = set(v_triangles).intersection(w_triangles)

    if not common_triangles:
        return None  # No common triangle found for the edge (v, w)

    # Iterate through the common triangles to find the opposite vertex
    for triangle_idx in common_triangles:
        tri = elem_nodes[triangle_idx]
        if tri is None:
            continue  # Skip deleted triangles

        # Find the vertex opposite to edge (v, w)
        for vertex in tri:
            if vertex != v and vertex != w:
                return vertex

    return None

def adjacent_2_vertex(
    u: int, 
    elem_nodes: List[Optional[Tuple[int, int, int]]], 
    node_elems: List[List[int]], 
    node_coords: List[Tuple[float, float]]
) -> Tuple[int, int]:
    """
    This function finds pairs of vertices `(v, w)` such that `(u, v, w)` forms a positively oriented triangle 
    (counter-clockwise) in the Delaunay triangulation. It inspects all triangles that share the vertex `u` and 
    ensures that the order `(u, v, w)` is positively oriented based on the orientation test.

    Parameters:
    - u (int): Vertex index for which adjacent pairs are to be found.
    - elem_nodes (List[Optional[Tuple[int, int, int]]]): List of existing triangles, where each triangle is represented 
      as a tuple of three vertex indices. A value of `None` indicates a deleted triangle.
    - node_elems (List[List[int]]): List of lists, where each sublist contains the indices of triangles connected 
      to a node at the corresponding index.
    - node_coords (List[Tuple[float, float]]): List of 2D coordinates of the points, where each element 
      is a tuple `(x, y)` representing a point's x and y coordinates.

    Returns:
    - Tuple[int, int]: A tuple of two vertex indices `(v, w)` forming a positively oriented triangle with `u`.
    """

    # Iterate through each triangle that contains vertex u
    for tri_idx in node_elems[u]:
        triangle = elem_nodes[tri_idx]
        if triangle is None:
            continue  # Skip deleted triangles

        # Extract vertices v and w such that u, v, and w are part of the triangle
        v, w = [vertex for vertex in triangle if vertex != u]

        # Determine the orientation of the triangle formed by (u, v, w)
        orientation = orient(u, v, w, node_coords)

        if orientation < 0:
            # Swap v and w to ensure a positively oriented triangle
            v, w = w, v

        return v, w

def get_one_triangle_of_vertex(
    vertex: int, 
    node_elems: List[List[int]], 
    elem_nodes: List[Optional[Tuple[int, int, int]]]
) -> int:
    """
    This function finds and returns the index of one active triangle (not deleted) that the given vertex is a part of. 
    It iterates through the list of triangles connected to the vertex and returns the first active triangle it finds.

    Parameters:
    - vertex (int): Index of the vertex to search for.
    - node_elems (List[List[int]]): A list of lists, where each sublist contains the indices of triangles 
      connected to the corresponding vertex.
    - elem_nodes (List[Optional[Tuple[int, int, int]]]): List of triangles, where each triangle is a tuple 
      of three vertex indices or `None` if deleted.

    Returns:
    - int: Index of the active triangle that the vertex is part of.

    Raises:
    - IndexError: If the vertex index is out of bounds.
    - ValueError: If the vertex is not part of any active triangle.
    """
    # Validate vertex index
    if vertex < 0 or vertex >= len(node_elems):
        raise IndexError(f"Vertex index {vertex} is out of bounds.")

    # Iterate through the list of triangle indices associated with the vertex
    for tri_idx in node_elems[vertex]:
        # Return the first active triangle found
        if elem_nodes[tri_idx] is not None:
            return tri_idx

    # If no active triangle is found, raise an error
    raise ValueError(f"Vertex {vertex} is not part of any active triangle.")

def get_neighbor_through_edge(
    current_triangle_idx: int, 
    edge: Tuple[int, int], 
    node_elems: List[List[int]]
) -> int:
    """
    This function identifies the neighboring triangle that shares a given edge with the current triangle. 
    It does so by finding triangles connected to both vertices of the edge and returns the index of the neighboring 
    triangle if it exists.

    Parameters:
    - current_triangle_idx (int): Index of the current triangle.
    - edge (Tuple[int, int]): A tuple containing the two vertex indices that form the edge `(v1, v2)`.
    - node_elems (List[List[int]]): A list of lists, where each sublist contains the indices of triangles 
      connected to the corresponding vertex.

    Returns:
    - int: Index of the neighboring triangle that shares the given edge, or -1 if no such neighbor exists.
    """
    v1, v2 = edge

    # Retrieve the list of triangles connected to each vertex of the edge
    v1_triangles = node_elems[v1]
    v2_triangles = node_elems[v2]

    # Find the common triangles between the two vertices, excluding the current triangle
    common_triangles = set(v1_triangles).intersection(v2_triangles)
    common_triangles.discard(current_triangle_idx)

    # If a neighboring triangle exists, return its index; otherwise, return -1
    return next(iter(common_triangles), -1)
