import random
from src.predicates import orient, in_circle


def random_permutation(n):
    """Generates a random permutation of integers from 0 to n-1."""
    perm = list(range(n))
    random.shuffle(perm)
    return perm

def add_triangle(u, v, w, triangles, delaunay_dic_edge_triangle, delaunay_node_elems):
    """
    Adds a new triangle (u, v, w) to the triangulation and updates the delaunay_dic_edge_triangle and delaunay_node_elems structure.

    Parameters:
    - u, v, w: Vertex indices of the new triangle.
    - triangles: List of triangles, where each triangle is a tuple of 3 vertex indices.
    - delaunay_dic_edge_triangle: Dictionary mapping each edge (tuple of 2 vertex indices) to one or two triangle indices.
    - delaunay_node_elems: List of triangle indices for each vertex.

    Returns:
    - new_triangle_idx: Index of the newly added triangle in the 'triangles' list.
    """
    # Step 1: Add the new triangle to the triangles list
    new_triangle_idx = len(triangles)
    triangles.append((u, v, w))

    # Step 2: Define the edges of the new triangle (sorted for consistency)
    edges = [
        tuple(sorted((u, v))),  # Edge 0
        tuple(sorted((v, w))),  # Edge 1
        tuple(sorted((w, u)))   # Edge 2
    ]

    # Step 3: Update delaunay_dic_edge_triangle
    for edge in edges:
        if edge in delaunay_dic_edge_triangle:
            existing = delaunay_dic_edge_triangle[edge]

            if isinstance(existing, int):
                # Edge is currently shared by one triangle
                existing_triangle_idx = existing

                # Update delaunay_dic_edge_triangle to reflect that this edge is now shared by two triangles
                delaunay_dic_edge_triangle[edge] = (existing_triangle_idx, new_triangle_idx)
            elif isinstance(existing, tuple):
                # Edge is already shared by two triangles; cannot add another
                raise ValueError(f"Edge {edge} is already shared by two triangles: {existing}.")
        else:
            # Edge is unique to this triangle; map it to this triangle
            delaunay_dic_edge_triangle[edge] = new_triangle_idx

    # Step 4: Update delaunay_node_elems
    for vertex in (u, v, w):
        delaunay_node_elems[vertex].append(new_triangle_idx)


    return new_triangle_idx

def delete_triangle(t_idx, triangles, delaunay_dic_edge_triangle, delaunay_node_elems):
    """
    Deletes a triangle from the triangulation and updates the delaunay_dic_edge_triangle structure.

    Parameters:
    - t_idx: Index of the triangle to delete.
    - triangles: List of existing triangles.
    - delaunay_dic_edge_triangle: Dictionary mapping edges to triangle indices.

    Returns:
    - None. Updates the structures in place.
    """
    if t_idx < 0 or t_idx >= len(triangles):
        raise IndexError(f"Triangle index {t_idx} is out of bounds.")

    tri = triangles[t_idx]
    if tri is None:
        return  # Triangle already deleted

    # Define the edges of the triangle (sorted)
    edges = [
        tuple(sorted((tri[0], tri[1]))),
        tuple(sorted((tri[1], tri[2]))),
        tuple(sorted((tri[2], tri[0])))
    ]

    # Step 1: Remove the triangle from delaunay_dic_edge_triangle
    for edge in edges:
        if edge in delaunay_dic_edge_triangle:
            existing = delaunay_dic_edge_triangle[edge]

            if existing == t_idx:
                # Edge is unique to this triangle; remove it
                del delaunay_dic_edge_triangle[edge]
            elif isinstance(existing, tuple) and t_idx in existing:
                # Edge is shared by two triangles; update to keep the other triangle
                other_triangle_idx = existing[0] if existing[1] == t_idx else existing[1]
                delaunay_dic_edge_triangle[edge] = other_triangle_idx
            elif isinstance(existing, int):
                # Edge is shared by another triangle; no action needed
                pass
            else:
                raise ValueError(f"Invalid entry in delaunay_dic_edge_triangle for edge {edge}.")


    # Step 2: Remove the triangle from delaunay_node_elems
    for vertex in tri:
        delaunay_node_elems[vertex].remove(t_idx)

    # Step 3: Remove the triangle by setting it to None
    triangles[t_idx] = None

def get_triangle_neighbors(triangle_idx, triangles, delaunay_dic_edge_triangle):
    """
    Retrieves the neighboring triangles of a given triangle by inspecting shared edges.

    Parameters:
    - triangle_idx: Index of the triangle whose neighbors are to be found.
    - triangles: List of existing triangles.
    - delaunay_dic_edge_triangle: Dictionary mapping edges to one or two triangle indices.

    Returns:
    - neighbors: List of neighboring triangle indices (length 3). A neighbor index is -1 if there is no neighbor across that edge.
                 The order corresponds to the edges of the triangle as follows:
                 Edge 0: (u, v), Edge 1: (v, w), Edge 2: (w, u)
    """
    if triangle_idx < 0 or triangle_idx >= len(triangles):
        raise IndexError(f"Triangle index {triangle_idx} is out of bounds.")

    tri = triangles[triangle_idx]
    if tri is None:
        raise ValueError(f"Triangle index {triangle_idx} has been deleted.")

    u, v, w = tri
    edges = [
        tuple(sorted((u, v))),  # Edge 0
        tuple(sorted((v, w))),  # Edge 1
        tuple(sorted((w, u)))   # Edge 2
    ]

    neighbors = []
    for edge in edges:
        if edge in delaunay_dic_edge_triangle:
            existing = delaunay_dic_edge_triangle[edge]
            # Handle different possible representations of existing triangles
            if isinstance(existing, int):
                if existing != triangle_idx:
                    neighbors.append(existing)
                else:
                    # The edge is only connected to this triangle
                    neighbors.append(-1)
            elif isinstance(existing, tuple):
                # Shared by two triangles
                if existing[0] == triangle_idx:
                    neighbors.append(existing[1])
                elif existing[1] == triangle_idx:
                    neighbors.append(existing[0])
                else:
                    # Neither of the triangles in the tuple is the current triangle
                    neighbors.append(-1)
            else:
                # Unexpected format
                neighbors.append(-1)
        else:
            # No triangle shares this edge
            neighbors.append(-1)

    return neighbors

def get_triangle_neighbors_constrained(triangle_idx, triangles, delaunay_dic_edge_triangle, constrained_edges_set):
    """
    Retrieves the neighboring triangles of a given triangle by inspecting shared edges,
    without crossing constrained edges.

    Parameters:
    - triangle_idx: Index of the triangle whose neighbors are to be found.
    - triangles: List of existing triangles.
    - delaunay_dic_edge_triangle: Dictionary mapping edges to one or two triangle indices.
    - constrained_edges_set: Set of edges (as sorted tuples) that are constrained.

    Returns:
    - neighbors: List of neighboring triangle indices (length 3). A neighbor index is -1 if there is no neighbor across that edge.
                 The order corresponds to the edges of the triangle as follows:
                 Edge 0: (u, v), Edge 1: (v, w), Edge 2: (w, u)
    """
    if triangle_idx < 0 or triangle_idx >= len(triangles):
        raise IndexError(f"Triangle index {triangle_idx} is out of bounds.")

    tri = triangles[triangle_idx]
    if tri is None:
        raise ValueError(f"Triangle index {triangle_idx} has been deleted.")

    u, v, w = tri
    edges = [
        tuple(sorted((u, v))),  # Edge 0
        tuple(sorted((v, w))),  # Edge 1
        tuple(sorted((w, u)))   # Edge 2
    ]

    neighbors = []
    for edge in edges:
        # Check if the edge is constrained
        if edge in constrained_edges_set:
            neighbors.append(-1)
            continue  # Do not traverse across constrained edges

        if edge in delaunay_dic_edge_triangle:
            existing = delaunay_dic_edge_triangle[edge]
            # Handle different possible representations of existing triangles
            if isinstance(existing, int):
                if existing != triangle_idx:
                    neighbors.append(existing)
                else:
                    # The edge is only connected to this triangle
                    neighbors.append(-1)
            elif isinstance(existing, (tuple, list)):
                # Shared by two triangles
                if existing[0] == triangle_idx:
                    neighbors.append(existing[1])
                elif existing[1] == triangle_idx:
                    neighbors.append(existing[0])
                else:
                    # Neither of the triangles in the tuple is the current triangle
                    neighbors.append(-1)
            else:
                # Unexpected format
                neighbors.append(-1)
        else:
            # No triangle shares this edge
            neighbors.append(-1)

    return neighbors

def adjacent(v, w, triangles, delaunay_dic_edge_triangle):
    """
    Finds the vertex opposite to edge (v, w) in the adjacent triangle.
    
    Parameters:
    - v, w: Vertex indices forming the edge.
    - triangles: List of existing triangles.
    - delaunay_dic_edge_triangle: Dictionary mapping edges to triangle indices.
    
    Returns:
    - The vertex index opposite to edge (v, w), or None if no adjacent triangle exists.
    """
    sorted_edge = tuple(sorted((v, w)))
    if sorted_edge not in delaunay_dic_edge_triangle:
        # print(f"Edge {sorted_edge} not in delaunay_dic_edge_triangle.")
        return None  # Edge does not exist in the triangulation
    
    triangles_idx = delaunay_dic_edge_triangle[sorted_edge]
    if isinstance(triangles_idx, int):
        tri = triangles[triangles_idx]
        if tri is None:
            return None

        # Find the vertex opposite to edge (v, w)
        for vertex in tri:
            if vertex != v and vertex != w:
                return vertex
    elif isinstance(triangles_idx, tuple):
        # Edge is shared by two triangles; return the vertex opposite to edge (v, w)
        tri1, tri2 = triangles[triangles_idx[0]], triangles[triangles_idx[1]]
        if tri1 is not None:
            for vertex in tri1:
                if vertex != v and vertex != w:
                    return vertex
        elif tri2 is not None:
            for vertex in tri2:
                if vertex != v and vertex != w:
                    return vertex

    return None

def find_containing_triangle(u_idx, triangles, delaunay_dic_edge_triangle, delaunay_node_coords):
    """
    Finds a triangle whose circumcircle contains the point u.
    
    Parameters:
    - u_idx: Index of the point to insert.
    - triangles: List of existing triangles.
    - delaunay_dic_edge_triangle: Dictionary mapping edges to triangle indices.
    - delaunay_node_coords: List of vertex coordinates.
    
    Returns:
    - The index of the containing triangle, or None if not found.
    """
    for idx, tri in enumerate(triangles):
        # Reorder the triangle delaunay_node_coords so that in_circle works correctly, and for it to work, the three points need occur in counterclockwise order around the circle.
        if tri is not None:
            a, b, c = tri
            if orient(a, b, c, delaunay_node_coords) < 0:
                a, b = b, a
            if in_circle(a, b, c, u_idx, delaunay_node_coords) > 0:
                return idx
    return None

def adjacent_2_vertex(u, triangles, delaunay_dic_edge_triangle, delaunay_node_elems):
    """
    Return delaunay_node_coords v, w such that uvw is a positively oriented triangle
    """
    adjacent_delaunay_node_coords = set()
    for tri_idx in delaunay_node_elems[u]:
        triangle = triangles[tri_idx]
        # Assuming triangle is a triplet of vertex indices
        v, w = [v for v in triangle if v != u]
        # Determine orientation if necessary
        # This example assumes you need to collect all adjacent pairs
        adjacent_delaunay_node_coords.add((v, w))
    
    # Convert to list or desired format
    return list(adjacent_delaunay_node_coords)

def get_one_triangle_of_vertex(vertex, delaunay_node_elems, triangles):
    """
    Returns one active triangle that the given vertex is part of.

    Parameters:
    - vertex (int): Index of the vertex.
    - delaunay_node_elems (List[List[int]]): Adjacency list mapping each vertex to its triangles.
    - triangles (List[Optional[Tuple[int, int, int]]]): List of triangles, where each triangle is a tuple of 3 vertex indices or None if deleted.

    Returns:
    - int: Index of the active triangle that the vertex is part of.
    Raises:
    - IndexError: If the vertex index is out of bounds.
    - ValueError: If the vertex is not part of any active triangle.
    """
    # Validate vertex index
    if vertex < 0 or vertex >= len(delaunay_node_elems):
        raise IndexError(f"Vertex index {vertex} is out of bounds.")

    # Iterate through the list of triangle indices associated with the vertex
    for tri_idx in delaunay_node_elems[vertex]:
        # Return the first active triangle found
        if triangles[tri_idx] is not None:
            return tri_idx
    # If no active triangle is found, raise an error
    raise ValueError(f"Vertex {vertex} is not part of any active triangle.")

def get_neighbor_through_edge(current_triangle_idx, edge, delaunay_dic_edge_triangle):
    """
    Finds the neighboring triangle across the given edge using delaunay_dic_edge_triangle.

    Parameters:
    - current_triangle_idx: Index of the current triangle.
    - edge: Tuple containing the two delaunay_node_coords that form the edge (v1, v2).
    - delaunay_dic_edge_triangle: Dictionary mapping each edge to one or two triangle indices.

    Returns:
    - The index of the neighboring triangle that shares the given edge, or -1 if no such neighbor exists.
    """
    # Sort the edge to match the convention used in delaunay_dic_edge_triangle
    sorted_edge = tuple(sorted(edge))

    triangles_sharing_edge = delaunay_dic_edge_triangle.get(sorted_edge, [])

    if isinstance(triangles_sharing_edge, int):
        # Only one triangle shares this edge; it's a boundary edge
        return -1
    elif isinstance(triangles_sharing_edge, tuple) or isinstance(triangles_sharing_edge, list):
        # Two triangles share this edge; return the one that's not the current triangle
        for tri in triangles_sharing_edge:
            if tri != current_triangle_idx:
                return tri
        return -1
    else:
        # Unexpected data structure
        raise ValueError("Invalid delaunay_dic_edge_triangle mapping.")
    