import random
from time import time
from typing import List, Tuple, Optional, Set

from .visualize import (
    plot_triangulation_with_points, 
    plot_triangulation,
    plot_points_ordered,
    plot_triangulation_and_edge
    )

from .operations import (
    adjacent, 
    delete_triangle, 
    add_triangle
    )

from .predicates import (
    orient, 
    in_circle, 
    )

from .utils import (
    nodes_to_triangle_idx
    )

from .cdt_utils import (
    find_intersecting_triangle, 
    find_cavities, 
    walk_to_point_constrained, 
    find_bad_elem_nodes_constrained,
    order_boundary_node_coords_ccw
    )

from .dt_utils import (
    delete_bad_elem_nodes_and_collect_edges, 
    find_boundary_edges, 
    triangulate_cavity_with_new_point,
    delete_bad_elem_nodes_and_collect_edges, 
    find_boundary_edges, 
    triangulate_cavity_with_new_point
    )

from .utils import (
    log
)   

# ----------------- Insert Vertex into CDT ----------------- #

def bowyer_watson_constrained(
    u_idx: int, 
    node_coords: List[Tuple[float, float]], 
    elem_nodes: List[Optional[Tuple[int, int, int]]], 
    node_elems: List[List[int]], 
    node_nodes: List[List[int]], 
    most_recent_idx: int, 
    triangle_most_recent_idx: int, 
    constrained_edges_set: Set[Tuple[int, int]], 
    visualize: bool = False
) -> Tuple[int, int]:
    """
    This function inserts a new point `u_idx` into an existing Delaunay triangulation using a constrained version 
    of the Bowyer-Watson algorithm. The constrained algorithm identifies triangles whose circumcircles contain the new 
    point and re-triangulates the cavity formed by the removed triangles while respecting the edges marked as constrained.

    Parameters:
    - u_idx (int): Index of the point to be inserted into the triangulation.
    - node_coords (List[Tuple[float, float]]): List of vertex coordinates `(x, y)`.
    - elem_nodes (List[Optional[Tuple[int, int, int]]]): List of existing triangles, where each triangle is represented 
      as a tuple of three vertex indices. A value of `None` indicates a deleted triangle.
    - node_elems (List[List[int]]): List of lists, where each sublist contains the indices of triangles connected 
      to a node at the corresponding index.
    - node_nodes (List[List[int]]): List of lists, where each sublist contains the indices of nodes connected to a node.
    - most_recent_idx (int): Index of the most recently inserted vertex.
    - triangle_most_recent_idx (int): Index of the triangle containing the most recent vertex.
    - constrained_edges_set (Set[Tuple[int, int]]): Set of edges (as sorted tuples) that are marked as constrained.
    - visualize (bool, optional): Boolean flag to enable or disable visualization of intermediate steps. Default is False.

    Returns:
    - Tuple[int, int]: A tuple containing the index of the newly created triangle and the index of the inserted point.

    Raises:
    - ValueError: If no new triangle is created during the re-triangulation step.
    """

    # Step 1: Find one triangle whose open circumdisk contains `u` using constrained walk-to-point, starting from the most recent vertex
    initial_bad = walk_to_point_constrained(
        most_recent_idx, 
        u_idx, 
        triangle_most_recent_idx, 
        node_coords, 
        elem_nodes, 
        node_elems, 
        constrained_edges_set, 
        visualize
    )

    # Step 2: Identify all the other bad triangles using a depth-first search, while respecting constrained edges
    bad_elem_nodes = find_bad_elem_nodes_constrained(
        initial_bad, 
        u_idx, 
        node_coords, 
        elem_nodes, 
        node_elems, 
        constrained_edges_set, 
        visualize
    )

    # Step 3: Delete bad triangles and collect cavity edges, then find the boundary edges of the cavity
    cavity_edges = delete_bad_elem_nodes_and_collect_edges(
        bad_elem_nodes, 
        elem_nodes, 
        node_elems, 
        node_nodes
    )
    boundary_edges = find_boundary_edges(cavity_edges)

    # Step 4: Re-triangulate the cavity with the new point, respecting constraints
    new_triangle_idx = triangulate_cavity_with_new_point(
        boundary_edges, 
        u_idx, 
        elem_nodes, 
        node_elems, 
        node_nodes
    )

    if new_triangle_idx is not None:
        return new_triangle_idx, u_idx
    else:
        raise ValueError("The new triangle index is None")

# ----------------- Retriangulation of a Cavity with Chew's algorithm----------------- #

def convex_insert_vertex(
    u: int, 
    v: int, 
    w: int, 
    node_coords: List[Tuple[float, float]], 
    elem_nodes: List[Optional[Tuple[int, int, int]]], 
    node_elems: List[List[int]], 
    node_nodes: List[List[int]], 
    S: Set[int]
) -> None:
    """
    Inserts a vertex into the triangulation, ensuring the Delaunay condition.

    Parameters:
    - u (int): Index of the vertex to insert.
    - v (int), w (int): Indices forming the edge into which `u` is being inserted.
    - node_coords (List[Tuple[float, float]]): List of vertex coordinates `(x, y)`.
    - elem_nodes (List[Optional[Tuple[int, int, int]]]): List of existing triangles, represented as tuples of three vertex indices.
    - node_elems (List[List[int]]): List of lists, where each sublist contains the indices of triangles connected to a node.
    - node_nodes (List[List[int]]): List of lists, where each sublist contains the indices of nodes connected to a node.
    - S (Set[int]): Set of vertex indices representing the vertices in the convex hull.

    Returns:
    - None. Updates the triangulation data structures in place.
    """
    # Find the adjacent vertex `x` across edge (v, w)
    x = adjacent(w, v, elem_nodes, node_elems)

    if x is None:
        add_triangle(u, v, w, elem_nodes, node_elems, node_nodes)
        return
    
    # Check if any of the vertices are collinear
    if orient(u, v, w, node_coords) == 0:
        print("u, v, w are aligned")
    if orient(u, v, x, node_coords) == 0:
        print("u, v, x are aligned")
    if orient(u, w, x, node_coords) == 0:
        print("u, w, x are aligned")

    # Check the Delaunay condition using the in_circle test
    if x in S and in_circle(u, v, w, x, node_coords) > 0:
        t_idx = nodes_to_triangle_idx([w, v, x], elem_nodes, node_elems)
        delete_triangle(t_idx, elem_nodes, node_elems, node_nodes)

        # Recursively insert the new vertex in the subdivided triangles
        convex_insert_vertex(u, v, x, node_coords, elem_nodes, node_elems, node_nodes, S)
        convex_insert_vertex(u, x, w, node_coords, elem_nodes, node_elems, node_nodes, S)
    else:
        add_triangle(u, v, w, elem_nodes, node_elems, node_nodes)

def convex_dt(
    S: List[int], 
    node_coords: List[Tuple[float, float]], 
    elem_nodes: List[Optional[Tuple[int, int, int]]], 
    node_elems: List[List[int]], 
    node_nodes: List[List[int]]
) -> None:
    """
    Constructs the Delaunay triangulation of a convex polygon.

    Parameters:
    - S (List[int]): List of vertex indices arranged in counterclockwise order.
    - node_coords (List[Tuple[float, float]]): List of vertex coordinates `(x, y)`.
    - elem_nodes (List[Optional[Tuple[int, int, int]]]): List of existing triangles, represented as tuples of three vertex indices.
    - node_elems (List[List[int]]): List of lists, where each sublist contains the indices of triangles connected to a node.
    - node_nodes (List[List[int]]): List of lists, where each sublist contains the indices of nodes connected to a node.

    Returns:
    - None. Updates the triangulation data structures in place.
    """
    k = len(S)
    if k < 3:
        raise ValueError("At least three vertices are required to form a triangulation.")

    # Step 1: Create a random permutation of the indices
    pi = list(range(k))
    random.shuffle(pi)

    # Step 2: Initialize the doubly-linked list for the polygon vertices
    next_vertex = [(i + 1) % k for i in range(k)]
    prev_vertex = [(i - 1) % k for i in range(k)]

    # Step 3: Remove vertices from the linked list in reverse order of permutation
    for i in range(k - 1, 2, -1):
        next_vertex[prev_vertex[pi[i]]] = next_vertex[pi[i]]
        prev_vertex[next_vertex[pi[i]]] = prev_vertex[pi[i]]

    # Step 4: Add the initial triangle between `pi[0]`, `next[pi[0]]`, and `prev[pi[0]]`
    add_triangle(S[pi[0]], S[next_vertex[pi[0]]], S[prev_vertex[pi[0]]], elem_nodes, node_elems, node_nodes)

    # Step 5: Insert remaining vertices using `convex_insert_vertex`
    for i in range(3, k):
        convex_insert_vertex(S[pi[i]], S[next_vertex[pi[i]]], S[prev_vertex[pi[i]]], node_coords, elem_nodes, node_elems, node_nodes, set(S))

def retriangulate_cavity(
    V: List[int], 
    common_vertex: int, 
    node_coords: List[Tuple[float, float]], 
    elem_nodes: List[Optional[Tuple[int, int, int]]], 
    node_elems: List[List[int]], 
    node_nodes: List[List[int]], 
    visualize: bool = False
) -> None:
    """
    Retriangulates a polygonal cavity by removing existing triangles formed exclusively by vertices in `V`
    and then retriangulating the cavity.

    Parameters:
    - V (List[int]): List of vertex indices defining the cavity (unordered).
    - common_vertex (int): The vertex index that is common among all triangles to be removed (typically the newly inserted vertex `u`).
    - node_coords (List[Tuple[float, float]]): List of vertex coordinates `(x, y)`.
    - elem_nodes (List[Optional[Tuple[int, int, int]]]): List of existing triangles, represented as tuples of three vertex indices.
    - node_elems (List[List[int]]): List of lists, where each sublist contains the indices of triangles connected to a node.
    - node_nodes (List[List[int]]): List of lists, where each sublist contains the indices of nodes connected to a node.
    - visualize (bool, optional): Boolean flag to enable or disable visualization of intermediate steps. Default is False.

    Returns:
    - None. Modifies the triangulation data structures in place.
    """
    V_set = set(V)
    elem_nodes_to_remove = set()

    # Step 1: Identify triangles to remove (those where all three vertices are in `V`)
    for t_idx in node_elems[common_vertex]:
        triangle = elem_nodes[t_idx]
        if triangle is None:
            continue
        if all(v in V_set for v in triangle):
            elem_nodes_to_remove.add(t_idx)

    # Collect and remove the triangles
    cavity_edges = delete_bad_elem_nodes_and_collect_edges(
        elem_nodes_to_remove, 
        elem_nodes, 
        node_elems, 
        node_nodes
    )
    boundary_edges = find_boundary_edges(cavity_edges)

    # Step 2: Extract the vertices from the boundary edges in counterclockwise order
    ordered_node_coords = order_boundary_node_coords_ccw(boundary_edges, node_coords)

    # Step 3: Retriangulate the cavity using convex Delaunay triangulation
    convex_dt(ordered_node_coords, node_coords, elem_nodes, node_elems, node_nodes)

    if visualize:
        plot_triangulation(node_coords, elem_nodes)

# ----------------- Cavities Constrained Triangulation ----------------- #

def cavity_constrained_insert_vertex(
    u: int, 
    v: int, 
    w: int, 
    node_coords: List[Tuple[float, float]], 
    elem_nodes: List[Optional[Tuple[int, int, int]]], 
    node_elems: List[List[int]], 
    node_nodes: List[List[int]], 
    marked_nodes: Set[int]
) -> None:
    """
    Inserts a vertex into the triangulation, maintaining the Delaunay condition.

    Parameters:
    - u (int): Index of the vertex to insert.
    - v, w (int): Indices of the constraint edge `(v, w)`.
    - node_coords (List[Tuple[float, float]]): List of vertex coordinates.
    - elem_nodes (List[Optional[Tuple[int, int, int]]]): List of existing triangles, where each triangle is represented 
      as a tuple of three vertex indices.
    - node_elems (List[List[int]]): List of lists, where each sublist contains the indices of triangles connected to a node.
    - node_nodes (List[List[int]]): List of lists, where each sublist contains the indices of nodes connected to a node.
    - marked_nodes (Set[int]): Set of nodes that have been marked for retriangulation.

    Returns:
    - None. Updates the triangulation data structures in place.
    """
    # Find the adjacent vertex `x` across edge (v, w)
    x = adjacent(w, v, elem_nodes, node_elems)

    if x is None:
        add_triangle(u, v, w, elem_nodes, node_elems, node_nodes)
        return

    # Check if the triangle satisfies the Delaunay condition
    incircle = in_circle(w, v, x, u, node_coords)
    orientation = orient(u, v, w, node_coords)

    if incircle <= 0 and orientation > 0:
        # Edge (v, w) survives; create triangle (u, v, w)
        add_triangle(u, v, w, elem_nodes, node_elems, node_nodes)
        return
    else:
        # Edge (v, w) does not satisfy Delaunay; flip the edge (v, w) -> (u, x)
        t_idx = nodes_to_triangle_idx([v, w, x], elem_nodes, node_elems)
        delete_triangle(t_idx, elem_nodes, node_elems, node_nodes)

        # Recursively insert `u` into the flipped triangles
        cavity_constrained_insert_vertex(u, v, x, node_coords, elem_nodes, node_elems, node_nodes, marked_nodes)
        cavity_constrained_insert_vertex(u, x, w, node_coords, elem_nodes, node_elems, node_nodes, marked_nodes)

        # Mark the involved nodes for retriangulation
        marked_nodes.update([u, v, w, x])

def CavityCDT(
    V: List[int], 
    node_coords: List[Tuple[float, float]], 
    elem_nodes: List[Optional[Tuple[int, int, int]]], 
    node_elems: List[List[int]], 
    node_nodes: List[List[int]], 
    visualize: bool = False
) -> None:
    """
    Retriangulates a polygonal cavity using a constrained Delaunay triangulation algorithm.
    Source : https://doi.org/10.1016/j.comgeo.2015.04.006 ( Fast segment insertion and incremental construction of constrained Delaunay triangulations )
    
    Parameters:
    - V (List[int]): List of vertex indices in counterclockwise order around the cavity, with `V[-1] - V[0]` forming the constraint edge.
    - node_coords (List[Tuple[float, float]]): List of vertex coordinates `(x, y)`.
    - elem_nodes (List[Optional[Tuple[int, int, int]]]): List of existing triangles, where each triangle is represented 
      as a tuple of three vertex indices.
    - node_elems (List[List[int]]): List of lists, where each sublist contains the indices of triangles connected to a node.
    - node_nodes (List[List[int]]): List of lists, where each sublist contains the indices of nodes connected to a node.
    - visualize (bool, optional): Boolean flag to enable or disable visualization. Default is False.

    Returns:
    - None. Updates the triangulation data structures in place.
    """
    m = len(V)
    if m < 3:
        return

    v0 = V[0]
    vm_1 = V[-1]

    # Initialize the next and prev pointers for the doubly-linked list
    next_ptr = [None] * m
    prev_ptr = [None] * m

    distance = [0] * m

    pi = list(range(0, m))


    for i in range(1, m - 1):
        next_ptr[i] = (i + 1)%m
        prev_ptr[i] = (i - 1)%m
        distance[i] = orient(v0, i, vm_1, node_coords) # distance[i] is proportional to the distance of vertex i to the line v0-vm_1
        pi[i] = i # Pi will always be a permutation of [1, 2, ..., m-2]

    distance[0] = 0
    distance[m - 1] = 0

    # Delete the delaunay_node_coords from the polygon in a random order
    for i in range(m - 2, 1, -1): # from m-2 downto 2
        # Select a vertex to delete that is not closer to v0-vm_1 than its neighbors
        j = random.randint(1,i)
        while distance[pi[j]] < distance[prev_ptr[pi[j]]] and distance[pi[j]] < distance[next_ptr[pi[j]]]:
            j = random.randint(1,i)
        # Point location: take the vertex v_pi[j] out of the doubly linked list
        next_ptr[prev_ptr[pi[j]]] = next_ptr[pi[j]]
        prev_ptr[next_ptr[pi[j]]] = prev_ptr[pi[j]]

        # Move the deleted vertex index pi[j] to follow the live delaunay_node_coords
        pi[i], pi[j] = pi[j], pi[i]
    
    # Initialize temporary data structures for retriangulation, as per the paper the algorithm might create temporary elem_nodes that would conflict with the main triangulation
    temp_elem_nodes = []
    temp_node_elems = [[] for _ in range(len(node_coords))]
    temp_node_nodes = [[] for _ in range(len(node_coords))]


    add_triangle(v0, V[pi[1]], vm_1, temp_elem_nodes, temp_node_elems, temp_node_nodes)

    # # Keep track of added points
    # added_points = [delaunay_node_coords[v0],delaunay_node_coords[V[pi[1]]], delaunay_node_coords[vm_1]]

    for i in range(2, m - 1):
        marked_nodes = set()
        cavity_constrained_insert_vertex(V[pi[i]], V[next_ptr[pi[i]]], V[prev_ptr[pi[i]]], node_coords, temp_elem_nodes, temp_node_elems, temp_node_nodes, marked_nodes)

        # added_points.append(delaunay_node_coords[V[pi[i]]])

        if V[pi[i]] in marked_nodes:
            # Use Chew's algorithm to retriangulate the cavity
            retriangulate_cavity(marked_nodes, V[pi[i]], node_coords, temp_elem_nodes, temp_node_elems, temp_node_nodes, visualize=visualize)
        

    # Update the main triangulation with the retriangulated cavity
    for tri in temp_elem_nodes:
        if tri is not None:
            add_triangle(tri[0], tri[1], tri[2], elem_nodes, node_elems, node_nodes)
    
    # plot_triangulation(delaunay_node_coords, elem_nodes)

# ---------- Constraint Edge Insertion ------------ #

def insert_constraint_edge(
    u: int, 
    v: int, 
    node_coords: List[Tuple[float, float]], 
    elem_nodes: List[Optional[Tuple[int, int, int]]], 
    node_elems: List[List[int]], 
    node_nodes: List[List[int]], 
    visualize: bool = False
) -> Tuple[List[Optional[Tuple[int, int, int]]], List[List[int]], List[List[int]]]:
    """
    Inserts a constraint edge `(u, v)` into the existing triangulation, resulting in a Constrained Delaunay Triangulation (CDT).

    Parameters:
    - u, v (int): Indices of the vertices defining the constraint edge.
    - node_coords (List[Tuple[float, float]]): List of vertex coordinates `(x, y)`.
    - elem_nodes (List[Optional[Tuple[int, int, int]]]): List of existing triangles, where each triangle is represented 
      as a tuple of three vertex indices.
    - node_elems (List[List[int]]): List of lists, where each sublist contains the indices of triangles connected to a node.
    - node_nodes (List[List[int]]): List of lists, where each sublist contains the indices of nodes connected to a node.
    - visualize (bool, optional): Boolean flag to enable or disable visualization of the process. Default is False.

    Returns:
    - Tuple[List[Optional[Tuple[int, int, int]]], List[List[int]], List[List[int]]]: Updated `elem_nodes`, `node_elems`, and `node_nodes`.
    """

    # Step 1: Check if the constraint edge already exists
    if v in node_nodes[u] or u in node_nodes[v]:
        return elem_nodes, node_elems, node_nodes

    # # Step 2: Find all elem_nodes intersected by the constraint edge s
    intersected_elem_nodes = find_intersecting_triangle(u, v, node_coords, elem_nodes, node_elems, visualize)

    if len(intersected_elem_nodes) == 0:
        return elem_nodes, node_elems, node_nodes

    # Step 3: Identify the polygonal cavities on both sides of s
    cavities = find_cavities(u, v, intersected_elem_nodes, node_coords, elem_nodes)
    for t_idx in intersected_elem_nodes:
        delete_triangle(t_idx, elem_nodes, node_elems, node_nodes)

    if visualize:
        plot_triangulation_and_edge(node_coords, elem_nodes, [u, v])

    # Step 4: Retriangulate each cavity using cavityCDT
    for cavity_nodes in cavities:
        # Retriangulate the cavity
        CavityCDT(cavity_nodes, node_coords, elem_nodes, node_elems, node_nodes, visualize)

    return elem_nodes, node_elems, node_nodes

def constrained_delaunay_triangulation(
    constraint_edges: List[Tuple[int, int]], 
    node_coords: List[Tuple[float, float]], 
    elem_nodes: List[Optional[Tuple[int, int, int]]], 
    node_elems: List[List[int]], 
    node_nodes: List[List[int]], 
    visualize: bool = False, 
    verbose: int = 1
) -> Tuple[List[Tuple[float, float]], List[Optional[Tuple[int, int, int]]], List[List[int]], List[List[int]]]:
    """
    Performs Constrained Delaunay Triangulation (CDT) on a given Delaunay Triangulation and constraint edges.

    Description:
    This function iteratively inserts constraint edges into an existing Delaunay Triangulation to create 
    a Constrained Delaunay Triangulation (CDT). It ensures that the resulting triangulation maintains 
    the Delaunay property, except along the specified constraint edges.

    Parameters:
    - constraint_edges (List[Tuple[int, int]]): List of edges (as tuples) that must appear in the final triangulation.
    - node_coords (List[Tuple[float, float]]): List of vertex coordinates `(x, y)`.
    - elem_nodes (List[Optional[Tuple[int, int, int]]]): List of existing triangles, where each triangle is represented 
      as a tuple of three vertex indices. A value of `None` indicates a deleted triangle.
    - node_elems (List[List[int]]): List of lists, where each sublist contains the indices of triangles connected to a node.
    - node_nodes (List[List[int]]): List of lists, where each sublist contains the indices of nodes connected to a node.
    - visualize (bool, optional): Boolean indicating whether to visualize the triangulation after each constraint insertion. Default is `False`.
    - verbose (int, optional): Integer indicating the verbosity level (0: silent, 1: basic, 2: detailed). Default is `1`.

    Returns:
    - Tuple[List[Tuple[float, float]], List[Optional[Tuple[int, int, int]]], List[List[int]], List[List[int]]]:
        - Updated list of `node_coords`.
        - Updated list of `elem_nodes` after inserting constraint edges.
        - Updated `node_elems` list mapping each vertex to the indices of triangles connected to it.
        - Updated `node_nodes` list mapping each vertex to the indices of neighboring vertices.
    """
    
    start_time = time()
    log(f"Starting Constrained Delaunay Triangulation", verbose, level=2)
    total_constraints = len(constraint_edges)

    constraints_processed = 0
    elem_nodes_before = len(elem_nodes)
    
    for idx, edge in enumerate(constraint_edges, 1):
        u, v = edge
        log(f"\nInserting constraint edge {idx}/{total_constraints}: ({u}, {v})", verbose, level=3)
        step_start = time()
        
        # Insert the constraint edge and update triangulation
        try:
            elem_nodes, node_elems, node_nodes = insert_constraint_edge(
                u, v, node_coords, elem_nodes, node_elems, node_nodes, visualize
            )
            constraints_processed += 1
            elem_nodes_after = len(elem_nodes)
            elem_nodes_inserted = elem_nodes_after - elem_nodes_before
            elem_nodes_before = elem_nodes_after
            log(f"Inserted edge ({u}, {v}) in {time() - step_start:.4f} seconds. "
                f"elem_nodes increased by {elem_nodes_inserted}. Total elem_nodes: {elem_nodes_after}", verbose, level=4)
        except Exception as e:
            raise ValueError(f"Error inserting edge ({u}, {v}): {e}")
            # log(f"Error inserting edge ({u}, {v}): {e}", verbose, level=2)
    
    total_time = time() - start_time
    log(f"\nConstrained Delaunay Triangulation completed in {total_time:.4f} seconds.", verbose, level=1)
    log(f"Total constraints processed: {constraints_processed}/{total_constraints}", verbose, level=1)
    log(f"Final number of elem_nodes: {len(elem_nodes)}", verbose, level=1)
    
    return node_coords, elem_nodes, node_elems, node_nodes

# ----------------- Main CDT Function ----------------- #

# def constrained_delaunay_triangulation(boundary_node_coords, delaunay_node_coords, elem_nodes, delaunay_dic_edge_triangle, delaunay_node_elems, most_recent_idx, triangle_most_recent_idx ,visualize=False):
#     """
#     Performs a Constrained Delaunay Triangulation by inserting boundary points and enforcing constrained edges.

#     Parameters:
#     - boundary_node_coords: List of boundary points (tuples of (x, y)).
#     - delaunay_node_coords: NumPy array or list of vertex coordinates.
#     - elem_nodes: List of existing elem_nodes (each triangle is a tuple of 3 vertex indices).
#     - delaunay_dic_edge_triangle: Dictionary mapping sorted edge tuples to lists of triangle indices.
#     - delaunay_node_elems: Dictionary mapping vertex indices to lists of triangle indices.
#     - most_recent_idx: Index of the most recently inserted vertex.
#     - triangle_most_recent_idx: Index of the triangle containing the most recent vertex.
#     - visualize: Boolean flag to enable or disable visualization.

#     Returns:
#     - Updated delaunay_node_coords, elem_nodes, delaunay_dic_edge_triangle, delaunay_node_elems.
#     """
#     # Generate the constrained edges set
#     constrained_edges = set()

#     # Insert the first boundary point
#     u_idx = len(delaunay_node_coords)

#     # Add the new point to the delaunay_node_coords list
#     delaunay_node_coords = np.concatenate((delaunay_node_coords, [boundary_node_coords[0]]))

#     # Insert the first boundary point via Bowyer-Watson
#     triangle_most_recent_idx, most_recent_idx = bowyer_watson_constrained(
#         u_idx=u_idx,
#         delaunay_node_coords=delaunay_node_coords,
#         elem_nodes=elem_nodes,
#         most_recent_idx=most_recent_idx,
#         delaunay_dic_edge_triangle=delaunay_dic_edge_triangle,
#         triangle_most_recent_idx=triangle_most_recent_idx,
#         delaunay_node_elems=delaunay_node_elems,
#         constrained_edges_set=constrained_edges, # When inserting the first point, there are no constrained edges
#         visualize=visualize
#     )

#     # Iterate through boundary points and enforce constrained edges
#     for i in range(1, len(boundary_node_coords)):
#         u_idx = len(delaunay_node_coords)

#         # Add the new point to the delaunay_node_coords list
#         delaunay_node_coords = np.concatenate((delaunay_node_coords, [boundary_node_coords[i]]))

#         # Insert the boundary point via Bowyer-Watson
#         triangle_most_recent_idx, most_recent_idx = bowyer_watson_constrained(
#             u_idx=u_idx,
#             delaunay_node_coords=delaunay_node_coords,
#             elem_nodes=elem_nodes,
#             delaunay_dic_edge_triangle=delaunay_dic_edge_triangle,
#             most_recent_idx=most_recent_idx,
#             triangle_most_recent_idx=triangle_most_recent_idx,
#             delaunay_node_elems=delaunay_node_elems,
#             constrained_edges_set=constrained_edges,
#             visualize=visualize
#         )

#         # Add the new edge to the constrained edges set
#         prev_idx = len(delaunay_node_coords) - 2
#         current_idx = len(delaunay_node_coords) - 1

#         # Enforce the constrained edge between prev_idx and current_idx
#         insert_constraint_edge(
#             u=prev_idx,
#             v=current_idx,
#             delaunay_node_coords=delaunay_node_coords,
#             elem_nodes=elem_nodes,
#             delaunay_dic_edge_triangle=delaunay_dic_edge_triangle,
#             delaunay_node_elems=delaunay_node_elems,
#             visualize=visualize
#         )

#         plot_triangulation_with_points(delaunay_node_coords, elem_nodes, [delaunay_node_coords[prev_idx], delaunay_node_coords[current_idx]])

#         constrained_edges.add((prev_idx, current_idx))

#     print("Constrained Delaunay Triangulation complete.")
#     return delaunay_node_coords, elem_nodes, delaunay_dic_edge_triangle, delaunay_node_elems, constrained_edges