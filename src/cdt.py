import random
import numpy as np
from time import time

from src.visualize import (
    plot_triangulation_with_points, 
    plot_triangulation,
    plot_points_ordered,
    plot_triangulation_and_edge
    )

from src.operations import (
    adjacent, 
    delete_triangle, 
    add_triangle
    )

from src.predicates import (
    orient, 
    in_circle, 
    compute_angle
    )

from src.utils import (
    convert_triangle_vertices_idx_to_triangle_idx
    )

from src.cdt_utils import (
    find_intersecting_triangle, 
    find_cavities, 
    walk_to_point_constrained, 
    find_bad_triangles_constrained,
    order_boundary_vertices_ccw
    )

from src.dt_utils import (
    delete_bad_triangles_and_collect_edges, 
    find_boundary_edges, 
    triangulate_cavity_with_new_point,
    delete_bad_triangles_and_collect_edges, 
    find_boundary_edges, 
    triangulate_cavity_with_new_point
    )

from src.utils import (
    log
)   

# ----------------- Insert Vertex into CDT ----------------- #

def bowyer_watson_constrained(u_idx, vertices, triangles, edge_to_triangle, most_recent_idx, triangle_most_recent_idx, vertex_to_triangles, constrained_edges_set,visualize=False):
    """
    Implements the Bowyer-Watson algorithm to insert a point into the Delaunay triangulation.

    Parameters:
    - u_idx: Vertex index of the point to be inserted.
    - vertices: List of vertex coordinates.
    - triangles: List of existing triangles.
    - edge_to_triangle: Dictionary mapping edges to triangle indices.
    - most_recent_idx: Index of the most recently inserted vertex.
    - triangle_most_recent_idx: Index of the triangle containing the most recent vertex.
    - vertex_to_triangles: Dictionary mapping vertices to triangle indices.

    Returns:
    - None. Updates the triangles and edge_to_triangle structures.
    """

    # 1. Find one triangle whose open circumdisk contains u using walk-to-point, starting from the most recent vertex
    initial_bad = walk_to_point_constrained(most_recent_idx, u_idx, vertices, triangles, edge_to_triangle, vertex_to_triangles ,triangle_most_recent_idx, visualize)

    # 2. Find all the others by a depth-first search in the triangulation.
    bad_triangles = find_bad_triangles_constrained(initial_bad, u_idx, vertices, triangles, edge_to_triangle , constrained_edges_set , visualize)

    # 3. Delete bad triangles and collect cavity edges, and find the doundary edges of the cavity
    cavity_edges = delete_bad_triangles_and_collect_edges(bad_triangles, triangles, edge_to_triangle, vertex_to_triangles)
    boundary_edges = find_boundary_edges(cavity_edges)


    # 4. Re-triangulate the cavity with the new point
    new_triangle_idx = triangulate_cavity_with_new_point(boundary_edges, u_idx, triangles, edge_to_triangle, vertex_to_triangles)



    if new_triangle_idx is not None:
        return new_triangle_idx, u_idx
    else:
        raise ValueError("The new triangle index is None")

# ----------------- Retriangulation of a Cavity with Chew's algorithm----------------- #

def convex_insert_vertex(u, v, w, vertices, triangles, edge_to_triangle, vertex_to_triangle, S):

    """
    Inserts a vertex into the triangulation, ensuring the Delaunay condition.
    
    Parameters:
    - u: Vertex index to insert.
    - v, w: Vertex indices forming the edge into which u is being inserted.
    - vertices: List of vertex coordinates.
    - triangles: List of existing triangles.
    - edge_to_triangle: List of lists, where each sublist contains triangle indices.
    - vertex_to_triangle: List of lists, where each sublist contains triangle indices.
    - S: 
    """
    # Find the adjacent vertex x across edge (v, w)
    x = adjacent(w, v, triangles, edge_to_triangle)

    if x is None:
        add_triangle(u, v, w, triangles, edge_to_triangle, vertex_to_triangle)
        return
    
    # check if the points are aligned 
    if orient(u,v,w, vertices) == 0:
        print("u,v,w are aligned")

    if orient(u,v,x, vertices) == 0:
        print("u,v,x are aligned")

    if orient(u,w,x, vertices) == 0:
        print("u,w,x are aligned")

    # Check the Delaunay condition
    incircle = in_circle(u, v, w, x, vertices)

    if x in S and incircle > 0 : 
        t_idx = convert_triangle_vertices_idx_to_triangle_idx([w, v, x], triangles, edge_to_triangle)
        delete_triangle(t_idx, triangles, edge_to_triangle, vertex_to_triangle)

        convex_insert_vertex(u, v, x, vertices, triangles, edge_to_triangle, vertex_to_triangle, S)
        convex_insert_vertex(u, x, w, vertices, triangles, edge_to_triangle, vertex_to_triangle, S)

        return
    
    else : 
        add_triangle(u, v, w, triangles, edge_to_triangle, vertex_to_triangle)

def convex_dt(S, vertices, triangles, edge_to_triangle, vertex_to_triangle):
    """
    Constructs the Delaunay triangulation of a convex polygon.
    
    Parameters:
    - S: List of vertex arranged in counterclockwise order.
    - vertices: List of vertex coordinates.
    - triangles: List to store triangles.
    - edge_to_triangle: Dictionary to map edges to triangles.
    - vertex_to_triangle: List of lists, where each sublist contains triangle indices.

    Returns:
    - None. Updates the triangles, triangle_neighbors, and edge_to_triangle structures.
    """
    k = len(S)
    if k < 3:
        raise ValueError("At least three vertices are required to form a triangulation.")

    # Step 1: Create a random permutation of the indices
    pi = list(range(k))
    random.shuffle(pi)

    # Step 2: Initialize the doubly-linked list LL for the polygon vertices
    next_vertex = [(i + 1) % k for i in range(k)]
    prev_vertex = [(i - 1) % k for i in range(k)]

    # Step 3: Remove vertices from the linked list in reverse order of permutation
    for i in range(k - 1, 2, -1):
        next_vertex[prev_vertex[pi[i]]] = next_vertex[pi[i]]
        prev_vertex[next_vertex[pi[i]]] = prev_vertex[pi[i]]


    # Step 4: Add the initial triangle between pi[0], next[pi[0]], prev[pi[0]]
    add_triangle(S[pi[0]], S[next_vertex[pi[0]]],S[prev_vertex[pi[0]]], triangles, edge_to_triangle, vertex_to_triangle)


    # Step 5: Insert remaining vertices using ConvexInsertVertex
    for i in range(3, k):
        convex_insert_vertex(S[pi[i]], S[next_vertex[pi[i]]],S[prev_vertex[pi[i]]], vertices, triangles, edge_to_triangle, vertex_to_triangle, S)

def retriangulate_cavity(V, common_vertex, vertices, triangles, edge_to_triangle, vertex_to_triangle, visualize=False):
    """
    Retriangulates a polygonal cavity by removing existing triangles formed exclusively by vertices in V
    and then retriangulating the cavity.

    Parameters:
    - V: List of vertex indices defining the cavity (unordered).
    - common_vertex: The vertex index that is common among all triangles to be removed (typically the newly inserted vertex `u`).
    - vertices: List of vertex coordinates.
    - triangles: List of existing triangles.
    - edge_to_triangle: Dictionary mapping sorted edge tuples to lists of triangle indices.
    - vertex_to_triangle: Dictionary mapping vertex indices to lists of triangle indices.
    - visualize: Boolean flag to enable or disable visualization.

    Returns:
    - None. Modifies triangles, edge_to_triangle, and vertex_to_triangle in place.
    """
    V_set = set(V)
    triangles_to_remove = set()

    # Step 1: Identify Triangles to Remove
    # Only triangles where all three vertices are in V
    for t_idx in vertex_to_triangle[common_vertex]:
        triangle = triangles[t_idx]
        if all(v in V_set for v in triangle):
            triangles_to_remove.add(t_idx)


    # Get the boundary edges of the cavity and order the vertices in CCW order
    triangle_to_remove = list(triangles_to_remove)

    cavity_edges = delete_bad_triangles_and_collect_edges(triangle_to_remove, triangles, edge_to_triangle, vertex_to_triangle)

    boundary_edges = find_boundary_edges(cavity_edges)

    # Step 2: Extract the vertices from the boundary edges in CCW order
    ordered_vertices = order_boundary_vertices_ccw(boundary_edges, vertices)

    # Step 4: Retriangulate the Cavity Using Convex Delaunay Triangulation
    convex_dt(ordered_vertices, vertices, triangles, edge_to_triangle, vertex_to_triangle)


    if visualize:
        plot_triangulation(vertices, triangles)


# ----------------- Cavities Constrained Triangulation ----------------- #

def cavity_constrained_insert_vertex(u, v, w, vertices, triangles, edge_to_triangle, vertex_to_triangle, marked_vertices):
    """
    Inserts a vertex into the triangulation, maintaining the Delaunay condition.

    Parameters:
    - u: Index of the vertex to insert.
    - v, w: Indices of the constraint edge (v, w).
    - vertices: List of vertex coordinates.
    - triangles: List of triangles (each triangle is represented as a tuple of 3 vertex indices).
    - edge_to_triangle: List of lists, where each sublist contains triangle indices.
    - vertex_to_triangle: List of lists, where each sublist contains triangle indices.
    - marked_vertices: Set of vertices that have been marked as "bad" during the walk, to later retriangulate.

    Returns:
    - None. Updates the triangles, edge_to_triangle, and vertex_to_triangle structures.
    """
    # Find the adjacent vertex x across edge (v, w)
    x = adjacent(w,v, triangles, edge_to_triangle)

    if x is None:
        add_triangle(u, v, w, triangles, edge_to_triangle, vertex_to_triangle)
        return

    # Check if the triangle satisfies the Delaunay condition
    incircle = in_circle(w, v, x, u, vertices)
    orientation = orient(u, v, w, vertices)

    if incircle <= 0 and orientation > 0:
        # Edge (v, w) survives; create triangle (u, v, w)
        add_triangle(u, v, w, triangles, edge_to_triangle, vertex_to_triangle)
        return
    else:
        # Edge (v, w) does not satisfy Delaunay; flip the edge (vw) -> (ux)
        t_idx = convert_triangle_vertices_idx_to_triangle_idx([v, w, x], triangles, edge_to_triangle)
        delete_triangle(t_idx, triangles, edge_to_triangle, vertex_to_triangle)
        
        # Recursively insert u into the flipped triangles
        cavity_constrained_insert_vertex(u, v, x, vertices, triangles, edge_to_triangle, vertex_to_triangle, marked_vertices)
        cavity_constrained_insert_vertex(u, x, w, vertices, triangles, edge_to_triangle, vertex_to_triangle, marked_vertices)

        # Reuse the in_circle computation
        if incircle <= 0:
            # Triangle (w,v,x) is a crossed triangle
            # Mark vertices for retriangulation
            marked_vertices.update([u, v, w, x])   

def CavityCDT(V, vertices, triangles, edge_to_triangle, vertex_to_triangle, visualize=False):
    """
    Retriangulates a polygonal cavity using the specified step-by-step algorithm. 
    Source : https://doi.org/10.1016/j.comgeo.2015.04.006 ( Fast segment insertion and incremental construction of constrained Delaunay triangulations )
    
    Parameters:
    - V: Tuple of vertex indices in CCW order around the cavity, with V[-1] - V[0] being the constraint edge.
    - vertices: List of vertex coordinates.
    - triangles: List of triangles in the main triangulation.
    - edge_to_triangle: Dictionary mapping edges to triangle indices in the main triangulation.
    - vertex_to_triangle: Dictionary mapping vertex indices to triangle indices in the main triangulation.
    - visualize: Boolean flag to enable or disable visualization.
    
    Returns:
    - None. Updates triangles, edge_to_triangle, and vertex_to_triangle in place.
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
        distance[i] = orient(v0, i, vm_1, vertices) # distance[i] is proportional to the distance of vertex i to the line v0-vm_1
        pi[i] = i # Pi will always be a permutation of [1, 2, ..., m-2]

    distance[0] = 0
    distance[m - 1] = 0

    # Delete the vertices from the polygon in a random order
    for i in range(m - 2, 1, -1): # from m-2 downto 2
        # Select a vertex to delete that is not closer to v0-vm_1 than its neighbors
        j = random.randint(1,i)
        while distance[pi[j]] < distance[prev_ptr[pi[j]]] and distance[pi[j]] < distance[next_ptr[pi[j]]]:
            j = random.randint(1,i)
        # Point location: take the vertex v_pi[j] out of the doubly linked list
        next_ptr[prev_ptr[pi[j]]] = next_ptr[pi[j]]
        prev_ptr[next_ptr[pi[j]]] = prev_ptr[pi[j]]

        # Move the deleted vertex index pi[j] to follow the live vertices
        pi[i], pi[j] = pi[j], pi[i]
    
    # Initialize temporary data structures for retriangulation, as per the paper the algorithm might create temporary triangles that would conflict with the main triangulation
    temp_triangles = []
    temp_edge_to_triangle = {}
    temp_vertex_to_triangle = [[] for _ in range(len(vertices))]


    add_triangle(v0, V[pi[1]], vm_1, temp_triangles, temp_edge_to_triangle, temp_vertex_to_triangle)

    # # Keep track of added points
    # added_points = [vertices[v0],vertices[V[pi[1]]], vertices[vm_1]]

    for i in range(2, m - 1):
        marked_vertices = set()
        cavity_constrained_insert_vertex(V[pi[i]], V[next_ptr[pi[i]]], V[prev_ptr[pi[i]]], vertices, temp_triangles, temp_edge_to_triangle, temp_vertex_to_triangle, marked_vertices)

        # added_points.append(vertices[V[pi[i]]])

        if V[pi[i]] in marked_vertices:
            # Use Chew's algorithm to retriangulate the cavity
            retriangulate_cavity(marked_vertices, V[pi[i]], vertices, temp_triangles, temp_edge_to_triangle, temp_vertex_to_triangle, visualize=visualize)
        

    # Update the main triangulation with the retriangulated cavity
    for tri in temp_triangles:
        if tri is not None:
            add_triangle(tri[0], tri[1], tri[2], triangles, edge_to_triangle, vertex_to_triangle)
    
    # plot_triangulation(vertices, triangles)

# ---------- Constraint Edge Insertion ------------ #

def insert_constraint_edge(u, v, vertices, triangles, edge_to_triangle, vertex_to_triangle ,visualize=False):
    """
    Inserts a constraint edge (u, v) into the existing triangulation, resulting in a Constrained Delaunay Triangulation (CDT).

    Parameters:
    - u, v: Indices of the vertices defining the constraint edge.
    - vertices: List of vertex coordinates.
    - triangles: List of triangles (each triangle is a tuple of 3 vertex indices).
    - edge_to_triangle: Dictionary mapping edges to one or two triangle indices.
    - visualize: Boolean flag to enable or disable visualization of the process.

    Returns:
    - Updated triangles and edge_to_triangle reflecting the inserted constraint edge.
    """
    # Step 1: Check if the constraint edge already exists
    s = tuple(sorted((u, v)))
    if s in edge_to_triangle:
        return triangles, edge_to_triangle, vertex_to_triangle


    print(f"Inserting constraint edge ({u}, {v})")
    # Step 2: Find all triangles intersected by the constraint edge s
    intersected_triangles = find_intersecting_triangle(u, v, vertices, triangles, edge_to_triangle, vertex_to_triangle, visualize)
    print(f"Intersected triangles: {intersected_triangles}")

    if len(intersected_triangles) == 0:
        return triangles, edge_to_triangle, vertex_to_triangle

    # Step 3: Identify the polygonal cavities on both sides of s
    cavities = find_cavities(u, v, intersected_triangles, vertices, triangles)
    for t_idx in intersected_triangles:
        delete_triangle(t_idx, triangles, edge_to_triangle, vertex_to_triangle)

    if visualize:
        plot_triangulation_and_edge(vertices, triangles, [u, v])

    # Step 4: Retriangulate each cavity using cavityCDT
    for cavity_vertices in cavities:
        # Retriangulate the cavity
        CavityCDT(cavity_vertices, vertices, triangles, edge_to_triangle, vertex_to_triangle, visualize)


    return triangles, edge_to_triangle, vertex_to_triangle

def constrained_delaunay_triangulation(constraint_edges, vertices, triangles, edge_to_triangle, vertex_to_triangle, visualize=False, verbose=1):
    """
    Performs Constrained Delaunay Triangulation (CDT) on a given Delaunay Triangulation and constraint edges.

    Parameters:
    - constraint_edges: List of edges (tuples) that must appear in the final triangulation.
    - vertices: List or array of vertex coordinates.
    - triangles: List of triangles (tuples of vertex indices) representing the current triangulation.
    - edge_to_triangle: Dictionary mapping edges to triangle indices.
    - vertex_to_triangle: List of lists, where each sublist contains triangle indices associated with a vertex.
    - visualize: Boolean indicating whether to visualize the triangulation after each constraint insertion.
    - verbose: Integer indicating the verbosity level (0: silent, 1: basic, 2: detailed).

    Returns:
    - vertices: Updated list of vertices.
    - triangles: Updated list of triangles after inserting constraint edges.
    - edge_to_triangle: Updated edge-to-triangle mapping.
    - vertex_to_triangle: Updated vertex-to-triangle list.
    """
    
    def log(message, verbose, level=1):
        """Helper function to print messages based on verbosity level."""
        if verbose >= level:
            print(message)
    
    start_time = time()
    log(f"Starting Constrained Delaunay Triangulation", verbose, level=2)
    total_constraints = len(constraint_edges)

    constraints_processed = 0
    triangles_before = len(triangles)
    
    for idx, edge in enumerate(constraint_edges, 1):
        u, v = edge
        log(f"\nInserting constraint edge {idx}/{total_constraints}: ({u}, {v})", verbose, level=3)
        step_start = time()
        
        # Insert the constraint edge and update triangulation
        try:
            triangles, edge_to_triangle, vertex_to_triangle = insert_constraint_edge(
                u, v, vertices, triangles, edge_to_triangle, vertex_to_triangle, visualize
            )
            constraints_processed += 1
            triangles_after = len(triangles)
            triangles_inserted = triangles_after - triangles_before
            triangles_before = triangles_after
            log(f"Inserted edge ({u}, {v}) in {time() - step_start:.4f} seconds. "
                f"Triangles increased by {triangles_inserted}. Total triangles: {triangles_after}", verbose, level=4)
        except Exception as e:
            log(f"Error inserting edge ({u}, {v}): {e}", verbose, level=2)
    
    total_time = time() - start_time
    log(f"\nConstrained Delaunay Triangulation completed in {total_time:.4f} seconds.", verbose, level=1)
    log(f"Total constraints processed: {constraints_processed}/{total_constraints}", verbose, level=1)
    log(f"Final number of triangles: {len(triangles)}", verbose, level=1)
    
    return vertices, triangles, edge_to_triangle, vertex_to_triangle


# ----------------- Main CDT Function ----------------- #

# def constrained_delaunay_triangulation(boundary_points, vertices, triangles, edge_to_triangle, vertex_to_triangles, most_recent_idx, triangle_most_recent_idx ,visualize=False):
#     """
#     Performs a Constrained Delaunay Triangulation by inserting boundary points and enforcing constrained edges.

#     Parameters:
#     - boundary_points: List of boundary points (tuples of (x, y)).
#     - vertices: NumPy array or list of vertex coordinates.
#     - triangles: List of existing triangles (each triangle is a tuple of 3 vertex indices).
#     - edge_to_triangle: Dictionary mapping sorted edge tuples to lists of triangle indices.
#     - vertex_to_triangles: Dictionary mapping vertex indices to lists of triangle indices.
#     - most_recent_idx: Index of the most recently inserted vertex.
#     - triangle_most_recent_idx: Index of the triangle containing the most recent vertex.
#     - visualize: Boolean flag to enable or disable visualization.

#     Returns:
#     - Updated vertices, triangles, edge_to_triangle, vertex_to_triangles.
#     """
#     # Generate the constrained edges set
#     constrained_edges = set()

#     # Insert the first boundary point
#     u_idx = len(vertices)

#     # Add the new point to the vertices list
#     vertices = np.concatenate((vertices, [boundary_points[0]]))

#     # Insert the first boundary point via Bowyer-Watson
#     triangle_most_recent_idx, most_recent_idx = bowyer_watson_constrained(
#         u_idx=u_idx,
#         vertices=vertices,
#         triangles=triangles,
#         most_recent_idx=most_recent_idx,
#         edge_to_triangle=edge_to_triangle,
#         triangle_most_recent_idx=triangle_most_recent_idx,
#         vertex_to_triangles=vertex_to_triangles,
#         constrained_edges_set=constrained_edges, # When inserting the first point, there are no constrained edges
#         visualize=visualize
#     )

#     # Iterate through boundary points and enforce constrained edges
#     for i in range(1, len(boundary_points)):
#         u_idx = len(vertices)

#         # Add the new point to the vertices list
#         vertices = np.concatenate((vertices, [boundary_points[i]]))

#         # Insert the boundary point via Bowyer-Watson
#         triangle_most_recent_idx, most_recent_idx = bowyer_watson_constrained(
#             u_idx=u_idx,
#             vertices=vertices,
#             triangles=triangles,
#             edge_to_triangle=edge_to_triangle,
#             most_recent_idx=most_recent_idx,
#             triangle_most_recent_idx=triangle_most_recent_idx,
#             vertex_to_triangles=vertex_to_triangles,
#             constrained_edges_set=constrained_edges,
#             visualize=visualize
#         )

#         # Add the new edge to the constrained edges set
#         prev_idx = len(vertices) - 2
#         current_idx = len(vertices) - 1

#         # Enforce the constrained edge between prev_idx and current_idx
#         insert_constraint_edge(
#             u=prev_idx,
#             v=current_idx,
#             vertices=vertices,
#             triangles=triangles,
#             edge_to_triangle=edge_to_triangle,
#             vertex_to_triangle=vertex_to_triangles,
#             visualize=visualize
#         )

#         plot_triangulation_with_points(vertices, triangles, [vertices[prev_idx], vertices[current_idx]])

#         constrained_edges.add((prev_idx, current_idx))

#     print("Constrained Delaunay Triangulation complete.")
#     return vertices, triangles, edge_to_triangle, vertex_to_triangles, constrained_edges