import numpy as np
import random
import math
from collections import defaultdict

from src.predicates import in_circle, orient, in_triangle

from src.operations import (
    delete_triangle,
    add_triangle, 
    get_triangle_neighbors,
    get_neighbor_through_edge,
    )

from src.visualize import (
    visualize_bad_triangles_step, 
    visualize_walk_step, 
    visualize_walk_to_point,
    visualize_bad_triangles,
    )

def delete_bad_triangles_and_collect_edges(bad_triangles, triangles, edge_to_triangle, vertex_to_triangles):
    """
    Deletes bad triangles and collects all their edges to form the cavity.

    Parameters:
    - bad_triangles: Set of triangle indices that are bad (to be removed).
    - triangles: List of existing triangles.
    - edge_to_triangle: Dictionary mapping edges to triangle indices or tuples of triangle indices.
    - vertex_to_triangles: Dictionary mapping vertices to triangle indices.

    Returns:
    - cavity_edges: List of boundary edges as sorted tuples (a, b).
    """
    cavity_edges = []

    for t_idx in bad_triangles:
        tri = triangles[t_idx]
        if tri is None:
            continue  # Triangle already deleted

        a, b, c = tri

        # Collect sorted edges to ensure consistency
        edges = [
            tuple(sorted((a, b))),
            tuple(sorted((b, c))),
            tuple(sorted((c, a)))
        ]

        # Add edges to cavity_edges
        cavity_edges.extend(edges)

        # Delete the triangle (assumes delete_triangle handles all necessary updates)
        delete_triangle(t_idx, triangles, edge_to_triangle, vertex_to_triangles)

    return cavity_edges

def find_boundary_edges(cavity_edges):
    """
    Identifies the boundary edges of the cavity.

    Parameters:
    - cavity_edges: List of edges (as sorted tuples) from the cavity.

    Returns:
    - boundary_edges: List of boundary edges as sorted tuples (a, b).
    """
    edge_count = {}

    for edge in cavity_edges:
        edge_count[edge] = edge_count.get(edge, 0) + 1

    # Boundary edges appear exactly once
    boundary_edges = [edge for edge, count in edge_count.items() if count == 1]

    return boundary_edges

def find_bad_triangles(initial_bad, u_idx, vertices, triangles, edge_to_triangle ,visualize=False):
    """
    Finds all the bad triangles whose circumcircles contain the new point u_idx.

    Parameters:
    - initial_bad: Index of the initial bad triangle.
    - u_idx: Index of the point to be inserted.
    - vertices: List of vertex coordinates.
    - triangles: List of existing triangles.
    - edge_to_triangle: Dictionary mapping edges to triangle indices or tuples of triangle indices.
    - visualize: Boolean flag to indicate whether to visualize each step.

    Returns:
    - bad_triangles: Set of indices of bad triangles.
    """
    bad_triangles = set()
    stack = [initial_bad]
    step_number = 0

    while stack:
        current_t_idx = stack.pop()
        if current_t_idx in bad_triangles:
            continue

        tri = triangles[current_t_idx]
        if tri is None:
            continue

        a, b, c = tri
        # The three points occur in counterclockwise order around the circle
        if orient(a, b, c, vertices) < 0:
            a, b = b, a

        step_number += 1
        # if visualize:
        #     visualize_bad_triangles_step(vertices, triangles, bad_triangles, current_t_idx, u_idx, step_number)

        if in_circle(a, b, c, u_idx, vertices) > 0:

            bad_triangles.add(current_t_idx)


            # Add neighbors to the stack for further exploration

            neighbors = get_triangle_neighbors(current_t_idx, triangles, edge_to_triangle)
            for neighbor_idx in neighbors:
                if neighbor_idx != -1 and neighbor_idx not in bad_triangles:
                    stack.append(neighbor_idx)

    # Final visualization
    if visualize:
        visualize_bad_triangles(vertices, triangles, bad_triangles, u_idx, step_number + 1)

    return bad_triangles

def triangulate_cavity_with_new_point(boundary_edges, u_idx, triangles, edge_to_triangle, vertex_to_triangles):
    """
    Retriangulates the cavity with the new point u_idx.

    Parameters:
    - boundary_edges: List of boundary edges as sorted tuples (a, b).
    - u_idx: Index of the new point to insert.
    - triangles: List of existing triangles.
    - edge_to_triangle: Dictionary mapping edges to triangle indices.
    - vertex_to_triangles: Dictionary mapping vertices to triangle indices.

    Returns:
    - None. Updates the triangles and edge_to_triangle structures.
    """

    for edge in boundary_edges:
        a, b = edge
        new_triangle_idx = add_triangle(a, b, u_idx, triangles, edge_to_triangle, vertex_to_triangles)

    # Check if boundary_edges is empty
    if boundary_edges == []:
        return None
    else:
        return new_triangle_idx

def interleave_bits(x, y, precision=32):
    """
    Interleaves the bits of two integers x and y.

    Parameters:
    - x: First integer.
    - y: Second integer.
    - precision: Number of bits in each integer.

    """
    result = 0
    for i in range(precision): 
        result |= ((x & (1 << i)) << i) | ((y & (1 << i)) << (i + 1))
    return result

def compute_z_order(vertices, precision=1000000):
    """
    Computes the Z-order (Morton order) for each vertex.
    
    Args:
    vertices (np.array): An Nx2 array of vertex coordinates (x, y)
    precision (int): Scaling factor to convert floats to ints
    
    Returns:
    np.array: An array of Z-order values for each vertex
    """
    # Scale and convert to integers
    scaled_vertices = (vertices * precision).astype(int)
    
    # Compute min values to normalize coordinates
    min_x, min_y = np.min(scaled_vertices, axis=0)
    
    # Normalize coordinates to start from (0, 0)
    normalized_vertices = scaled_vertices - [min_x, min_y]
    
    # Compute Z-order for each vertex
    z_orders = np.zeros(len(vertices), dtype=np.uint64)
    for i, (x, y) in enumerate(normalized_vertices):
        z_orders[i] = interleave_bits(x, y)
    
    return z_orders

def brio_ordering(vertices):
    """
    Generates a BRIO ordering for the given vertices.
    
    Parameters:
    - vertices: List of (x, y) tuples.
    
    Returns:
    - List of vertex indices in BRIO order.
    """
    n = len(vertices)
    max_round = math.floor(math.log2(n)) if n > 0 else 0
    rounds = defaultdict(list)
    
    for idx in range(n):
        r = 0
        while r < max_round and random.random() < 0.5:
            r += 1
        rounds[r].append(idx)
    
    # Order within each round using Z-order
    z_orders = compute_z_order(vertices)
    
    ordered_vertices = []
    for r in range(max_round + 1):
        round_vertices = rounds[r]
        # Sort vertices in the round based on Z-order
        round_vertices_sorted = sorted(round_vertices, key=lambda idx: z_orders[idx])
        # Reverse order on even-numbered rounds to enhance locality
        if r % 2 == 0:
            round_vertices_sorted = list(reversed(round_vertices_sorted))
        ordered_vertices.extend(round_vertices_sorted)
    
    return ordered_vertices

def walk_to_point(start_idx, target_idx, vertices, triangles, edge_to_triangle, vertex_to_triangle,start_triangle_idx, visualize=False):
    """
    This function finds the triangle that contains the point specified by target_idx, starting from a triangle containing 
    the point at start_idx. It uses a straight walk algorithm based on orientation tests to traverse the triangulation.
    
    Reference: "Straight Walk Algorithm" described in Roman Soukal's paper 
    (http://graphics.zcu.cz/files/106_REP_2010_Soukal_Roman.pdf)

    The initialization step identifies the orientation of the starting triangle and sets up left (l) and right (r) pointers 
    relative to the point. Then the straight walk traverses triangles until it finds the target point.

    Parameters:
    - start_idx: Index of the starting vertex.
    - target_idx: Index of the point to locate.
    - vertices: List of vertex coordinates.
    - triangles: List of triangles (each triangle is represented as a tuple of 3 vertex indices).
    - edge_to_triangle: Dictionary mapping edges (tuples of vertex indices) to the indices of one or two triangles.
    - start_triangle_idx: Index of the triangle containing the starting vertex.
    - visualize: Boolean flag to enable or disable visualization of the walk steps.

    Returns:
    - The index of the triangle that contains the target point, or raises an error if no such triangle is found.
    """
    
    current_triangle_idx = start_triangle_idx  # Start with the triangle containing the start point
    step_number = 0  # Initialize step counter for the walk
    initilization_triangles = [current_triangle_idx]  # Store the triangles visited during initialization
    main_traversal_triangles = []  # Store the triangles visited during the main traversal

    if current_triangle_idx == -1:
        raise ValueError("Starting vertex is not part of any triangle in the triangulation.")

    current_triangle = triangles[current_triangle_idx]  # Get the triangle containing the start vertex

    # Get the two other vertices of the current triangle (excluding start_idx)
    other_vertices = [v for v in current_triangle if v != start_idx]
    if len(other_vertices) != 2:
        raise ValueError("Invalid triangle configuration; a triangle must have exactly 3 vertices.")

    v1_idx, v2_idx = other_vertices

    # Determine the orientation of the triangle formed by start_idx, v1_idx, and v2_idx
    orientation = orient(start_idx, v1_idx, v2_idx, vertices)
    
    if orientation < 0:
        # If the triangle is counter-clockwise, assign the right and left vertices
        r_idx = v1_idx
        l_idx = v2_idx
    else:
        # If the triangle is clockwise, swap the assignment
        r_idx = v2_idx
        l_idx = v1_idx

    step_number += 1  # Increment the step counter for initialization
    # if visualize:
    #     visualize_walk_step(vertices, triangles, current_triangle_idx, start_idx, target_idx, r_idx, l_idx,
    #                         step_number, "Initial setup", initilization_triangles, main_traversal_triangles)

    # Check if the target point is already inside the current triangle
    if in_triangle(r_idx, start_idx, l_idx, target_idx, vertices):
        return current_triangle_idx

    # Initialization phase: rotate around the start vertex to align with the target point
    if orient(start_idx, target_idx, r_idx, vertices) > 0:
        # While the orientation indicates the target is to the right of the current edge
        while orient(start_idx, target_idx, l_idx, vertices) > 0:
            step_number += 1
            r_idx = l_idx  # Shift right

            neighbor_triangle_idx = get_neighbor_through_edge(current_triangle_idx, (start_idx, l_idx), edge_to_triangle)
            
            if neighbor_triangle_idx == -1:
                raise ValueError("Reached a boundary while traversing; point may be outside triangulation.")
            
            current_triangle_idx = neighbor_triangle_idx  # Move to the neighbor triangle
            initilization_triangles.append(current_triangle_idx)
            current_triangle = triangles[current_triangle_idx]
            l_idx = next(v for v in current_triangle if v != start_idx and v != r_idx)  # Update the left vertex
            if visualize:
                visualize_walk_step(vertices, triangles, current_triangle_idx, start_idx, target_idx, r_idx, l_idx, 
                                    step_number, "Rotating 1", initilization_triangles, main_traversal_triangles)
    else:
        # The opposite direction (left side)
        cond = True
        while cond:
            step_number += 1
            l_idx = r_idx  # Shift left
            neighbor_triangle_idx = get_neighbor_through_edge(current_triangle_idx, (start_idx, r_idx), edge_to_triangle)
            
            if neighbor_triangle_idx == -1:
                raise ValueError("Reached a boundary while traversing; point may be outside triangulation.")
            
            current_triangle_idx = neighbor_triangle_idx  # Move to the neighbor triangle
            initilization_triangles.append(current_triangle_idx)
            current_triangle = triangles[current_triangle_idx]
            r_idx = next(v for v in current_triangle if v != start_idx and v != l_idx)  # Update the right vertex
            
            cond = orient(start_idx, target_idx, r_idx, vertices) <= 0  # Continue as long as the condition holds
            if visualize:
                visualize_walk_step(vertices, triangles, current_triangle_idx, start_idx, target_idx, r_idx, l_idx, 
                                    step_number, "Rotating 2", initilization_triangles, main_traversal_triangles)

    initilization_triangles.pop()  # Remove the last triangle from initialization list
    main_traversal_triangles = [current_triangle_idx]  # Start tracking the main traversal path

    # Switch left and right pointers for the main traversal
    l_idx, r_idx = r_idx, l_idx

    # Check if the target point is already inside the current triangle
    if in_triangle(r_idx, start_idx, l_idx, target_idx, vertices):
        return current_triangle_idx
    
    # Main traversal phase: perform a straight walk through the triangulation
    while orient(target_idx, r_idx, l_idx, vertices) <= 0:
        step_number += 1
        neighbor_triangle_idx = get_neighbor_through_edge(current_triangle_idx, (r_idx, l_idx), edge_to_triangle)
        if neighbor_triangle_idx == -1:
            # If no neighbor is found, check if the target point lies inside the current triangle
            if in_triangle(r_idx, start_idx, l_idx, target_idx, vertices):
                if visualize:
                    visualize_walk_step(vertices, triangles, current_triangle_idx, start_idx, target_idx, r_idx, l_idx, 
                                        step_number, "Main path finding", initilization_triangles, main_traversal_triangles)
                return current_triangle_idx
            else:
                raise ValueError("Reached a boundary while traversing; point may be outside triangulation.")
        
        current_triangle_idx = neighbor_triangle_idx  # Move to the next triangle
        main_traversal_triangles.append(current_triangle_idx)
        current_triangle = triangles[current_triangle_idx]
        s_idx = next(v for v in current_triangle if v != r_idx and v != l_idx)  # Find the remaining vertex

        # Update the right or left pointer based on the orientation
        if orient(s_idx, start_idx, target_idx, vertices) <= 0:
            r_idx = s_idx
        else:
            l_idx = s_idx

        if in_triangle(r_idx, start_idx, l_idx, target_idx, vertices):
            if visualize:
                visualize_walk_to_point(vertices, triangles, current_triangle_idx, start_idx, target_idx, r_idx, l_idx, 
                            step_number, "Main path finding", initilization_triangles, main_traversal_triangles)

            return current_triangle_idx
        # if visualize:
        #     visualize_walk_step(vertices, triangles, current_triangle_idx, start_idx, target_idx, r_idx, l_idx, 
        #                         step_number, "Main path finding", initilization_triangles, main_traversal_triangles)

    
    if visualize:
        visualize_walk_to_point(vertices, triangles, current_triangle_idx, start_idx, target_idx, r_idx, l_idx, 
                            step_number, "Main path finding", initilization_triangles, main_traversal_triangles)


    return current_triangle_idx  # Return the index of the triangle containing the target point

