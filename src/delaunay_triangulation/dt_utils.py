import numpy as np
import random
from collections import defaultdict
from typing import List, Tuple, Optional, Set

from .predicates import (
    in_circle, 
    orient, 
    in_triangle
    )

from .operations import (
    delete_triangle,
    add_triangle, 
    get_triangle_neighbors,
    get_neighbor_through_edge,
    )

from .visualize import (
    visualize_bad_elem_nodes_step, 
    visualize_walk_step, 
    visualize_walk_to_point,
    visualize_bad_elem_nodes,
    )


# ------------- Function for Bowyer-Watson Algorithm ------------- #

def delete_bad_elem_nodes_and_collect_edges(
    bad_elem_nodes: Set[int], 
    elem_nodes: List[Optional[Tuple[int, int, int]]], 
    node_elems: List[List[int]], 
    node_nodes: List[List[int]]
) -> List[Tuple[int, int]]:
    """
    This function deletes triangles identified as "bad" and collects the edges of these triangles. 
    The collected edges are used to determine the boundary of the cavity created by deleting the bad triangles.

    Parameters:
    - bad_elem_nodes (Set[int]): Set of triangle indices that are to be removed.
    - elem_nodes (List[Optional[Tuple[int, int, int]]]): List of existing triangles, where each triangle is represented 
      as a tuple of three vertex indices. A value of `None` indicates a deleted triangle.
    - node_elems (List[List[int]]): List of lists, where each sublist contains the indices of triangles connected 
      to a node at the corresponding index.
    - node_nodes (List[List[int]]): List of lists, where each sublist contains the indices of nodes connected to each node.

    Returns:
    - List[Tuple[int, int]]: List of boundary edges as sorted tuples `(a, b)`.
    """
    cavity_edges = []

    for t_idx in bad_elem_nodes:
        tri = elem_nodes[t_idx]
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
        delete_triangle(t_idx, elem_nodes, node_elems, node_nodes)

    return cavity_edges

def find_boundary_edges(
        cavity_edges: List[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    """
    This function identifies the boundary edges of a cavity by counting the occurrence of each edge 
    in the list of cavity edges. An edge that appears exactly once is considered a boundary edge.

    Parameters:
    - cavity_edges (List[Tuple[int, int]]): List of edges (as sorted tuples) from the cavity.

    Returns:
    - List[Tuple[int, int]]: List of boundary edges as sorted tuples `(a, b)`.
    """
    edge_count = {}

    for edge in cavity_edges:
        edge_count[edge] = edge_count.get(edge, 0) + 1

    # Boundary edges appear exactly once
    boundary_edges = [edge for edge, count in edge_count.items() if count == 1]

    return boundary_edges

def find_bad_elem_nodes(
    initial_bad: int, 
    u_idx: int, 
    node_coords: List[Tuple[float, float]], 
    elem_nodes: List[Optional[Tuple[int, int, int]]], 
    node_elems: List[List[int]], 
    visualize: bool = False
) -> Set[int]:
    """
    This function identifies all the triangles that need to be removed due to the insertion of a new point `u_idx`. 
    It starts with an initial bad triangle and uses a depth-first search to find all triangles whose circumcircles 
    contain the new point.

    Parameters:
    - initial_bad (int): Index of the initial bad triangle.
    - u_idx (int): Index of the point to be inserted.
    - node_coords (List[Tuple[float, float]]): List of vertex coordinates.
    - elem_nodes (List[Optional[Tuple[int, int, int]]]): List of existing triangles, where each triangle is represented 
      as a tuple of three vertex indices. A value of `None` indicates a deleted triangle.
    - node_elems (List[List[int]]): List of lists, where each sublist contains the indices of triangles connected 
      to a node at the corresponding index.
    - visualize (bool, optional): Boolean flag to indicate whether to visualize each step. Default is False.

    Returns:
    - Set[int]: Set of indices of bad triangles.
    """
    bad_elem_nodes = set()
    stack = [initial_bad]
    step_number = 0

    while stack:
        current_t_idx = stack.pop()
        if current_t_idx in bad_elem_nodes:
            continue

        tri = elem_nodes[current_t_idx]
        if tri is None:
            continue

        a, b, c = tri
        # The three points occur in counterclockwise order around the circle
        if orient(a, b, c, node_coords) < 0:
            a, b = b, a

        step_number += 1
        # if visualize:
        #     visualize_bad_elem_nodes_step(node_coords, elem_nodes, bad_elem_nodes, current_t_idx, u_idx, step_number)

        if in_circle(a, b, c, u_idx, node_coords) > 0:

            bad_elem_nodes.add(current_t_idx)


            # Add neighbors to the stack for further exploration

            neighbors = get_triangle_neighbors(current_t_idx, elem_nodes, node_elems)
            for neighbor_idx in neighbors:
                if neighbor_idx != -1 and neighbor_idx not in bad_elem_nodes:
                    stack.append(neighbor_idx)

    # Final visualization
    if visualize:
        visualize_bad_elem_nodes(node_coords, elem_nodes, bad_elem_nodes, u_idx, step_number + 1)

    return bad_elem_nodes

def triangulate_cavity_with_new_point(
    boundary_edges: List[Tuple[int, int]], 
    u_idx: int, 
    elem_nodes: List[Optional[Tuple[int, int, int]]], 
    node_elems: List[List[int]], 
    node_nodes: List[List[int]]
) -> Optional[int]:
    """
    This function creates new triangles by connecting the new point `u_idx` with each of the boundary edges of the cavity.
    The boundary edges are defined as pairs of vertex indices. For each boundary edge `(a, b)`, a new triangle `(a, b, u_idx)` 
    is created and added to the triangulation data structures.

    Parameters:
    - boundary_edges (List[Tuple[int, int]]): List of boundary edges represented as sorted tuples `(a, b)`.
    - u_idx (int): Index of the new point to insert.
    - elem_nodes (List[Optional[Tuple[int, int, int]]]): List of existing triangles, where each triangle is represented 
      as a tuple of three vertex indices. A value of `None` indicates a deleted triangle.
    - node_elems (List[List[int]]): List of lists, where each sublist contains the indices of triangles connected 
      to a node at the corresponding index.
    - node_nodes (List[List[int]]): List of lists, where each sublist contains the indices of nodes connected to a node.

    Returns:
    - Optional[int]: Index of the last newly created triangle or `None` if no new triangles were created.
    """
    new_triangle_idx = None

    for edge in boundary_edges:
        a, b = edge
        new_triangle_idx = add_triangle(a, b, u_idx, elem_nodes, node_elems, node_nodes)

    return new_triangle_idx

# ------------- Function for Walk-to-Point Algorithm ------------- #

def interleave_bits(
        x: int, 
        y: int, 
        precision: int = 32
) -> int:
    """
    This function interleaves the bits of two integers `x` and `y`, producing a new integer by alternating the bits of `x` and `y`.
    This technique is often used in spatial indexing, such as creating Morton codes.

    Parameters:
    - x (int): The first integer.
    - y (int): The second integer.
    - precision (int, optional): The number of bits in each integer. Default is 32.

    Returns:
    - int: An integer resulting from interleaving the bits of `x` and `y`.
    """
    result = 0
    for i in range(precision):
        result |= ((x & (1 << i)) << i) | ((y & (1 << i)) << (i + 1))
    return result

def compute_z_order(
        node_coords: np.ndarray, 
        precision: int = 1000000
) -> np.ndarray:
    """
    This function computes the Z-order value (or Morton code) for each vertex based on its 
    2D coordinates `(x, y)`. Z-order values are calculated by interleaving the bits of the 
    normalized x and y coordinates.

    Parameters:
    - node_coords (np.ndarray): An Nx2 array of vertex coordinates `(x, y)`.
    - precision (int, optional): Scaling factor to convert float coordinates to integers. 
      Default is 1,000,000.

    Returns:
    - np.ndarray: An array of Z-order values for each vertex.
    """
    # Scale and convert coordinates to integers
    scaled_node_coords = (node_coords * precision).astype(int)
    
    # Compute min values to normalize coordinates
    min_x, min_y = np.min(scaled_node_coords, axis=0)
    
    # Normalize coordinates to start from (0, 0)
    normalized_node_coords = scaled_node_coords - [min_x, min_y]
    
    # Compute Z-order for each vertex
    z_orders = np.zeros(len(node_coords), dtype=np.uint64)
    for i, (x, y) in enumerate(normalized_node_coords):
        z_orders[i] = interleave_bits(x, y)
    
    return z_orders

def brio_ordering(
        node_coords: np.ndarray
) -> List[int]:
    """
    This function generates a BRIO ordering for the input node coordinates, which is a hierarchical 
    ordering technique that sorts points in successive rounds, applying a Z-order sorting within each 
    round. This ordering improves locality when inserting points in Delaunay triangulation algorithms.

    Parameters:
    - node_coords (np.ndarray): An Nx2 array of vertex coordinates `(x, y)`.

    Returns:
    - List[int]: List of vertex indices in BRIO order.
    """
    n = len(node_coords)
    max_round = int(np.floor(np.log2(n))) if n > 0 else 0
    rounds = defaultdict(list)
    
    # Assign points to rounds based on biased probability
    for idx in range(n):
        r = 0
        while r < max_round and random.random() < 0.5:
            r += 1
        rounds[r].append(idx)
    
    # Compute Z-order for all node coordinates
    z_orders = compute_z_order(node_coords)
    
    # Order within each round using Z-order
    ordered_node_coords = []
    for r in range(max_round + 1):
        round_node_coords = rounds[r]
        # Sort node coordinates in the round based on Z-order
        round_node_coords_sorted = sorted(round_node_coords, key=lambda idx: z_orders[idx])
        # Reverse order on even-numbered rounds to enhance locality
        if r % 2 == 0:
            round_node_coords_sorted = list(reversed(round_node_coords_sorted))
        ordered_node_coords.extend(round_node_coords_sorted)
    
    return ordered_node_coords

def walk_to_point(
    start_idx: int, 
    target_idx: int, 
    start_triangle_idx: int, 
    node_coords: List[Tuple[float, float]], 
    elem_nodes: List[Optional[Tuple[int, int, int]]], 
    node_elems: List[List[int]], 
    visualize: bool = False
) -> int:
    """
    The function begins at a starting triangle containing the point `start_idx` and traverses the triangulation using 
    orientation tests. The goal is to find the triangle containing the point `target_idx`. The algorithm leverages 
    local adjacency information to efficiently walk towards the target point.

    Reference:
    - "Straight Walk Algorithm" described in Roman Soukal's paper (http://graphics.zcu.cz/files/106_REP_2010_Soukal_Roman.pdf).

    Parameters:
    - start_idx (int): Index of the starting vertex.
    - target_idx (int): Index of the point to locate.
    - node_coords (List[Tuple[float, float]]): List of vertex coordinates `(x, y)`.
    - elem_nodes (List[Optional[Tuple[int, int, int]]]): List of triangles, where each triangle is represented as a tuple 
      of three vertex indices. A value of `None` indicates a deleted triangle.
    - node_elems (List[List[int]]): List of lists, where each sublist contains the indices of triangles connected 
      to a node at the corresponding index.
    - start_triangle_idx (int): Index of the triangle containing the starting vertex.
    - visualize (bool, optional): Boolean flag to enable or disable visualization of the walk steps. Default is False.

    Returns:
    - int: The index of the triangle that contains the target point.

    Raises:
    - ValueError: If no triangle is found containing the target point.
    """
    current_triangle_idx = start_triangle_idx  # Start with the triangle containing the start point
    step_number = 0  # Initialize step counter for the walk
    initilization_elem_nodes = [current_triangle_idx]  # Store the elem_nodes visited during initialization
    main_traversal_elem_nodes = []  # Store the elem_nodes visited during the main traversal

    if current_triangle_idx == -1:
        raise ValueError("Starting vertex is not part of any triangle in the triangulation.")

    current_triangle = elem_nodes[current_triangle_idx]  # Get the triangle containing the start vertex

    # Get the two other node_coords of the current triangle (excluding start_idx)
    other_node_coords = [v for v in current_triangle if v != start_idx]
    if len(other_node_coords) != 2:
        raise ValueError("Invalid triangle configuration; a triangle must have exactly 3 node_coords.")

    v1_idx, v2_idx = other_node_coords

    # Determine the orientation of the triangle formed by start_idx, v1_idx, and v2_idx
    orientation = orient(start_idx, v1_idx, v2_idx, node_coords)
    
    if orientation < 0:
        # If the triangle is counter-clockwise, assign the right and left node_coords
        r_idx = v1_idx
        l_idx = v2_idx
    else:
        # If the triangle is clockwise, swap the assignment
        r_idx = v2_idx
        l_idx = v1_idx

    step_number += 1  # Increment the step counter for initialization
    # if visualize:
    #     visualize_walk_step(node_coords, elem_nodes, current_triangle_idx, start_idx, target_idx, r_idx, l_idx,
    #                         step_number, "Initial setup", initilization_elem_nodes, main_traversal_elem_nodes)

    # Check if the target point is already inside the current triangle
    if in_triangle(r_idx, start_idx, l_idx, target_idx, node_coords):
        return current_triangle_idx

    # Initialization phase: rotate around the start vertex to align with the target point
    if orient(start_idx, target_idx, r_idx, node_coords) > 0:
        # While the orientation indicates the target is to the right of the current edge
        while orient(start_idx, target_idx, l_idx, node_coords) > 0:
            step_number += 1
            r_idx = l_idx  # Shift right

            neighbor_triangle_idx = get_neighbor_through_edge(current_triangle_idx, (start_idx, l_idx), node_elems)
            
            if neighbor_triangle_idx == -1:
                raise ValueError("Reached a boundary while traversing; point may be outside triangulation.")
            
            current_triangle_idx = neighbor_triangle_idx  # Move to the neighbor triangle
            initilization_elem_nodes.append(current_triangle_idx)
            current_triangle = elem_nodes[current_triangle_idx]
            l_idx = next(v for v in current_triangle if v != start_idx and v != r_idx)  # Update the left vertex
            if visualize:
                visualize_walk_step(node_coords, elem_nodes, current_triangle_idx, start_idx, target_idx, r_idx, l_idx, 
                                    step_number, "Rotating 1", initilization_elem_nodes, main_traversal_elem_nodes)
    else:
        # The opposite direction (left side)
        cond = True
        while cond:
            step_number += 1
            l_idx = r_idx  # Shift left
            neighbor_triangle_idx = get_neighbor_through_edge(current_triangle_idx, (start_idx, r_idx), node_elems)
            
            if neighbor_triangle_idx == -1:
                raise ValueError("Reached a boundary while traversing; point may be outside triangulation.")
            
            current_triangle_idx = neighbor_triangle_idx  # Move to the neighbor triangle
            initilization_elem_nodes.append(current_triangle_idx)
            current_triangle = elem_nodes[current_triangle_idx]
            r_idx = next(v for v in current_triangle if v != start_idx and v != l_idx)  # Update the right vertex
            
            cond = orient(start_idx, target_idx, r_idx, node_coords) <= 0  # Continue as long as the condition holds
            if visualize:
                visualize_walk_step(node_coords, elem_nodes, current_triangle_idx, start_idx, target_idx, r_idx, l_idx, 
                                    step_number, "Rotating 2", initilization_elem_nodes, main_traversal_elem_nodes)

    initilization_elem_nodes.pop()  # Remove the last triangle from initialization list
    main_traversal_elem_nodes = [current_triangle_idx]  # Start tracking the main traversal path

    # Switch left and right pointers for the main traversal
    l_idx, r_idx = r_idx, l_idx

    # Check if the target point is already inside the current triangle
    if in_triangle(r_idx, start_idx, l_idx, target_idx, node_coords):
        return current_triangle_idx
    
    # Main traversal phase: perform a straight walk through the triangulation
    while orient(target_idx, r_idx, l_idx, node_coords) <= 0:
        step_number += 1
        neighbor_triangle_idx = get_neighbor_through_edge(current_triangle_idx, (r_idx, l_idx), node_elems)
        if neighbor_triangle_idx == -1:
            # If no neighbor is found, check if the target point lies inside the current triangle
            if in_triangle(r_idx, start_idx, l_idx, target_idx, node_coords):
                if visualize:
                    visualize_walk_step(node_coords, elem_nodes, current_triangle_idx, start_idx, target_idx, r_idx, l_idx, 
                                        step_number, "Main path finding", initilization_elem_nodes, main_traversal_elem_nodes)
                return current_triangle_idx
            else:
                raise ValueError("Reached a boundary while traversing; point may be outside triangulation.")
        
        current_triangle_idx = neighbor_triangle_idx  # Move to the next triangle
        main_traversal_elem_nodes.append(current_triangle_idx)
        current_triangle = elem_nodes[current_triangle_idx]
        s_idx = next(v for v in current_triangle if v != r_idx and v != l_idx)  # Find the remaining vertex

        # Update the right or left pointer based on the orientation
        if orient(s_idx, start_idx, target_idx, node_coords) <= 0:
            r_idx = s_idx
        else:
            l_idx = s_idx

        if in_triangle(r_idx, start_idx, l_idx, target_idx, node_coords):
            if visualize:
                visualize_walk_to_point(node_coords, elem_nodes, current_triangle_idx, start_idx, target_idx, r_idx, l_idx, 
                            step_number, "Main path finding", initilization_elem_nodes, main_traversal_elem_nodes)

            return current_triangle_idx
        # if visualize:
        #     visualize_walk_step(node_coords, elem_nodes, current_triangle_idx, start_idx, target_idx, r_idx, l_idx, 
        #                         step_number, "Main path finding", initilization_elem_nodes, main_traversal_elem_nodes)

    
    if visualize:
        visualize_walk_to_point(node_coords, elem_nodes, current_triangle_idx, start_idx, target_idx, r_idx, l_idx, 
                            step_number, "Main path finding", initilization_elem_nodes, main_traversal_elem_nodes)


    return current_triangle_idx  # Return the index of the triangle containing the target point

