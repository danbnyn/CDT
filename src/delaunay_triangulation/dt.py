import numpy as np
from time import time
from typing import List, Tuple, Optional

from .utils import (
    log
    )

from .pre_process import (
    initialize_triangulation, 
    initialize_node_elems,
    initialize_node_nodes
    )

from .dt_utils import (
    find_bad_elem_nodes, 
    delete_bad_elem_nodes_and_collect_edges, 
    find_boundary_edges, 
    triangulate_cavity_with_new_point, 
    walk_to_point,
    brio_ordering
    )


def bowyer_watson(
    u_idx: int, 
    most_recent_idx: int, 
    triangle_most_recent_idx: int, 
    node_coords: List[Tuple[float, float]], 
    elem_nodes: List[Optional[Tuple[int, int, int]]], 
    node_elems: List[List[int]], 
    node_nodes: List[List[int]], 
    visualize: bool = False
) -> Tuple[int, int]:
    """
    This function inserts a new point into an existing Delaunay triangulation using the Bowyer-Watson algorithm. 
    The algorithm identifies and removes triangles whose circumcircles contain the new point and then re-triangulates 
    the cavity formed by the deleted triangles to maintain a Delaunay triangulation.

    Parameters:
    - u_idx (int): Index of the point to be inserted into the triangulation.
    - most_recent_idx (int): Index of the most recently inserted vertex.
    - triangle_most_recent_idx (int): Index of the triangle containing the most recent vertex.
    - node_coords (List[Tuple[float, float]]): List of coordinates of all nodes (vertices).
    - elem_nodes (List[Optional[Tuple[int, int, int]]]): List of existing triangles, where each triangle is represented 
      as a tuple of three vertex indices. A value of `None` indicates a deleted triangle.
    - node_elems (List[List[int]]): List of lists, where each sublist contains the indices of triangles connected 
      to a node at the corresponding index.
    - node_nodes (List[List[int]]): List of lists, where each sublist contains the indices of nodes connected to a node.
    - visualize (bool, optional): Whether to visualize the intermediate steps of the algorithm. Default is False.

    Returns:
    - Tuple[int, int]: A tuple containing the index of the new triangle created and the index of the inserted point.

    Raises:
    - ValueError: If the new triangle index is None.
    """

    # Step 1: Find one triangle whose open circumcircle contains u using walk-to-point, starting from the most recent vertex
    initial_bad = walk_to_point(
        most_recent_idx, 
        u_idx, 
        triangle_most_recent_idx, 
        node_coords, 
        elem_nodes, 
        node_elems, 
        visualize
    )

    # Step 2: Identify all the other triangles whose circumcircles contain u by a depth-first search in the triangulation
    bad_elem_nodes = find_bad_elem_nodes(
        initial_bad, 
        u_idx, 
        node_coords, 
        elem_nodes, 
        node_elems, 
        visualize
    )

    # Step 3: Delete bad triangles and collect cavity edges, and find the boundary edges of the cavity
    cavity_edges = delete_bad_elem_nodes_and_collect_edges(
        bad_elem_nodes, 
        elem_nodes, 
        node_elems, 
        node_nodes
    )
    boundary_edges = find_boundary_edges(cavity_edges)

    # Step 4: Re-triangulate the cavity with the new point
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

def delaunay_triangulation(
    cloud_node_coords: List[Tuple[float, float]], 
    visualize: bool = False, 
    verbose: int = 1
) -> Tuple[np.ndarray, List[Tuple[int, int, int]], List[List[int]], List[List[int]], Tuple[int, int]]:
    """
    This function performs Delaunay triangulation on a set of 2D points using the Bowyer-Watson algorithm. 
    It initializes the triangulation with a super-triangle that encompasses all points, and then iteratively inserts 
    points using the Bowyer-Watson algorithm. The function supports optional visualization and varying verbosity levels.

    Parameters:
    - cloud_node_coords (List[Tuple[float, float]]): List of 2D points (x, y) to triangulate.
    - visualize (bool, optional): Boolean indicating whether to visualize the triangulation after each insertion. Default is False.
    - verbose (int, optional): Integer indicating the verbosity level (0: silent, 1: basic, 2: detailed, 3: highly detailed). Default is 1.

    Returns:
    - Tuple[np.ndarray, List[Tuple[int, int, int]], List[List[int]], List[List[int]], Tuple[int, int]]:
        - node_coords (np.ndarray): NumPy array of node coordinates including the original point cloud and super-triangle nodes.
        - elem_nodes (List[Tuple[int, int, int]]): List of elements (triangles) representing the triangulated mesh.
        - node_elems (List[List[int]]): List of lists, where each sublist contains the indices of triangles connected to each vertex.
        - node_nodes (List[List[int]]): List of lists, where each sublist contains the indices of nodes connected to each vertex.
        - indices (Tuple[int, int]): A tuple containing the index of the last triangle inserted and the most recently added vertex index.
    """

    start_time = time()
    log(f"Starting Delaunay Triangulation", verbose, level=1)

    elem_nodes = []

    # Initialize the vertex to elem_nodes adjacency list
    num_initial_node_coords = len(cloud_node_coords) + 3  # Assuming super-triangle adds 3 nodes
    node_elems = initialize_node_elems(num_initial_node_coords)
    node_nodes = initialize_node_nodes(num_initial_node_coords)

    log(f"Initialized node_elems with {num_initial_node_coords} entries.", verbose, level=2)
    log(f"Initialized node_nodes with {num_initial_node_coords} entries.", verbose, level=2)

    # Step 1: Initialize triangulation with a super-triangle
    step_start = time()
    node_coords = initialize_triangulation(cloud_node_coords, elem_nodes, node_elems, node_nodes)
    step_time = time() - step_start
    log(f"Step 1: Initialized triangulation with super-triangle in {step_time:.4f} seconds.", verbose, level=1)

    # Super-triangle nodes are added at the end of the node_coords list, so the most recent vertex will be the last
    most_recent_idx = len(node_coords) - 1  # Index of the last vertex
    triangle_most_recent_idx = len(elem_nodes) - 1  # Index of the last triangle

    # Use the Brio ordering to sort the point cloud
    biased_random_ordering = brio_ordering(cloud_node_coords)
    log(f"Point cloud ordered using Brio ordering.", verbose, level=2)

    # Step 2: Insert each point using the Bowyer-Watson algorithm
    total_points = len(biased_random_ordering)
    log(f"Step 2: Beginning point insertion for {total_points} points.", verbose, level=1)

    for insertion_count, point_idx in enumerate(biased_random_ordering, 1):
        step_start = time()

        # The index of the new point is the current length of the node_coords list
        u_idx = len(node_coords)

        # Add the new point to the node_coords list
        new_point = np.array(cloud_node_coords[point_idx])
        node_coords = np.vstack((node_coords, new_point))
        log(f"Inserting point {insertion_count}/{total_points}: {new_point}", verbose, level=3)

        # Perform the Bowyer-Watson step to insert the new point into the triangulation
        try:
            triangle_most_recent_idx, most_recent_idx = bowyer_watson(
                u_idx, most_recent_idx, triangle_most_recent_idx, node_coords, elem_nodes, node_elems, node_nodes, visualize
            )
            step_time = time() - step_start
            log(f"Inserted point {insertion_count}/{total_points} in {step_time:.4f} seconds.", verbose, level=4)
            log(f"Total node_coords: {len(node_coords)} | Total elem_nodes: {len(elem_nodes)}", verbose, level=4)
        except Exception as e:
            log(f"Error inserting point {point_idx} ({new_point}): {e}", verbose, level=2)

    total_time = time() - start_time
    log(f"Step 2: Completed point insertion in {total_time:.4f} seconds.", verbose, level=1)
    log(f"Final triangulation has {len(node_coords)} node_coords and {len(elem_nodes)} elem_nodes.", verbose, level=1)

    return node_coords, elem_nodes, node_elems, node_nodes, (triangle_most_recent_idx, most_recent_idx)
