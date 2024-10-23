import numpy as np
import math
from time import time

from src.utils import (
    log
    )

from src.pre_process import (
    initialize_triangulation, 
    initialize_delaunay_node_elems
    )

from src.dt_utils import (
    find_bad_triangles, 
    delete_bad_triangles_and_collect_edges, 
    find_boundary_edges, 
    triangulate_cavity_with_new_point, 
    walk_to_point,
    brio_ordering
    )

from src.visualize import (
    plot_triangulation,
    visualize_brio
    )

def bowyer_watson(u_idx, delaunay_node_coords, triangles, delaunay_dic_edge_triangle, most_recent_idx, triangle_most_recent_idx, delaunay_node_elems, visualize=False):
    """
    Implements the Bowyer-Watson algorithm to insert a point into the Delaunay triangulation.

    Parameters:
    - u_idx: Vertex index of the point to be inserted.
    - delaunay_node_coords: List of vertex coordinates.
    - triangles: List of existing triangles.
    - delaunay_dic_edge_triangle: Dictionary mapping edges to triangle indices.
    - most_recent_idx: Index of the most recently inserted vertex.
    - triangle_most_recent_idx: Index of the triangle containing the most recent vertex.
    - delaunay_node_elems: Dictionary mapping delaunay_node_coords to triangle indices.

    Returns:
    - None. Updates the triangles and delaunay_dic_edge_triangle structures.
    """

    # 1. Find one triangle whose open circumdisk contains u using walk-to-point, starting from the most recent vertex
    initial_bad = walk_to_point(most_recent_idx, u_idx, delaunay_node_coords, triangles, delaunay_dic_edge_triangle, delaunay_node_elems ,triangle_most_recent_idx, visualize)

    # 2. Find all the others by a depth-first search in the triangulation.
    bad_triangles = find_bad_triangles(initial_bad, u_idx, delaunay_node_coords, triangles, delaunay_dic_edge_triangle , visualize)

    # 3. Delete bad triangles and collect cavity edges, and find the doundary edges of the cavity
    cavity_edges = delete_bad_triangles_and_collect_edges(bad_triangles, triangles, delaunay_dic_edge_triangle, delaunay_node_elems)
    boundary_edges = find_boundary_edges(cavity_edges)


    # 4. Re-triangulate the cavity with the new point
    new_triangle_idx = triangulate_cavity_with_new_point(boundary_edges, u_idx, triangles, delaunay_dic_edge_triangle, delaunay_node_elems)



    if new_triangle_idx is not None:
        return new_triangle_idx, u_idx
    else:
        raise ValueError("The new triangle index is None")



def delaunay_triangulation(cloud_node_coords, visualize=False, verbose=1):
    """
    Performs Delaunay triangulation on the input point cloud.

    Parameters:
    - cloud_node_coords: List of points (tuples or lists of (x, y)) to triangulate.
    - visualize: Boolean indicating whether to visualize the triangulation after each insertion.
    - verbose: Integer indicating the verbosity level (0: silent, 1: basic, 2: detailed, 3: highly detailed).

    Returns:
    - delaunay_node_coords: NumPy array of delaunay_node_coords including the original point cloud and super-triangle delaunay_node_coords.
    - triangles: List of triangles representing the triangulated mesh.
    - delaunay_dic_edge_triangle: Dictionary mapping edges to triangle indices.
    - delaunay_node_elems: List of lists, where each sublist contains triangle indices.
    - indices: Tuple containing (triangle_most_recent_idx, most_recent_idx).
    """
    
    start_time = time()
    log(f"Starting Delaunay Triangulation", verbose, level=1)
    
    triangles = []
    delaunay_dic_edge_triangle = {}
    
    # Initialize the vertex to triangles adjacency list
    num_initial_delaunay_node_coords = len(cloud_node_coords) + 3  # Assuming super-triangle adds 3 delaunay_node_coords
    delaunay_node_elems = initialize_delaunay_node_elems(num_initial_delaunay_node_coords)
    log(f"Initialized delaunay_node_elems with {num_initial_delaunay_node_coords} entries.", verbose, level=2)
    
    # Step 1: Initialize triangulation with a super-triangle
    step_start = time()
    delaunay_node_coords = initialize_triangulation(cloud_node_coords, triangles, delaunay_dic_edge_triangle, delaunay_node_elems)
    step_time = time() - step_start
    log(f"Step 1: Initialized triangulation with super-triangle in {step_time:.4f} seconds.", verbose, level=1)

    # Super-triangle delaunay_node_coords are added at the end of the delaunay_node_coords list, so the most recent vertex will be the last
    most_recent_idx = len(delaunay_node_coords) - 1  # Index of the last vertex
    triangle_most_recent_idx = len(triangles) - 1  # Index of the last triangle
    
    # Use the Brio ordering to sort the point cloud
    biaised_random_ordering = brio_ordering(cloud_node_coords)
    log(f"Point cloud ordered using Brio ordering.", verbose, level=2)
    
    # Step 2: Insert each point using the Bowyer-Watson algorithm
    total_points = len(biaised_random_ordering)
    log(f"Step 2: Beginning point insertion for {total_points} points.", verbose, level=1)
    
    for insertion_count, point_idx in enumerate(biaised_random_ordering, 1):
        step_start = time()
        
        # The index of the new point is the current length of the delaunay_node_coords list
        u_idx = len(delaunay_node_coords)
        
        # Add the new point to the delaunay_node_coords list
        new_point = np.array(cloud_node_coords[point_idx])
        delaunay_node_coords = np.vstack((delaunay_node_coords, new_point))
        log(f"Inserting point {insertion_count}/{total_points}: {new_point}", verbose, level=3)
        
        # Perform the Bowyer-Watson step to insert the new point into the triangulation
        try:
            triangle_most_recent_idx, most_recent_idx = bowyer_watson(
                u_idx, delaunay_node_coords, triangles, delaunay_dic_edge_triangle, most_recent_idx, triangle_most_recent_idx, delaunay_node_elems, visualize
            )
            step_time = time() - step_start
            log(f"Inserted point {insertion_count}/{total_points} in {step_time:.4f} seconds.", verbose, level=4)
            log(f"Total delaunay_node_coords: {len(delaunay_node_coords)} | Total triangles: {len(triangles)}", verbose, level=4)
        except Exception as e:
            log(f"Error inserting point {point_idx} ({new_point}): {e}", verbose, level=2)
    
    total_time = time() - start_time
    log(f"Step 2: Completed point insertion in {total_time:.4f} seconds.", verbose, level=1)
    log(f"Final triangulation has {len(delaunay_node_coords)} delaunay_node_coords and {len(triangles)} triangles.", verbose, level=1)
    
    return delaunay_node_coords, triangles, delaunay_dic_edge_triangle, delaunay_node_elems, (triangle_most_recent_idx, most_recent_idx)
