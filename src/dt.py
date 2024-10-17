import numpy as np
import math
from time import time

from src.utils import (
    log
    )

from src.pre_process import (
    initialize_triangulation, 
    initialize_vertex_to_triangles
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

def bowyer_watson(u_idx, vertices, triangles, edge_to_triangle, most_recent_idx, triangle_most_recent_idx, vertex_to_triangles, visualize=False):
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
    initial_bad = walk_to_point(most_recent_idx, u_idx, vertices, triangles, edge_to_triangle, vertex_to_triangles ,triangle_most_recent_idx, visualize)

    # 2. Find all the others by a depth-first search in the triangulation.
    bad_triangles = find_bad_triangles(initial_bad, u_idx, vertices, triangles, edge_to_triangle , visualize)

    # 3. Delete bad triangles and collect cavity edges, and find the doundary edges of the cavity
    cavity_edges = delete_bad_triangles_and_collect_edges(bad_triangles, triangles, edge_to_triangle, vertex_to_triangles)
    boundary_edges = find_boundary_edges(cavity_edges)


    # 4. Re-triangulate the cavity with the new point
    new_triangle_idx = triangulate_cavity_with_new_point(boundary_edges, u_idx, triangles, edge_to_triangle, vertex_to_triangles)



    if new_triangle_idx is not None:
        return new_triangle_idx, u_idx
    else:
        raise ValueError("The new triangle index is None")



def delaunay_triangulation(point_cloud, visualize=False, verbose=1):
    """
    Performs Delaunay triangulation on the input point cloud.

    Parameters:
    - point_cloud: List of points (tuples or lists of (x, y)) to triangulate.
    - visualize: Boolean indicating whether to visualize the triangulation after each insertion.
    - verbose: Integer indicating the verbosity level (0: silent, 1: basic, 2: detailed, 3: highly detailed).

    Returns:
    - vertices: NumPy array of vertices including the original point cloud and super-triangle vertices.
    - triangles: List of triangles representing the triangulated mesh.
    - edge_to_triangle: Dictionary mapping edges to triangle indices.
    - vertex_to_triangles: List of lists, where each sublist contains triangle indices.
    - indices: Tuple containing (triangle_most_recent_idx, most_recent_idx).
    """
    
    start_time = time()
    log(f"Starting Delaunay Triangulation", verbose, level=1)
    
    triangles = []
    edge_to_triangle = {}
    
    # Initialize the vertex to triangles adjacency list
    num_initial_vertices = len(point_cloud) + 3  # Assuming super-triangle adds 3 vertices
    vertex_to_triangles = initialize_vertex_to_triangles(num_initial_vertices)
    log(f"Initialized vertex_to_triangles with {num_initial_vertices} entries.", verbose, level=2)
    
    # Step 1: Initialize triangulation with a super-triangle
    step_start = time()
    vertices = initialize_triangulation(point_cloud, triangles, edge_to_triangle, vertex_to_triangles)
    step_time = time() - step_start
    log(f"Step 1: Initialized triangulation with super-triangle in {step_time:.4f} seconds.", verbose, level=1)

    # Super-triangle vertices are added at the end of the vertices list, so the most recent vertex will be the last
    most_recent_idx = len(vertices) - 1  # Index of the last vertex
    triangle_most_recent_idx = len(triangles) - 1  # Index of the last triangle
    
    # Use the Brio ordering to sort the point cloud
    biaised_random_ordering = brio_ordering(point_cloud)
    log(f"Point cloud ordered using Brio ordering.", verbose, level=2)
    
    # Step 2: Insert each point using the Bowyer-Watson algorithm
    total_points = len(biaised_random_ordering)
    log(f"Step 2: Beginning point insertion for {total_points} points.", verbose, level=1)
    
    for insertion_count, point_idx in enumerate(biaised_random_ordering, 1):
        step_start = time()
        
        # The index of the new point is the current length of the vertices list
        u_idx = len(vertices)
        
        # Add the new point to the vertices list
        new_point = np.array(point_cloud[point_idx])
        vertices = np.vstack((vertices, new_point))
        log(f"Inserting point {insertion_count}/{total_points}: {new_point}", verbose, level=3)
        
        # Perform the Bowyer-Watson step to insert the new point into the triangulation
        try:
            triangle_most_recent_idx, most_recent_idx = bowyer_watson(
                u_idx, vertices, triangles, edge_to_triangle, most_recent_idx, triangle_most_recent_idx, vertex_to_triangles, visualize
            )
            step_time = time() - step_start
            log(f"Inserted point {insertion_count}/{total_points} in {step_time:.4f} seconds.", verbose, level=4)
            log(f"Total vertices: {len(vertices)} | Total triangles: {len(triangles)}", verbose, level=4)
        except Exception as e:
            log(f"Error inserting point {point_idx} ({new_point}): {e}", verbose, level=2)
    
    total_time = time() - start_time
    log(f"Step 2: Completed point insertion in {total_time:.4f} seconds.", verbose, level=1)
    log(f"Final triangulation has {len(vertices)} vertices and {len(triangles)} triangles.", verbose, level=1)
    
    return vertices, triangles, edge_to_triangle, vertex_to_triangles, (triangle_most_recent_idx, most_recent_idx)
