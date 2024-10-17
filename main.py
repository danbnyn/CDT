from random import seed
from time import time
import numpy as np
import matplotlib.pyplot as plt

from src.visualize import (
    plot_triangulation, 
    plot_triangulation_with_points_ordered,
    plot_triangulation_with_points,
    plot_points,
    plot_adjancy_matrix
    )

from src.pre_process import (
    generate_cloud, 
    generate_mandelbrot_boundary, 
    remove_duplicate_points,
    initialize_triangulation
    )

from src.dt import delaunay_triangulation
 
from src.cdt import (
    constrained_delaunay_triangulation, 
    CavityCDT, 
    convex_dt
    )

from src.utils import (
    generate_edges_from_points, 
    convert_edges_to_ids, 
    convert_points_to_ids
    )

from src.post_process import (
    clean_mesh,
    convert_to_mesh_format,
    apply_rcm,
    build_adjacency_list
    )

def main():

    seed(18)

    resolution = 200
    max_iteration = 100
    min_distance = 0.02

    verbose = 2


    # Generate a cloud of points and a boundary
    boundary = generate_mandelbrot_boundary(resolution, max_iteration, verbose=verbose)

    # plot_points(boundary, title="Mandelbrot Boundary with resolution " + str(resolution) + " and max iteration " + str(max_iteration))

    cloud_points, boundary_points, interior_points = generate_cloud(boundary, min_distance, verbose = verbose)
    # #reverse the order of the boundary points to be ccw
    boundary_points = boundary_points[::-1]

    plot_points(cloud_points, title="Cloud Points based on Mandelbrot Boundary with resolution " + str(resolution) + " and min distance " + str(min_distance))

    # Step 1 : Perform the DT on the convex hull to generate the bulk of the mesh
    vertices, triangles, edge_to_triangle, vertex_to_triangles, (triangle_most_recent_idx, most_recent_idx) = delaunay_triangulation(cloud_points, verbose=verbose) # the vertices has the super triangle vertices at the beginning

    plot_triangulation(vertices, triangles, title="Delaunay Triangulation based on Cloud Points")

    # Step 2 : Perform the CDT on the boundary points to constrain the mesh and ensure that the futur boundary egdes are present
    polygon_edges = generate_edges_from_points(boundary_points)
    boundary_constrained_edges = convert_edges_to_ids(polygon_edges, vertices)


    vertices, triangles, edge_to_triangle, vertex_to_triangles = constrained_delaunay_triangulation(boundary_constrained_edges, vertices, triangles, edge_to_triangle, vertex_to_triangles, visualize=False, verbose =verbose)


    # Step 3 : Clean the mesh
    super_vertices = [0, 1, 2]
    vertices, new_triangles, edge_to_triangle, vertex_to_triangles = clean_mesh(vertices, triangles, edge_to_triangle, super_vertices, vertex_to_triangles, boundary_constrained_edges, verbose = verbose)

    plot_triangulation(vertices, new_triangles, title="Constrained Delaunay Triangulation based on Boundary Points")

    # Step 4 : Convert mesh to data structure
    node_coord, numb_elem, elem2node, p_elem2node = convert_to_mesh_format(vertices, new_triangles)

    # Transform to np array
    node_coord = np.array(node_coord)
    elem2node = np.array(elem2node)
    p_elem2node = np.array(p_elem2node)

    plot_adjancy_matrix(node_coord, elem2node, p_elem2node, title="Adjacency Matrix of the Mesh")

    # Step 5 : Apply RCM
    new_node_coord, new_elem2node = apply_rcm(node_coord, elem2node, p_elem2node)

    plot_adjancy_matrix(new_node_coord, new_elem2node, p_elem2node, title="Adjacency Matrix of the Mesh after RCM")

    plt.show()

if __name__ == "__main__":
    main()