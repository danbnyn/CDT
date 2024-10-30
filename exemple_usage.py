from random import seed
import matplotlib.pyplot as plt

from src.delaunay_triangulation import (
    plot_triangulation, 
    plot_triangulation_with_points_ordered,
    plot_triangulation_with_points,
    plot_triangulation_with_node_nodes,
    plot_points,
    plot_adjancy_matrix,
    generate_cloud, 
    generate_mandelbrot_boundary, 
    remove_duplicate_points,
    initialize_triangulation,
    delaunay_triangulation,
    constrained_delaunay_triangulation, 
    CavityCDT, 
    convex_dt,
    generate_edges_from_points, 
    convert_edges_to_ids, 
    clean_mesh,
    convert_to_mesh_format,
    apply_rcm,
    build_adjacency_list
)

def main():

    seed(18)

    # Resolution for the generationof the fractal boundary 
    resolution = 200 
    max_iteration = 100

    # Input minimum distance between 2 points in the cloud of points
    min_distance = 0.02 

    verbose = 2

    # Not supporting holes for now

    # Generate a polygon boundary 
    boundary_node_coords = generate_mandelbrot_boundary(resolution, max_iteration, verbose=verbose) # Shape : len()

    # plot_points(boundary, title="Mandelbrot Boundary with resolution " + str(resolution) + " and max iteration " + str(max_iteration))

    # Generate points on boundary for refinement and interior points
    cloud_node_coords, boundary_node_coords, interior_node_coords = generate_cloud(boundary_node_coords, min_distance, verbose = verbose)
    # #reverse the order of the boundary points to be ccw
    boundary_node_coords = boundary_node_coords[::-1]

    plot_points(cloud_node_coords, title="Cloud Points based on Mandelbrot Boundary with resolution " + str(resolution) + " and min distance " + str(min_distance))

    # Step 1 : Perform the DT on the convex hull to generate the bulk of the mesh
    delaunay_node_coords, delaunay_elem_nodes, delaunay_node_elems, delaunay_node_nodes, (triangle_most_recent_idx, most_recent_idx) = delaunay_triangulation(cloud_node_coords, verbose=verbose) # the delaunay_node_coords has the super triangle delaunay_node_coords at the beginning

    # points = np.concatenate((delaunay_node_nodes[3539], [3539]))
    # points_coords = delaunay_node_coords[points]
    # print("plotting the triangulation with the node_nodes[v] and v")
    # plot_triangulation_with_points(delaunay_node_coords, delaunay_elem_nodes, points_coords)

    # delaunay_node_coords = delaunay_node_coords
    # elem_nodes = elem_nodes (homogene)

    # delaunay_dic_edge_triangle
    # (1,2) indice des noeuds qui composent l'arrete
    # {(1,2) : (5,8) } (5,8) indince des elem_nodes

    # delaunay_dic_edge_triangle[(1,2)] = (5,8)

    # delaunay_node_elems
    # [[1,2,4], [1,4,7]]

    # delaunay_node_elems[0] = [1,2,4]

    # plot_triangulation(delaunay_node_coords, delaunay_elem_nodes, title="Delaunay Triangulation based on Cloud Points")

    # Step 2 : Perform the CDT on the boundary points to constrain the mesh and ensure that the futur boundary egdes are present
    polygon_edges = generate_edges_from_points(boundary_node_coords)
    boundary_constrained_edges = convert_edges_to_ids(polygon_edges, delaunay_node_coords)


    # # Display the boundary points 
    # plot_triangulation_with_points(delaunay_node_coords, delaunay_elem_nodes, boundary_node_coords)

    # # Check the node_nodes of the boundary points
    # for v in boundary_nodes:
    #     plot_triangulation_with_node_nodes(delaunay_node_coords, delaunay_elem_nodes, v, delaunay_node_nodes)

    delaunay_node_coords, delaunay_elem_nodes, delaunay_node_elems, delaunay_node_nodes = constrained_delaunay_triangulation(boundary_constrained_edges, delaunay_node_coords, delaunay_elem_nodes, delaunay_node_elems, delaunay_node_nodes, verbose=verbose)

    plot_triangulation(delaunay_node_coords, delaunay_elem_nodes, title="Delaunay Triangulation based on Cloud Points")
    
    # Step 3 : Clean the mesh
    super_delaunay_node_coords = [0, 1, 2]
    delaunay_node_coords, delaunay_elem_nodes, delaunay_node_elems = clean_mesh(delaunay_node_coords, delaunay_elem_nodes, delaunay_node_elems, delaunay_node_nodes, super_delaunay_node_coords, boundary_constrained_edges, verbose = verbose)

    # # clean_dt

    plot_triangulation(delaunay_node_coords, delaunay_elem_nodes, title="Constrained Delaunay Triangulation based on Boundary Points")

    # # Step 4 : Convert mesh to data structure
    # node_coords, numb_elems, elem2nodes, p_elem2nodes = convert_to_mesh_format(delaunay_node_coords, delaunay_elem_nodes)

    # # Transform to np array ( a mettre dans convert mesh)
    # node_coords = np.array(node_coords)
    # elem2nodes = np.array(elem2nodes)
    # p_elem2nodes = np.array(p_elem2nodes)

    # plot_adjancy_matrix(node_coords, elem2nodes, p_elem2nodes, title="Adjacency Matrix of the Mesh")

    # # Step 5 : Apply RCM
    # new_node_coords, new_elem2nodes = apply_rcm(node_coords, elem2nodes, p_elem2nodes)

    # plot_adjancy_matrix(new_node_coords, new_elem2nodes, p_elem2nodes, title="Adjacency Matrix of the Mesh after RCM")

    plt.show()

if __name__ == "__main__":
    main()