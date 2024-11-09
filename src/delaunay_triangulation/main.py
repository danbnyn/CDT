import numpy as np

from .pre_process import generate_cloud
from .dt import delaunay_triangulation
from .utils import generate_edges_from_points, convert_edges_to_ids
from .cdt import constrained_delaunay_triangulation
from .post_process import clean_mesh, convert_to_mesh_format, apply_rcm
from .visualize import plot_triangulation

def generate_mesh(polygon_outer, polygons_holes, min_distance_outer, verbose=1):

    # Generate points on boundary for refinement and interior points
    cloud_node_coords, outer_boundary_node_coords, hole_boundaries_node_coords, interior_node_coords = generate_cloud(polygon_outer, polygons_holes, min_distance_outer, verbose = verbose)

    # #reverse the order of the boundary points to be ccw
    outer_boundary_node_coords = outer_boundary_node_coords[::-1]
    hole_boundaries_node_coords = [hole[::-1] for hole in hole_boundaries_node_coords]

    # Step 1 : Perform the DT on the convex hull to generate the bulk of the mesh
    delaunay_node_coords, delaunay_elem_nodes, delaunay_node_elems, delaunay_node_nodes, (triangle_most_recent_idx, most_recent_idx) = delaunay_triangulation(cloud_node_coords, verbose=verbose) # the delaunay_node_coords has the super triangle delaunay_node_coords at the beginning

    # Step 2 : Perform the CDT on the boundary points to constrain the mesh and ensure that the futur boundary egdes are present
    outer_boundary_edges = generate_edges_from_points(outer_boundary_node_coords)
    hole_boundaries_edges = [generate_edges_from_points(hole_boundary) for hole_boundary in hole_boundaries_node_coords]

    outer_boundary_constrained_edges = convert_edges_to_ids(outer_boundary_edges, delaunay_node_coords)
    hole_boundaries_constrained_edges = [convert_edges_to_ids(hole_boundary_edges, delaunay_node_coords) for hole_boundary_edges in hole_boundaries_edges]
    hole_boundaries_constrained_edges_flattened = [item for sublist in hole_boundaries_constrained_edges for item in sublist]

    # Make a single list of all the constrained edges
    boundary_constrained_edges = outer_boundary_constrained_edges + hole_boundaries_constrained_edges_flattened

    delaunay_node_coords, delaunay_elem_nodes, delaunay_node_elems, delaunay_node_nodes = constrained_delaunay_triangulation(boundary_constrained_edges, delaunay_node_coords, delaunay_elem_nodes, delaunay_node_elems, delaunay_node_nodes, verbose=verbose)

    # Step 3 : Clean the mesh
    super_delaunay_node_coords = [0, 1, 2]
    delaunay_node_coords, delaunay_elem_nodes, delaunay_node_elems, delaunay_node_nodes = clean_mesh(delaunay_node_coords, delaunay_elem_nodes, delaunay_node_elems, delaunay_node_nodes, super_delaunay_node_coords, outer_boundary_constrained_edges, hole_boundaries_constrained_edges, verbose = verbose)

    plot_triangulation(delaunay_node_coords, delaunay_elem_nodes, title="Constrained Delaunay Triangulation based on Boundary Points")

    # # Step 4 : Convert mesh to data structure
    node_coords, numb_elems, elem2nodes, p_elem2nodes, node2elems, p_node2elems, node2nodes, p_node2nodes = convert_to_mesh_format(delaunay_node_coords, delaunay_elem_nodes, delaunay_node_elems, delaunay_node_nodes)

    # Step 5 : Apply RCM
    node_coords, elem2nodes, node2elems, p_node2elems, node2nodes, p_node2nodes = apply_rcm(node_coords, elem2nodes, p_elem2nodes, node2elems, p_node2elems, node2nodes, p_node2nodes)

    return  np.array(node_coords), np.array(elem2nodes), np.array(p_elem2nodes), np.array(node2elems), np.array(p_node2elems), np.array(node2nodes), np.array(p_node2nodes)

