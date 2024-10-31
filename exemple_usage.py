from random import seed
import matplotlib.pyplot as plt
import numpy as np

from src.delaunay_triangulation import (
    generate_mesh,
    generate_mandelbrot_boundary
)



def main():
    resolution = 100
    min_distance = 0.02
    max_iterations = 50
    verbose = 1

    boundary_polygon = generate_mandelbrot_boundary(resolution=resolution, max_iterations=max_iterations, verbose=1)

    # simple non convex hole
    holes_boundary = [[ [-0.5,-0.25], [0.1,-0.25], [0.1,0.25], [-0.5,0.25], [-0.25,0] ]]

    
    node_coords, elem2nodes, node2elems, node2nodes, p_node2nodes = generate_mesh(boundary_polygon, holes_boundary, min_distance, verbose)


if __name__ == "__main__":
    main()