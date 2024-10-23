import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from src.predicates import get_circumcircle 
import networkx as nx
from collections import defaultdict

import pickle


# Visualization
def plot_triangulation(delaunay_node_coords, triangles, title='Delaunay Triangulation'):
    """
    Plots the triangulation using matplotlib.
    """
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # Plot edges of the triangles
    for tri in triangles:
        if tri is not None:
            u, v, w = tri
            triangle = [delaunay_node_coords[u], delaunay_node_coords[v], delaunay_node_coords[w], delaunay_node_coords[u]]
            xs, ys = zip(*triangle)
            ax.plot(xs, ys, 'k-')
    
    # # Plot delaunay_node_coords
    # xs, ys = zip(*delaunay_node_coords)
    # ax.plot(xs, ys, 'ro')
    
    ax.set_aspect('equal')
    plt.title(title)

    # # Save the figure using pickle
    # pickle.dump((fig,ax), open("myplot.pickle", "wb"))

    plt.show()

def visualize_walk_step(delaunay_node_coords, triangles, current_triangle_idx, q_idx, p_idx, r_idx, l_idx, step_number, action, initilization_triangles, main_traversal_triangles):
    plt.figure(figsize=(14, 6))

    # Plot all triangles
    for i, triangle in enumerate(triangles):
        if triangle is None:
            continue
        else:
            tri_x = [delaunay_node_coords[v][0] for v in triangle]
            tri_y = [delaunay_node_coords[v][1] for v in triangle]
            if i in initilization_triangles:
                plt.fill(tri_x, tri_y, color='lightcoral')
            elif i in main_traversal_triangles:
                plt.fill(tri_x, tri_y, color='cornflowerblue')
            else:
                plt.fill(tri_x, tri_y, 'ghostwhite')
            plt.plot(tri_x + [tri_x[0]], tri_y + [tri_y[0]], 'k-')
    
    # # Highlight the current triangle
    # if current_triangle_idx is not None:
    #     current_triangle = triangles[current_triangle_idx]
    #     tri_x = [delaunay_node_coords[v][0] for v in current_triangle]
    #     tri_y = [delaunay_node_coords[v][1] for v in current_triangle]
    #     plt.fill(tri_x, tri_y, 'yellow', alpha=0.3)
    
    # # Plot all delaunay_node_coords
    # x, y = zip(*delaunay_node_coords)
    # plt.scatter(x, y, c='blue')
    
    # Highlight special points
    plt.scatter(*delaunay_node_coords[q_idx], c='red', s=100, label='q (start)')
    plt.scatter(*delaunay_node_coords[p_idx], c='green', s=100, label='p (target)')
    # plt.scatter(*delaunay_node_coords[r_idx], c='purple', s=100, label='r')
    # plt.scatter(*delaunay_node_coords[l_idx], c='orange', s=100, label='l')
    
    # # Add labels
    # for i, (x, y) in enumerate(delaunay_node_coords):
    #     plt.annotate(f'{i}', (x, y), xytext=(5, 5), textcoords='offset points')
    
    # # Draw the oriented ray from q to p
    # q_x, q_y = delaunay_node_coords[q_idx]
    # p_x, p_y = delaunay_node_coords[p_idx]
    # dx, dy = p_x - q_x, p_y - q_y
    # plt.arrow(q_x, q_y, dx, dy, head_width=0.05, head_length=0.1, fc='k', ec='k')
    
    plt.legend()
    plt.title(f'Step {step_number}: {action}\nCurrent Triangle: {current_triangle_idx}')
    plt.axis('equal')
    plt.show()

def plot_triangulation_with_cdt_cavity(delaunay_node_coords, triangles, cavity):
    """
    Plots the triangulation using matplotlib and highlights the CDT cavity.
    """
    plt.figure(figsize=(15, 7))
    fig, ax = plt.subplots()
    
    # Plot edges of the triangles
    for tri in triangles:
        if tri is not None:
            u, v, w = tri
            triangle = [delaunay_node_coords[u], delaunay_node_coords[v], delaunay_node_coords[w], delaunay_node_coords[u]]
            xs, ys = zip(*triangle)
            ax.plot(xs, ys, 'k-')
    
    # Plot cavity points
    cavity_points = [delaunay_node_coords[idx] for idx in cavity]
    xs, ys = zip(*cavity_points)
    ax.plot(xs, ys, 'b-')

    ax.set_aspect('equal')
    plt.title('Delaunay Triangulation with CDT Cavity')
    plt.show()

def plot_triangulation_with_points(delaunay_node_coords, triangles, points):
    """
    Plots the triangulation using matplotlib and highlights a specific point.
    """
    fig, ax = plt.subplots()
    
    # Plot edges of the triangles
    for tri in triangles:
        if tri is not None:
            u, v, w = tri
            triangle = [delaunay_node_coords[u], delaunay_node_coords[v], delaunay_node_coords[w], delaunay_node_coords[u]]
            xs, ys = zip(*triangle)
            ax.plot(xs, ys, 'k-')
    
    # # Plot delaunay_node_coords
    # xs, ys = zip(*delaunay_node_coords)
    # ax.plot(xs, ys, 'ro')
    
    # Highlight the specific points
    for point in points:
        ax.plot(point[0], point[1], 'bo', markersize=10)
        
    ax.set_aspect('equal')
    plt.title('Delaunay Triangulation with New Point')
    plt.show()

def plot_points_ordered(points):
    """
    Plots a list of points in order. Annotate them with the order.
    """
    plt.figure(figsize=(6, 6))
    for i, point in enumerate(points):
        plt.plot(point[0], point[1], 'bo', markersize=10)
        plt.annotate(f'{i}', (point[0], point[1]), xytext=(5, 5), textcoords='offset points')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Ordered Points')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.show()

def plot_triangulation_with_points_ordered(delaunay_node_coords, triangles, points):
    """
    Plots the triangulation using matplotlib and highlights a specific point.
    """
    plt.figure(figsize=(15, 7))
    fig, ax = plt.subplots()
    
    # Plot edges of the triangles
    for tri in triangles:
        if tri is not None:
            u, v, w = tri
            triangle = [delaunay_node_coords[u], delaunay_node_coords[v], delaunay_node_coords[w], delaunay_node_coords[u]]
            xs, ys = zip(*triangle)
            ax.plot(xs, ys, 'k-')
    
    # Plot delaunay_node_coords
    xs, ys = zip(*delaunay_node_coords)
    ax.plot(xs, ys, 'ro')
    
    # Highlight the specific points in order
    for i, point in enumerate(points):
        ax.plot(point[0], point[1], 'bo', markersize=10)
        ax.annotate(f'{i}', (point[0], point[1]), xytext=(5, 5), textcoords='offset points')
    
    ax.set_aspect('equal')
    plt.title('Delaunay Triangulation with Points')
    plt.show()

def visualize_intersecting_triangle(delaunay_node_coords, triangles, bad_triangles, start_idx, end_idx):

    # Plot all triangles
    for i, triangle in enumerate(triangles):
        if triangle is not None:
            a, b, c = triangle
            tri_x = [delaunay_node_coords[v][0] for v in [a, b, c, a]]
            tri_y = [delaunay_node_coords[v][1] for v in [a, b, c, a]]
            
            # Highlight bad triangles in red
            if i in bad_triangles:
                plt.fill(tri_x, tri_y, 'red', alpha=0.3)
                plt.plot(tri_x, tri_y, 'k-')

            else:
                plt.fill(tri_x, tri_y, 'lightgray', alpha=0.3)
                plt.plot(tri_x, tri_y, 'k-')

    # Highlight the start and end points
    plt.scatter(*delaunay_node_coords[start_idx], c='red', s=100, label='Start point')
    plt.scatter(*delaunay_node_coords[end_idx], c='red', s=100, label='End point')

    # Draw the line from start to end
    plt.plot([delaunay_node_coords[start_idx][0], delaunay_node_coords[end_idx][0]], [delaunay_node_coords[start_idx][1], delaunay_node_coords[end_idx][1]], 'k-', lw=2, color='red')

    plt.legend()
    plt.title('Finding Intersecting Triangle')
    plt.axis('equal')
    plt.show()

def visualize_bad_triangles(delaunay_node_coords, triangles, bad_triangles, u_idx, step_number):

    # Plot all triangles
    for i, triangle in enumerate(triangles):
        if triangle is not None:
            a, b, c = triangle
            tri_x = [delaunay_node_coords[v][0] for v in [a, b, c, a]]
            tri_y = [delaunay_node_coords[v][1] for v in [a, b, c, a]]
            
            # Highlight bad triangles in red
            if i in bad_triangles:
                plt.fill(tri_x, tri_y, 'red', alpha=0.3)
                plt.plot(tri_x, tri_y, 'k-')

                # Draw circumcircle for bad triangles
                center, radius = get_circumcircle(delaunay_node_coords[a], delaunay_node_coords[b], delaunay_node_coords[c])
                circle = Circle(center, radius, fill=False, linestyle='--', color='blue')
                plt.gca().add_artist(circle)
            else:
                plt.fill(tri_x, tri_y, 'lightgray', alpha=0.3)
                plt.plot(tri_x, tri_y, 'k-')

    # Highlight the new point u
    plt.scatter(*delaunay_node_coords[u_idx], c='green', s=100, label='New point u')

    plt.legend()
    plt.title(f'Step {step_number}: Displaying Bad Triangles and Circumcircles\nBad Triangles: {bad_triangles}')
    plt.axis('equal')
    plt.show()

def visualize_bad_triangles_step(delaunay_node_coords, triangles, bad_triangles, current_triangle, u_idx, step_number):

    # Plot all triangles
    for i, triangle in enumerate(triangles):
        if triangle is not None:
            a, b, c = triangle
            tri_x = [delaunay_node_coords[v][0] for v in [a, b, c, a]]
            tri_y = [delaunay_node_coords[v][1] for v in [a, b, c, a]]
            if i in bad_triangles:
                plt.fill(tri_x, tri_y, 'red', alpha=0.3)
            else:
                plt.fill(tri_x, tri_y, 'lightgray', alpha=0.3)
            plt.plot(tri_x, tri_y, 'k-')

    # Highlight the current triangle
    if current_triangle is not None:
        a, b, c = triangles[current_triangle]
        tri_x = [delaunay_node_coords[v][0] for v in [a, b, c, a]]
        tri_y = [delaunay_node_coords[v][1] for v in [a, b, c, a]]
        plt.fill(tri_x, tri_y, 'yellow', alpha=0.5)

        # Draw circumcircle
        center, radius = get_circumcircle(delaunay_node_coords[a], delaunay_node_coords[b], delaunay_node_coords[c])
        circle = Circle(center, radius, fill=False, linestyle='--', color='blue')
        plt.gca().add_artist(circle)

    # Plot all delaunay_node_coords
    x, y = zip(*delaunay_node_coords)
    plt.scatter(x, y, c='blue', s=50)

    # Highlight the new point u
    plt.scatter(*delaunay_node_coords[u_idx], c='green', s=100, label='New point u')

    # Add labels to delaunay_node_coords
    for i, (x, y) in enumerate(delaunay_node_coords):
        plt.annotate(f'{i}', (x, y), xytext=(5, 5), textcoords='offset points')

    plt.legend()
    plt.title(f'Step {step_number}: Finding Bad Triangles\nCurrent Triangle: {current_triangle}, Bad Triangles: {bad_triangles}')
    plt.axis('equal')
    plt.show()

def plot_edges_from_idx(delaunay_node_coords, edges):
    """
    Plots the edges of a polygon given the vertex indices.
    """
    plt.figure(figsize=(6, 6))
    for edge in edges:
        u, v = edge
        plt.plot([delaunay_node_coords[u][0], delaunay_node_coords[v][0]], [delaunay_node_coords[u][1], delaunay_node_coords[v][1]], 'b-')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Polygon Edges')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.show()

def plot_points(delaunay_node_coords, point_size=0.5, point_color='blue', alpha=1.0, title='Points'):
    """
    Plots a list of points with adjustable size, color, and transparency.
    
    Args:
        delaunay_node_coords (list of tuples): List of (x, y) coordinates to plot.
        point_size (int, optional): Size of the points. Default is 5.
        point_color (str, optional): Color of the points. Default is 'blue'.
        alpha (float, optional): Transparency level of the points. Default is 1.0 (opaque).
    """
    plt.figure(figsize=(6, 6))
    x, y = zip(*delaunay_node_coords)
    plt.scatter(x, y, s=point_size, c=point_color, alpha=alpha)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.show()

def plot_edges_from_coords(delaunay_node_coords, edges):
    """
    Plots the edges of a polygon given the vertex coordinates.
    """
    plt.figure(figsize=(6, 6))
    for edge in edges:
        (x1, y1), (x2, y2) = edge
        plt.plot([x1, x2], [y1, y2], 'b-')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Polygon Edges')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.show()


def visualize_brio(biaised_random_ordering_idx, cloud_node_coords):
    """
    Visualizes the BRIO (Biased Random Insertion Order) process.
    
    Parameters:
    - biaised_random_ordering_idx: List of indices representing the BRIO order.
    - cloud_node_coords: List of points (tuples of (x, y)) to visualize.
    """
    
    # Ensure that the point cloud is a NumPy array for easier indexing
    cloud_node_coords = np.array(cloud_node_coords)

    # Create a figure for the visualization
    plt.figure(figsize=(8, 8))
    plt.title("BRIO Visualization", fontsize=16)
    
    # Plot the points step-by-step according to the BRIO order
    for i, idx in enumerate(biaised_random_ordering_idx):
        # Get the current point to plot
        current_point = cloud_node_coords[idx]
        
        # Plot all points added so far
        plt.scatter(cloud_node_coords[biaised_random_ordering_idx[:i+1], 0], 
                    cloud_node_coords[biaised_random_ordering_idx[:i+1], 1], 
                    c='blue', label=f'Step {i+1}' if i == 0 else "")
        
        # Highlight the current point being added
        plt.scatter(current_point[0], current_point[1], c='red', s=100, label="Current Point")
        
        # Label the current point index
        plt.text(current_point[0] + 0.02, current_point[1] + 0.02, f'{idx}', fontsize=12, color='red')
        
        # Set labels and grid
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        
        # Add a pause to simulate the step-by-step process
        plt.pause(0.5)
        
    # Display the final result
    plt.scatter(cloud_node_coords[:, 0], cloud_node_coords[:, 1], c='blue')
    plt.show()


def visualize_walk_to_point(delaunay_node_coords, triangles, current_triangle_idx, q_idx, p_idx, r_idx, l_idx, step_number, action, initilization_triangles, main_traversal_triangles):
    # Plot all triangles
    for i, triangle in enumerate(triangles):
        if triangle is None:
            continue
        else:
            tri_x = [delaunay_node_coords[v][0] for v in triangle]
            tri_y = [delaunay_node_coords[v][1] for v in triangle]
            if i in initilization_triangles:
                plt.fill(tri_x, tri_y, color='lightcoral')
            elif i in main_traversal_triangles:
                plt.fill(tri_x, tri_y, color='cornflowerblue')
            else:
                plt.fill(tri_x, tri_y, 'ghostwhite')
            plt.plot(tri_x + [tri_x[0]], tri_y + [tri_y[0]], 'k-')

    # Highlight special points
    plt.scatter(*delaunay_node_coords[q_idx], c='red', s=100, label='q (start)')
    plt.scatter(*delaunay_node_coords[p_idx], c='green', s=100, label='p (target)')

    # Draw the ray from q to p with an arrowhead that stops exactly at point p
    q_x, q_y = delaunay_node_coords[q_idx]
    p_x, p_y = delaunay_node_coords[p_idx]

    # Plot the line from q to p
    plt.plot([q_x, p_x], [q_y, p_y], 'k-', lw=2, color='green')

    # Add the arrowhead at point p, in green color
    arrow_dx, arrow_dy = p_x - q_x, p_y - q_y
    plt.arrow(q_x, q_y, arrow_dx, arrow_dy, head_width=0.03, head_length=0.03, fc='green', ec='green', length_includes_head=True)

    plt.legend()
    plt.title("Illustration of Walk-to-Point Algorithm")
    plt.axis('equal')
    plt.show()


def plot_adjancy_matrix(node_coord, elem2node, p_elem2node, title="Adjacency Matrix"):

    def build_adjacency_list(elem2node, p_elem2node):
        """Build adjacency list from elem2node and p_elem2node."""
        adjacency = defaultdict(set)
        for elem_idx in range(len(p_elem2node) - 1):
            start = p_elem2node[elem_idx]
            end = p_elem2node[elem_idx + 1]
            triangle = elem2node[start:end]
            for i in range(3):
                for j in range(i + 1, 3):
                    adjacency[triangle[i]].add(triangle[j])
                    adjacency[triangle[j]].add(triangle[i])
        return adjacency

    adjacency = build_adjacency_list(elem2node, p_elem2node)
    n_nodes = len(node_coord)
    adjacency_matrix = np.zeros((n_nodes, n_nodes))
    for node, neighbors in adjacency.items():
        for neighbor in neighbors:
            adjacency_matrix[node, neighbor] = 1
    plt.matshow(adjacency_matrix)
    plt.title(title)
    plt.show()

def plot_triangulation_and_edge(delaunay_node_coords, triangles, edge, title="Triangulation with Edge"):
    """
    Plots the triangulation using matplotlib and highlights a specific edge.
    """
    fig, ax = plt.subplots()
    
    # Plot edges of the triangles
    for tri in triangles:
        if tri is not None:
            u, v, w = tri
            triangle = [delaunay_node_coords[u], delaunay_node_coords[v], delaunay_node_coords[w], delaunay_node_coords[u]]
            xs, ys = zip(*triangle)
            ax.plot(xs, ys, 'k-')
    
    # Highlight the specific edge
    u, v = edge
    edge_x = [delaunay_node_coords[u][0], delaunay_node_coords[v][0]]
    edge_y = [delaunay_node_coords[u][1], delaunay_node_coords[v][1]]
    ax.plot(edge_x, edge_y, 'r-', linewidth=2)
    
    ax.set_aspect('equal')
    plt.title(title)
    plt.show()

def plot_triangulation_with_points_and_marked_triangles(delaunay_node_coords, triangles, points, marked_triangles):
    """
    Plots the triangulation using matplotlib and highlights a specific point and triangles.
    """
    fig, ax = plt.subplots()
    
    # Plot edges of the triangles
    for tri in triangles:
        if tri is not None:
            u, v, w = tri
            triangle = [delaunay_node_coords[u], delaunay_node_coords[v], delaunay_node_coords[w], delaunay_node_coords[u]]
            xs, ys = zip(*triangle)
            ax.plot(xs, ys, 'k-')
    
    # Plot delaunay_node_coords
    xs, ys = zip(*delaunay_node_coords)
    ax.plot(xs, ys, 'ro')
    
    # Highlight the specific points
    for point in points:
        ax.plot(point[0], point[1], 'bo', markersize=10)
    
    # Highlight the marked triangles
    for tri_idx in marked_triangles:
        u, v, w = triangles[tri_idx]
        triangle = [delaunay_node_coords[u], delaunay_node_coords[v], delaunay_node_coords[w], delaunay_node_coords[u]]
        xs, ys = zip(*triangle)
        ax.fill(xs, ys, 'r', alpha=0.3)
    
    ax.set_aspect('equal')
    plt.title('Delaunay Triangulation with Points and Marked Triangles')
    plt.show()