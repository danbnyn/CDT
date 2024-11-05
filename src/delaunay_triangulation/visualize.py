import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from collections import defaultdict
import pickle

from .predicates import get_circumcircle 

# Visualization
def plot_triangulation_with_elem(node_coords, elem_nodes, elem ,title='Delaunay Triangulation with Node Nodes'):
    """
    Plots the triangulation using matplotlib and highlights a specific elem.
    """
    fig, ax = plt.subplots()
    
    # Plot edges of the elem_nodes
    for tri in elem_nodes:
        if tri is not None:
            u, v, w = tri
            triangle = [node_coords[u], node_coords[v], node_coords[w], node_coords[u]]
            xs, ys = zip(*triangle)
            ax.plot(xs, ys, 'k-')
    
    # Highlight the specific elem
    u, v, w = elem_nodes[elem]
    tri_x = [node_coords[u][0], node_coords[v][0], node_coords[w][0], node_coords[u][0]]
    tri_y = [node_coords[u][1], node_coords[v][1], node_coords[w][1], node_coords[u][1]]
    ax.fill(tri_x, tri_y, 'r', alpha=0.3)
    
    ax.set_aspect('equal')
    plt.title(title)
    plt.show()

def plot_triangulation(node_coords, elem_nodes, title='Delaunay Triangulation'):
    """
    Plots the triangulation using matplotlib.
    """
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # Plot edges of the elem_nodes
    for tri in elem_nodes:
        if tri is not None:
            u, v, w = tri
            triangle = [node_coords[u], node_coords[v], node_coords[w], node_coords[u]]
            xs, ys = zip(*triangle)
            ax.plot(xs, ys, 'k-')
    
    # # Plot node_coords
    # xs, ys = zip(*node_coords)
    # ax.plot(xs, ys, 'ro')
    
    ax.set_aspect('equal')
    plt.title(title)

    # # Save the figure using pickle
    # pickle.dump((fig,ax), open("myplot.pickle", "wb"))

    plt.show()

def visualize_walk_step(node_coords, elem_nodes, current_triangle_idx, q_idx, p_idx, r_idx, l_idx, step_number, action, initilization_elem_nodes, main_traversal_elem_nodes):
    plt.figure(figsize=(14, 6))

    # Plot all elem_nodes
    for i, triangle in enumerate(elem_nodes):
        if triangle is None:
            continue
        else:
            tri_x = [node_coords[v][0] for v in triangle]
            tri_y = [node_coords[v][1] for v in triangle]
            if i in initilization_elem_nodes:
                plt.fill(tri_x, tri_y, color='lightcoral')
            elif i in main_traversal_elem_nodes:
                plt.fill(tri_x, tri_y, color='cornflowerblue')
            else:
                plt.fill(tri_x, tri_y, 'ghostwhite')
            plt.plot(tri_x + [tri_x[0]], tri_y + [tri_y[0]], 'k-')
    
    # # Highlight the current triangle
    # if current_triangle_idx is not None:
    #     current_triangle = elem_nodes[current_triangle_idx]
    #     tri_x = [node_coords[v][0] for v in current_triangle]
    #     tri_y = [node_coords[v][1] for v in current_triangle]
    #     plt.fill(tri_x, tri_y, 'yellow', alpha=0.3)
    
    # # Plot all node_coords
    # x, y = zip(*node_coords)
    # plt.scatter(x, y, c='blue')
    
    # Highlight special points
    plt.scatter(*node_coords[q_idx], c='red', s=100, label='q (start)')
    plt.scatter(*node_coords[p_idx], c='green', s=100, label='p (target)')
    # plt.scatter(*node_coords[r_idx], c='purple', s=100, label='r')
    # plt.scatter(*node_coords[l_idx], c='orange', s=100, label='l')
    
    # # Add labels
    # for i, (x, y) in enumerate(node_coords):
    #     plt.annotate(f'{i}', (x, y), xytext=(5, 5), textcoords='offset points')
    
    # # Draw the oriented ray from q to p
    # q_x, q_y = node_coords[q_idx]
    # p_x, p_y = node_coords[p_idx]
    # dx, dy = p_x - q_x, p_y - q_y
    # plt.arrow(q_x, q_y, dx, dy, head_width=0.05, head_length=0.1, fc='k', ec='k')
    
    plt.legend()
    plt.title(f'Step {step_number}: {action}\nCurrent Triangle: {current_triangle_idx}')
    plt.axis('equal')
    plt.show()

def plot_triangulation_with_cdt_cavity(node_coords, elem_nodes, cavity):
    """
    Plots the triangulation using matplotlib and highlights the CDT cavity.
    """
    plt.figure(figsize=(15, 7))
    fig, ax = plt.subplots()
    
    # Plot edges of the elem_nodes
    for tri in elem_nodes:
        if tri is not None:
            u, v, w = tri
            triangle = [node_coords[u], node_coords[v], node_coords[w], node_coords[u]]
            xs, ys = zip(*triangle)
            ax.plot(xs, ys, 'k-')
    
    # Plot cavity points
    cavity_points = [node_coords[idx] for idx in cavity]
    xs, ys = zip(*cavity_points)
    ax.plot(xs, ys, 'b-')

    ax.set_aspect('equal')
    plt.title('Delaunay Triangulation with CDT Cavity')
    plt.show()

def plot_triangulation_with_node_nodes(node_coords, elem_nodes, node, node_nodes):
    """
    For a given node will plot the corresponding node_nodes.
    """
    fig, ax = plt.subplots()
    
    # Plot edges of the elem_nodes
    for tri in elem_nodes:
        if tri is not None:
            u, v, w = tri
            triangle = [node_coords[u], node_coords[v], node_coords[w], node_coords[u]]
            xs, ys = zip(*triangle)
            ax.plot(xs, ys, 'k-')
    

    # Highlight the neighbor nodes
    for neigh_node in node_nodes[node]:
        ax.plot(node_coords[neigh_node][0], node_coords[neigh_node][1], 'bo', markersize=10)
    
    # Highlight the specific node
    ax.plot(node_coords[node][0], node_coords[node][1], 'ro', markersize=10)


    ax.set_aspect('equal')
    plt.title('Delaunay Triangulation with Node Nodes')
    plt.show()

def plot_triangulation_with_points(node_coords, elem_nodes, points):
    """
    Plots the triangulation using matplotlib and highlights a specific point.
    """
    fig, ax = plt.subplots()
    
    # Plot edges of the elem_nodes
    for tri in elem_nodes:
        if tri is not None:
            u, v, w = tri
            triangle = [node_coords[u], node_coords[v], node_coords[w], node_coords[u]]
            xs, ys = zip(*triangle)
            ax.plot(xs, ys, 'k-')
    
    # # Plot node_coords
    # xs, ys = zip(*node_coords)
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

def plot_triangulation_with_points_ordered(node_coords, elem_nodes, points):
    """
    Plots the triangulation using matplotlib and highlights a specific point.
    """
    plt.figure(figsize=(15, 7))
    fig, ax = plt.subplots()
    
    # Plot edges of the elem_nodes
    for tri in elem_nodes:
        if tri is not None:
            u, v, w = tri
            triangle = [node_coords[u], node_coords[v], node_coords[w], node_coords[u]]
            xs, ys = zip(*triangle)
            ax.plot(xs, ys, 'k-')
    
    # Plot node_coords
    xs, ys = zip(*node_coords)
    ax.plot(xs, ys, 'ro')
    
    # Highlight the specific points in order
    for i, point in enumerate(points):
        ax.plot(point[0], point[1], 'bo', markersize=10)
        ax.annotate(f'{i}', (point[0], point[1]), xytext=(5, 5), textcoords='offset points')
    
    ax.set_aspect('equal')
    plt.title('Delaunay Triangulation with Points')
    plt.show()

def visualize_intersecting_triangle(node_coords, elem_nodes, bad_elem_nodes, start_idx, end_idx):

    # Plot all elem_nodes
    for i, triangle in enumerate(elem_nodes):
        if triangle is not None:
            a, b, c = triangle
            tri_x = [node_coords[v][0] for v in [a, b, c, a]]
            tri_y = [node_coords[v][1] for v in [a, b, c, a]]
            
            # Highlight bad elem_nodes in red
            if i in bad_elem_nodes:
                plt.fill(tri_x, tri_y, 'red', alpha=0.3)
                plt.plot(tri_x, tri_y, 'k-')

            else:
                plt.fill(tri_x, tri_y, 'lightgray', alpha=0.3)
                plt.plot(tri_x, tri_y, 'k-')

    # Highlight the start and end points
    plt.scatter(*node_coords[start_idx], c='red', s=100, label='Start point')
    plt.scatter(*node_coords[end_idx], c='red', s=100, label='End point')

    # Draw the line from start to end
    plt.plot([node_coords[start_idx][0], node_coords[end_idx][0]], [node_coords[start_idx][1], node_coords[end_idx][1]], 'k-', lw=2, color='red')

    plt.legend()
    plt.title('Finding Intersecting Triangle')
    plt.axis('equal')
    plt.show()

def visualize_bad_elem_nodes(node_coords, elem_nodes, bad_elem_nodes, u_idx, step_number):

    # Plot all elem_nodes
    for i, triangle in enumerate(elem_nodes):
        if triangle is not None:
            a, b, c = triangle
            tri_x = [node_coords[v][0] for v in [a, b, c, a]]
            tri_y = [node_coords[v][1] for v in [a, b, c, a]]
            
            # Highlight bad elem_nodes in red
            if i in bad_elem_nodes:
                plt.fill(tri_x, tri_y, 'red', alpha=0.3)
                plt.plot(tri_x, tri_y, 'k-')

                # Draw circumcircle for bad elem_nodes
                center, radius = get_circumcircle(node_coords[a], node_coords[b], node_coords[c])
                circle = Circle(center, radius, fill=False, linestyle='--', color='blue')
                plt.gca().add_artist(circle)
            else:
                plt.fill(tri_x, tri_y, 'lightgray', alpha=0.3)
                plt.plot(tri_x, tri_y, 'k-')

    # Highlight the new point u
    plt.scatter(*node_coords[u_idx], c='green', s=100, label='New point u')

    plt.legend()
    plt.title(f'Step {step_number}: Displaying Bad elem_nodes and Circumcircles\nBad elem_nodes: {bad_elem_nodes}')
    plt.axis('equal')
    plt.show()

def visualize_bad_elem_nodes_step(node_coords, elem_nodes, bad_elem_nodes, current_triangle, u_idx, step_number):

    # Plot all elem_nodes
    for i, triangle in enumerate(elem_nodes):
        if triangle is not None:
            a, b, c = triangle
            tri_x = [node_coords[v][0] for v in [a, b, c, a]]
            tri_y = [node_coords[v][1] for v in [a, b, c, a]]
            if i in bad_elem_nodes:
                plt.fill(tri_x, tri_y, 'red', alpha=0.3)
            else:
                plt.fill(tri_x, tri_y, 'lightgray', alpha=0.3)
            plt.plot(tri_x, tri_y, 'k-')

    # Highlight the current triangle
    if current_triangle is not None:
        a, b, c = elem_nodes[current_triangle]
        tri_x = [node_coords[v][0] for v in [a, b, c, a]]
        tri_y = [node_coords[v][1] for v in [a, b, c, a]]
        plt.fill(tri_x, tri_y, 'yellow', alpha=0.5)

        # Draw circumcircle
        center, radius = get_circumcircle(node_coords[a], node_coords[b], node_coords[c])
        circle = Circle(center, radius, fill=False, linestyle='--', color='blue')
        plt.gca().add_artist(circle)

    # Plot all node_coords
    x, y = zip(*node_coords)
    plt.scatter(x, y, c='blue', s=50)

    # Highlight the new point u
    plt.scatter(*node_coords[u_idx], c='green', s=100, label='New point u')

    # Add labels to node_coords
    for i, (x, y) in enumerate(node_coords):
        plt.annotate(f'{i}', (x, y), xytext=(5, 5), textcoords='offset points')

    plt.legend()
    plt.title(f'Step {step_number}: Finding Bad elem_nodes\nCurrent Triangle: {current_triangle}, Bad elem_nodes: {bad_elem_nodes}')
    plt.axis('equal')
    plt.show()

def plot_edges_from_idx(node_coords, edges):
    """
    Plots the edges of a polygon given the vertex indices.
    """
    plt.figure(figsize=(6, 6))
    for edge in edges:
        u, v = edge
        plt.plot([node_coords[u][0], node_coords[v][0]], [node_coords[u][1], node_coords[v][1]], 'b-')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Polygon Edges')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.show()

def plot_points(node_coords, point_size=0.5, point_color='blue', alpha=1.0, title='Points'):
    """
    Plots a list of points with adjustable size, color, and transparency.
    
    Args:
        node_coords (list of tuples): List of (x, y) coordinates to plot.
        point_size (int, optional): Size of the points. Default is 5.
        point_color (str, optional): Color of the points. Default is 'blue'.
        alpha (float, optional): Transparency level of the points. Default is 1.0 (opaque).
    """
    plt.figure(figsize=(6, 6))
    x, y = zip(*node_coords)
    plt.scatter(x, y, s=point_size, c=point_color, alpha=alpha)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.show()

def plot_edges_from_coords(node_coords, edges):
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

def visualize_walk_to_point(node_coords, elem_nodes, current_triangle_idx, q_idx, p_idx, r_idx, l_idx, step_number, action, initilization_elem_nodes, main_traversal_elem_nodes):
    # Plot all elem_nodes
    for i, triangle in enumerate(elem_nodes):
        if triangle is None:
            continue
        else:
            tri_x = [node_coords[v][0] for v in triangle]
            tri_y = [node_coords[v][1] for v in triangle]
            if i in initilization_elem_nodes:
                plt.fill(tri_x, tri_y, color='lightcoral')
            elif i in main_traversal_elem_nodes:
                plt.fill(tri_x, tri_y, color='cornflowerblue')
            else:
                plt.fill(tri_x, tri_y, 'ghostwhite')
            plt.plot(tri_x + [tri_x[0]], tri_y + [tri_y[0]], 'k-')

    # Highlight special points
    plt.scatter(*node_coords[q_idx], c='red', s=100, label='q (start)')
    plt.scatter(*node_coords[p_idx], c='green', s=100, label='p (target)')

    # Draw the ray from q to p with an arrowhead that stops exactly at point p
    q_x, q_y = node_coords[q_idx]
    p_x, p_y = node_coords[p_idx]

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

def plot_triangulation_and_edge(node_coords, elem_nodes, edge, title="Triangulation with Edge"):
    """
    Plots the triangulation using matplotlib and highlights a specific edge.
    """
    fig, ax = plt.subplots()
    
    # Plot edges of the elem_nodes
    for tri in elem_nodes:
        if tri is not None:
            u, v, w = tri
            triangle = [node_coords[u], node_coords[v], node_coords[w], node_coords[u]]
            xs, ys = zip(*triangle)
            ax.plot(xs, ys, 'k-')
    
    # Highlight the specific edge
    u, v = edge
    edge_x = [node_coords[u][0], node_coords[v][0]]
    edge_y = [node_coords[u][1], node_coords[v][1]]
    ax.plot(edge_x, edge_y, 'r-', linewidth=2)
    
    ax.set_aspect('equal')
    plt.title(title)
    plt.show()

def plot_triangulation_with_points_and_marked_elem_nodes(node_coords, elem_nodes, points, marked_elem_nodes):
    """
    Plots the triangulation using matplotlib and highlights a specific point and elem_nodes.
    """
    fig, ax = plt.subplots()
    
    # Plot edges of the elem_nodes
    for tri in elem_nodes:
        if tri is not None:
            u, v, w = tri
            triangle = [node_coords[u], node_coords[v], node_coords[w], node_coords[u]]
            xs, ys = zip(*triangle)
            ax.plot(xs, ys, 'k-')
    
    # Plot node_coords
    xs, ys = zip(*node_coords)
    ax.plot(xs, ys, 'ro')
    
    # Highlight the specific points
    for point in points:
        ax.plot(point[0], point[1], 'bo', markersize=10)
    
    # Highlight the marked elem_nodes
    for tri_idx in marked_elem_nodes:
        u, v, w = elem_nodes[tri_idx]
        triangle = [node_coords[u], node_coords[v], node_coords[w], node_coords[u]]
        xs, ys = zip(*triangle)
        ax.fill(xs, ys, 'r', alpha=0.3)
    
    ax.set_aspect('equal')
    plt.title('Delaunay Triangulation with Points and Marked elem_nodes')
    plt.show()