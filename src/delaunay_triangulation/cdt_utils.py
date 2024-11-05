from collections import defaultdict
from typing import List, Tuple, Optional, Set, Dict

from .predicates import (
    orient, 
    in_triangle, 
    is_point_on_edge, 
    in_circle
    )

from .visualize import (
    plot_triangulation_with_points, 
    visualize_walk_step, 
    visualize_bad_elem_nodes_step,
    visualize_intersecting_triangle
)

from .operations import (
    get_neighbor_through_edge, 
    get_one_triangle_of_vertex, 
    get_triangle_neighbors_constrained
    )

from .dt_utils import (
    walk_to_point
    )

def ignore_during_traversal(
    r_idx: int, 
    l_idx: int, 
    start_idx: int, 
    target_idx: int, 
    node_coords: List[Tuple[float, float]]
) -> bool:
    """
    Checks if a point is collinear with the starting and target point. If so, it should be ignored during the traversal, 
    as it does not contribute to the intersection with the triangle.

    Parameters:
    - r_idx (int): Index of the right endpoint of the edge.
    - l_idx (int): Index of the left endpoint of the edge.
    - start_idx (int): Index of the starting vertex.
    - target_idx (int): Index of the target vertex.
    - node_coords (List[Tuple[float, float]]): List of vertex coordinates.

    Returns:
    - bool: True if the point is collinear, False otherwise.
    """
    if orient(l_idx, start_idx, target_idx, node_coords) == 0 or orient(r_idx, start_idx, target_idx, node_coords) == 0:
        return True
    return False

def find_intersecting_triangle(
    start_idx: int, 
    target_idx: int, 
    node_coords: List[Tuple[float, float]], 
    elem_nodes: List[Optional[Tuple[int, int, int]]], 
    node_elems: List[List[int]], 
    visualize: bool = False
) -> Set[int]:
    """
    Finds all triangles in the triangulation that are intersected by the interior of the given edge `(start_idx, target_idx)`.

    Parameters:
    - start_idx (int): Index of the starting vertex `p`.
    - target_idx (int): Index of the target vertex `q`.
    - node_coords (List[Tuple[float, float]]): List of vertex coordinates.
    - elem_nodes (List[Optional[Tuple[int, int, int]]]): List of existing triangles, where each triangle is represented 
      as a tuple of three vertex indices.
    - node_elems (List[List[int]]): List of lists, where each sublist contains the indices of triangles connected to a node.
    - visualize (bool, optional): Boolean flag to enable or disable visualization of the walk steps. Default is False.

    Returns:
    - Set[int]: Set of triangle indices that are intersected by the interior of the edge.
    """
    current_triangle_idx = get_one_triangle_of_vertex(start_idx, node_elems, elem_nodes)

    if current_triangle_idx == target_idx or start_idx == target_idx:
        return [current_triangle_idx]

    step_number = 0
    initilization_elems = [current_triangle_idx]
    main_traversal_elems = []

    if current_triangle_idx == -1:
        raise ValueError("Starting vertex q is not part of any triangle in the triangulation.")

    current_triangle = elem_nodes[current_triangle_idx]

    # Get the two other node_coords of the current triangle
    other_nodes = [v for v in current_triangle if v != start_idx]
    if len(other_nodes) != 2:
        raise ValueError(f"Invalid triangle configuration, expected 2 vertices but got {len(other_nodes)}.")

    v1_idx, v2_idx = other_nodes

    # Determine the orientation of the triangle formed by start_idx, v1_idx, v2_idx
    orientation = orient(start_idx, v1_idx, v2_idx, node_coords)
    if orientation < 0:
        # Triangle is counter-clockwise
        r_idx = v1_idx
        l_idx = v2_idx
    else:
        # Triangle is clockwise
        r_idx = v2_idx
        l_idx = v1_idx

    step_number += 1
    # if visualize:
    #     visualize_walk_step(node_coords, elem_nodes, current_triangle_idx, start_idx, target_idx, r_idx, l_idx,
    #                         step_number, "Initial setup", initilization_elems, main_traversal_elems)


    if orient(start_idx, target_idx, r_idx, node_coords) > 0:
        while orient(start_idx, target_idx, l_idx, node_coords) > 0:
            step_number += 1
            r_idx = l_idx
            neighbor_triangle_idx = get_neighbor_through_edge(current_triangle_idx, (start_idx, l_idx), node_elems)
            if neighbor_triangle_idx == -1:
                # Check if neighbor_triangle_idx is has the target_idx
                if target_idx in elem_nodes[neighbor_triangle_idx]:
                    return [neighbor_triangle_idx]
                else:
                    raise ValueError("Reached a boundary while traversing; point may be outside triangulation.")
            
            current_triangle_idx = neighbor_triangle_idx
            initilization_elems.append(current_triangle_idx)
            current_triangle = elem_nodes[current_triangle_idx]
            l_idx = next(v for v in current_triangle if v != start_idx and v != r_idx)
            # if visualize:
            #     visualize_walk_step(node_coords, elem_nodes, current_triangle_idx, start_idx, target_idx, r_idx, l_idx, 
            #                         step_number, "Rotating", initilization_elems, main_traversal_elems)
                
            if in_triangle(start_idx, r_idx, l_idx, target_idx, node_coords):
                return [current_triangle_idx]
    else:
        cond = True
        while cond:
            step_number += 1
            l_idx = r_idx
            neighbor_triangle_idx = get_neighbor_through_edge(current_triangle_idx, (start_idx, r_idx), node_elems)
            
            if neighbor_triangle_idx == -1:
                # Check if neighbor_triangle_idx is has the target_idx
                if target_idx in elem_nodes[neighbor_triangle_idx]:
                    return [neighbor_triangle_idx]
                else:
                    print(start_idx, target_idx, r_idx, l_idx)
                    raise ValueError("Reached a boundary while traversing; point may be outside triangulation.")
            
            current_triangle_idx = neighbor_triangle_idx
            initilization_elems.append(current_triangle_idx)
            current_triangle = elem_nodes[current_triangle_idx]
            r_idx = next(v for v in current_triangle if v != start_idx and v != l_idx)
            
            cond = orient(start_idx, target_idx, r_idx, node_coords) <= 0
            # if visualize:
            #     visualize_walk_step(node_coords, elem_nodes, current_triangle_idx, start_idx, target_idx, r_idx, l_idx, 
            #                         step_number, "Rotating", initilization_elems, main_traversal_elems)
            
            if in_triangle(start_idx, r_idx, l_idx, target_idx, node_coords):
                return [current_triangle_idx]


    initilization_elems.pop()  # Remove the last triangle from the initialization list

    main_traversal_elems = []

    if not ignore_during_traversal(r_idx, l_idx, start_idx, target_idx, node_coords):
        main_traversal_elems.append(current_triangle_idx)


    # Switch l and r
    l_idx, r_idx = r_idx, l_idx


    # Straight walk along the triangulation
    while orient(target_idx, r_idx, l_idx, node_coords) <= 0:

        if in_triangle(start_idx, r_idx, l_idx, target_idx, node_coords):
            if visualize:
                visualize_intersecting_triangle(node_coords, elem_nodes, main_traversal_elems, start_idx, target_idx)
            return main_traversal_elems

        step_number += 1
        neighbor_triangle_idx = get_neighbor_through_edge(current_triangle_idx, (r_idx, l_idx), node_elems)

        if neighbor_triangle_idx == -1:
            # Check if neighbor_triangle_idx is has the target_idx
            if target_idx in elem_nodes[neighbor_triangle_idx]:
                return [neighbor_triangle_idx]
            else:
                raise ValueError("Reached a boundary while traversing; point may be outside triangulation.")
        
        current_triangle_idx = neighbor_triangle_idx

        if not ignore_during_traversal(r_idx, l_idx, start_idx, target_idx, node_coords):
            main_traversal_elems.append(current_triangle_idx)
        current_triangle = elem_nodes[current_triangle_idx]
        s_idx = next(v for v in current_triangle if v != r_idx and v != l_idx)

        if orient(s_idx, start_idx, target_idx, node_coords) <= 0:
            r_idx = s_idx
        else:
            l_idx = s_idx
        # if visualize:
        #     visualize_walk_step(node_coords, elem_nodes, current_triangle_idx, start_idx, target_idx, r_idx, l_idx, 
        #                         step_number, "Main path finding", initilization_elems, main_traversal_elems)


    return main_traversal_elems 

def find_cavities(
    u: int, 
    v: int, 
    intersected_elem_nodes: Set[int], 
    node_coords: List[Tuple[float, float]], 
    elem_nodes: List[Optional[Tuple[int, int, int]]]
) -> List[List[int]]:
    """
    Identifies the two polygonal cavities formed on each side of the constraint edge `s = (u, v)`.

    Parameters:
    - u, v (int): Vertex indices defining the constraint edge `s`.
    - intersected_elem_nodes (Set[int]): Set of triangle indices that were intersected by `s`.
    - node_coords (List[Tuple[float, float]]): List of vertex coordinates.
    - elem_nodes (List[Optional[Tuple[int, int, int]]]): List of existing triangles, where each triangle is represented 
      as a tuple of three vertex indices.

    Returns:
    - List[List[int]]: A list containing two cavities, each defined as an ordered list of vertex indices.
    """
    # Step 1: Collect Boundary Edges
    boundary_edges = collect_boundary_edges(intersected_elem_nodes, elem_nodes)

    # Step 2: Remove the Constraint Edge from Boundary Edges
    s = tuple(sorted((u, v)))
    if s in boundary_edges:
        boundary_edges.remove(s)

    # Step 3: Build Adjacency Map from Boundary Edges
    adjacency = build_adjacency_map(boundary_edges)


    # Step 4: Find Two Separate Paths (Cavities) from u to v
    try:
        cavity1 = traverse_path(u, v, adjacency)
    except ValueError as e:
        plot_triangulation_with_points(node_coords, elem_nodes, [node_coords[u], node_coords[v]])
        raise ValueError(f"Error while traversing first cavity: {e}")

    # Remove the edges of cavity1 from the adjacency to isolate cavity2
    remove_edges_from_adjacency(cavity1, adjacency)

    try:
        cavity2 = traverse_path(u, v, adjacency)
    except ValueError as e:
        raise ValueError(f"Error while traversing second cavity: {e}")

    # Step 5: Order the Cavity node_coords in CCW Order
    cavity1_ordered = order_cavity_node_coords(cavity1, node_coords)
    cavity2_ordered = order_cavity_node_coords(cavity2, node_coords)

    return [cavity1_ordered, cavity2_ordered]

def collect_boundary_edges(
    intersected_elem_nodes: Set[int], 
    elem_nodes: List[Optional[Tuple[int, int, int]]]
) -> Set[Tuple[int, int]]:
    """
    Collects boundary edges from the set of intersected triangles.

    Parameters:
    - intersected_elem_nodes (Set[int]): Set of triangle indices that were intersected by the constraint edge.
    - elem_nodes (List[Optional[Tuple[int, int, int]]]): List of existing triangles, where each triangle is represented 
      as a tuple of three vertex indices.

    Returns:
    - Set[Tuple[int, int]]: A set of boundary edges (sorted tuples of vertex indices).
    """
    edge_count = defaultdict(int)

    for t_idx in intersected_elem_nodes:
        tri = elem_nodes[t_idx]
        if tri is None:
            continue  # Triangle has been deleted

        edges = [
            tuple(sorted((tri[0], tri[1]))),
            tuple(sorted((tri[1], tri[2]))),
            tuple(sorted((tri[2], tri[0])))
        ]

        for edge in edges:
            edge_count[edge] += 1

    # Boundary edges are those that appear exactly once in intersected elem_nodes
    boundary_edges = set(edge for edge, count in edge_count.items() if count == 1)

    return boundary_edges

def build_adjacency_map(
        boundary_edges: Set[Tuple[int, int]]
) -> Dict[int, Set[int]]:
    """
    Builds an adjacency map from the set of boundary edges.

    Parameters:
    - boundary_edges (Set[Tuple[int, int]]): Set of boundary edges (sorted tuples of vertex indices).

    Returns:
    - Dict[int, Set[int]]: A dictionary mapping each vertex to a set of its adjacent vertices via boundary edges.
    """
    adjacency = {}

    for edge in boundary_edges:
        a, b = edge
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)

    return adjacency

def traverse_path(
        start: int, 
        end: int, 
        adjacency: Dict[int, Set[int]]
) -> List[int]:
    """
    Traverses from start to end using DFS and returns the path.

    Parameters:
    - start (int): Starting vertex index.
    - end (int): Ending vertex index.
    - adjacency (Dict[int, Set[int]]): Adjacency dictionary.

    Returns:
    - List[int]: An ordered list of vertex indices representing the path from start to end.

    Raises:
    - ValueError: If no path exists between start and end vertices.
    """
    stack = [(start, [start])]
    visited = set()

    while stack:
        (current, path) = stack.pop()
        if current == end:
            return path
        if current in visited:
            continue
        visited.add(current)
        for neighbor in adjacency.get(current, []):
            if neighbor not in visited:
                stack.append((neighbor, path + [neighbor]))

    raise ValueError(f"No path found from {start} to {end}.")

def remove_edges_from_adjacency(
        path: List[int], 
        adjacency: Dict[int, Set[int]]
) -> None:
    """
    Removes the edges of the given path from the adjacency map.

    Parameters:
    - path (List[int]): An ordered list of vertex indices representing a path.
    - adjacency (Dict[int, Set[int]]): Adjacency dictionary to be modified.

    Returns:
    - None. The adjacency map is modified in place.
    """
    for i in range(len(path) - 1):
        a, b = path[i], path[i+1]
        if b in adjacency.get(a, set()):
            adjacency[a].remove(b)
        if a in adjacency.get(b, set()):
            adjacency[b].remove(a)

def compute_polygon_orientation(
        polygon: List[int], 
        node_coords: List[Tuple[float, float]]
) -> str:
    """
    Computes the orientation of a polygon.

    Parameters:
    - polygon (List[int]): An ordered list of vertex indices representing the polygon.
    - node_coords (List[Tuple[float, float]]): List of vertex coordinates.

    Returns:
    - str: "CCW" if the polygon is counterclockwise, "CW" if the polygon is clockwise, or "COLINEAR" if the polygon is collinear.
    """
    area = 0.0
    n = len(polygon)

    for i in range(n):
        x1, y1 = node_coords[polygon[i]]
        x2, y2 = node_coords[polygon[(i + 1) % n]]
        area += (x1 * y2) - (x2 * y1)

    if area > 0:
        return "CCW"
    elif area < 0:
        return "CW"
    else:
        return "COLINEAR"

def order_cavity_node_coords(
        cavity_path: List[int], 
        node_coords: List[Tuple[float, float]]
)-> List[int]:
    """
    Orders the cavity nodes in counterclockwise (CCW) order around the cavity.

    Description:
    This function takes a list of vertex indices representing a path around a cavity and orders them in a counterclockwise 
    direction. The last edge (v, u) forms the constraint edge.

    Parameters:
    - cavity_path (List[int]): An ordered list of vertex indices representing the cavity path from u to v.
    - node_coords (List[Tuple[float, float]]): List of vertex coordinates as tuples of (x, y).

    Returns:
    - List[int]: An ordered list of vertex indices in CCW order, starting at u and ending at v.
    """
    # Create a closed loop by appending the start vertex to the end
    closed_loop = cavity_path + [cavity_path[0]]

    orientation = compute_polygon_orientation(closed_loop, node_coords)

    if orientation == "CW":
        # If the loop is clockwise, reverse it to make it counterclockwise
        ordered_path = cavity_path[::-1]
    else:
        # If already counterclockwise or colinear, keep as is
        ordered_path = cavity_path

    return ordered_path

def walk_to_point_constrained(
    start_idx: int, 
    target_idx: int, 
    start_triangle_idx: int, 
    node_coords: List[Tuple[float, float]], 
    elem_nodes: List[Tuple[int, int, int]], 
    node_elems: List[List[int]], 
    visualize: bool = False
) -> List[int]:
    """
    This function initiates a traversal from a starting triangle and moves towards a target point while checking if the point lies on an edge 
    of the current triangle. The traversal respects constraints and visualizations if specified.

    Parameters:
    - start_idx (int): The index of the starting point.
    - target_idx (int): The index of the target point.
    - start_triangle_idx (int): The index of the triangle containing the starting point.
    - node_coords (List[Tuple[float, float]]): List of node coordinates as tuples of (x, y).
    - elem_nodes (List[Tuple[int, int, int]]): List of existing elem_nodes where each element is a tuple of three vertex indices.
    - node_elems (List[List[int]]): List of lists where node_elems[node] contains triangle indices that include the node.
    - visualize (bool): Boolean flag to enable or disable visualization (default: False).

    Returns:
    - List[int]: A list of triangle indices that contain the target point or are connected through a shared edge.
    """
    # Start by calling walk_to_point to find the initial triangle
    initial_bad_idx = walk_to_point(start_idx, target_idx, start_triangle_idx, node_coords, elem_nodes, node_elems, visualize)

    initial_bad = elem_nodes[initial_bad_idx]

    # Generate the list of edges in the bad triangle
    bad_edges = [
        sorted((initial_bad[0], initial_bad[1])),
        sorted((initial_bad[1], initial_bad[2])),
        sorted((initial_bad[2], initial_bad[0]))
    ]

    for edge in bad_edges:
        if is_point_on_edge(target_idx, edge, node_coords):
            # Retrieve triangles that share this edge by finding common triangles between the two nodes of the edge
            node1, node2 = edge
            triangles_node1 = set(node_elems[node1])
            triangles_node2 = set(node_elems[node2])

            # Find the common triangles between the two nodes (edge) that are not the current triangle
            common_triangles = triangles_node1.intersection(triangles_node2)
            common_triangles.discard(initial_bad_idx)

            # If a shared triangle exists, it must be the "other" bad triangle
            if common_triangles:
                other_bad_triangle_idx = next(iter(common_triangles))
                return [initial_bad_idx, other_bad_triangle_idx]

    return [initial_bad_idx]

def find_bad_elem_nodes_constrained(
    initial_bads: List[int], 
    u_idx: int, 
    node_coords: List[Tuple[float, float]], 
    elem_nodes: List[Tuple[int, int, int]], 
    node_elems: List[List[int]], 
    constrained_edges_set: Set[Tuple[int, int]], 
    visualize: bool = False
) -> Set[int]:
    """
    This function identifies all triangles (elem_nodes) whose circumcircles contain a newly inserted point, respecting 
    the constrained edges set. The search starts from an initial list of bad triangles and avoids crossing constrained edges.

    Parameters:
    - initial_bads (List[int]): List of indices of initial bad elem_nodes.
    - u_idx (int): Index of the point to be inserted.
    - node_coords (List[Tuple[float, float]]): List of vertex coordinates as tuples of (x, y).
    - elem_nodes (List[Tuple[int, int, int]]): List of existing elem_nodes, where each element is a tuple of three vertex indices.
    - node_elems (List[List[int]]): List of lists where each sublist contains the indices of triangles connected to a node.
    - constrained_edges_set (Set[Tuple[int, int]]): Set of edges (as sorted tuples) that are constrained.
    - visualize (bool): Boolean flag to indicate whether to visualize each step.

    Returns:
    - Set[int]: A set of indices of bad elem_nodes that need to be re-triangulated.
    """
    bad_elem_nodes = set()
    stack = initial_bads  # Initialize stack with all initial bad elem_nodes
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
        if visualize:
            visualize_bad_elem_nodes_step(node_coords, elem_nodes, bad_elem_nodes, current_t_idx, u_idx, step_number)

        if in_circle(a, b, c, u_idx, node_coords) > 0:

            bad_elem_nodes.add(current_t_idx)


            # Retrieve neighbors without crossing constrained edges
            neighbors = get_triangle_neighbors_constrained(current_t_idx, elem_nodes, node_elems, constrained_edges_set)


            for neighbor_idx in neighbors:

                if neighbor_idx != -1 and neighbor_idx not in bad_elem_nodes:

                    stack.append(neighbor_idx)

    # Final visualization
    if visualize:
        visualize_bad_elem_nodes_step(node_coords, elem_nodes, bad_elem_nodes, None, u_idx, step_number + 1)

    return bad_elem_nodes

def order_boundary_node_coords_ccw(
    boundary_edges: List[Tuple[int, int]], 
    node_coords: List[Tuple[float, float]]
) -> List[int]:
    """
    Orders the nodes from boundary edges in counter-clockwise (CCW) order.

    Description:
    This function takes a list of boundary edges and orders the nodes in a counter-clockwise direction 
    around the boundary of a polygonal region. It ensures that the resulting list represents a simple polygon in CCW order.

    Parameters:
    - boundary_edges (List[Tuple[int, int]]): List of tuples representing edges `(v1, v2)` in the boundary.
    - node_coords (List[Tuple[float, float]]): List of vertex coordinates as tuples of `(x, y)`.

    Returns:
    - List[int]: A list of vertex indices ordered in a counter-clockwise (CCW) direction.
    """

    # Step 1: Create adjacency map
    adjacency = defaultdict(list)
    for v1, v2 in boundary_edges:
        adjacency[v1].append(v2)
        adjacency[v2].append(v1)

    # Step 2: Traverse the boundary to order node_coords
    # Start from a vertex with degree 2 (assuming a simple polygon)
    start_vertex = None
    for v, neighbors in adjacency.items():
        if len(neighbors) == 2:
            start_vertex = v
            break

    if start_vertex is None:
        raise ValueError("No suitable starting vertex found. The boundary might not form a simple polygon.")

    ordered_node_coords = [start_vertex]
    current = start_vertex
    prev = None

    while True:
        neighbors = adjacency[current]
        # Choose the neighbor that's not the previous vertex
        next_node_coords = [v for v in neighbors if v != prev]
        if not next_node_coords:
            break  # Completed the loop
        next_vertex = next_node_coords[0]
        if next_vertex == start_vertex:
            break  # Completed the loop
        ordered_node_coords.append(next_vertex)
        prev, current = current, next_vertex

    # Step 3: Ensure CCW orientation
    if not is_ccw(ordered_node_coords, node_coords):
        ordered_node_coords.reverse()

    return ordered_node_coords

def is_ccw(
    ordered_node_coords: List[int], 
    node_coords: List[Tuple[float, float]]
) -> bool:
    """
    Determines if the ordered list of node coordinates is in counter-clockwise (CCW) order.

    Description:
    This function checks whether the given list of vertex indices represents a counter-clockwise (CCW) 
    oriented polygon based on the provided node coordinates. The function calculates the signed area 
    to determine the orientation of the polygon.

    Parameters:
    - ordered_node_coords (List[int]): List of vertex indices ordered around the polygon.
    - node_coords (List[Tuple[float, float]]): List of vertex coordinates as tuples of `(x, y)`.

    Returns:
    - bool: `True` if the nodes are ordered in a counter-clockwise direction, `False` otherwise.
    """
    area = 0.0
    n = len(ordered_node_coords)
    for i in range(n):
        v_current = node_coords[ordered_node_coords[i]]
        v_next = node_coords[ordered_node_coords[(i + 1) % n]]
        area += (v_current[0] * v_next[1]) - (v_next[0] * v_current[1])
    return area > 0