from collections import defaultdict

from src.predicates import (
    orient, 
    in_triangle, 
    is_point_on_edge, 
    in_circle
    )

from src.visualize import (
    plot_triangulation_with_points, 
    visualize_walk_step, 
    visualize_bad_triangles_step,
    visualize_intersecting_triangle
)

from src.operations import (
    get_neighbor_through_edge, 
    get_one_triangle_of_vertex, 
    get_triangle_neighbors_constrained
    )

from src.dt_utils import walk_to_point

def ignore_during_traversal(r_idx, l_idx, start_idx, target_idx, vertices):
    """
    Check if one of the point is collinear with the starting and target point. If so we ignore it during the traversal, since it means it doesn't intersect the triangle.
    """

    if orient(l_idx, start_idx, target_idx, vertices) == 0 or orient(r_idx, start_idx, target_idx, vertices) == 0:
        return True
    return False

def find_intersecting_triangle(start_idx, target_idx, vertices, triangles, edge_to_triangle, vertex_to_triangles, visualize=False):
    """
    Finds all triangles in the triangulation that are intersected by the interior of the given edge s = (start_idx, target_idx).
    
    Parameters:
    - start_idx: Index of the starting vertex p.
    - target_idx: Index of the point q to locate.
    - vertices: List of vertex coordinates.
    - triangles: List of triangles (each triangle is represented as a tuple of 3 vertex indices).
    - edge_to_triangle: Dictionary mapping edges to one or two triangle indices.
    - vertex_to_triangles: List of lists, where each sublist contains triangle indices.
    - visualize: Boolean flag to enable or disable visualization of the walk steps.

    
    Returns:
    - intersecting_triangles: Set of triangle indices that are intersected by the interior of the edge.
    """

    current_triangle_idx = get_one_triangle_of_vertex(start_idx, vertex_to_triangles, triangles)

    if current_triangle_idx == target_idx or start_idx == target_idx:
        return [current_triangle_idx]

    step_number = 0
    initilization_triangles = [current_triangle_idx]
    main_traversal_triangles = []

    if current_triangle_idx == -1:
        raise ValueError("Starting vertex q is not part of any triangle in the triangulation.")

    current_triangle = triangles[current_triangle_idx]

    # Get the two other vertices of the current triangle
    other_vertices = [v for v in current_triangle if v != start_idx]
    if len(other_vertices) != 2:
        raise ValueError("Invalid triangle configuration.")

    v1_idx, v2_idx = other_vertices

    # Determine the orientation of the triangle formed by start_idx, v1_idx, v2_idx
    orientation = orient(start_idx, v1_idx, v2_idx, vertices)
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
    #     visualize_walk_step(vertices, triangles, current_triangle_idx, start_idx, target_idx, r_idx, l_idx,
    #                         step_number, "Initial setup", initilization_triangles, main_traversal_triangles)


    if orient(start_idx, target_idx, r_idx, vertices) > 0:
        while orient(start_idx, target_idx, l_idx, vertices) > 0:
            step_number += 1
            r_idx = l_idx
            neighbor_triangle_idx = get_neighbor_through_edge(current_triangle_idx, (start_idx, l_idx), edge_to_triangle)
            
            if neighbor_triangle_idx == -1:
                # Check if neighbor_triangle_idx is has the target_idx
                if target_idx in triangles[neighbor_triangle_idx]:
                    return [neighbor_triangle_idx]
                else:
                    raise ValueError("Reached a boundary while traversing; point may be outside triangulation.")
            
            current_triangle_idx = neighbor_triangle_idx
            initilization_triangles.append(current_triangle_idx)
            current_triangle = triangles[current_triangle_idx]
            l_idx = next(v for v in current_triangle if v != start_idx and v != r_idx)
            # if visualize:
            #     visualize_walk_step(vertices, triangles, current_triangle_idx, start_idx, target_idx, r_idx, l_idx, 
            #                         step_number, "Rotating", initilization_triangles, main_traversal_triangles)
                
            if in_triangle(start_idx, r_idx, l_idx, target_idx, vertices):
                return [current_triangle_idx]
    else:
        cond = True
        while cond:
            step_number += 1
            l_idx = r_idx
            neighbor_triangle_idx = get_neighbor_through_edge(current_triangle_idx, (start_idx, r_idx), edge_to_triangle)
            
            if neighbor_triangle_idx == -1:
                # Check if neighbor_triangle_idx is has the target_idx
                if target_idx in triangles[neighbor_triangle_idx]:
                    return [neighbor_triangle_idx]
                else:
                    print(start_idx, target_idx, r_idx, l_idx)
                    raise ValueError("Reached a boundary while traversing; point may be outside triangulation.")
            
            current_triangle_idx = neighbor_triangle_idx
            initilization_triangles.append(current_triangle_idx)
            current_triangle = triangles[current_triangle_idx]
            r_idx = next(v for v in current_triangle if v != start_idx and v != l_idx)
            
            cond = orient(start_idx, target_idx, r_idx, vertices) <= 0
            # if visualize:
            #     visualize_walk_step(vertices, triangles, current_triangle_idx, start_idx, target_idx, r_idx, l_idx, 
            #                         step_number, "Rotating", initilization_triangles, main_traversal_triangles)
            
            if in_triangle(start_idx, r_idx, l_idx, target_idx, vertices):
                return [current_triangle_idx]


    initilization_triangles.pop()  # Remove the last triangle from the initialization list

    main_traversal_triangles = []

    if not ignore_during_traversal(r_idx, l_idx, start_idx, target_idx, vertices):
        main_traversal_triangles.append(current_triangle_idx)


    # Switch l and r
    l_idx, r_idx = r_idx, l_idx


    # Straight walk along the triangulation
    while orient(target_idx, r_idx, l_idx, vertices) <= 0:

        if in_triangle(start_idx, r_idx, l_idx, target_idx, vertices):
            visualize_intersecting_triangle(vertices, triangles, main_traversal_triangles, start_idx, target_idx)
            return main_traversal_triangles

        step_number += 1
        neighbor_triangle_idx = get_neighbor_through_edge(current_triangle_idx, (r_idx, l_idx), edge_to_triangle)

        if neighbor_triangle_idx == -1:
            # Check if neighbor_triangle_idx is has the target_idx
            if target_idx in triangles[neighbor_triangle_idx]:
                return [neighbor_triangle_idx]
            else:
                raise ValueError("Reached a boundary while traversing; point may be outside triangulation.")
        
        current_triangle_idx = neighbor_triangle_idx

        if not ignore_during_traversal(r_idx, l_idx, start_idx, target_idx, vertices):
            main_traversal_triangles.append(current_triangle_idx)
        current_triangle = triangles[current_triangle_idx]
        s_idx = next(v for v in current_triangle if v != r_idx and v != l_idx)

        if orient(s_idx, start_idx, target_idx, vertices) <= 0:
            r_idx = s_idx
        else:
            l_idx = s_idx
        # if visualize:
        #     visualize_walk_step(vertices, triangles, current_triangle_idx, start_idx, target_idx, r_idx, l_idx, 
        #                         step_number, "Main path finding", initilization_triangles, main_traversal_triangles)


    return main_traversal_triangles 

def find_cavities(u, v, intersected_triangles, vertices, triangles):
    """
    Identifies the two polygonal cavities formed on each side of the constraint edge s = (u, v).

    Parameters:
    - u, v: Vertex indices defining the constraint edge s.
    - intersected_triangles: Set of triangle indices that were intersected by s.
    - vertices: List of vertex coordinates.
    - triangles: List of triangles (each triangle is represented as a tuple of 3 vertex indices).


    Returns:
    - A list containing two cavities, each defined as an ordered list of vertex indices.
    """
    # Step 1: Collect Boundary Edges
    boundary_edges = collect_boundary_edges(intersected_triangles, triangles)

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
        plot_triangulation_with_points(vertices, triangles, [vertices[u], vertices[v]])
        raise ValueError(f"Error while traversing first cavity: {e}")

    # Remove the edges of cavity1 from the adjacency to isolate cavity2
    remove_edges_from_adjacency(cavity1, adjacency)

    try:
        cavity2 = traverse_path(u, v, adjacency)
    except ValueError as e:
        raise ValueError(f"Error while traversing second cavity: {e}")

    # Step 5: Order the Cavity Vertices in CCW Order
    cavity1_ordered = order_cavity_vertices(cavity1, vertices)
    cavity2_ordered = order_cavity_vertices(cavity2, vertices)

    return [cavity1_ordered, cavity2_ordered]

def collect_boundary_edges(intersected_triangles, triangles):
    """
    Collects boundary edges from the set of intersected triangles.

    Parameters:
    - intersected_triangles: Set of triangle indices that were intersected by s.
    - triangles: List of triangles.

    Returns:
    - A set of boundary edges (sorted tuples of vertex indices).
    """

    edge_count = defaultdict(int)

    for t_idx in intersected_triangles:
        tri = triangles[t_idx]
        if tri is None:
            continue  # Triangle has been deleted

        edges = [
            tuple(sorted((tri[0], tri[1]))),
            tuple(sorted((tri[1], tri[2]))),
            tuple(sorted((tri[2], tri[0])))
        ]

        for edge in edges:
            edge_count[edge] += 1

    # Boundary edges are those that appear exactly once in intersected triangles
    boundary_edges = set(edge for edge, count in edge_count.items() if count == 1)

    return boundary_edges

def build_adjacency_map(boundary_edges):
    """
    Builds an adjacency map from the set of boundary edges.

    Parameters:
    - boundary_edges: Set of boundary edges (sorted tuples of vertex indices).

    Returns:
    - A dictionary mapping each vertex to a set of its adjacent vertices via boundary edges.
    """
    adjacency = {}

    for edge in boundary_edges:
        a, b = edge
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)

    return adjacency

def traverse_path(start, end, adjacency):
    """
    Traverses from start to end using DFS and returns the path.

    Parameters:
    - start: Starting vertex index.
    - end: Ending vertex index.
    - adjacency: Adjacency dictionary.

    Returns:
    - An ordered list of vertex indices representing the path from start to end.

    Raises:
    - ValueError if no path exists.
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

def remove_edges_from_adjacency(path, adjacency):
    """
    Removes the edges of the given path from the adjacency map.

    Parameters:
    - path: An ordered list of vertex indices representing a path.
    - adjacency: Adjacency dictionary to be modified.

    Returns:
    - None. The adjacency map is modified in place.
    """
    for i in range(len(path) - 1):
        a, b = path[i], path[i+1]
        edge = tuple(sorted((a, b)))
        if b in adjacency.get(a, set()):
            adjacency[a].remove(b)
        if a in adjacency.get(b, set()):
            adjacency[b].remove(a)

def compute_polygon_orientation(polygon, vertices):
    """
    Computes the orientation of a polygon.

    Parameters:
    - polygon: An ordered list of vertex indices representing the polygon.
    - vertices: List of vertex coordinates.

    Returns:
    - "CCW" if the polygon is counterclockwise.
    - "CW" if the polygon is clockwise.
    - "COLINEAR" if the polygon is colinear.
    """
    area = 0.0
    n = len(polygon)

    for i in range(n):
        x1, y1 = vertices[polygon[i]]
        x2, y2 = vertices[polygon[(i + 1) % n]]
        area += (x1 * y2) - (x2 * y1)

    if area > 0:
        return "CCW"
    elif area < 0:
        return "CW"
    else:
        return "COLINEAR"

def order_cavity_vertices(cavity_path, vertices):
    """
    Orders the cavity vertices in counterclockwise order around the cavity.

    Parameters:
    - cavity_path: An ordered list of vertex indices representing the cavity path from u to v.
    - vertices: List of vertex coordinates.

    Returns:
    - An ordered list of vertex indices in CCW order, starting at u and ending at v.
      The last edge (v, u) forms the constraint edge.
    """
    # Create a closed loop by appending the start vertex to the end
    closed_loop = cavity_path + [cavity_path[0]]

    orientation = compute_polygon_orientation(closed_loop, vertices)

    if orientation == "CW":
        # If the loop is clockwise, reverse it to make it counterclockwise
        ordered_path = cavity_path[::-1]
    else:
        # If already counterclockwise or colinear, keep as is
        ordered_path = cavity_path

    return ordered_path

def walk_to_point_constrained(start_idx, target_idx, vertices, triangles, edge_to_triangle, vertex_to_triangle, start_triangle_idx, visualize=False):

    # Start by callking walk_to_point to find the initial triangle
    initial_bad_idx = walk_to_point(start_idx, target_idx, vertices, triangles, edge_to_triangle, vertex_to_triangle, start_triangle_idx, visualize)

    initial_bad = triangles[initial_bad_idx]

    # Generate the list of edges in the bad triangle 
    bad_edges = [sorted((initial_bad[0], initial_bad[1])), sorted((initial_bad[1], initial_bad[2])), sorted((initial_bad[2], initial_bad[0]))]

    for edge in bad_edges:
        if is_point_on_edge(target_idx, edge, vertices):
            # Get the other triangle that shares the edge from the edge_to_triangle
            triangles_idx = edge_to_triangle[tuple(edge)]
            # Get the triangle that is not the initial_bad triangle from triangles_idx, knowing an edge is shared by only two triangles
            other_bad_triangle_idx = triangles_idx[0] if triangles_idx[0] != initial_bad else triangles_idx[1]
            
            return [initial_bad_idx, other_bad_triangle_idx]
    
    return [initial_bad_idx]

def find_bad_triangles_constrained(initial_bads, u_idx, vertices, triangles, edge_to_triangle, constrained_edges_set, visualize=False):
    """
    Finds all the bad triangles whose circumcircles contain the new point u_idx,
    without crossing constrained edges.

    Parameters:
    - initial_bads: List of indices of initial bad triangles.
    - u_idx: Index of the point to be inserted.
    - vertices: List of vertex coordinates.
    - triangles: List of existing triangles.
    - edge_to_triangle: Dictionary mapping edges to triangle indices or tuples of triangle indices.
    - constrained_edges_set: Set of edges (as sorted tuples) that are constrained.
    - visualize: Boolean flag to indicate whether to visualize each step.

    Returns:
    - bad_triangles: Set of indices of bad triangles.
    """
    bad_triangles = set()
    stack = initial_bads  # Initialize stack with all initial bad triangles
    step_number = 0

    while stack:
        current_t_idx = stack.pop()
        if current_t_idx in bad_triangles:
            continue

        tri = triangles[current_t_idx]
        if tri is None:
            continue

        a, b, c = tri
        # The three points occur in counterclockwise order around the circle
        if orient(a, b, c, vertices) < 0:
            a, b = b, a

        step_number += 1
        if visualize:
            visualize_bad_triangles_step(vertices, triangles, bad_triangles, current_t_idx, u_idx, step_number)

        if in_circle(a, b, c, u_idx, vertices) > 0:

            bad_triangles.add(current_t_idx)


            # Retrieve neighbors without crossing constrained edges
            neighbors = get_triangle_neighbors_constrained(current_t_idx, triangles, edge_to_triangle, constrained_edges_set)


            for neighbor_idx in neighbors:

                if neighbor_idx != -1 and neighbor_idx not in bad_triangles:

                    stack.append(neighbor_idx)

    # Final visualization
    if visualize:
        visualize_bad_triangles_step(vertices, triangles, bad_triangles, None, u_idx, step_number + 1)

    return bad_triangles

def order_boundary_vertices_ccw(boundary_edges, vertices):
    """
    Orders the vertices from boundary_edges in counter-clockwise (CCW) order.

    Parameters:
    - boundary_edges: List of tuples representing edges (v1, v2).
    - vertices: List of vertex coordinates.

    Returns:
    - List of vertex indices ordered in CCW.
    """
    from collections import defaultdict

    # Step 1: Create adjacency map
    adjacency = defaultdict(list)
    for v1, v2 in boundary_edges:
        adjacency[v1].append(v2)
        adjacency[v2].append(v1)

    # Step 2: Traverse the boundary to order vertices
    # Start from a vertex with degree 2 (assuming a simple polygon)
    start_vertex = None
    for v, neighbors in adjacency.items():
        if len(neighbors) == 2:
            start_vertex = v
            break

    if start_vertex is None:
        raise ValueError("No suitable starting vertex found. The boundary might not form a simple polygon.")

    ordered_vertices = [start_vertex]
    current = start_vertex
    prev = None

    while True:
        neighbors = adjacency[current]
        # Choose the neighbor that's not the previous vertex
        next_vertices = [v for v in neighbors if v != prev]
        if not next_vertices:
            break  # Completed the loop
        next_vertex = next_vertices[0]
        if next_vertex == start_vertex:
            break  # Completed the loop
        ordered_vertices.append(next_vertex)
        prev, current = current, next_vertex

    # Step 3: Ensure CCW orientation
    if not is_ccw(ordered_vertices, vertices):
        ordered_vertices.reverse()

    return ordered_vertices

def is_ccw(ordered_vertices, vertices):
    """
    Determines if the ordered list of vertices is in counter-clockwise order.

    Parameters:
    - ordered_vertices: List of vertex indices ordered around the polygon.
    - vertices: List of vertex coordinates.

    Returns:
    - True if the vertices are ordered CCW, False otherwise.
    """
    area = 0.0
    n = len(ordered_vertices)
    for i in range(n):
        v_current = vertices[ordered_vertices[i]]
        v_next = vertices[ordered_vertices[(i + 1) % n]]
        area += (v_current[0] * v_next[1]) - (v_next[0] * v_current[1])
    return area > 0