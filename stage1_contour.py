import os
import pickle
from collections import namedtuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.lines import Line2D
from shapely.geometry import LineString, Point, Polygon

ContourTreeResult = namedtuple('ContourTreeResult', ['LEVEL_INDEX', 'DIRECTED_EDGES', 'UNDIRECTED_EDGES'])

def create_contour_tree(grid):
    rows, cols = grid.shape

    G = nx.Graph()

    # Add nodes to the graph
    for i in range(rows):
        for j in range(cols):
            node_id = i * cols + j
            G.add_node(node_id, label=grid[i][j])

    # Add edges to the graph based on horizontal and vertical adjacency and equal values
    for i in range(rows):
        for j in range(cols):
            node_id = i * cols + j
            
            # Check adjacent nodes (horizontally and vertically only)
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                ni, nj = i + dx, j + dy
                
                # Check if the adjacent node is within the grid bounds
                if 0 <= ni < rows and 0 <= nj < cols:
                    adjacent_node_id = ni * cols + nj
                    
                    # Add an edge if the values are equal
                    if grid[i][j] == grid[ni][nj]:
                        G.add_edge(node_id, adjacent_node_id)

    # Set up the plot
    # Set up the plot with extra space for the legend
    # plt.figure(figsize=(5, 5))
    pos = {node_id: (j, -i) for i in range(rows) for j in range(cols) for node_id in [i * cols + j]}

    # Define colors for H, L, and M
    color_map = {'H': 'cyan', 'L': '#ffe900', 'M': 'red'}

    # Assign colors and sizes to nodes
    node_colors = [color_map[G.nodes[node]['label']] for node in G.nodes()]
    node_sizes = [100 + 50 * len(list(G.neighbors(node))) for node in G.nodes()]

    # Draw the graph
    nx.draw(G, pos, node_color=node_colors, node_size=node_sizes, 
            with_labels=False, edge_color='gray', width=1.0, alpha=0.7)

    # Set background color
    plt.gca().set_facecolor('#f0f0f0')

    # # Add labels to the nodes
    labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_labels(G, pos, labels, font_size=8)

    # # Create legend
    # legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[label], markersize=10, label=label) for label in color_map]
    # # plt.legend(handles=legend_elements, loc='upper right', title="Labels")
    # plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), title="Labels")


    # Show the plot
    plt.title("Grid Graph Visualization", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.subplots_adjust(right=0.8)  # Adjust the right margin to make space for the legend
    # plt.show()

    # # Find connected components
    connected_components = list(nx.connected_components(G))

    # Initialize dictionaries
    level_index = {}
    level_cells = {}

    # Process connected components
    for component in connected_components:
        # Determine the level of the component
        node_id = next(iter(component))
        level = G.nodes[node_id]['label']
        
        # Initialize level index if not already done
        if level not in level_index:
            level_index[level] = []
        
        # Get the index for this component
        component_index = len(level_index[level])
        level_index[level].append(component_index)
        
        # Get the cell coordinates for this component
        cell_coords = {(node_id // cols, node_id % cols) for node_id in component}
        
        # Store the component's coordinates
        level_cells[(level, component_index)] = cell_coords


    enclosure_comps = {}
    for key, cells in level_cells.items():
        if len(cells) >= 3:
            polygon = Polygon(cells)
            enclosure_comps[key] = polygon.convex_hull
        elif len(cells) == 2:
            enclosure_comps[key] = LineString(cells)
        else:
            enclosure_comps[key] = Point(cells[0])

    # Compute directed edges based on enclosure
    directed_edges = set()

    for key1, enclcomp1 in enclosure_comps.items():
        for key2, enclcomp2 in enclosure_comps.items():
            if key1 != key2 and enclcomp1.contains(enclcomp2):
                directed_edges.add((key1, key2))


    def compute_undirected_edges(level_index, level_cells):
        level_order = ['L', 'M', 'H']

        # Get the unique levels and sort them based on the desired order
        # print(level_index)
        # print(level_cells)
        levels = sorted(set(level for level, _ in level_cells.keys()), key=lambda x: level_order.index(x))
        undirected_edges = set()

        def are_adjacent(cells1, cells2):
            return any(abs(cell1[0] - cell2[0]) + abs(cell1[1] - cell2[1]) == 1 
                    for cell1 in cells1 for cell2 in cells2)

        def find_next_nearest_highest_nodes(node, levels):
            current_level = node[0]
            current_cells = level_cells[node]

            next_nearest_nodes = set()

            current_level_index = levels.index(current_level)
            if current_level_index < len(levels) - 1:
                next_highest_level = levels[current_level_index + 1]
                for index in level_index[next_highest_level]:
                    higher_node = (next_highest_level, index)
                    higher_cells = level_cells[higher_node]
                    if are_adjacent(current_cells, higher_cells):
                        next_nearest_nodes.add(higher_node)

            return next_nearest_nodes

        # Check adjacency for all pairs of components
        for i, level1 in enumerate(levels):
            for index1 in level_index[level1]:
                node1 = (level1, index1)
                cells1 = level_cells[node1]
                next_nearest_nodes = find_next_nearest_highest_nodes(node1, levels)

                for higher_node in next_nearest_nodes:
                    undirected_edges.add((node1, higher_node))

        return undirected_edges
    
    # Compute undirected edges
    undirected_edges = compute_undirected_edges(level_index, level_cells)
    plt.close('all')

    return ContourTreeResult(level_index, directed_edges, undirected_edges)

def draw_contour_tree(level_index, directed_edges, undirected_edges, title):
    # Create a new graph
    G = nx.MultiDiGraph()  # Use MultiDiGraph instead of DiGraph

    # Add nodes
    for level, indices in level_index.items():
        for index in indices:
            G.add_node((level, index), level=level)

    # Add directed edges
    for edge in directed_edges:
        G.add_edge(edge[0], edge[1], color='black', style='dotted', connection='directed')

    # Add undirected edges
    for edge in undirected_edges:
        G.add_edge(edge[0], edge[1], color='red', style='solid', connection='undirected')
        G.add_edge(edge[1], edge[0], color='red', style='solid', connection='undirected')

    # Set up the plot
    plt.figure(figsize=(8, 5))

    # Use spring layout for node positioning
    pos = nx.spring_layout(G)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=['skyblue' if node[0] == 'H' else 'lightgreen' for node in G.nodes()], 
                        node_size=3000, alpha=0.8)

    # Draw node labels
    nx.draw_networkx_labels(G, pos, {node: f"{node[0]}{node[1]}" for node in G.nodes()}, font_size=10)

    # Draw edges
    for (u, v, data) in G.edges(data=True):
        if data['connection'] == 'directed':
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], 
                                arrows=True, 
                                connectionstyle="arc3,rad=0.2",
                                style=data['style'],
                                edge_color=data['color'],
                                arrowsize=20)
        else:
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], 
                                arrows=False, 
                                connectionstyle="arc3,rad=0.2",
                                style=data['style'],
                                edge_color=data['color'])

    # Add the title using the 'title' parameter
    plt.title(title, fontsize=16)

    # Remove axis
    plt.axis('off')

    # Show the plot
    plt.tight_layout()
    # plt.show()

def process_and_save_grid(name, grid, output_dir):
    # print(f"\nProcessing {name} grid:")
    contour_tree = create_contour_tree(grid)
    # print("Level Index:", contour_tree.LEVEL_INDEX)
    # print(" Edges:", contour_tree.DIRECTED_EDGES)
    # print("Undirected Edges:", contour_tree.UNDIRECTED_EDGES)
    
    # Create a MultiDiGraph from the contour tree data
    G = nx.MultiDiGraph()
    for level, indices in contour_tree.LEVEL_INDEX.items():
        for index in indices:
            G.add_node((level, index), level=level)
    for edge in contour_tree.DIRECTED_EDGES:
        G.add_edge(edge[0], edge[1], connection='directed')
    for edge in contour_tree.UNDIRECTED_EDGES:
        G.add_edge(edge[0], edge[1], connection='undirected')
        G.add_edge(edge[1], edge[0], connection='undirected')

    # Save the graph and namedtuple
    # output_file = os.path.join(output_dir, f"{name}_contour_tree.pkl")
    # with open(output_file, 'wb') as f:
    #     pickle.dump((G, contour_tree), f)
    
    # print(f"Saved contour tree to {output_file}")

    # Draw and save the graph
    # draw_contour_tree(contour_tree.LEVEL_INDEX, contour_tree.DIRECTED_EDGES, contour_tree.UNDIRECTED_EDGES, f"Contour Tree for {name.capitalize()} Grid")
    # plt.savefig(os.path.join(output_dir, f"{name}_contour_tree.png"))
    # plt.show()
    # plt.close()
    return(G, contour_tree)

def process_multiple_grids(grids, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for name, grid in grids.items():
        (G, contour_tree) = process_and_save_grid(name, grid, output_dir)
        return (G, contour_tree)
    

def gen_contour(threshold_regA,threshold_regB):
    # Define multiple grids
    grids = {
        "threshold_regA": threshold_regA,
        "threshold_regB": threshold_regB
    }
   
    output_directory = "contour_trees"
    G, contour_tree = process_multiple_grids(grids, output_directory)
    return(G, contour_tree)
# Example usage:
if __name__ == "__main__":
    # Define multiple grids
    grids = {
        "extended": np.array(
                    [['H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'],
                    ['H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'],
                    ['H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'],
                    ['H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'],
                    ['H', 'H', 'H', 'H', 'L', 'L', 'M', 'H', 'H', 'H'],
                    ['H', 'H', 'H', 'H', 'L', 'L', 'M', 'H', 'H', 'H'],
                    ['H', 'H', 'H', 'H', 'L', 'L', 'M', 'H', 'H', 'H'],
                    ['H', 'H', 'H', 'H', 'L', 'L', 'M', 'H', 'H', 'H'],
                    ['H', 'H', 'H', 'H', 'H', 'L', 'H', 'H', 'H', 'H'],
                    ['H', 'H', 'H', 'H', 'H', 'L', 'H', 'H', 'H', 'H']]),
        "zigzag": np.array([['H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'],
                            ['H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'],
                            ['H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'],
                            ['H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'],
                            ['H', 'H', 'H', 'H', 'H', 'L', 'L', 'L', 'H', 'H'],
                            ['H', 'H', 'H', 'H', 'H', 'L', 'H', 'L', 'H', 'H'],
                            ['H', 'H', 'H', 'L', 'L', 'L', 'H', 'M', 'H', 'H'],
                            ['H', 'H', 'H', 'M', 'H', 'H', 'H', 'M', 'H', 'H'],
                            ['H', 'H', 'H', 'M', 'M', 'M', 'M', 'M', 'H', 'H'],
                            ['H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H']]),
        "around":np.array( 
                    [['H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'],
                    ['H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'],
                    ['H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'],
                    ['H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'],
                    ['H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'],
                    ['H', 'H', 'M', 'L', 'L', 'H', 'H', 'H', 'H', 'H'],
                    ['H', 'H', 'M', 'H', 'L', 'H', 'H', 'H', 'H', 'H'],
                    ['H', 'H', 'M', 'H', 'L', 'H', 'H', 'H', 'H', 'H'],
                    ['H', 'H', 'M', 'M', 'L', 'H', 'H', 'H', 'H', 'H'],
                    ['H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H']]),

        "parallel":np.array(
                    [['H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'],
                    ['H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'],
                    ['H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'],
                    ['H', 'L', 'L', 'L', 'L', 'H', 'H', 'H', 'H', 'H'],
                    ['H', 'L', 'H', 'H', 'L', 'H', 'H', 'H', 'H', 'H'],
                    ['H', 'L', 'H', 'H', 'L', 'H', 'H', 'H', 'H', 'H'],
                    ['H', 'L', 'L', 'L', 'L', 'H', 'H', 'H', 'H', 'H'],
                    ['H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'],
                    ['H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'],
                    ['H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H']])
    }


    output_directory = "contour_trees"
    process_multiple_grids(grids, output_directory)

    # Example of how to load and use the saved data
    print("\nExample of loading and using saved data:")
    with open(os.path.join(output_directory, "extended_contour_tree.pkl"), 'rb') as f:
        loaded_graph, loaded_contour_tree = pickle.load(f)
    
    print("Loaded Graph Nodes:", loaded_graph.nodes())
    print("Loaded Contour Tree Level Index:", loaded_contour_tree.LEVEL_INDEX)

