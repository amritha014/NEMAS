import networkx as nx
from networkx.algorithms import isomorphism

def print_tree_debug(name, tree):
    # print(f"\n===== {name} =====")
    # print("Nodes:")
    for node, data in tree.nodes(data=True):
        print(f"  {node} -> {data}")

    print("Edges:")
    for u, v, data in tree.edges(data=True):
        print(f"  {u} -> {v} -> {data}")


def compare_contour_trees(tree1, tree2, tree1_id="new_tree", tree2_id="existing_tree"):
    # print("\n\n================ CONTOUR TREE COMPARISON ================")
    # print(f"Comparing: {tree1_id}  VS  {tree2_id}")

    print_tree_debug(tree1_id, tree1)
    print_tree_debug(tree2_id, tree2)

    node_match = isomorphism.categorical_node_match("level", None)
    edge_match = isomorphism.categorical_edge_match("connection", None)

    result = nx.is_isomorphic(
        tree1,
        tree2,
        node_match=node_match,
        edge_match=edge_match
    )

    if result:
        print("\nRESULT: ISOMORPHIC")
        print("The contour trees are equal.")
    else:
        print("\nRESULT: NOT ISOMORPHIC")
        print("The contour trees are not equal.")

    print("========================================================\n")

    return result
