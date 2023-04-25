from collections import defaultdict
from alphabet import sidechain_chemistry_per_residue, backbone_chemistry


def traverse(neighbor_dict, max_depth, depth=0, paths=None):
    """
    Recursive function: given a dictionary of neighbors, find all paths of depth `max_depth`
    """
    if paths is None:
        paths = list([[node] for node in neighbor_dict.keys()])
    if depth == max_depth - 1:
        return paths
    else:
        new_paths = []
        for path in paths:
            for neighbor in neighbor_dict[path[-1]]:
                if neighbor not in path:
                    new_path = path + [neighbor]
                    new_paths.append(new_path)
        return traverse(neighbor_dict, max_depth, depth=depth + 1, paths=new_paths)


for residue, chemistry in sidechain_chemistry_per_residue.items():
    bonds = backbone_chemistry["bonds"] + chemistry["bonds"]
    flippable = chemistry["flippable"]
    bonds_dict = defaultdict(list)
    for bond in bonds:
        bonds_dict[bond[0]].append(bond[1])
        bonds_dict[bond[1]].append(bond[0])

    print(f"{residue}=dict(")

    print(f"bonds={bonds}, ")
    print(f"flippable={flippable}, ")

    print("angles=[")
    unique_paths = set()
    for path in traverse(bonds_dict, max_depth=3):
        if tuple(path[::-1]) not in unique_paths:
            has_sidechain = False
            for atom in path:
                if not (atom in ["C", "O", "CA", "N"]):
                    has_sidechain = True
            if has_sidechain:
                unique_paths.add(tuple(path))
                print(f"{path},")
    print("],")
    print("dihedrals=[")
    unique_paths = set()
    for path in traverse(bonds_dict, max_depth=4):
        if tuple(path[::-1]) not in unique_paths:
            has_sidechain = False
            for atom in path:
                if not (atom in ["C", "O", "CA", "N"]):
                    has_sidechain = True
            if has_sidechain:
                unique_paths.add(tuple(path))
                print(f"{path},")
    print("],")

    print("), ")
