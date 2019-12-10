import os


def get_id_maps(id_path):
    id_map = {}
    with open(id_path, 'r') as handle:
        for idx, name in handle:
            id_map[idx] = name
    return id_map