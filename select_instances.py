import random
import os

seed = 42
num_select = 2
num_total_select = 10
directory = './maxcut-instances'


def get_instances(seed=seed, amount=num_select, add_low=False, add_mid=False, add_high=False):
    instance_dict = select_instances(seed)
    instances = []
    
    for set_inst in instance_dict.keys():
        # per each number of vertices, get the corresponding files
        vertices = instance_dict[set_inst]
        print(f'Set {set_inst} has the vertices {sorted(vertices, key=lambda it: it[0])}')
        sorted_vertices= list(sorted(vertices, key=lambda it: it[0]))

        if add_low:
            vertex_amount_low = sorted_vertices[0][0]
            files_low = sorted_vertices[0][1]
            files_low = files_low[:amount]
            instances.append((vertex_amount_low, set_inst, files_low))
        if add_mid:
            vertex_amount_mid = sorted_vertices[len(sorted_vertices) // 2][0]
            files_mid = sorted_vertices[len(sorted_vertices) // 2][1]
            files_mid = files_mid[:amount]
            instances.append((vertex_amount_mid, set_inst, files_mid))
        if add_high:
            vertex_amount_high = sorted_vertices[len(sorted_vertices) - 1][0]
            files_high = sorted_vertices[len(sorted_vertices) - 1][1]
            files_high = files_high[:amount]
            instances.append((vertex_amount_high, set_inst, files_high))

    return instances


def select_instances(seed=seed):
    selection_random = random.Random(seed)
    instance_dict = {}
    for set_inst in [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]:
        # per each number of vertices, get the corresponding files
        vertices = {}
        for file in os.listdir(os.path.join(directory, set_inst)):
            if file.endswith('.txt'):
                joint_path = os.path.join(directory, set_inst, file)
                with open(joint_path, 'r') as f:
                    lines = f.read().split('\n')
                    v, e = map(int, lines[0].split())
                    if v not in vertices:
                        vertices[v] = []
                    vertices[v].append(file)
        print(f'Set {set_inst} has the vertices {sorted(list(vertices.keys()))}')

        vertices = list(sorted(vertices.items(), key=lambda it: it[0]))
        selected_vertices = []
        for v, files in vertices:
            selection_random.shuffle(files)
            files = files[:num_total_select]
            selected_vertices.append((v, files))

        instance_dict[set_inst] = vertices
    return instance_dict
         





if __name__ == "__main__":
    selection_random = random.Random(seed)
    for set_inst in [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]:
        # per each number of vertices, get the corresponding files
        vertices = {}
        for file in os.listdir(os.path.join(directory, set_inst)):
            if file.endswith('.txt'):
                joint_path = os.path.join(directory, set_inst, file)
                with open(joint_path, 'r') as f:
                    lines = f.read().split('\n')
                    v, e = map(int, lines[0].split())
                    if v not in vertices:
                        vertices[v] = []
                    vertices[v].append(file)
        print(f'Set {set_inst} has the vertices {sorted(list(vertices.keys()))}')

        vertices = list(sorted(vertices.items(), key=lambda it: it[0]))
        for v, files in vertices:
            selection_random.shuffle(files)
            files = files[:num_select]
            print(f'Vertices {v}: {files}')
