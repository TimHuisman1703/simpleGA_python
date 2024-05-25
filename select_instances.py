import random
import os

seed = 42
num_select = 2
directory = './maxcut-instances'

random.seed(seed)

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
        random.shuffle(files)
        files = files[:num_select]
        print(f'Vertices {v}: {files}')
