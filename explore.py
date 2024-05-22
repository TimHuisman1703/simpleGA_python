import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

set_inst = 'setE'
directory = os.path.join('./maxcut-instances', set_inst)
out_dir = './out'

edges = []
vertices = []
for file in os.listdir(directory):
    if file.endswith('.txt'):
        joint_path = os.path.join(directory, file)
        with open(joint_path, 'r') as f:
            lines = f.read().split('\n')
            v, e = map(int, lines[0].split())
            edges.append(e)
            vertices.append(v)

binwidth = 1
edges_set = sorted(list(set(edges)))
vertices_set = sorted(list(set(vertices)))
print(f'{set_inst}:')
for e in edges_set:
    print(f'Edges: {e}: {edges.count(e)}')
print()
for v in vertices_set:
    print(f'Vertices: {v}: {vertices.count(v)}')