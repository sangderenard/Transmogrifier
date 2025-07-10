#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# --------------------------------
# Physics-based spring layout engine
# --------------------------------
def spring_layout_step(nodes, edges, k=0.1, damping=0.9, rest_length=1.0, dt=0.1):
    # Reset forces
    for node in nodes.values():
        node['force'][:] = 0

    # Apply spring forces on edges
    for a, b in edges:
        delta = nodes[b]['pos'] - nodes[a]['pos']
        dist = np.linalg.norm(delta)
        if dist == 0:
            continue
        direction = delta / dist
        force = k * (dist - rest_length) * direction
        nodes[a]['force'] += force
        nodes[b]['force'] -= force

    # Integrate motion
    for node in nodes.values():
        node['vel'] += node['force'] * dt
        node['vel'] *= damping
        node['pos'] += node['vel'] * dt

# --------------------------------
# Initialize positions randomly on a sphere
# --------------------------------
def initialize_positions_on_sphere(nodes, radius=5):
    for node_data in nodes.values():
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.arccos(np.random.uniform(-1,1))
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        node_data['pos'] = np.array([x, y])
        node_data['vel'] = np.zeros(2)
        node_data['force'] = np.zeros(2)

# --------------------------------
# Main driver
# --------------------------------
def main():
    # Load or create a networkx graph
    G = nx.erdos_renyi_graph(20, 0.1)  # e.g. 20 nodes, sparse random

    # Build node storage
    nodes = {n: {} for n in G.nodes()}
    edges = list(G.edges())

    # Random spherical initial positions
    initialize_positions_on_sphere(nodes)

    # Setup matplotlib interactive
    plt.ion()
    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_aspect('equal')

    # Main loop
    for step in range(500):
        spring_layout_step(nodes, edges, k=0.2, damping=0.95, rest_length=2.0, dt=0.2)

        ax.clear()
        ax.set_xlim(-10,10)
        ax.set_ylim(-10,10)

        # Draw edges
        for a, b in edges:
            pa = nodes[a]['pos']
            pb = nodes[b]['pos']
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], 'gray', lw=0.8)

        # Draw nodes
        positions = np.array([nodes[n]['pos'] for n in G.nodes()])
        ax.scatter(positions[:,0], positions[:,1], color='blue', s=30)

        ax.set_title(f"Physics Graph Layout, step {step}")
        plt.pause(0.01)

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
