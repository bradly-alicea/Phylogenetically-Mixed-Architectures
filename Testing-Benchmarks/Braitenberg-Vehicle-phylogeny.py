import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.special import expit # sigmoid
import torch
import random

# --- Helper functions for Braitenberg Vehicle definition and drawing (adapted from previous cells) ---

def make_reservoir(n_res=200, spectral_radius=0.9, sparsity=0.1, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 1.0, size=(n_res, n_res))
    mask = rng.random(size=W.shape) < sparsity
    W *= mask
    # scale to spectral radius
    eigs = np.linalg.eigvals(W)
    max_eig = max(abs(eigs)) if len(eigs) > 0 else 1.0
    if max_eig > 0:
        W *= (spectral_radius / max_eig)
    Win = rng.normal(0, 1.0, size=(n_res, 8)) # input dim 8 (random control)
    return W, Win

def run_reservoir(W, Win, inputs, leak=0.3):
    n_res = W.shape[0]
    x = np.zeros(n_res)
    states = []
    for u in inputs:
        pre = W.dot(x) + Win.dot(u)
        x = (1 - leak) * x + leak * np.tanh(pre)
        states.append(x.copy())
    return np.stack(states, axis=0) # (T, n_res)

def map_state_to_morph(state, rng=None):
    """
    Map a reservoir state vector to a Braitenberg vehicle morphology.
    Returns a dict with:
      - n_sensors (1..8)
      - n_motors (1..8)
      - sensor_positions: list of (x,y) on unit circle
      - motor_positions: list of (x,y) on unit circle
      - wiring: matrix shape (n_sensors, n_motors) of weights (positive excitatory, negative inhibitory)
      - body_radius: float
    """
    if rng is None:
        rng = np.random.default_rng(int(abs(state.sum()*1e6) % (2**31-1)))
    proj = np.tanh(state[:64]) # take first 64 dims

    # counts
    n_sensors = int(1 + np.clip(int(expit(proj[0]) * 7), 0, 7))
    n_motors = int(1 + np.clip(int(expit(proj[1]) * 7), 0, 7))

    # body radius
    body_radius = 0.25 + 0.25 * expit(proj[2])

    # positions around circle with jitter
    def positions(k, offset_seed):
        if k == 0: return []
        base_angles = np.linspace(0, 2*np.pi, k, endpoint=False)
        angles = []
        for i, a in enumerate(base_angles):
            jitter = (expit(proj[(3 + offset_seed + i) % len(proj)]) - 0.5) * 0.6
            angles.append(a + jitter)
        pts = [(0.5 + 0.35 * np.cos(a), 0.5 + 0.35 * np.sin(a)) for a in angles]
        return pts

    sensor_positions = positions(n_sensors, 0)
    motor_positions = positions(n_motors, 8)

    # wiring weights: use remaining proj entries to create weights
    wiring = np.zeros((n_sensors, n_motors), dtype=np.float32)
    idx = 12
    for i in range(n_sensors):
        for j in range(n_motors):
            val = proj[idx % len(proj)]
            # map to [-1.5, 1.5]
            w = (expit(val) - 0.5) * 3.0
            wiring[i, j] = w
            idx += 1
    # add small random perturbation
    wiring += (rng.normal(scale=0.05, size=wiring.shape))

    return {
        "n_sensors": n_sensors,
        "n_motors": n_motors,
        "sensor_positions": sensor_positions,
        "motor_positions": motor_positions,
        "wiring": wiring,
        "body_radius": body_radius
    }

def draw_vehicle(ax, morph, title=None, alpha=1.0):
    # body
    cx, cy = 0.5, 0.5
    r = morph["body_radius"]
    body = Circle((cx, cy), r, facecolor="#f0f0f0", edgecolor="#333333", linewidth=1.0, alpha=alpha)
    ax.add_patch(body)
    # sensors (red)
    for (sx, sy) in morph["sensor_positions"]:
        ax.plot(sx, sy, 'o', color="#d9534f", markersize=6, markeredgecolor='k', markeredgewidth=0.4, alpha=alpha)
    # motors (blue)
    for (mx, my) in morph["motor_positions"]:
        ax.plot(mx, my, 'o', color="#4f81bd", markersize=6, markeredgecolor='k', markeredgewidth=0.4, alpha=alpha)
    # wiring lines
    n_s, n_m = morph["wiring"].shape
    for i in range(n_s):
        sx, sy = morph["sensor_positions"][i]
        for j in range(n_m):
            mx, my = morph["motor_positions"][j]
            w = morph["wiring"][i, j]
            # line color: red for inhibitory (negative), green for excitatory (positive)
            if w >= 0:
                color = "#2ca02c"
            else:
                color = "#d62728"
            lw = 0.5 + 3.0 * (min(1.0, abs(w) / 1.5))
            ax.plot([sx, mx], [sy, my], color=color, linewidth=lw, alpha=alpha * 0.85)
    # optional title
    if title:
        ax.text(0.5, 0.02, title, ha='center', va='bottom', fontsize=7, alpha=alpha)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')


def draw_vehicle_thumbnail(ax, pos_xy, morph, scale=0.08, alpha=0.9, edgecolor='k'):
    """
    Draw a small thumbnail of the vehicle at normalized axes coordinates pos_xy=(x,y).
    scale controls thumbnail size relative to the original drawing size.
    """
    x_orig, y_orig = morph["sensor_positions"][0] if morph["n_sensors"] > 0 else (0.5,0.5)
    x_offset = pos_xy[0] - x_orig * scale
    y_offset = pos_xy[1] - y_orig * scale

    # Transform original drawing commands to the thumbnail space
    # Body
    r = morph["body_radius"] * scale
    body = Circle((x_offset + 0.5*scale, y_offset + 0.5*scale), r, facecolor="white", edgecolor=edgecolor, linewidth=0.5, alpha=alpha)
    ax.add_patch(body)

    # Sensors
    for (sx, sy) in morph["sensor_positions"]:
        ax.plot(x_offset + sx*scale, y_offset + sy*scale, 'o', color="#d9534f", markersize=3*scale/0.08, markeredgecolor='k', markeredgewidth=0.2, alpha=alpha)
    # Motors
    for (mx, my) in morph["motor_positions"]:
        ax.plot(x_offset + mx*scale, y_offset + my*scale, 'o', color="#4f81bd", markersize=3*scale/0.08, markeredgecolor='k', markeredgewidth=0.2, alpha=alpha)
    # Wiring
    n_s, n_m = morph["wiring"].shape
    for i in range(n_s):
        sx, sy = morph["sensor_positions"][i]
        for j in range(n_m):
            mx, my = morph["motor_positions"][j]
            w = morph["wiring"][i, j]
            color = "#2ca02c" if w >= 0 else "#d62728"
            lw = 0.2 + 1.5 * (min(1.0, abs(w) / 1.5)) * scale/0.08
            ax.plot([x_offset + sx*scale, x_offset + mx*scale], [y_offset + sy*scale, y_offset + my*scale], color=color, linewidth=lw, alpha=alpha * 0.7)


# --- New functions for phylogenetic tree ---

def generate_evolved_braitenberg_vehicles(num_generations=5, vehicles_per_generation=5, latent_dim=300, mutation_strength=0.1, initial_seed=42):
    rng_np = np.random.default_rng(initial_seed)
    torch.manual_seed(initial_seed)

    history_records = []
    reservoir_W, reservoir_Win = make_reservoir(n_res=latent_dim, seed=initial_seed)

    current_generation_latents = []
    current_generation_ids = []

    # Generation 0: Initial Ancestors
    for i in range(vehicles_per_generation):
        # Use reservoir to generate an initial diverse set of latents
        initial_input = rng_np.normal(size=(20, reservoir_Win.shape[1]))
        initial_states = run_reservoir(reservoir_W, reservoir_Win, initial_input)
        latent_state_np = initial_states[-1] # Take the final state
        latent_state_torch = torch.tensor(latent_state_np, dtype=torch.float32)

        morph = map_state_to_morph(latent_state_np, rng=rng_np)
        veh_id = f"gen0_veh{i}"
        history_records.append({
            "id": veh_id,
            "generation": 0,
            "latent": latent_state_torch,
            "decoded_morphology": morph,
            "parent_id": None
        })
        current_generation_latents.append(latent_state_torch)
        current_generation_ids.append(veh_id)

    # Subsequent Generations
    for gen in range(1, num_generations + 1):
        next_generation_latents = []
        next_generation_ids = []

        # For each new vehicle, select a parent from the previous generation
        for i in range(vehicles_per_generation):
            parent_idx = rng_np.choice(len(current_generation_latents))
            parent_latent = current_generation_latents[parent_idx]
            parent_id = current_generation_ids[parent_idx]

            # Mutate parent's latent to create child's latent
            mutation_noise = torch.randn_like(parent_latent) * mutation_strength
            child_latent = parent_latent + mutation_noise

            # Map child latent to morphology
            child_latent_np = child_latent.cpu().numpy()
            morph = map_state_to_morph(child_latent_np, rng=rng_np)

            veh_id = f"gen{gen}_veh{i}"
            history_records.append({
                "id": veh_id,
                "generation": gen,
                "latent": child_latent,
                "decoded_morphology": morph,
                "parent_id": parent_id
            })
            next_generation_latents.append(child_latent)
            next_generation_ids.append(veh_id)

        current_generation_latents = next_generation_latents
        current_generation_ids = next_generation_ids

    return history_records

def build_phylogenetic_tree(history_records):
    """
    Builds a directed graph representing the phylogenetic tree based on parent_id.
    """
    G = nx.DiGraph()
    for record in history_records:
        G.add_node(record["id"], **record) # Store all record info in the node attributes
        if record["parent_id"] is not None:
            G.add_edge(record["parent_id"], record["id"])
    return G

def plot_phylogenetic_tree_with_thumbnails(G, figsize=(15, 10), save_path=None):
    """
    Plots the phylogenetic tree with vehicle thumbnails at each node.
    """
    plt.figure(figsize=figsize)
    ax = plt.gca()
    ax.set_title("Phylogenetic Tree of Braitenberg Vehicle Evolution")

    # Use a hierarchical layout, e.g., Sugiyama layout (graphviz dot)
    # It's better to use a layout that respects the directed nature and generations
    # For this, we'll try to explicitly arrange nodes by generation

    generations = sorted(list(set(nx.get_node_attributes(G, 'generation').values())))
    if not generations: # Handle empty graph case
        print("No generations found in graph.")
        plt.axis('off')
        plt.show()
        return

    pos = {} # Dictionary to store node positions
    # Calculate positions based on generation (y-axis) and index within generation (x-axis)
    y_spacing = 1.0 / (len(generations) + 1)

    for i, gen in enumerate(generations):
        gen_nodes = [n for n, data in G.nodes(data=True) if data.get('generation') == gen]
        gen_nodes.sort() # Consistent ordering for horizontal placement
        x_spacing = 1.0 / (len(gen_nodes) + 1)
        for j, node_id in enumerate(gen_nodes):
            pos[node_id] = np.array([(j + 1) * x_spacing, 1.0 - (i + 1) * y_spacing])

    # If some nodes still don't have positions (e.g., disconnected or error), use spring_layout as fallback
    if len(pos) < G.number_of_nodes():
        remaining_nodes = [n for n in G.nodes() if n not in pos]
        if remaining_nodes:
            # Create a subgraph for remaining nodes and get their layout
            subgraph_for_layout = G.subgraph(remaining_nodes)
            fallback_pos = nx.spring_layout(subgraph_for_layout, k=0.5, iterations=50, seed=42)
            # Scale fallback positions to fit within overall layout if needed
            for n, p in fallback_pos.items():
                pos[n] = p # Simply add them, might overlap with structured layout

    # Draw edges first
    nx.draw_networkx_edges(G, pos, ax=ax, arrowstyle='->', arrowsize=10, edge_color='gray', alpha=0.6)

    # Draw thumbnails and labels
    for node_id, p in pos.items():
        node_data = G.nodes[node_id]
        morph = node_data["decoded_morphology"]
        label = f"Gen {node_data['generation']}\n{node_id}"

        # Draw thumbnail
        draw_vehicle_thumbnail(ax, p, morph, scale=0.07, alpha=0.9)

        # Draw label (text)
        ax.text(p[0], p[1] - 0.04, label, ha='center', va='top', fontsize=6, color='black', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


# --- Main Execution ---
if __name__ == "__main__":
    # Generate evolved vehicles
    print("Generating evolved Braitenberg vehicles...")
    evolved_vehicles = generate_evolved_braitenberg_vehicles(
        num_generations=4,
        vehicles_per_generation=4,
        latent_dim=300,
        mutation_strength=0.15,
        initial_seed=100
    )
    print(f"Generated {len(evolved_vehicles)} vehicles across generations.")

    # Build the phylogenetic tree
    print("Building phylogenetic tree...")
    phylogenetic_tree = build_phylogenetic_tree(evolved_vehicles)
    print(f"Tree has {phylogenetic_tree.number_of_nodes()} nodes and {phylogenetic_tree.number_of_edges()} edges.")

    # Plot the tree with thumbnails
    print("Plotting phylogenetic tree with thumbnails...")
    plot_phylogenetic_tree_with_thumbnails(phylogenetic_tree, figsize=(14, 10), save_path='phylogenetic_tree.png')
    print("Done.")
