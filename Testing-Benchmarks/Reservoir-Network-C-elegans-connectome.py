import numpy as np
import networkx as nx
import plotly.graph_objects as go

# -----------------------------
# 1. Reservoir construction
# -----------------------------
np.random.seed(42)

n_res = 302          # number of neurons
spectral_radius = 0.9
density = 0.03       # sparsity of the connectome
input_dim = 1

# Random sparse recurrent weight matrix (generic connectome)
W = np.zeros((n_res, n_res))
mask = np.random.rand(n_res, n_res) < density
W[mask] = np.random.uniform(-1, 1, size=np.count_nonzero(mask))

# Normalize to desired spectral radius
eigvals = np.linalg.eigvals(W)
current_radius = max(abs(eigvals)) if np.count_nonzero(W) > 0 else 1.0
if current_radius > 0:
    W *= spectral_radius / current_radius

# Input weights
Win = np.random.uniform(-1, 1, size=(n_res, input_dim)) * 0.5

# -----------------------------
# 2. Run reservoir dynamics
# -----------------------------
T = 200  # timesteps
x = np.zeros((n_res,))  # reservoir state

# simple input: sine wave
t = np.linspace(0, 4 * np.pi, T)
u = np.sin(t).reshape(-1, 1)

states = []

for k in range(T):
    x = np.tanh(W @ x + Win @ u[k])
    states.append(x.copy())

states = np.array(states)  # shape: (T, n_res)

# -----------------------------
# 3. Build graph layout
# -----------------------------
G = nx.from_numpy_array((W != 0).astype(int), create_using=nx.DiGraph)

# Use a 2D layout for visualization
pos = nx.spring_layout(G, seed=42, dim=2)  # dict: node -> (x, y)
node_x = []
node_y = []
for i in range(n_res):
    node_x.append(pos[i][0])
    node_y.append(pos[i][1])

# Edge coordinates
edge_x = []
edge_y = []
for i, j in G.edges():
    edge_x.append(pos[i][0])
    edge_y.append(pos[i][1])
    edge_x.append(pos[j][0])
    edge_y.append(pos[j][1])
    edge_x.append(None)
    edge_y.append(None)

# -----------------------------
# 4. Prepare animation frames
# -----------------------------
# Normalize activities for color mapping
min_act = states.min()
max_act = states.max()
eps = 1e-9

def normalize(a):
    return (a - min_act) / (max_act - min_act + eps)

frames = []
for k in range(T):
    act_norm = normalize(states[k])
    frame = go.Frame(
        data=[
            # nodes
            go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers",
                marker=dict(
                    size=6,
                    color=act_norm,
                    colorscale="Viridis",
                    cmin=0,
                    cmax=1,
                    showscale=True,
                    colorbar=dict(title="Activity")
                ),
                hoverinfo="text",
                text=[f"Neuron {i}<br>Activity: {states[k, i]:.3f}" for i in range(n_res)],
            ),
            # edges (static)
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line=dict(width=0.5, color="rgba(150,150,150,0.4)"),
                hoverinfo="none",
            ),
        ],
        name=str(k),
    )
    frames.append(frame)

# -----------------------------
# 5. Create interactive figure
# -----------------------------
# Initial frame (k=0)
init_act = normalize(states[0])

node_trace = go.Scatter(
    x=node_x,
    y=node_y,
    mode="markers",
    marker=dict(
        size=6,
        color=init_act,
        colorscale="Viridis",
        cmin=0,
        cmax=1,
        showscale=True,
        colorbar=dict(title="Activity"),
    ),
    hoverinfo="text",
    text=[f"Neuron {i}<br>Activity: {states[0, i]:.3f}" for i in range(n_res)],
)

edge_trace = go.Scatter(
    x=edge_x,
    y=edge_y,
    mode="lines",
    line=dict(width=0.5, color="rgba(150,150,150,0.4)"),
    hoverinfo="none",
)

fig = go.Figure(
    data=[node_trace, edge_trace],
    layout=go.Layout(
        title="Reservoir Network (302 neurons) – Activity over Time",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=50, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(frame=dict(duration=0, redraw=False),
                                 mode="immediate",
                                 transition=dict(duration=0)),
                        ],
                    ),
                ],
            )
        ],
    ),
    frames=frames,
)

fig.show()
