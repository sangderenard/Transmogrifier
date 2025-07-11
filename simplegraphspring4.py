import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import networkx as nx
import graph_express2
from graph_express2 import ProcessGraph, test_suite
from orbital import Orbit
import torch
import numpy as np
import random
from collections import deque
import colorsys

# Debugging mode flag: toggles fixed camera and axis visualization
DEBUG_MODE = False

VERTEX_SHADER = """
#version 330
in vec3 in_position;
in vec3 in_color;
in float in_alpha;
in float in_radius;
in float in_ke;

out vec3 frag_color;
out float frag_alpha;
out float frag_ke;
uniform mat4 u_mvp;
uniform float u_time;

mat3 rotX(float a) {
    float c = cos(a), s = sin(a);
    return mat3(1,0,0,  0,c,-s,  0,s,c);
}
mat3 rotY(float a) {
    float c = cos(a), s = sin(a);
    return mat3(c,0,s,  0,1,0,  -s,0,c);
}
mat3 rotZ(float a) {
    float c = cos(a), s = sin(a);
    return mat3(c,-s,0, s,c,0, 0,0,1);
}


void main() {
    // apply combined MVP passed from CPU
    float t = u_time * 0.0001; 
    vec3 rotated = rotX(t) * rotY(0.8*t) * rotZ(1.5*t) * in_position;
    gl_Position = u_mvp * vec4(rotated, 1.0);

    gl_PointSize = in_radius * 20.0 + in_ke * 5.0; // customizable
    frag_color = in_color;
    frag_alpha = in_alpha;
    frag_ke = in_ke;
}
"""

FRAGMENT_SHADER = """
#version 330
in vec3 frag_color;
in float frag_alpha;
in float frag_ke;

out vec4 out_color;

void main() {
    vec2 coord = gl_PointCoord - vec2(0.5);
    float dist = length(coord);
    if (dist > 0.5) discard;
    // smooth edge for anti-aliasing
    float alpha = frag_alpha * smoothstep(0.5, 0.45, dist);
    out_color = vec4(frag_color, alpha);
}
"""

def compile_shader(source, shader_type):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        raise RuntimeError(glGetShaderInfoLog(shader).decode())
    return shader

def create_program(vertex_src, fragment_src):
    program = glCreateProgram()
    vs = compile_shader(vertex_src, GL_VERTEX_SHADER)
    fs = compile_shader(fragment_src, GL_FRAGMENT_SHADER)
    glAttachShader(program, vs)
    glAttachShader(program, fs)
    glLinkProgram(program)
    if not glGetProgramiv(program, GL_LINK_STATUS):
        raise RuntimeError(glGetProgramInfoLog(program).decode())
    return program


WIDTH, HEIGHT = 800, 600
NODE_RADIUS = .15
SPEED_FACTOR = 1.0  # slow down simulation speed for visibility
FPS = 60
SPRING_K, REPULSION_K, DAMPING = 0.1, 0.01, 00.970
EDGE_BASE_WIDTH, EDGE_GLOW_WIDTH = 2, 2
GLOW_RISE, GLOW_DECAY = 0.3, 0.05
GLOW_PEAK_ALPHA, GLOW_FLOOR_ALPHA = 1.0, 0.1
GLOW_PEAK_RADIUS, GLOW_FLOOR_RADIUS = 0.2, .10
BETA_LEVEL, BETA_TYPE, BETA_ROLE, ALPHA_IDLE = 0.5, 0.7, 0.9, 0.1
LEVEL_TARGET_FACTOR, TYPE_TARGET_FACTOR, ROLE_TARGET_FACTOR = 0.9, 0.75, 0.5
CAMERA_DISTANCE, FOCAL_LENGTH = 100, 50
DEFAULT_EDGE_LENGTH = 10.0
CANVAS_COLOR = (0.1, 0.1, 0.1, 1.0)  # Background color
EDGE_ALPHA = 0.3  # default edge alpha blending

# Parameters for state buffering and best state tracking
BUFFER_SIZE = 300  # number of recent frames to keep
ACTIVE_EDGE_COUNT = 5  # number of top active edges to consider per frame
GHOST_COUNT = 50  # number of ghosts to keep in rainbow trail
GHOST_INIT_ALPHA = 0.5
GHOST_MIN_DECAY = 0.90
GHOST_MAX_DECAY = 0.99
GHOST_PEAK_RADIUS = 0.3  # radius for ghost nodes
# Rainbow trail config
ENABLE_RAINBOW_TRAIL = True  # toggle rainbow ghost trail
TRAIL_COUNT = GHOST_COUNT
TRAIL_PRUNE_THRESHOLD = 0.01  # minimum alpha before pruning
TRAIL_MODE = "trail"
BEST_MODE  = "best"
RAINBOW_MODE = TRAIL_MODE  # set to TRAIL_MODE or BEST_MODE


BEST_SCALES = [10, 50, 250, 1000, 5000]  # or as desired, powers-of-n
BEST_TIMEOUT = 50   # Don't admit bests before this many frames
# Rolling-state buffers (must be defined at module scope for full_reset to clear)
from collections import deque

def project_positions(positions):
    # no longer used: leave 3D positions for OpenGL
    return positions

def load_graph(dataG):
    init_pos = nx.spring_layout(dataG, scale=1.0)
    nodes = list(dataG.nodes)
    pos_array = fibonacci_sphere(len(nodes), radius=10.0)

    velocities = torch.zeros_like(pos_array)
    forces = torch.zeros_like(pos_array)
    glow_alpha = torch.full((len(nodes),1), GLOW_FLOOR_ALPHA)
    glow_radius = torch.full((len(nodes),1), GLOW_FLOOR_RADIUS)
    return nodes, pos_array, velocities, forces, glow_alpha, glow_radius

def build_edges(nodes, dataG):
    edges = []
    rest_lengths = []
    for u,v in dataG.edges:
        edges.append([nodes.index(u), nodes.index(v)])
        rest_lengths.append(DEFAULT_EDGE_LENGTH)
    return torch.tensor(edges, dtype=torch.long), torch.tensor(rest_lengths, dtype=torch.float32)

def setup_vbo():
    vbo = glGenBuffers(1)
    return vbo

def update_vbo(vbo, data):
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_DYNAMIC_DRAW)
def fibonacci_sphere(samples=100, radius=10.0):
    points = []
    offset = 2.0 / samples
    increment = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(samples):
        y = ((i * offset) - 1) + (offset / 2)
        r = np.sqrt(1 - y*y)
        phi = i * increment
        x = np.cos(phi) * r
        z = np.sin(phi) * r
        points.append((x * radius, y * radius, z * radius))
    return torch.tensor(points, dtype=torch.float32)

def main():
    
    # BUFFER_SIZE is already defined above
    state_buffer   = deque(maxlen=BUFFER_SIZE)
    best_score     = float('inf')
    ghost_configs  = []

    # TRAIL_LENGTH must match what you use inside main()
    TRAIL_LENGTH   = 24
    rainbow_trail  = deque(maxlen=TRAIL_LENGTH)

    # VBO handles (initialized later via setup_vbo)
    vbo_nodes        = None
    vbo_node_colors  = None
    vbo_edges        = None
    vbo_edge_colors  = None
    vbo_ghosts       = None

    decay_speeds = np.linspace(GHOST_MAX_DECAY, GHOST_MIN_DECAY, GHOST_COUNT)
    rainbow_colors = [colorsys.hsv_to_rgb(h, 1.0, 1.0)
                      for h in np.linspace(0, 1, GHOST_COUNT, endpoint=False)]
    # For trail mode: a fixed-length deque of past frame positions
    TRAIL_LENGTH = 24
    rainbow_trail = deque(maxlen=TRAIL_LENGTH)
    trail_colors = [colorsys.hsv_to_rgb(h, 1.0, 1.0) for h in np.linspace(0, 1, TRAIL_LENGTH, endpoint=False)]

    # For best mode: as before, but persistent, not fading
    BEST_SCALES = [10, 50, 250, 1000, 5000]
    best_per_scale = [None] * len(BEST_SCALES)
    hues = np.linspace(0, 1, len(BEST_SCALES), endpoint=False)
    rainbow_best_colors = [colorsys.hsv_to_rgb(h, 1, 1) for h in hues]

    # Select demo from graph_express2.test_suite
    demo = test_suite[-1]
    expr = demo.get('expr_fn') or demo
    recomb = 0

    pg = ProcessGraph(recomb)
    pg.build_from_expression(expr, *demo.get('dims',(1,)))
    pg.compute_levels(method='alap')
    dataG, grouped = pg.dataG, pg.group_edges_by_dataset(pg.dataG)
    ordered_keys = [(lvl, typ, role)
        for lvl in sorted(grouped)
        for typ in sorted(grouped[lvl])
        for role in sorted(grouped[lvl][typ])]
    
    best_per_scale = [None] * len(BEST_SCALES)  # stores dicts: {'score', 'frame', 'positions'}
    best_per_scale_trail = [None] * len(BEST_SCALES)  # for rainbow ghosts
    hues = np.linspace(0, 1, len(BEST_SCALES), endpoint=False)
    rainbow_colors = [colorsys.hsv_to_rgb(h, 1, 1) for h in hues]


    def edges_at_level(l): return set.union(*(set(grouped[l][t][r]) for t in grouped[l] for r in grouped[l][t]))
    def edges_at_type(l,t): return set.union(*(set(grouped[l][t][r]) for r in grouped[l][t]))
    def edges_at_role(r): return set.union(*(set(grouped[l][t].get(r,[])) for l in grouped for t in grouped[l]))

    # Load graph
    nodes, positions, velocities, forces, glow_alpha, glow_radius = load_graph(dataG)
    edges, base_lengths = build_edges(nodes, dataG)
    edge_rest_lengths = base_lengths.clone()
    N, E = len(nodes), len(edges)

    # Precompute role masks
    edge_keys = [(nodes[u],nodes[v]) for u,v in edges]
    lvl_mask = torch.zeros((len(ordered_keys), E))
    typ_mask = torch.zeros_like(lvl_mask)
    role_mask= torch.zeros_like(lvl_mask)
    node_lvl = torch.zeros((len(ordered_keys),N))
    node_typ = torch.zeros_like(node_lvl)
    node_role= torch.zeros_like(node_lvl)
    for idx,(lvl,typ,role) in enumerate(ordered_keys):
        lvl_set, typ_set, role_set = edges_at_level(lvl), edges_at_type(lvl,typ), set(grouped[lvl][typ].get(role,[]))
        for e_idx,key in enumerate(edge_keys):
            if key in lvl_set: lvl_mask[idx][e_idx]=1
            if key in typ_set: typ_mask[idx][e_idx]=1
            if key in role_set:role_mask[idx][e_idx]=1
        for n_idx,nid in enumerate(nodes):
            if any(nid==u or nid==v for (u,v) in lvl_set): node_lvl[idx][n_idx]=1
            if any(nid==u or nid==v for (u,v) in typ_set): node_typ[idx][n_idx]=1
            if any(nid==u or nid==v for (u,v) in role_set):node_role[idx][n_idx]=1

    # Setup OpenGL
    pygame.init()
    pygame.display.set_mode((WIDTH, HEIGHT), DOUBLEBUF|OPENGL)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(60, WIDTH/HEIGHT, 1.0, 1000.0)
    gluLookAt(0, 0, 100.0,   0, 0, 0,   0, 1, 0)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    gluLookAt(0, 0, 100.0,   0, 0, 0,   0, 1, 0)
    glEnable(GL_DEPTH_TEST)
    # allow shader-controlled point sizes for larger vertices
    glEnable(GL_PROGRAM_POINT_SIZE)
    glEnable(GL_POINT_SPRITE)
    glEnable(GL_POINT_SMOOTH)
    # enable blending for proper round point rendering
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    vbo_nodes = setup_vbo()
    vbo_node_colors = setup_vbo()
    vbo_edges = setup_vbo()
    vbo_edge_colors = setup_vbo()
    vbo_ghosts = setup_vbo()

    shader_program = create_program(VERTEX_SHADER, FRAGMENT_SHADER)
    # retrieve MVP and time uniform locations
    mvp_loc = glGetUniformLocation(shader_program, "u_mvp")
    time_loc = glGetUniformLocation(shader_program, "u_time")
    clock = pygame.time.Clock()
    frame = 0
    # vertex count for drawing edges (two vertices per edge)
    edge_count = E * 2
    menu_index = 0  # Track the current menu index


    def full_reset(menu_index):
        """
        Fully reset the program state for the selected menu item,
        including error-caught demo advancement, CPU- and GPU-side buffers,
        and all rolling state.
        """

        nonlocal dataG, grouped, ordered_keys
        nonlocal nodes, positions, velocities, forces, glow_alpha, glow_radius
        nonlocal edges, base_lengths, edge_rest_lengths, N, E
        nonlocal lvl_mask, typ_mask, role_mask, node_lvl, node_typ, node_role
        nonlocal ghost_configs, rainbow_trail
        nonlocal state_buffer, vbo_nodes, vbo_edges, vbo_edge_colors, vbo_ghosts, vbo_node_colors
        # Ensure all variables in full_reset are declared as global
        

        # 1) Try to initialize the selected demo, skip to next on failure
        while True:
            try:
                demo = test_suite[menu_index]
                expr = demo.get('expr_fn') or demo

                pg = ProcessGraph(0)
                pg.build_from_expression(expr, *demo.get('dims', (1,)))
                pg.compute_levels(method='alap')

                dataG, grouped = pg.dataG, pg.group_edges_by_dataset(pg.dataG)
                if len(dataG.nodes) == 0:
                    raise ValueError("Graph is empty")

                # Rebuild the ordered key list
                ordered_keys = [
                    (lvl, typ, role)
                    for lvl in sorted(grouped)
                    for typ in sorted(grouped[lvl])
                    for role in sorted(grouped[lvl][typ])
                ]

                # Recompute all masks
                nodes, positions, velocities, forces, glow_alpha, glow_radius = load_graph(dataG)
                edges, base_lengths = build_edges(nodes, dataG)
                edge_rest_lengths = base_lengths.clone()
                N, E = len(nodes), len(edges)

                # Masks: lvl_mask, typ_mask, role_mask, node_lvl, node_typ, node_role
                edge_keys = [(nodes[u], nodes[v]) for u, v in edges]
                lvl_mask = torch.zeros((len(ordered_keys), E))
                typ_mask = lvl_mask.clone()
                role_mask = lvl_mask.clone()
                node_lvl = torch.zeros((len(ordered_keys), N))
                node_typ = node_lvl.clone()
                node_role = node_lvl.clone()

                for idx, (lvl, typ, role) in enumerate(ordered_keys):
                    lvl_set  = set.union(*(set(grouped[lvl][t][r])
                                        for t in grouped[lvl] for r in grouped[lvl][t]))
                    typ_set  = set.union(*(set(grouped[lvl][typ][r])
                                        for r in grouped[lvl][typ]))
                    role_set = set(grouped[lvl][typ].get(role, []))

                    for e_idx, key in enumerate(edge_keys):
                        if key in lvl_set:  lvl_mask[idx, e_idx]  = 1
                        if key in typ_set:  typ_mask[idx, e_idx]  = 1
                        if key in role_set: role_mask[idx, e_idx] = 1

                    for n_idx, nid in enumerate(nodes):
                        if any(nid in edge for edge in lvl_set):  node_lvl[idx, n_idx]  = 1
                        if any(nid in edge for edge in typ_set):  node_typ[idx, n_idx]  = 1
                        if any(nid in edge for edge in role_set): node_role[idx, n_idx] = 1

                break  # success

            except Exception as e:
                print(f"Initialization failed: {e}. Skipping to next test.")
                menu_index = (menu_index + 1) % len(test_suite)

        # 2) Clear rolling/state buffers
        state_buffer.clear()
        #state_buffer.append({'sum': 0, 'edges': [], 'positions': positions.clone()})
        best_score = float('inf')
        ghost_configs.clear()
        rainbow_trail.clear()

        # 3) Re-allocate (or clear) all VBOs to match new sizes
        import numpy as np

        # Nodes: 9 floats per vertex
        empty_node_data = np.zeros((N, 9), dtype=np.float32)
        update_vbo(vbo_nodes, empty_node_data)

        # (If used) separate color buffer for nodes
        empty_node_colors = np.zeros((N, 3), dtype=np.float32)
        update_vbo(vbo_node_colors, empty_node_colors)

        # Edges: 2 endpoints × 2 floats each, plus RGBA colors
        empty_edge_pts   = np.zeros((E * 2, 2), dtype=np.float32)
        empty_edge_cols  = np.zeros((E * 2, 4), dtype=np.float32)
        update_vbo(vbo_edges, empty_edge_pts)
        update_vbo(vbo_edge_colors, empty_edge_cols)

        # Ghosts: same layout as nodes
        update_vbo(vbo_ghosts, np.zeros((0, 9), dtype=np.float32))


    full_reset(menu_index)
    while True:
        for e in pygame.event.get():
            if e.type == QUIT:
                pygame.quit()
                return
            elif e.type == KEYDOWN:
                if e.key == K_LEFT:
                    menu_index = (menu_index - 1) % len(test_suite)
                    full_reset(menu_index)
                elif e.key == K_RIGHT:
                    menu_index = (menu_index + 1) % len(test_suite)
                    full_reset(menu_index)

        idx = frame % len(ordered_keys)

        # --- Physics updates, exactly as before ---
        adjustments = (base_lengths*(1-LEVEL_TARGET_FACTOR)).unsqueeze(1)*lvl_mask[idx].unsqueeze(1)*BETA_LEVEL \
                     +(base_lengths*(1-TYPE_TARGET_FACTOR)).unsqueeze(1)*typ_mask[idx].unsqueeze(1)*BETA_TYPE \
                     +(base_lengths*(1-ROLE_TARGET_FACTOR)).unsqueeze(1)*role_mask[idx].unsqueeze(1)*BETA_ROLE
        edge_rest_lengths -= adjustments.sum(dim=1)
        edge_rest_lengths += (base_lengths - edge_rest_lengths) * ALPHA_IDLE

        u, v = edges[:,0], edges[:,1]
        delta = positions[v] - positions[u]
        dist = torch.norm(delta, dim=1, keepdim=True)
        dir_v = delta / (dist + 1e-8)
        spring_force = SPRING_K * (dist - edge_rest_lengths.unsqueeze(1))
        # Determine active edges by current grouping masks
        # active_mask = (lvl_mask[idx] + typ_mask[idx] + role_mask[idx]) > 0
        # Determine most active edges via role mask only
        active_mask = role_mask[idx] > 0
        active_edges = torch.where(active_mask)[0]
        # Sum of edge lengths for active edges as traversal time proxy
        active_lengths = dist.squeeze()[active_edges]
        current_active_sum = active_lengths.sum().item()
        # Buffer detailed state: sum, active edge indices, and positions snapshot
        state_buffer.append({
            'sum': current_active_sum,
            'edges': active_edges.cpu().tolist(),
            'positions': positions.clone()
        })
        # Rolling sum over the buffered sums
        rolling_sum = sum(item['sum'] for item in state_buffer)
        # when rolling sum hits new low, record ghost state
        if rolling_sum < best_score:
            best_score = rolling_sum
            # select next ghost index cyclically
            idx_g = len(ghost_configs) % GHOST_COUNT
            ghost_configs.append({
                'positions': positions.clone(),
                'alpha': GHOST_INIT_ALPHA,
                'decay': float(decay_speeds[idx_g]),
                'color': rainbow_colors[idx_g]
            })
            if len(ghost_configs) > GHOST_COUNT:
                ghost_configs.pop(0)
            print(f"New best rolling sum: {best_score} at frame {frame}")

        forces.zero_()
        forces.index_add_(0, u, spring_force * dir_v)
        forces.index_add_(0, v, -spring_force * dir_v)

        diff = positions.unsqueeze(1) - positions.unsqueeze(0)
        dist2 = (diff**2).sum(-1) + torch.eye(N)
        rep_force = REPULSION_K / dist2
        forces += (rep_force.unsqueeze(2) * diff / dist2.unsqueeze(2).sqrt()).sum(1)

        velocities = (velocities + forces) * DAMPING
        positions += velocities * SPEED_FACTOR
        #radii = torch.norm(positions, dim=1, keepdim=True)
        #positions = positions / (radii + 1e-8) * 10.0  # keep on sphere

        # project velocity to be tangential
        #to_center = positions / (10.0 + 1e-8)
        #velocities -= (velocities * to_center).sum(dim=1, keepdim=True) * to_center

        # --- Derived values ---
        # project to pixel-space for edges and normalize to NDC for shaders
        proj_pixels = project_positions(positions)
        proj_ndc = positions
        kinetic_energy = torch.norm(velocities, dim=1, keepdim=True)

        # glow updates
        alpha_target = GLOW_FLOOR_ALPHA + (GLOW_PEAK_ALPHA-GLOW_FLOOR_ALPHA) * \
                       (0.3*node_lvl[idx] + 0.5*node_typ[idx] + 1.0*node_role[idx]).unsqueeze(1)
        radius_target = GLOW_FLOOR_RADIUS + (GLOW_PEAK_RADIUS-GLOW_FLOOR_RADIUS) * \
                       (0.3*node_lvl[idx] + 0.5*node_typ[idx] + 1.0*node_role[idx]).unsqueeze(1)
        glow_alpha += (alpha_target - glow_alpha) * torch.where(alpha_target > glow_alpha, GLOW_RISE, GLOW_DECAY)
        glow_radius += (radius_target - glow_radius) * torch.where(radius_target > glow_radius, GLOW_RISE, GLOW_DECAY)

        # --- Prepare interleaved node data with proper colors and 3D positions ---
        # world-space positions
        pos_np = positions.cpu().numpy().astype(np.float32)  # N x 3
        # compute node colors for this frame
        colors = np.stack([node_role[idx].cpu().numpy(),
                           node_typ[idx].cpu().numpy(),
                           node_lvl[idx].cpu().numpy()], axis=1).astype(np.float32)  # N x 3
        alpha_np = glow_alpha.cpu().numpy().astype(np.float32)      # N x 1
        radius_np = glow_radius.cpu().numpy().astype(np.float32)    # N x 1
        ke_np = kinetic_energy.cpu().numpy().astype(np.float32)      # N x 1
        # concatenate into N x 9 array: pos(3), color(3), alpha, radius, ke
        node_data = np.concatenate([pos_np, colors, alpha_np, radius_np, ke_np], axis=1)
        update_vbo(vbo_nodes, node_data)

        # --- Legacy edge drawing data ---
        # draw edges in pixel-space to align with orthographic projection
        edge_points = np.vstack([proj_pixels[edges[:,0]], proj_pixels[edges[:,1]]]).astype(np.float32)
        # build edge color RGBA for blending
        edge_rgb = np.vstack([np.stack([role_mask[idx], typ_mask[idx], lvl_mask[idx]], axis=1)] * 2).astype(np.float32)
        edge_alphas = np.full((edge_rgb.shape[0], 1), EDGE_ALPHA, dtype=np.float32)
        edge_colors = np.hstack([edge_rgb, edge_alphas])
        update_vbo(vbo_edges, edge_points)
        update_vbo(vbo_edge_colors, edge_colors)

        # --- Render ---
        glClearColor(*CANVAS_COLOR)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        # set camera for shader & debugging
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        if DEBUG_MODE:
            # fixed eye positioned along Z to see entire graph
            #gluLookAt(0, 0, 300.0,   0, 0, 0,   0, 1, 0)
            # draw world axes
            glBegin(GL_LINES)
            # X axis (red)
            glColor3f(1, 0, 0)
            glVertex3f(0, 0, 0)
            glVertex3f(100, 0, 0)
            # Y axis (green)
            glColor3f(0, 1, 0)
            glVertex3f(0, 0, 0)
            glVertex3f(0, 100, 0)
            # Z axis (blue)
            glColor3f(0, 0, 1)
            glVertex3f(0, 0, 0)
            glVertex3f(0, 0, 100)
            glEnd()

        # ---- Draw edges with legacy pipeline ----
        glEnableClientState(GL_VERTEX_ARRAY)
        glLineWidth(EDGE_BASE_WIDTH * 2)
        glEnableClientState(GL_COLOR_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_edges)
        glVertexPointer(2, GL_FLOAT, 0, None)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_edge_colors)
        glColorPointer(4, GL_FLOAT, 0, None)  # RGBA
        #glDrawArrays(GL_LINES, 0, edge_count)
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)

        # ---- Draw nodes with shader pipeline ----
        # compute and upload combined ModelViewProjection matrix
        proj_mat = (GLfloat * 16)()
        model_mat = (GLfloat * 16)()
        glGetFloatv(GL_PROJECTION_MATRIX, proj_mat)
        glGetFloatv(GL_MODELVIEW_MATRIX, model_mat)
        # convert to numpy, reshape and transpose to column-major
        proj_np = np.array(proj_mat, dtype=np.float32).reshape(4,4)
        model_np = np.array(model_mat, dtype=np.float32).reshape(4,4)
        mvp_np = proj_np.dot(model_np)
        # upload uniform
        glUseProgram(shader_program)
        # send time uniform for rotation
        current_time = pygame.time.get_ticks()
        glUniform1f(time_loc, current_time)
        glUniformMatrix4fv(mvp_loc, 1, GL_FALSE, mvp_np.flatten())

        glBindBuffer(GL_ARRAY_BUFFER, vbo_nodes)

        stride = 9 * 4  # 9 floats (pos3 + color3 + alpha + radius + ke) * 4 bytes
        # in_position (vec3)
        loc = glGetAttribLocation(shader_program, "in_position")
        glEnableVertexAttribArray(loc)
        glVertexAttribPointer(loc, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        # in_color (vec3 based on role/type/level masks)
        loc_c = glGetAttribLocation(shader_program, "in_color")
        glEnableVertexAttribArray(loc_c)
        glVertexAttribPointer(loc_c, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(3*4))
        # in_alpha
        loc_a = glGetAttribLocation(shader_program, "in_alpha")
        glEnableVertexAttribArray(loc_a)
        glVertexAttribPointer(loc_a, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(6*4))
        # in_radius
        loc_r = glGetAttribLocation(shader_program, "in_radius")
        glEnableVertexAttribArray(loc_r)
        glVertexAttribPointer(loc_r, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(7*4))
        # in_ke
        loc_k = glGetAttribLocation(shader_program, "in_ke")
        glEnableVertexAttribArray(loc_k)
        glVertexAttribPointer(loc_k, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(8*4))

        glDrawArrays(GL_POINTS, 0, N)

        glDisableVertexAttribArray(loc)
        glDisableVertexAttribArray(loc_c)
        glDisableVertexAttribArray(loc_a)
        glDisableVertexAttribArray(loc_r)
        glDisableVertexAttribArray(loc_k)
        glUseProgram(0)


        
        if RAINBOW_MODE == TRAIL_MODE:
            rainbow_trail.append({
                'positions': positions.clone(),
                'frame': frame  # or pygame.time.get_ticks() for ms-precise
            })

        elif RAINBOW_MODE == BEST_MODE:
            
            # Update best per scale, but do not fade, always persistent
            for i, window in enumerate(BEST_SCALES):
                if len(state_buffer) >= window:
                    window_scores = [item['sum'] for item in list(state_buffer)[-window:]]
                    window_min = min(window_scores)
                    min_idx = window_scores.index(window_min)
                    min_item = list(state_buffer)[-window + min_idx]
                    # Only update if this is a new best for the window
                    last_best = best_per_scale[i]
                    if last_best is None or window_min < last_best['score']:
                        best_per_scale[i] = {
                            'score': window_min,
                            'frame': frame,
                            'positions': min_item['positions'].clone()
                        }

        # After updating trail or best buffers:
        ghost_pos, ghost_col, ghost_alpha, ghost_age = [], [], [], []
        if RAINBOW_MODE == TRAIL_MODE:
            for i, trail_state in enumerate(rainbow_trail):
                pos = trail_state['positions'].cpu().numpy()
                r, g, b = trail_colors[i]
                alpha = 1.0 - (i / TRAIL_LENGTH)
                ghost_pos.append(pos)
                ghost_col.append(np.tile((r, g, b), (pos.shape[0], 1)))
                ghost_alpha.append(np.full((pos.shape[0], 1), alpha * 0.6))
                ghost_age.append(np.full((pos.shape[0], 1), trail_state['frame']))
        elif RAINBOW_MODE == BEST_MODE:
            for i, best in enumerate(best_per_scale):
                if best is not None:
                    pos = best['positions'].cpu().numpy()
                    r, g, b = rainbow_best_colors[i]
                    alpha = 0.92
                    ghost_pos.append(pos)
                    ghost_col.append(np.tile((r, g, b), (pos.shape[0], 1)))
                    ghost_alpha.append(np.full((pos.shape[0], 1), alpha))
                    ghost_age.append(np.full((pos.shape[0], 1), best['frame']))
        if ghost_pos:
            ghost_pos = np.concatenate(ghost_pos, axis=0)
            ghost_col = np.concatenate(ghost_col, axis=0)
            ghost_alpha = np.concatenate(ghost_alpha, axis=0)
            ghost_age = np.concatenate(ghost_age, axis=0)
            ghost_data = np.concatenate([ghost_pos, ghost_col, ghost_alpha, ghost_age], axis=1)
            # sort by age (last col)
            sort_idx = np.argsort(ghost_data[:,-1])
            ghost_data = ghost_data[sort_idx]
        else:
            ghost_data = np.zeros((0,8), dtype=np.float32)

        update_vbo(vbo_ghosts, ghost_data.astype(np.float32))

        # --- Draw ghosts with the same shader as nodes ---
            # --- Draw ghosts with the same shader as nodes ---
        if ghost_data.shape[0] > 0:
            # 1) Repack into pos(3), color(3), alpha, radius, ke
            N_ghosts = ghost_data.shape[0]
            pos    = ghost_data[:, 0:3]
            col    = ghost_data[:, 3:6]
            alpha  = ghost_data[:, 6:7]
            radius = np.full((N_ghosts, 1), GHOST_PEAK_RADIUS, dtype=np.float32)
            ke     = np.zeros((N_ghosts, 1), dtype=np.float32)
            ghost_node_data = np.hstack([pos, col, alpha, radius, ke]).astype(np.float32)
            update_vbo(vbo_ghosts, ghost_node_data)

            # 2) Bind shader & uniforms
            glUseProgram(shader_program)
            glUniform1f(time_loc, pygame.time.get_ticks())
            glUniformMatrix4fv(mvp_loc, 1, GL_FALSE, mvp_np.flatten())

            # 3) Enable attributes exactly as for nodes
            glBindBuffer(GL_ARRAY_BUFFER, vbo_ghosts)
            stride = 9 * 4
            # in_position
            loc = glGetAttribLocation(shader_program, "in_position")
            glEnableVertexAttribArray(loc)
            glVertexAttribPointer(loc, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
            # in_color
            loc_c = glGetAttribLocation(shader_program, "in_color")
            glEnableVertexAttribArray(loc_c)
            glVertexAttribPointer(loc_c, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(3*4))
            # in_alpha
            loc_a = glGetAttribLocation(shader_program, "in_alpha")
            glEnableVertexAttribArray(loc_a)
            glVertexAttribPointer(loc_a, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(6*4))
            # in_radius
            loc_r = glGetAttribLocation(shader_program, "in_radius")
            glEnableVertexAttribArray(loc_r)
            glVertexAttribPointer(loc_r, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(7*4))
            # in_ke
            loc_k = glGetAttribLocation(shader_program, "in_ke")
            glEnableVertexAttribArray(loc_k)
            glVertexAttribPointer(loc_k, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(8*4))

            # 4) Draw them all in one go
            glDrawArrays(GL_POINTS, 0, N_ghosts)

            # 5) Tear down
            glDisableVertexAttribArray(loc)
            glDisableVertexAttribArray(loc_c)
            glDisableVertexAttribArray(loc_a)
            glDisableVertexAttribArray(loc_r)
            glDisableVertexAttribArray(loc_k)
            glUseProgram(0)

        pygame.display.flip()
        clock.tick(FPS)
        dataG.run_at(ordered_keys[idx][0], ordered_keys[idx][1], ordered_keys[idx][2])
        frame += 1




if __name__=="__main__":
    main()
