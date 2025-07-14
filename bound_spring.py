# bound_spring.py – spring/repulsion simulator + **symbolic gas/pressure force library**
"""
This file keeps the fast **PyTorch** `BoundSpringNetwork`, *but* now plugs in a
small SymPy → Graph‑Express bridge so you can add physically‑meaningful force
terms (ideal‑gas pressure, van‑der‑Waals, Lennard‑Jones, etc.) without ever
running `sympy.lambdify` in the hot‑loop.

*Every* expression is stored symbolically, turned into a **ProcessGraph** once
(lighting‑fast), and the compiled callable is cached.  The core solver can
sample any of those forces each frame just by calling `net.apply_symbolic()`.

Key additions
-------------
1. **Symbolic catalogue** in `SYM_FORCE_EXPR` – start with ideal‑gas pressure
   and a linear drag term; extend as you like.
2. **`compile_force(name)`** – builds a *single* `ProcessGraph`, lowers it to
   Python, and memoises the resulting callable.
3. **`BoundSpringNetwork.apply_symbolic(name, **par)`** – enqueue extra forces
   that are evaluated inside the force hook every step.

Nothing else in the original network changed – if you ignore the new helpers,
*API compatibility is 100 %*.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple
import math
import torch
import sympy as sp
import numpy as np

# Set torch device (CPU or CUDA if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
#  ── 1.  Symbolic force library + Graph‑Express compilation  ────────────────
# ---------------------------------------------------------------------------
from graph_express2 import ProcessGraph  # your DAG engine

# --- Define symbols
r, V, n, R_, T, P, k_drag, v = sp.symbols("r V n R_ T P k_drag v", positive=True)

# --- Raw expressions -------------------------------------------------------
SYM_FORCE_EXPR: Dict[str, sp.Expr] = {
    # Ideal‑gas pressure difference (scalar).  Users supply ΔP and outward unit normal.
    "ideal_gas_pressure" : n*R_*T / V,  # P = nRT / V   (ΔP computed outside)
    # Simple Stokes drag  F = −k·v  (vector handled separately)
    "stokes_drag"        : -k_drag * v,
}

# --- Compilation cache -----------------------------------------------------
_COMPILED: Dict[str, Callable[..., float]] = {}

def compile_force(name: str) -> Callable[..., float]:
    """Return a fast numeric callable produced by Graph‑Express."""
    if name in _COMPILED:
        return _COMPILED[name]
    if name not in SYM_FORCE_EXPR:
        raise KeyError(name)

    expr = SYM_FORCE_EXPR[name]
    pg = ProcessGraph()
    pg.build_from_expression(expr, expand_complex=False)
    pg.finalize_graph_with_outputs()
    fn = pg.python_callable()
    _COMPILED[name] = fn
    return fn

# Convenience wrapper -------------------------------------------------------
_DEF_ARGS = {
    "ideal_gas_pressure": (n, R_, T, V),
    "stokes_drag"       : (k_drag, v),
}

def evaluate_force(name: str, **params):
    fn   = compile_force(name)
    args = [params[str(s)] for s in _DEF_ARGS[name]]
    return fn(*args)

# ---------------------------------------------------------------------------
#  ── 2.  Fast all‑pairs repulsion kernel (unchanged)  ───────────────────────
# ---------------------------------------------------------------------------
@torch.jit.script
def _repel(x: torch.Tensor, c: float, eps: float):
    N = x.size(0)
    F = torch.zeros_like(x)
    for i in range(N):
        d = x[i] - x
        dist2 = (d * d).sum(dim=1) + eps
        inv = 1.0 / dist2
        inv[i] = 0.0
        F[i] += (c * inv.unsqueeze(1) * d).sum(dim=0)
    return F

# ---------------------------------------------------------------------------
#  ── 3.  BoundSpringNetwork with symbolic hooks  ────────────────────────────
# ---------------------------------------------------------------------------
from graph_express2 import ProcessGraph

import torch

import torch

class BoundaryMaskHelper:
    def __init__(
        self,
        hull_vertices: torch.Tensor,    # (Vh,3)
        hull_triangles: torch.Tensor,   # (Th,3)
    ):
        self.update_hull(hull_vertices, hull_triangles)

    def update_hull(
        self,
        hull_vertices: torch.Tensor,  # (Vh,3)
        hull_triangles: torch.Tensor | None # (Th,3)
    ):
        """
        Call this whenever your hull mesh changes.  Recomputes:
         • self.hull_vertices, self.hull_triangles
         • per-face normals, per-vertex normals
         • self.hull_radius (for contraction offsets)
        """
        self.hull_vertices  = hull_vertices
        self.hull_triangles = hull_triangles

        center = hull_vertices.mean(dim=0, keepdim=True)
        # if triangles given, average face‐normals; else use radial normals
        if hull_triangles is not None and len(hull_triangles)>0:
            v0 = hull_vertices[hull_triangles[:,0]]
            v1 = hull_vertices[hull_triangles[:,1]]
            v2 = hull_vertices[hull_triangles[:,2]]
            fn = torch.cross(v1 - v0, v2 - v0, dim=1)
            fn = fn / (fn.norm(dim=1, keepdim=True) + 1e-9)
            vn = torch.zeros_like(hull_vertices)
            for i, tri in enumerate(hull_triangles):
                vn[tri] += fn[i]
            self.vertex_normals = vn / (vn.norm(dim=1, keepdim=True) + 1e-9)
        else:
            # pure radial normals
            diff = hull_vertices - center
            self.vertex_normals = diff / (diff.norm(dim=1, keepdim=True) + 1e-9)
        

        # 3) hull “radius” (for converting percent→absolute)
        center = hull_vertices.mean(dim=0, keepdim=True)
        d2     = ((hull_vertices - center)**2).sum(dim=1)
        self.hull_radius = float(d2.max().sqrt().item())

    def masks(
        self,
        pos: torch.Tensor,               # (N,3) current network verts
        contract_percent: float = 0.0,
        eps: float = 1e-6
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # compute distances to hull and contracted hull
        d_orig = torch.cdist(pos, self.hull_vertices)
        dmin_orig = d_orig.min(dim=1).values
        offset = contract_percent * self.hull_radius
        contracted = self.hull_vertices - self.vertex_normals * offset
        d_contr = torch.cdist(pos, contracted)
        dmin_contr = d_contr.min(dim=1).values
        mask_hull       = dmin_orig <= eps
        mask_contracted = dmin_contr <= eps
        return mask_hull, mask_contracted



Tensor = torch.Tensor

@dataclass
class BoundSpringNetwork:
    # mandatory input: the ProcessGraph you built for this demo
    pg: ProcessGraph
    # if you pass None, we’ll auto-generate a Fibonacci sphere hull
    hull_vertices: Tensor | None  = None  # (Vh,3) or None
    hull_triangles: Tensor | None = None  # (Th,3) or None
    # we’ll fill these in __post_init__
    boundary_center:  Tensor      = field(init=False)
    boundary_radius:  float       = field(init=False)
    # physics params
    k_stretch: float = 8.0
    c_repulse: float = 0.4
    p_target: float = 0.0
    mass: Tensor | None = None
    damping: float = 0.902
    eps_rep: float = 1e-4

    # ── glow & color parameters ──────────────────────────────────────────
    beta_level: float     = 0.5
    beta_type:  float     = 0.7
    beta_role:  float     = 0.9
    GLOW_RISE:  float     = 0.3
    GLOW_DECAY: float     = 0.05
    GLOW_PEAK_ALPHA: float = 1.0
    GLOW_FLOOR_ALPHA: float = 0.1
    GLOW_PEAK_RADIUS: float = 0.2
    GLOW_FLOOR_RADIUS: float = 0.1

    # internal/state  -----------------------------------------------------
    vel: Tensor = field(init=False)
    pos: Tensor = field(init=False)  # (N,3) positions
    # glow‐state
    glow_alpha:  Tensor = field(init=False)   # (N,1)
    glow_radius: Tensor = field(init=False)   # (N,1)

    # graph metadata
    nodes:          list[str]        = field(init=False)
    edges:          Tensor           = field(init=False)   # shape [E,2]
    ordered_keys:   list[tuple]      = field(init=False)   # list of (level,type,role)
    lvl_mask:       Tensor           = field(init=False)   # [L, E]
    typ_mask:       Tensor           = field(init=False)   # [L, E]
    role_mask:      Tensor           = field(init=False)   # [L, E]
    node_lvl:       Tensor           = field(init=False)   # [L, N]
    node_typ:       Tensor           = field(init=False)   # [L, N]
    node_role:      Tensor           = field(init=False)   # [L, N]

    # hooks (ensure all phases are defined) ------------------------------------------
    hooks_pre:             List[Callable[["BoundSpringNetwork"], None]]                  = field(default_factory=list, init=False)
    hooks_force:           List[Callable[["BoundSpringNetwork", Tensor], None]]          = field(default_factory=list, init=False)
    hooks_force_pre:       List[Callable[["BoundSpringNetwork", Tensor], None]]          = field(default_factory=list, init=False)
    hooks_post:            List[Callable[["BoundSpringNetwork"], None]]                  = field(default_factory=list, init=False)
    hooks_force_negotiate: List[Callable[["BoundSpringNetwork", torch.Tensor], torch.Tensor]] = field(default_factory=list, init=False)
    hooks_force_commit:    List[Callable[["BoundSpringNetwork", torch.Tensor], torch.Tensor]] = field(default_factory=list, init=False)

    frame: int = field(default=0, init=False)
    _sym_queue: List[Tuple[str, Dict[str,float]]] = field(default_factory=list, init=False)
    # ——— stanchion dynamics ————————————————————————————————————
    # how far to contract towards each target per cycle
    level_target: float = 0.9       # ← LEVEL_TARGET_FACTOR
    type_target:  float = 0.75      # ← TYPE_TARGET_FACTOR
    role_target:  float = 0.5       # ← ROLE_TARGET_FACTOR
    relax_rate:   float = 0.1       # ← ALPHA_IDLE
    
    growth_rate: float = 0.1  # ← from initial to natural rest length
    natural_rest_length: Tensor = field(init=False)  # will be set in __post_init__

    # how many step() calls to hold each active group
    cycle_steps: int = 1

    # internal rest-lengths (dynamic)
    base_lengths:  Tensor = field(init=False)  # constant snapshot of L0
    rest_lengths:  Tensor = field(init=False)  # will be updated every step

    # cycling state
    group_idx: int = field(default=0, init=False)
    group_step: int = field(default=0, init=False)

    # ── safety caps for sub-dt loop ─────────────────────────────────────
    max_force:        float | None = None     # e.g. 10.0
    max_velocity:     float | None = None     # e.g. 5.0
    max_displacement: float | None = None     # e.g. 0.1
    c_frac: float = 0.5  # relativistic correction factor   
    outer_transient: float = 1.0      # thickness of outer transient layer
    inner_transient: float = 1.0      # thickness of inner transient layer
    k_transient: float = 8.0          # spring constant for transients
    slip_embed: bool = True           # toggle slip‐only on inner wall
    # -------------------------------------------------------------- setup
    def __post_init__(self):
        self.c_frac = 0.5  # relativistic correction factor
        G = self.pg.dataG
        # 1) nodes list + raw edges as index pairs
        self.nodes = list(G.nodes)
        self.source_id_to_index = {name: i for i, name in enumerate(self.nodes)}
        print(self.nodes)
        edge_list = [(self.source_id_to_index[u], self.source_id_to_index[v]) for u, v in G.edges]
        self.edges = torch.tensor(edge_list, dtype=torch.long, device=device)
        initial_positions = self.fibonacci_sphere(len(self.nodes), radius=10.0)
        init_lengths = initial_positions[self.edges[:, 1]] - initial_positions[self.edges[:, 0]]
        init_lengths = init_lengths.norm(dim=1)  # edge lengths
        # original “rest length” from Graph edges (or default)
        self.rest_lengths = torch.tensor(
            [G[u][v].get("rest_length", init_lengths[i]) for i, (u,v) in enumerate(G.edges)],
            dtype=torch.float32,
            device=device
        )
        # stash a constant copy for all future “spring stanchion” targets
        self.base_lengths = self.rest_lengths.clone().to(device)
        # ── sensible defaults for sub‐dt caps ──────────────────────────
        # base_lengths is your “L0” per‐edge; use its mean as a scale
        mean_L0 = float(self.base_lengths.mean().item())
        # no displacement > 50% of a typical spring length
        if self.max_displacement is None:
            self.max_displacement = 0.5 * mean_L0
        # no force > what a full‐stretch spring would exert
        if self.max_force is None:
            self.max_force = self.k_stretch * mean_L0
        # no velocity > displacement-per-step at default dt
        default_dt = 1
        if self.max_velocity is None:
            self.max_velocity = self.max_displacement / default_dt
        # === pull our topology & masks from ProcessGraph ===
        # remove static L0 singular; force uses dynamic self.rest_lengths instead
        # 2) grouping & ordered roles
        grouped = self.pg.group_edges_by_dataset(G)
        self.ordered_keys = self.pg.sort_roles(grouped)
        L = len(self.ordered_keys); E = self.edges.size(0); N = len(self.nodes)

        # 3) allocate masks
        self.lvl_mask  = torch.zeros(L, E, dtype=torch.bool, device=device)
        self.typ_mask  = torch.zeros(L, E, dtype=torch.bool, device=device)
        self.role_mask = torch.zeros(L, E, dtype=torch.bool, device=device)
        self.node_lvl  = torch.zeros(L, N, dtype=torch.bool, device=device)
        self.node_typ  = torch.zeros(L, N, dtype=torch.bool, device=device)
        self.node_role = torch.zeros(L, N, dtype=torch.bool, device=device)

        # 4) fill masks exactly as simplegraphspring4 did
        for idx, (lvl, typ, role) in enumerate(self.ordered_keys):
            lvl_set  = set.union(*(set(grouped[lvl][t][r]) 
                                  for t in grouped[lvl] for r in grouped[lvl][t]))
            typ_set  = set.union(*(set(grouped[lvl][typ][r]) 
                                  for r in grouped[lvl][typ]))
            role_set = set(grouped[lvl][typ].get(role, []))

            # edge masks
            for e, (u, v) in enumerate(self.edges.tolist()):
                key = (self.nodes[u], self.nodes[v])
                if key in lvl_set:   self.lvl_mask[idx, e]  = True
                if key in typ_set:   self.typ_mask[idx, e]  = True
                if key in role_set:  self.role_mask[idx, e] = True

            # node masks: incident on any edge in those sets
            for n, name in enumerate(self.nodes):
                if any(name in ed for ed in lvl_set):  self.node_lvl[idx, n]  = True
                if any(name in ed for ed in typ_set):  self.node_typ[idx, n]  = True
                if any(name in ed for ed in role_set): self.node_role[idx, n] = True

        # === now continue existing sensor/spring setup ===
        self.pos = self.fibonacci_sphere(N, radius=10.0)  # you'll need to expose this

        self.vel = torch.zeros_like(self.pos, device=device)

        # initialize glow state
        self.glow_alpha  = torch.full((N,1), self.GLOW_FLOOR_ALPHA, dtype=torch.float32, device=device)
        self.glow_radius = torch.full((N,1), self.GLOW_FLOOR_RADIUS, dtype=torch.float32, device=device)

        if self.mass is None:
            self.mass = torch.ones(self.pos.size(0), device=device)
        else:
            self.mass = self.mass.clone().float().to(device)

        # ── 1) if no hull was passed, build a fallback Fibonacci sphere hull ──
        N = self.pos.size(0)
        if self.hull_vertices is None:
            # estimate network radius from initial pos
            radius_est = self.pos.norm(dim=1).max().item()
            hull_radius = 2.0 * radius_est
            # twice as many hull verts
            self.hull_vertices = self.fibonacci_sphere(2*N, radius=hull_radius)
            self.hull_triangles = None

        # ── 2) instantiate boundary helper & initialize center/radius ────────
        self.boundary_helper = BoundaryMaskHelper(
            hull_vertices  = self.hull_vertices,
            hull_triangles = self.hull_triangles,
        )
        # center & radius for any quick-use
        self.boundary_center = self.hull_vertices.mean(dim=0)
        self.boundary_radius = self.boundary_helper.hull_radius
        self.natural_rest_length = torch.ones_like(self.rest_lengths, device=device)
        self.done_growing = torch.zeros_like(self.rest_lengths, dtype=torch.bool, device=device)
    def node_labels(self):     # e.g. ['A','B',...]
        return self.nodes

    def edge_list(self):
        return self.edges

    def level_mask(self):      # shape [N], int levels
        return self.lvl_mask, self.node_lvl

    def type_mask(self):      # shape [N], int types
        return self.typ_mask, self.node_typ

    def rol_mask(self):      # shape [N], int roles
        return self.role_mask, self.node_role

    def kinetic_energy(self):
        """Compute the kinetic energy of all nodes."""
        return 0.5 * self.mass * self.vel.norm(dim=1)**2

    # -------------------------------------------------------------- DSL
    def hook(self, phase: str):
        def _decor(fn):
            {
                "pre":               self.hooks_pre,
                "force":             self.hooks_force,
                "force_pre":         self.hooks_force_pre,
                "force_negotiate":   self.hooks_force_negotiate,
                "force_commit":      self.hooks_force_commit,
                "post":              self.hooks_post,
            }[phase].append(fn)
            return fn
        return _decor

    def apply_symbolic(self, name: str, **params):
        """Queue a symbolic force for the next `step()` call."""
        self._sym_queue.append((name, params))

    
    def fibonacci_sphere(self, samples=100, radius=10.0):
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
        return torch.tensor(points, dtype=torch.float32, device=device)

    # -------------------------------------------------------------- step

    @torch.no_grad()
    def step(self, dt: float = 1/60.):
        # 1) pre-hooks
        for fn in self.hooks_pre:
            fn(self)

        # 2) advance active-group clock
        self.group_step += 1
        if self.group_step >= self.cycle_steps:
            self.group_step = 0
            self.group_idx = (self.group_idx + 1) % len(self.ordered_keys)
        idx = self.group_idx

        # 3) dynamic rest-length contraction & relaxation
        lvl_m = self.lvl_mask[idx].to(self.rest_lengths.dtype)
        typ_m = self.typ_mask[idx].to(self.rest_lengths.dtype)
        rol_m = self.role_mask[idx].to(self.rest_lengths.dtype)
        c1 = self.base_lengths * (1 - self.level_target) * lvl_m
        c2 = self.base_lengths * (1 - self.type_target)  * typ_m
        c3 = self.base_lengths * (1 - self.role_target)  * rol_m
        # apply contraction
        self.rest_lengths = self.rest_lengths - (c1 + c2 + c3)
        # relax back toward base
        self.rest_lengths = self.rest_lengths + \
            (self.base_lengths - self.rest_lengths) * self.relax_rate

        # nested force evaluator with per-vertex final_mask
        def force(self, pos, edges, final_mask: torch.BoolTensor | None = None):
            # ——— 1) build raw force F ————————————————————————
            F = torch.zeros_like(pos)
            i, j = edges[:,0], edges[:,1]
            d    = pos[i] - pos[j]
            length = d.norm(dim=1, keepdim=True) + 1e-9
            dir_   = d / length
            delta  = (length.squeeze(1) - self.rest_lengths).unsqueeze(1)
            f      = self.k_stretch * delta * dir_
            F.index_add_(0, i, -f)
            F.index_add_(0, j,  f)

            # ——— 2) repulsion ————————————————————————————————
            if self.c_repulse:
                F += _repel(pos, self.c_repulse, self.eps_rep)

            # ——— 3) symbolic queued forces ————————————————————
            if self._sym_queue:
                centre = pos.mean(dim=0)
                r_vec  = pos - centre
                r_norm = r_vec.norm(dim=1, keepdim=True)
                for name, par in self._sym_queue:
                    val = evaluate_force(name, **par)
                    if name == "ideal_gas_pressure":
                        F += -val * r_vec / (r_norm + 1e-9)
                    elif name == "stokes_drag":
                        F += val * self.vel
                # clear queue only for finalized vertices
                if final_mask is not None and final_mask.any():
                    self._sym_queue.clear()

            # ——— 4) negotiation hooks ————————————————————————
            for fn in self.hooks_force:
                fn(self, F)
            for fn in self.hooks_force_pre:
                fn(self, F)
            F_theory = F.clone()
            for fn in self.hooks_force_negotiate:
                F_theory = fn(self, F_theory)
            # commit negotiations for this sub-step
            if final_mask is not None and final_mask.any():
                F = F_theory
                for fn in self.hooks_force_commit:
                    F = fn(self, F)

            # ——— 5) mesh-based boundary & captive-hold —————————
            mask_hull, mask_contract = self.boundary_helper.masks(
                pos, contract_percent=self.inner_transient, eps=1e-6
            )
            # slip & spring-reject in inner transient
            if mask_contract.any():
                d2h = torch.cdist(pos[mask_contract],
                                  self.boundary_helper.hull_vertices).min(dim=1).values.unsqueeze(1)
                normals = self.boundary_helper.vertex_normals[
                    torch.cdist(pos[mask_contract],
                                self.boundary_helper.hull_vertices)
                    .argmin(dim=1)
                ]
                fnormal = (F[mask_contract] * normals).sum(dim=1, keepdim=True) * normals
                Frej   = -self.k_transient * d2h * normals
                F[mask_contract] = (F[mask_contract] - fnormal) + Frej
            # clamp penetrators back onto mesh
            if mask_hull.any():
                closest = torch.cdist(pos[mask_hull],
                                      self.boundary_helper.hull_vertices).argmin(dim=1)
                pos[mask_hull] = self.boundary_helper.hull_vertices[closest]

            # ——— 6) relativistic acceleration ————————————————
            prop_a = F / self.mass.unsqueeze(1)
            c_abs  = self.c_frac * (self.max_velocity or 1.0)
            v_mag  = self.vel.norm(dim=1, keepdim=True)
            γ      = 1.0 / torch.sqrt(torch.clamp(1 - (v_mag / c_abs)**2, min=1e-9))
            a_rel  = prop_a / (γ**3)

            # ——— 7) per-edge growth toward natural_rest_length —————
            delta = self.base_lengths - self.natural_rest_length
            c0    = delta * self.growth_rate
            tol   = 1e-2
            grow_mask = (~self.done_growing) & (delta.abs() > tol)
            c0 = c0 * grow_mask.to(c0.dtype)
            # commit growth only for finalized vertices
            if final_mask is not None and final_mask.any():
                self.base_lengths = self.base_lengths - c0
            # update per-edge done_growing
            self.done_growing = self.done_growing | (delta.abs() <= tol)

            return a_rel, F

        # ——— per-vertex dt negotiation & sub-step planning ——————
        prop_a, F = force(self, self.pos, self.edges, final_mask=None)
        N = self.pos.size(0)
        eps = 1e-9
        fmag = F.norm(dim=1)
        amag = prop_a.norm(dim=1)
        vmag = self.vel.norm(dim=1)

        dt_f = self.max_force    / (fmag + eps) if self.max_force    else torch.full((N,), float('inf'))
        dt_a = self.max_velocity / (amag + eps) if self.max_velocity else torch.full((N,), float('inf'))
        dt_v = self.max_displacement / (vmag + eps) if self.max_displacement else torch.full((N,), float('inf'))

        dt_allowed_per_vertex = torch.stack([dt_f, dt_a, dt_v], dim=1).min(dim=1).values
        # each vertex’s required steps
        n_steps = torch.ceil(dt / dt_allowed_per_vertex).to(torch.int64)
        n_steps = torch.clamp(n_steps, min=1)
        # global sub-step count by mean
        n_sub  = int(n_steps.float().mean().ceil().item())
        dt_sub = dt / n_sub

        # ——— sub-dt integration with per-vertex final masks ——————
        for j in range(1, n_sub + 1):
            final_mask  = (n_steps == j)
            active_mask = (n_steps >= j)

            sub_a, sub_F = force(self, self.pos, self.edges, final_mask=final_mask)

            v_old  = self.vel[active_mask]
            v_new  = (v_old + sub_a[active_mask] * dt_sub) * math.exp(-self.damping * dt_sub)
            p_new  = self.pos[active_mask] + v_new * dt_sub

            self.vel[active_mask] = v_new
            self.pos[active_mask] = p_new

        # ——— glow update ——————————————————————————————
        idx = self.frame % len(self.ordered_keys)
        lvl = self.node_lvl[idx].to(torch.float32).unsqueeze(1)
        typ = self.node_typ[idx].to(torch.float32).unsqueeze(1)
        rol = self.node_role[idx].to(torch.float32).unsqueeze(1)
        weight = lvl*self.beta_level + typ*self.beta_type + rol*self.beta_role

        alpha_tgt  = self.GLOW_FLOOR_ALPHA + (self.GLOW_PEAK_ALPHA - self.GLOW_FLOOR_ALPHA)*weight
        radius_tgt = self.GLOW_FLOOR_RADIUS + (self.GLOW_PEAK_RADIUS - self.GLOW_FLOOR_RADIUS)*weight

        rise_m   = (alpha_tgt > self.glow_alpha).float()
        decay_m  = 1.0 - rise_m
        self.glow_alpha  += (alpha_tgt  - self.glow_alpha)  * (rise_m*self.GLOW_RISE  + decay_m*self.GLOW_DECAY)
        rise_m_r = (radius_tgt > self.glow_radius).float()
        decay_m_r= 1.0 - rise_m_r
        self.glow_radius += (radius_tgt - self.glow_radius) * (rise_m_r*self.GLOW_RISE + decay_m_r*self.GLOW_DECAY)

        # 9) post-hooks, frame++ and return
        for fn in self.hooks_post:
            fn(self)
        self.frame += 1
        return self.pos, self.vel


    def node_colors(self) -> np.ndarray:
        """
        Returns an (N×3) array of floats in [0,1], where each channel
        is the boolean node_* mask for the *active* group at current frame.
        """
        idx = (self.frame - 1) % len(self.ordered_keys)
        r = self.node_lvl[idx].to(torch.float32)
        g = self.node_typ[idx].to(torch.float32)
        b = self.node_role[idx].to(torch.float32)
        rgb = torch.stack([r, g, b], dim=1)
        return rgb.cpu().numpy().astype(np.float32)
# ---------------------------------------------------------------------------
#  Quick demo ----------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    P0 = torch.randn(32,3)*0.3
    E  = [(i,(i+1)%32) for i in range(32)]
    net = BoundSpringNetwork(P0, E, k_stretch=6., c_repulse=0.3)

    # add an ideal‑gas outward pressure that fades over time
    for step in range(240):
        net.apply_symbolic("ideal_gas_pressure", n=40., R_=0.082, T=293., V=1.0)
        pos,_ = net.step(1/120.)
        if step%60==0:
            print(f"step {step:3d}  mean‑|x| =", pos.norm(dim=1).mean().item())
