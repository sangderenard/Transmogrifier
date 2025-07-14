

"""
binding_memranes_sympy.py
=========================
Definitive repository of symbolic expressions, evaluation registry, mesh utilities, and demo routines for BindingMembrane.
This file is organized as:
  1. Symbolic/numeric registry and mesh utilities (library)
  2. Demo and evaluation routines (demo/entry-point)
"""
from __future__ import annotations
from types import MappingProxyType
import math, random, abc
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Any, Tuple, List

import numpy as np
import sympy as sp
import networkx as nx
import torch


class BindingMembrane:
    """
    Multipurpose container that can evolve from a simple hard sphere into
    a full physically-based membrane.

    Public attributes (initialised to sane defaults or empty tensors)
    -----------------------------------------------------------------
    centre            (3,)  – current geometric centre
    radius            float – for the fallback spherical collider

    mesh_verts        (V,3) – surface vertices  (empty if unused)
    mesh_faces        (F,3) – index triples     (empty if unused)

    voxel_pressure    (*N,3) – scalar field inside the membrane
    voxel_species     dict[str, Tensor] – extra scalar fields (e.g. ions)

    face_permeability (F,)  – 0-1 coefficient per mesh face / DEC cell

    Notes
    -----
    • All tensors live on CPU by default; move them to CUDA if required.
    • Geometry helpers are written for *batch* inputs: `pos`, `vel`,
      `force` are (N,3) tensors.
    """

    # ---------------------------------------------------------------- init
    def __init__(self,
                 centre: torch.Tensor | tuple = (0., 0., 0.),
                 radius: float = 50.0):

        self.centre = torch.as_tensor(centre, dtype=torch.float32)    # (3,)
        self.radius = float(radius)                                   # scalar

        # --- future-heavy data, pre-allocated as empty --------------------
        self.mesh_verts        = torch.empty((0, 3))        # (V,3)
        self.mesh_faces        = torch.empty((0, 3), dtype=torch.long)
        self.face_permeability = torch.empty((0,))          # (F,)

        self.voxel_pressure    = torch.empty((0, 0, 0))     # 3-D grid
        self.voxel_species     = {}                         # name → Tensor

    # =============================================================== helpers
    def _distance_from_centre(self, pos: torch.Tensor) -> torch.Tensor:
        """Return (N,1) Euclidean distance to the *current* centre."""
        return torch.norm(pos - self.centre, dim=-1, keepdim=True)

    # ================================================================ public
    # --- full-trajectory gate ---------------------------------------------
    def check_bounds(self,
                     pos:   torch.Tensor,
                     vel:   torch.Tensor,
                     force: torch.Tensor
                     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Entry-point called each frame **before** positions are committed.

        You can pipe the data through more steps:

            pos, vel, force = self.apply_slip_boundary(... )
            pos, vel, force = self.respond_transgression(... )
            pos, vel, force = self.apply_osmosis(... )

        For now we delegate only to the conservative collider.
        """
        return self.respond_transgression(pos, vel, force)

    # --- collider (kept from the minimal version) -------------------------
    def respond_transgression(self,
                              pos:   torch.Tensor,
                              vel:   torch.Tensor,
                              force: torch.Tensor
                              ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Hard-sphere “freeze on outward motion”.

        TODO: upgrade to mesh-aware SDF or triangle tests when
        `self.mesh_verts` / `self.mesh_faces` are populated.
        """
        r_vec = pos - self.centre                    # (N,3)
        dist  = torch.norm(r_vec, dim=-1, keepdim=True)

        inside = dist <= self.radius
        if inside.all():
            return pos, vel, force                  # fast path

        # outward unit normal
        n = r_vec / (dist + 1e-9)

        v_out = (vel   * n).sum(-1, keepdim=True).clamp(min=0.)
        f_out = (force * n).sum(-1, keepdim=True).clamp(min=0.)

        vel   = vel   - n * v_out                   # cancel outward comp.
        force = force - n * f_out
        pos   = self.centre + n * self.radius       # clamp to surface

        return pos, vel, force

    # ===================================================== future extensions
    # ---------- geometry ---------------------------------------------------
    def inflate(self, dr: float):
        """Uniformly grow / shrink by *dr* ( >0 inflate, <0 deflate )."""
        self.radius = max(1e-6, self.radius + dr)          # keep >0
        # If you have a mesh: scale vertices *about* the centre
        if self.mesh_verts.numel():
            self.mesh_verts = (self.mesh_verts - self.centre) * \
                               ((self.radius) / (self.radius - dr)) + self.centre

    def build_convex_hull(self, points: torch.Tensor):
        """
        Given an (M,3) cloud, compute and store a convex hull mesh.

        Stub only – call out to `scipy.spatial.ConvexHull` or `trimesh`
        in a real implementation.
        """
        # TODO: fill in
        raise NotImplementedError

    # ---------- voxel fields ----------------------------------------------
    def set_voxel_pressure(self, grid: torch.Tensor):
        """Replace internal pressure field with *grid* (H,W,D)."""
        self.voxel_pressure = grid.clone()

    # ---------- permeability / osmosis ------------------------------------
    def compute_osmotic_flux(self, ext_concentration: torch.Tensor):
        """
        Placeholder: compute per-face flux given an outside concentration
        field.  Store the result or return it for the caller to apply.

        Returns
        -------
        flux : Tensor (F,) – positive = outward, negative = inward
        """
        # TODO: realistic implementation
        return torch.zeros_like(self.face_permeability)

    def apply_osmotic_exchange(self, dt: float,
                               ext_concentration: torch.Tensor | None = None):
        """
        Adjust internal voxel_species / pressure according to fluxes.

        At present this is a stub that does nothing.
        """
        if ext_concentration is None or not self.face_permeability.numel():
            return
        # TODO: integrate flux over dt and update self.voxel_species
        pass

    # ---------- slip-boundary (for lipid membranes) ------------------------
    def apply_slip_boundary(self,
                            pos:   torch.Tensor,
                            vel:   torch.Tensor,
                            mu:    float = 0.1):
        """
        Tangential slip: damp only the *normal* component of velocity by
        factor `mu` (0 = full slip, 1 = stick).

        Currently active only for the spherical fallback.
        """
        r_vec = pos - self.centre
        dist  = torch.norm(r_vec, dim=-1, keepdim=True)
        on_surface = torch.isclose(dist, torch.tensor(self.radius),
                                   atol=1e-3, rtol=0.)
        if not on_surface.any():
            return vel                                 # nothing to do

        n = r_vec / (dist + 1e-9)
        v_n = (vel * n).sum(-1, keepdim=True)
        vel = vel - mu * n * v_n                       # damp normal comp.
        return vel

class MembraneGraph:
    """
    Scene-wide manager holding **all** BindingMembrane instances and
    running their boundary logic in one vectorised sweep.

    Typical use
    -----------
    >>> memG = MembraneGraph()
    >>> mid = memG.add_membrane(BindingMembrane((0,0,0), 40.))
    >>> memG.add_membrane_child(parent=mid,
    ...       membrane=BindingMembrane((10,0,0), 8.0))
    >>> pos, vel, frc = memG.step(pos, vel, frc, dt)

    Internally
    ----------
    • Nodes:    graph node-id  → {"mem": BindingMembrane, "level": int}
    • Edges:    parent → child (`relation="contains"`) or lateral
    • All tensors are *not* copied – only views / in-place writes.
    """

    # ............................................................. lifecycle
    def __init__(self):
        self.G    : nx.DiGraph = nx.DiGraph()
        self._ids : Dict[int, BindingMembrane] = {}   # nid → object

    # ............................................................. adders
    def add_membrane(self,
                     membrane: BindingMembrane,
                     tag: str | None = None) -> int:
        nid = id(membrane)
        if nid in self.G:
            raise ValueError("Membrane already registered")
        self.G.add_node(nid,
                        mem   = membrane,
                        tag   = tag,
                        level = 0)
        self._ids[nid] = membrane
        return nid

    def add_membrane_child(self,
                           parent:   int,
                           membrane: BindingMembrane,
                           relation: str = "contains",
                           tag: str | None = None) -> int:
        child = self.add_membrane(membrane, tag)
        lvl   = self.G.nodes[parent]["level"] + 1
        self.G.nodes[child]["level"] = lvl
        self.G.add_edge(parent, child, relation=relation)
        return child

    # ............................................................. core step
    @torch.no_grad()
    def step(self,
             pos   : torch.Tensor,
             vel   : torch.Tensor,
             force : torch.Tensor,
             dt    : float = 1.0
             ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Vectorised membrane processing.

        Parameters
        ----------
        pos, vel, force : (N,3) tensors – *shared* arrays for every particle.
        dt              : timestep, forwarded to future physics (osmosis, …).

        Returns the modified tensors (same objects, just for chaining).
        """

        # 1) Topological order – parents first (nesting matters)
        for nid in nx.topological_sort(self.G):
            mem = self.G.nodes[nid]["mem"]

            # slice-mask for particles this membrane owns/responsible for.
            # Today → all particles.  Tomorrow → use spatial hash / tags.
            mask = torch.ones(pos.shape[0], dtype=torch.bool,
                              device=pos.device)

            # batch-apply
            _p, _v, _f = mem.check_bounds(pos[mask], vel[mask], force[mask])

            # in-place write-back (views share storage – cheap)
            pos[mask]   = _p
            vel[mask]   = _v
            force[mask] = _f

            # ---- optional future hooks -----------------------------------
            # mem.apply_osmotic_exchange(dt, ext_conc)   # not yet
            # mem.apply_slip_boundary(...)

        return pos, vel, force

import math
import torch

# -----------------------------------------------------------
#  High-level actuator / façade for BindingMembrane
# -----------------------------------------------------------
class MembraneActions:
    """
    Thin wrapper that offers semantic operations for gameplay /
    simulation scripts.  Everything ultimately delegates back to
    the underlying BindingMembrane instance.

    • grow_surface(dA) / shrink_surface(dA) – change shell area
    • apply_internal_pressure(dp)           – uniform pressure bump
    • leak(species, dmol)                   – stub for ion exchange
    • slip(mu)                              – adjust slip coefficient
    """
    def __init__(self, membrane: BindingMembrane):
        self.m = membrane

    # --------------- geometry ------------------------------------------------
    def grow_surface(self, dA: float):
        """Increase surface area by *dA* (same sign-convention as inflate())."""
        if dA <= 0:
            raise ValueError("dA must be positive; use shrink_surface() instead.")
        self._area_delta(dA)

    def shrink_surface(self, dA: float):
        """Decrease surface area by *dA* until minimum radius is reached."""
        if dA <= 0:
            raise ValueError("dA must be positive.")
        self._area_delta(-dA)

    def _area_delta(self, dA: float):
        r  = self.m.radius
        dr = dA / (8.0 * math.pi * r)            # dA/dR = 8πR  →  dR = dA / 8πR
        self.m.inflate(dr)

    # --------------- pressure -----------------------------------------------
    def apply_internal_pressure(self, dp: float):
        """Uniformly add *dp* to voxel_pressure (creates a 1-cell grid if empty)."""
        if not self.m.voxel_pressure.numel():
            self.m.voxel_pressure = torch.zeros((1, 1, 1))
        self.m.voxel_pressure += dp

    # --------------- species exchange ----------------------------------------
    def leak(self, species: str, dmol: float):
        """Very crude placeholder: subtract / add *dmol* from a scalar store."""
        store = self.m.voxel_species.setdefault(species, torch.tensor(0.0))
        store += dmol
        self.m.voxel_species[species] = store

    # --------------- slip toggle --------------------------------------------
    def slip(self, mu: float):
        """Set global slip factor 0–1 (for now stored on the object)."""
        self.m.slip_mu = float(mu)


# -----------------------------------------------------------
#  Minimal text-based smoke test
# -----------------------------------------------------------
def random_particles(n, spread=60.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    pos   = (torch.rand((n, 3), generator=g) - 0.5) * spread
    vel   = torch.zeros_like(pos)
    force = torch.zeros_like(pos)
    return pos, vel, force


def main():
    # ------------------------------------------------------------------
    # 1) build a tiny hierarchy (unchanged)
    # ------------------------------------------------------------------
    outer = BindingMembrane((0, 0, 0), 40.0)
    inner = BindingMembrane((10, 0, 0), 8.0)

    world = MembraneGraph()
    outer_id = world.add_membrane(outer, tag="outer")
    world.add_membrane_child(outer_id, inner, tag="inner")

    # optional: seed internal state so fluxes have something to work with
    for m in (outer, inner):
        m.voxel_pressure = torch.full((1, 1, 1), 2.0)     # 2 Pa everywhere
        m.voxel_species["Na⁺"] = torch.full((1, 1, 1), 0.2)  # 0.2 mol L⁻¹

    # ------------------------------------------------------------------
    # 2) external bath conditions   (could be tensors of any shape)
    # ------------------------------------------------------------------
    bath_pressure = torch.tensor(1.0)      # scalar solvent pressure
    bath_conc     = {"Na⁺": torch.tensor(0.05)}   # 0.05 mol L⁻¹ outside

    # coupling coefficients  (tune freely or make tensors)
    k_p = 0.3   # pressure relaxation rate   [s⁻¹]
    k_c = 0.1   # concentration relaxation   [s⁻¹]

    # ------------------------------------------------------------------
    # 3) particles and loop
    # ------------------------------------------------------------------
    pos, vel, frc = random_particles(200, spread=120.0)
    n_steps, dt = 200, 0.05

    for step in range(n_steps):
        # --- membrane geometry interaction (vectorised)
        pos, vel, frc = world.step(pos, vel, frc, dt)

        # --- continuum exchange for every membrane --------------------
        for mem in world._ids.values():
            # ---------- pressure -------------------------------------
            intP = mem.voxel_pressure.mean() if mem.voxel_pressure.numel() else 0.
            dP   = intP - bath_pressure        # +ve ⇒ inside > outside
            mem.voxel_pressure += (-k_p * dP * dt)

            # ---------- every tracked species -------------------------
            for sp, ext_val in bath_conc.items():
                internal = mem.voxel_species.get(sp, torch.zeros(()))
                mean_int = internal.mean() if internal.numel() else internal
                dC       = mean_int - ext_val
                mem.voxel_species[sp] = internal + (-k_c * dC * dt)

        # (optional) diagnostics every few frames
        if step % 40 == 0:
            far = pos.norm(dim=1).max().item()
            print(f"t = {step*dt:6.2f}s | farthest particle r = {far:6.2f}"
                  f" | outer P = {outer.voxel_pressure.mean():5.2f}"
                  f" | outer [Na⁺] = {outer.voxel_species['Na⁺'].mean():5.3f}")

    # ------------------------------------------------------------------
    # 4) high-level actions demo (still valid)
    # ------------------------------------------------------------------
    acts = MembraneActions(outer)
    acts.grow_surface(100.0)
    acts.apply_internal_pressure(+5.0)
    acts.leak("Na⁺", -0.02)

    print("\n--- final outer stats ---")
    print("radius            :", outer.radius)
    print("mean pressure     :", outer.voxel_pressure.mean().item())
    print("mean [Na⁺]        :", outer.voxel_species['Na⁺'].mean().item())


if __name__ == "__main__":
    main()
# ---------------------------------------------------------------------------
#  Minimal Queue / Boundary shells (stand-ins until your full versions land)
# ---------------------------------------------------------------------------
class BoundarySubgraph:
    def __init__(self, verts: torch.Tensor, faces: torch.Tensor):
        self.verts = verts            # (V,3)
        self.faces = faces            # (F,3)
        self.occ   = torch.zeros(len(verts), dtype=torch.long)   # occupancy counter
        # helper for stretch-energy demo
        self.assigned_vidx = torch.arange(0)   # filled by kernels later

class QueueSubgraph:
    def __init__(self, pos: torch.Tensor, state: torch.Tensor, viewer_idx: torch.Tensor = None):
        self.pos   = pos              # (N,3)
        self.state = state            # ±1  (outer / inner)
        if viewer_idx is None:
            self.viewer_idx = torch.arange(pos.shape[0], device=pos.device)
        else:
            self.viewer_idx = viewer_idx

# --- Parametric surface support ---
@dataclass(slots=True)
class ParametricSurface:
    name: str
    sym_expr: Tuple[sp.Expr, sp.Expr, sp.Expr] | None
    numeric: Callable[[np.ndarray, np.ndarray], np.ndarray]
    def eval(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        return self.numeric(u, v)
    def sample_grid(self, res: int = 64) -> Tuple[np.ndarray, np.ndarray]:
        uu, vv = np.meshgrid(np.linspace(0, 1, res), np.linspace(0, 1, res), indexing="ij")
        pts  = self.eval(uu.ravel(), vv.ravel())
        verts = pts.astype(np.float32)
        faces = []
        def idx(i, j): return i * res + j
        for i in range(res - 1):
            for j in range(res - 1):
                faces.append((idx(i, j), idx(i+1, j),   idx(i, j+1)))
                faces.append((idx(i+1, j), idx(i+1, j+1), idx(i, j+1)))
        return verts, np.asarray(faces, dtype=np.int32)

SURF_REGISTRY: Dict[str, ParametricSurface] = {}
def register_surface(name: str, surf: ParametricSurface):
    SURF_REGISTRY[name] = surf
def get_surface(name: str) -> ParametricSurface:
    return SURF_REGISTRY[name]
def mesh_from_parametric(surf: ParametricSurface, res: int = 64):
    return surf.sample_grid(res)

# --- Symbolic catalogue ---
R, t = sp.symbols("R t", positive=True)
P_int, P_ext = sp.symbols("P_int P_ext", real=True)
C_int, C_ext = sp.symbols("C_int C_ext", positive=True)
k_p, k_c = sp.symbols("k_p k_c", positive=True)
mu_k, mu_s_max, tau_mu = sp.symbols("mu_k mu_s_max tau_mu", positive=True)
mu, gamma = sp.symbols("mu gamma", positive=True)
H = sp.symbols("H", real=True)
Fx, Fy, Fz = sp.symbols("Fx Fy Fz", real=True)
n_x, n_y, n_z = sp.symbols("n_x n_y n_z", real=True)
F_vec = sp.Matrix([Fx, Fy, Fz])
n_vec = sp.Matrix([n_x, n_y, n_z])
u, v = sp.symbols("u v", real=True)
param_x = sp.Function("param_x")(u, v)
param_y = sp.Function("param_y")(u, v)
param_z = sp.Function("param_z")(u, v)
x, y, z, eps = sp.symbols("x y z eps", real=True)
phi_rbf = sp.Function("phi_rbf")
rbf_surface = sp.Function("rbf_surface")(x, y, z)
f_bi = sp.Function("f_bi")(u, v)
bi_eq = sp.Eq(sp.diff(f_bi, u, 4) + 2*sp.diff(f_bi, u, 2, v, 2) + sp.diff(f_bi, v, 4), 0)
mls_surface = sp.Function("mls_surface")(x, y, z)
J_solvent      = k_p * (P_int - P_ext)
J_solute       = k_c * (C_int - C_ext)
DeltaP_laplace = 2 * gamma * H
F_tangential   = F_vec - (F_vec.dot(n_vec)) * n_vec
mu_static_time = mu_k + (mu_s_max - mu_k) * (1 - sp.exp(-t / tau_mu))
SYM_EXPR: Dict[str, sp.Expr] = {
    "solvent_flux"   : J_solvent,
    "solute_flux"    : J_solute,
    "laplace_pressure": DeltaP_laplace,
    "slip_force"     : F_tangential,
    "mu_static_time" : mu_static_time,
    "param_x"        : param_x,
    "param_y"        : param_y,
    "param_z"        : param_z,
    "rbf_surface"    : rbf_surface,
    "biharmonic_eq"  : bi_eq,
    "mls_surface"    : mls_surface,
}
SYM_EXPR = MappingProxyType(SYM_EXPR)

# --- Evaluation registry ---
EVAL_REGISTRY: Dict[str, Callable[..., Any]] = {}
_NUMERIC_CACHE: Dict[str, Callable[..., Any]] = {}
def register_evaluator(name: str, func: Callable[..., Any]):
    if name not in SYM_EXPR:
        raise KeyError(name)
    EVAL_REGISTRY[name] = func
def _lambdify(name: str):
    expr = SYM_EXPR[name]
    if isinstance(expr, sp.Equality):
        _NUMERIC_CACHE[name] = lambda **kw: (_ for _ in ()).throw(NotImplementedError(
            f"{name} is a PDE – please register an evaluator."))
        return
    args = sorted(expr.free_symbols, key=lambda s: s.name)
    _NUMERIC_CACHE[name] = sp.lambdify(args, expr, modules="numpy")
def evaluate(name: str, **params):
    if name in EVAL_REGISTRY:
        return EVAL_REGISTRY[name](**params)
    if name not in _NUMERIC_CACHE:
        _lambdify(name)
    f = _NUMERIC_CACHE[name]
    expr = SYM_EXPR[name]
    if isinstance(expr, sp.Equality):
        return f()  # will raise
    argvals = [params[s.name] for s in sorted(expr.free_symbols, key=lambda s: s.name)]
    return f(*argvals)
num = evaluate  # alias

# quick analytic sphere  (u,v)∈[0,1]² → ℝ³
def _sphere_num(u, v):
    theta = v * np.pi
    phi   = u * 2*np.pi
    x = np.sin(theta)*np.cos(phi)
    y = np.sin(theta)*np.sin(phi)
    z = np.cos(theta)
    return np.stack([x, y, z], axis=1)

register_surface("unit_sphere",
    ParametricSurface("unit_sphere", None, _sphere_num))

def _enrol_into_slip(layer: SlipLayer,
                     bonded_mask: torch.Tensor,
                     queue: QueueSubgraph):
    """
    • `bonded_mask`  : boolean (Nq,) over *queue* rows
    • `queue`        : the queue object, must have .viewer_idx
    """
    if layer is None or not bonded_mask.any():
        return
    viewer_ids = queue.viewer_idx[bonded_mask]
    layer.enrol(bonded_mask, viewer_ids)

# ---------------------------------------------------------------------------
#  Diffusion kernels – base class + new MembraneWalkerKernel
# ---------------------------------------------------------------------------
import abc

class DiffusionKernel(abc.ABC):
    """
    Stateless (or light-state) object that mutates `boundary` & `queue`
    in-place.  Must be GPU-safe and autograd-friendly unless decorated
    with @torch.no_grad().
    """
    @abc.abstractmethod
    def step(self, boundary, queue, dt: float = 1.0, **kw):
        ...
# --- Mesh utilities ---
class BoltzmannKernel(DiffusionKernel):
    def __init__(self, beta: float = 1.5): self.beta = beta

    @torch.no_grad()   # toggle if you need autograd
    def step(self, boundary, queue, dt, **kw):
        """
        • boundary.occ is updated in-place
        • queue.pos & queue.state jump to new locations
        """
        verts = boundary.verts            # (V,3)
        occ   = boundary.occ              # (V,)
        pos   = queue.pos                 # (N,3)

        # 1) find nearest vertex for every agent   -------------------------
        d     = torch.cdist(pos, verts)           # (N,V)
        vidx  = d.argmin(dim=1)                   # (N,)
        
        # 2) decide whether to bond or stay in slip-layer  -----------------
        free  = (occ[vidx] == 0)
        p_bond = torch.exp(-self.beta * d.min(dim=1).values)  # shape (N,)

        will_bond = free & (torch.rand_like(p_bond) < p_bond)

        # ---- bond: clamp pos, mark occ  ---------------------------------
        pos[will_bond] = verts[vidx[will_bond]]
        # >>> tell SlipLayer about newly-bonded rows
        _enrol_into_slip(
            kw["slip_layer"],
            will_bond,
            queue)

        occ.index_add_(0, vidx[will_bond], torch.where(
            queue.state[will_bond] > 0, torch.ones_like(vidx[will_bond]),
                                      -torch.ones_like(vidx[will_bond])) )

        # ---- non-bond: do a tangential slide  ---------------------------
        not_bonded = ~will_bond
        if not_bonded.any():
            slide = torch.randn_like(pos[not_bonded])            # random dir
            centre = pos[not_bonded] * 0                         # assume sphere @ 0
            normal = pos[not_bonded] - centre
            slide -= (slide * normal).sum(-1, keepdim=True) * normal
            slide = slide / torch.norm(slide, dim=-1, keepdim=True)
            pos[not_bonded] += slide * dt * 5.0                  # drift speed
            pos[not_bonded] = pos[not_bonded] / \
                              torch.norm(pos[not_bonded], dim=-1, keepdim=True) \
                              * torch.norm(verts[vidx[not_bonded]], dim=-1, keepdim=True)

class MembraneWalker:
    def __init__(self):
        self.N, self.V = sp.symbols("N V", integer=True, positive=True)
        self.beta, self.gamma, self.kappa = sp.symbols("beta gamma kappa", positive=True)
        self.x = sp.IndexedBase("x")      # (N,3) agent positions
        self.v = sp.IndexedBase("v")      # (V,3) mesh vertices
        self.rho = sp.IndexedBase("rho")  # (V,) occupancy count
        self.eps = 1e-3
        def l1(i, j):
            dx = sp.Abs(self.x[i,0] - self.v[j,0] + self.eps)
            dy = sp.Abs(self.x[i,1] - self.v[j,1] + self.eps)
            dz = sp.Abs(self.x[i,2] - self.v[j,2] + self.eps)
            return dx + dy + dz
        self.l1 = l1
        # Manhattan-like (smoothed) distance to every vertex
        i, j = sp.symbols('i j', integer=True)
        axis = sp.symbols('axis', integer=True)
        # E_pos: sum over all agents of the min cost to any vertex
        self.E_pos = sp.Sum(
            sp.Min(
                sp.Sum(self.l1(i, j), (axis, 0, 0)),  # dummy axis loop
                (j, 0, self.V-1)
            ), (i, 0, self.N-1)
        )
        self.E_crowd = self.gamma * sp.Sum(self.rho[j]**2, (j, 0, self.V-1))
        self.E_glue  = -self.kappa * sp.Sum(self.rho[j]**2, (j, 0, self.V-1))
        self.E = sp.simplify(self.E_pos + self.E_crowd + self.E_glue)

class MembraneWalkerKernel(DiffusionKernel):
    """
    Numerical counterpart to the symbolic `MembraneWalker`.

        E(i→j) = β·‖x_i – v_j‖₂
               + (γ – κ)·ρ_j²      with  ρ_j = |occ[j]|

    • Attractive term pulls every agent toward *some* vertex.
    • Crowding term pushes them away from vertices that are busy.
    • Glue term (κ) rewards re-using an already-bonded vertex when κ > γ.

    Parameters
    ----------
    beta   : weight of distance (>=0)
    gamma  : crowding penalty     (>=0)
    kappa  : glue reward/penalty  (>=0; κ > γ favours bonding)
    """
    def __init__(self, beta: float = 1.0,
                       gamma: float = 0.5,
                       kappa: float = 0.0):
        self.beta  = float(beta)
        self.gamma = float(gamma)
        self.kappa = float(kappa)

    @torch.no_grad()                 # drop the decorator if you need grads
    def step(self, boundary, queue, dt: float = 1.0, **kws):

        verts = boundary.verts                      # (V,3)
        occ   = boundary.occ                        # (V,)
        pos   = queue.pos                           # (N,3)
        state = queue.state                         # (N,)  ±1

        # 1) pair-wise L2 distances -------------------------------
        d      = torch.cdist(pos, verts)            # (N,V)

        # 2) vertex-level energy  -------------------------------
        rho2   = occ.pow(2)                         # (V,)
        E      = self.beta * d + (self.gamma - self.kappa) * rho2

        # 3) pick minimal-energy vertex for every agent ----------
        vidx   = E.argmin(dim=1)                    # (N,)

        # 4) decide who can bond (free vertex) -------------------
        free   = (occ[vidx] == 0)


        # ---- bond -------------------------------------------------------
        if free.any():
            pos[free] = verts[vidx[free]]
            delta     = torch.where(state[free] > 0,
                                    torch.ones_like(vidx[free], dtype=occ.dtype),
                                   -torch.ones_like(vidx[free], dtype=occ.dtype))
            occ.index_add_(0, vidx[free], delta)

            # tell SlipLayer
            _enrol_into_slip(
                kws.get("slip_layer", None),   # kw passed by EmbeddingManager
                free,
                queue)

        # ---- not bonded → gradient drift toward chosen vertex -----------
        not_bonded = ~free
        if not_bonded.any():
            drift = verts[vidx[not_bonded]] - pos[not_bonded]
            pos[not_bonded] += 0.3 * drift * dt          # 0.3 = arbitrary speed





@dataclass(slots=True)
class VertexState:
    id: int
    pos: np.ndarray
    vel: np.ndarray
    force: np.ndarray
    extras: Dict[str, Any] = field(default_factory=dict)
    def as_tensor(self, backend=np):
        return backend.asarray(self.pos), backend.asarray(self.vel), backend.asarray(self.force)

class MeshProjection:
    @staticmethod
    def spherical_uv(verts: np.ndarray, centre: np.ndarray) -> np.ndarray:
        v = verts - centre
        x_, y_, z_ = v.T
        r_ = np.linalg.norm(v, axis=1) + 1e-12
        theta_ = np.arccos(np.clip(z_ / r_, -1.0, 1.0))
        phi_   = np.arctan2(y_, x_)
        u_ = (phi_ + np.pi) / (2*np.pi)
        v_ = theta_ / np.pi
        return np.stack([u_, v_], axis=1)
    @staticmethod
    def parameterise_mesh(verts: np.ndarray, centre: np.ndarray):
        return MeshProjection.spherical_uv(verts, centre)

class MeshGraphBridge:
    def __init__(self, verts: np.ndarray, faces: np.ndarray):
        self.verts = verts
        self.faces = faces
        self.vertex_states: Dict[int, VertexState] = {}
        self.G: nx.Graph = nx.Graph()
    def pull_from_mesh(self, indices: Iterable[int]):
        for i in indices:
            vs = self.vertex_states.setdefault(i, VertexState(i, self.verts[i].copy(), np.zeros(3), np.zeros(3)))
            vs.pos = self.verts[i].copy()
    def push_to_mesh(self):
        for i, vs in self.vertex_states.items():
            self.verts[i] = vs.pos
    def embed_complete_graph(self, indices: Iterable[int]):
        idx = list(indices)
        self.G.add_nodes_from(idx)
        for u_i in idx:
            for v_i in idx:
                if u_i < v_i:
                    self.G.add_edge(u_i, v_i)

class ElevatedRegistry:
    def __init__(self):
        self.pre_step: Dict[int, Callable[[VertexState], None]] = {}
        self.post_step: Dict[int, Callable[[VertexState], None]] = {}
    def elevate(self, vid: int, *, when: str, handler: Callable[[VertexState], None]):
        if when == "pre":
            self.pre_step[vid] = handler
        elif when == "post":
            self.post_step[vid] = handler
        else:
            raise ValueError("when must be 'pre' or 'post'")

def build_adjacency(faces: torch.Tensor) -> Dict[int, List[int]]:
    adj: Dict[int, List[int]] = {}
    for tri in faces.tolist():
        for i in range(3):
            a, b = tri[i], tri[(i+1)%3]
            adj.setdefault(a, []).append(b)
            adj.setdefault(b, []).append(a)
    for k, lst in adj.items():
        adj[k] = list(set(lst))
    return adj

# --- Diffusion kernels ---
def edge_prob_step(mem, agents_idx, occupancy, beta=1.5):
    """Default: Boltzmann kernel, p(i→j) ∝ exp(−β·occ[j])"""
    V = occupancy.numel()
    adj = mem._cached_adj
    N  = agents_idx.numel()
    bonded = torch.full((N,), -1, dtype=torch.long)
    for a in range(N):
        v_curr = int(agents_idx[a])
        neigh  = adj[v_curr] + [v_curr]
        occ    = occupancy[neigh]
        w      = torch.exp(-beta * occ)
        probs  = w / w.sum()
        next_v = int(torch.multinomial(probs, 1).item())
        next_vid = neigh[next_v]
        if occupancy[next_vid] < 0.5:
            occupancy[next_vid] += 1.0
            bonded[a] = next_vid
        agents_idx[a] = next_vid
    return agents_idx, bonded

def edge_uniform_step(mem, agents_idx, occupancy, **kwargs):
    """Uniform random walk kernel (ignores occupancy)."""
    adj = mem._cached_adj
    N  = agents_idx.numel()
    bonded = torch.full((N,), -1, dtype=torch.long)
    for a in range(N):
        v_curr = int(agents_idx[a])
        neigh  = adj[v_curr] + [v_curr]
        next_vid = random.choice(neigh)
        if occupancy[next_vid] < 0.5:
            occupancy[next_vid] += 1.0
            bonded[a] = next_vid
        agents_idx[a] = next_vid
    return agents_idx, bonded



# === 2. Demo and evaluation routines (demo/entry-point) ===
from binding_membrane import BindingMembrane
from graph_express2 import ProcessGraph
from graph_express2printing import GraphExpresss2Printer

class PrintableProcessGraph(GraphExpresss2Printer, ProcessGraph):
    def __init__(self, recomb_level: int = 0, *, expand_complex: bool = False):
        ProcessGraph.__init__(self, recomb_level, expand_complex)

def demo_membrane_instance() -> None:
    mem = BindingMembrane(centre=(0., 0., 0.), radius=50.0)
    print("\n=== Collider sanity‑check ===")
    pos   = torch.tensor([[60., 0., 0.], [10., 0., 0.]])
    vel   = torch.tensor([[ 1., 0., 0.], [ 0., 0., 0.]])
    force = torch.zeros_like(pos)
    p2, v2, f2 = mem.check_bounds(pos, vel, force)
    for i in range(len(pos)):
        state = "outside" if pos[i,0] > mem.radius else "inside"
        print(f"• particle {i}: {state} → new_pos = {p2[i].tolist()}")

def demo_symbolic_graphs(recomb: int = 0) -> None:
    print("\n=== Symbolic graph demos ===")
    for name, expr in SYM_EXPR.items():
        print(f"\n--- {name} ---")
        pg = PrintableProcessGraph(recomb)
        if hasattr(pg, "build_from_expression"):
            pg.build_from_expression(expr, 1)
        else:
            pg.build_graph(expr)
            pg.finalize_graph_with_outputs()
        pg.compute_levels(method="alap")
        pg.print_bands_and_ops()

def surface_step(mem: BindingMembrane,
                 agents_pos: torch.Tensor,
                 occupancy: torch.Tensor,
                 bond_dist: float = 5.0,
                 drift: float = 2.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Single Euler‑step for agents living on the membrane surface."""
    radius = mem.radius
    mesh   = mem.mesh_verts
    N = agents_pos.shape[0]
    bonded = torch.full((N,), -1, dtype=torch.long)
    for i in range(N):
        if occupancy.any() and (mesh == agents_pos[i]).all(dim=1).any():
            continue
        d = torch.linalg.norm(mesh - agents_pos[i], dim=1)
        v_idx = int(torch.argmin(d))
        if occupancy[v_idx]:
            tangent = torch.randn(3)
            tangent -= (tangent @ agents_pos[i]) / (radius**2) * agents_pos[i]
            tangent = tangent / torch.linalg.norm(tangent) * drift
            new_pos = agents_pos[i] + tangent
            agents_pos[i] = radius * new_pos / torch.linalg.norm(new_pos)
        else:
            if d[v_idx] < bond_dist:
                occupancy[v_idx] = True
                agents_pos[i]    = mesh[v_idx]
                bonded[i]        = v_idx
            else:
                direction = (mesh[v_idx] - agents_pos[i])
                agents_pos[i] += 0.5 * direction / torch.linalg.norm(direction)
                agents_pos[i]  = radius * agents_pos[i] / torch.linalg.norm(agents_pos[i])
    return agents_pos, bonded

def demo_surface_evolution(steps: int = 10) -> None:
    print("\n=== Surface evolution toy‑sim ===")
    base = torch.tensor([[ 1, 0, 0], [-1, 0, 0],
                         [ 0, 1, 0], [ 0,-1, 0],
                         [ 0, 0, 1], [ 0, 0,-1]], dtype=torch.float32)
    mem = BindingMembrane(centre=(0., 0., 0.), radius=50.0)
    mem.mesh_verts = base * mem.radius
    V = mem.mesh_verts.shape[0]
    occupancy = torch.zeros(V, dtype=torch.bool)
    rng = torch.rand(3, 2)
    phi   = rng[:, 0] * 2 * math.pi
    costh = 2 * rng[:, 1] - 1.0
    sinth = torch.sqrt(1 - costh**2)
    agents_pos = mem.radius * torch.stack([sinth * torch.cos(phi),
                                           sinth * torch.sin(phi),
                                           costh], dim=1)
    for t in range(steps):
        agents_pos, bonded = surface_step(mem, agents_pos, occupancy)
        print(f"step {t:2d}: bonds → {bonded.tolist()}")
    print("\nFinal bonds:")
    for i, b in enumerate(bonded):
        if b >= 0:
            coord = mem.mesh_verts[b].tolist()
            print(f"  agent {i} bonded to vertex {b} at {coord}")
        else:
            print(f"  agent {i} still searching…")

def demo_diffusion(steps: int = 20, kernel: str = "boltzmann", beta: float = 1.2) -> None:
    print(f"\n=== Diffusion demo (kernel: {kernel}) ===")
    base = torch.tensor([[ 1, 0, 0], [-1, 0, 0],
                         [ 0, 1, 0], [ 0,-1, 0],
                         [ 0, 0, 1], [ 0, 0,-1]], dtype=torch.float32)
    faces = torch.tensor([[0,4,2],[2,4,1],[1,4,3],[3,4,0],
                          [0,2,5],[2,1,5],[1,3,5],[3,0,5]])
    mem = BindingMembrane(centre=(0.,0.,0.), radius=50.0)
    mem.mesh_verts = base * mem.radius
    mem.mesh_faces = faces
    mem._cached_adj = build_adjacency(mem.mesh_faces)
    V = mem.mesh_verts.shape[0]
    occupancy = torch.zeros(V)
    agents_idx = torch.randint(0, V, (3,))
    kernel_fn = DIFFUSION_KERNELS.get(kernel, edge_prob_step)
    for t in range(steps):
        if kernel == "boltzmann":
            agents_idx, bonded = kernel_fn(mem, agents_idx, occupancy, beta=beta)
        else:
            agents_idx, bonded = kernel_fn(mem, agents_idx, occupancy)
        print(f"step {t:02d}: agents @ {agents_idx.tolist()}  bonds → {bonded.tolist()}")
        if (bonded >= 0).all():
            break
    print("\nFinal occupancy:")
    for v_id, occ in enumerate(occupancy.tolist()):
        print(f"  v{v_id}: {int(occ)} agent(s)")

class EmbeddingManager:
    def __init__(self, boundary: BoundarySubgraph, queue: QueueSubgraph,
                 kernel: DiffusionKernel = BoltzmannKernel(beta=1.5)):
        self.boundary = boundary
        self.queue    = queue
        self.kernel   = kernel
        self.slip = SlipLayer(boundary, queue)


    # --- main entry -------------------------------------------------------

    def step(self, dt: float = 1.0, **kw) -> dict[str, torch.Tensor]:
        """
        Vectorised single-timestep update.
        Returns a dict of *views*, so the caller can sync back selectively.
        """
        # 1) let the diffusion kernel decide *where* each agent moves
        self.kernel.step(self.boundary, self.queue, dt,
                         slip_layer=self.slip,         # NEW
                         **kw)

        # 2) split or duplicate vertices if over-occupied
        self._resolve_vertex_overflow()

        # 3) optional physics (stretch energy, etc.)
        self._apply_stretch_energy(dt)
        self.slip.step(dt, force_field=kw.get("force_field"))

        # 4) package views for caller; nothing is cloned or detached
        return {
            "verts" : self.boundary.verts,
            "occ"   : self.boundary.occ,
            "pos"   : self.queue.pos,
            "state" : self.queue.state,
        }

    # ---------------------------------------------------------------------
    def run(self, n_steps: int, dt: float = 1.0, **kw):
        for _ in range(n_steps):
            self.step(dt, **kw)

    # -------- helpers -----------------------------------------------------
    def _resolve_vertex_overflow(self):
        """
        Any vertex with |occ| > 1 spawns a new vertex *in-place*.
        The new vertex inherits the original position plus an infinitesimal
        outward normal offset so the mesh remains manifold.
        """
        occ = self.boundary.occ
        multi = (occ.abs() > 1).nonzero(as_tuple=False).flatten()
        if not multi.numel():
            return

        v_new = self.boundary.verts[multi]         # (M,3) positions
        occ_delta = torch.where(occ[multi] > 0, 1, -1)
        self.boundary.verts = torch.cat([self.boundary.verts, v_new], dim=0)
        self.boundary.occ   = torch.cat([self.boundary.occ, occ_delta], dim=0)

        # NOTE: faces are *not* split here; call MeshGraphBridge if needed.

    def _apply_stretch_energy(self, dt):
        """
        Hook for quadratic edge-stretch penalty:
        E = k Σ‖v_i − v_j‖²  for every queue-to-vertex assignment.
        """
        # Build once, reuse
        if not hasattr(self, "_k_stretch"):
            self._k_stretch = 1.0
        qpos = self.queue.pos
        if (not hasattr(self.boundary, "assigned_vidx") or
            self.boundary.assigned_vidx.numel() != self.queue.pos.size(0)):
            # build / refresh vertex assignment (nearest-neighbour)
            d = torch.cdist(self.queue.pos, self.boundary.verts)
            self.boundary.assigned_vidx = d.argmin(dim=1)      # (N,)

        vpos = self.boundary.verts[self.boundary.assigned_vidx]  # (N,3)
        diff = qpos - vpos
        frc  = -2 * self._k_stretch * diff
        self.queue.pos = qpos + frc * dt  # explicit Euler


# ---------------------------------------------------------------------------
# NEW DEMO: full embedding manager interacting with legacy components
# ---------------------------------------------------------------------------

def demo_embedding_manager(steps: int = 25, res: int = 32) -> None:
    print("\n=== EmbeddingManager end-to-end demo ===")

    # 1) build a parametric sphere surface as the membrane reference --------
    sph = get_surface("unit_sphere")  # assume you registered earlier
    verts_np, faces_np = mesh_from_parametric(sph, res)
    verts = torch.tensor(verts_np, dtype=torch.float32) * 50.      # radius 50
    faces = torch.tensor(faces_np, dtype=torch.long)

    # 2) create Boundary & Queue sub-graphs --------------------------------
    boundary = BoundarySubgraph(verts, faces)
    #
    # queue starts with 6 agents (3 outside, 3 inside) at random directions
    rng   = torch.rand(6, 2)
    phi   = rng[:, 0] * 2 * math.pi
    costh = 2 * rng[:, 1] - 1.0
    sinth = torch.sqrt(1 - costh**2)
    pos0  = torch.stack([sinth * torch.cos(phi),
                         sinth * torch.sin(phi),
                         costh], dim=1)
    pos0[:3] *= 60.     # outer queue (radius > 50)
    pos0[3:] *= 40.     # inner queue (radius < 50)
    queue  = QueueSubgraph(pos0,
                           state=torch.tensor([+1,+1,+1, -1,-1,-1]))

    # 3) wiring through EmbeddingManager + Boltzmann kernel ----------------
    em = EmbeddingManager(boundary, queue,
                          kernel=BoltzmannKernel(beta=1.2))

    # 4) optional: world graph container -----------------------------------
    world = nx.Graph()
    world.add_node("boundary", obj=boundary)
    world.add_node("queue",    obj=queue)
    world.add_edge("boundary", "queue")

    # 5) time integration ---------------------------------------------------
    for t in range(steps):
        views = em.step(dt=1.0)
        n_bonded = (views["occ"].abs() > 0).sum().item()
        print(f"step {t:02d}:  bonded vertices = {n_bonded:>2}")

        # early exit if everyone bonded
        if n_bonded >= len(queue.pos):
            break

    # 6) summary ------------------------------------------------------------
    print("\nFinal vertex occupancy (id → occ):")
    for vid, occ in enumerate(boundary.occ.tolist()):
        if occ != 0:
            print(f"  {vid:4d} → {int(occ):+d}")

    print("\nQueue agent final radii:")
    radii = torch.linalg.norm(queue.pos, dim=1).tolist()
    for i, r in enumerate(radii):
        print(f"  agent {i}: r = {r:6.2f}")

# --- keep all kernel code above here --------------------------------------
DIFFUSION_KERNELS = {
    "walker"   : MembraneWalkerKernel,
    "boltzmann": edge_prob_step,
    "uniform"  : edge_uniform_step,
}

# ---------------------------------------------------------------------------
# Hook into main script guard
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
#  SlipLayer  – bonded-but-mobile agents that may “plooOIIP!” through
# ---------------------------------------------------------------------------
class SlipLayer:
    """
    Owns ONLY the queue rows that are already bonded to a membrane vertex
    but still allowed to slide tangentially.

    Workflow each frame
    -------------------
    1. `enrol(mask)`          – kernel passes which freshly-bonded agents
       should join the slip set.
    2. `step(dt, force_field)` – performs:
         • local one-ring vertex swaps to lower spring stretch
         • calls `self.escape_probability(E_bar, pressure)`   ← overridable!
         • probabilistically expels some agents to the opposite side

    You can subclass SlipLayer and override *either* of:
        escape_probability(E_bar, pressure)
        pressure(force_field)
    to plug in osmotic or stoichiometric logic.
    """

    def __init__(self,
                 boundary: BoundarySubgraph,
                 queue:    QueueSubgraph,
                 capture_thickness: float = 3.0,
                 k_spring: float = 1.0,
                 comfort_E0: float = 1.0,
                 alpha: float = 4.0):
        self.boundary = boundary
        self.queue    = queue

        self.k_spring = float(k_spring)
        self.E0       = float(comfort_E0)
        self.alpha    = float(alpha)

        self.R_outer   = float(torch.norm(boundary.verts, dim=1).max())
        self.R_capture = self.R_outer - capture_thickness

        # dynamic buffers
        self._active     = torch.zeros(0, dtype=torch.bool)  # queue rows
        self._idx_buffer = torch.empty(0, dtype=torch.long)  # viewer idx

        # pre-compute adjacency once
        self._adj = build_adjacency(boundary.faces)

    # ------------------------------------------------------------------ api
    def enrol(self, mask: torch.Tensor, viewer_indices: torch.Tensor):
        """
        mask            : boolean (Nq,) over queue rows that just bonded.
        viewer_indices  : same length, giving the parent-simulation row id.
        """
        if not mask.any():
            return
        if not self._active.numel():
            self._active     = mask.clone()
            self._idx_buffer = viewer_indices.clone()
        else:
            self._active     = torch.cat([self._active,     mask])
            self._idx_buffer = torch.cat([self._idx_buffer, viewer_indices])

    def step(self, dt: float = 1.0,
                   force_field: torch.Tensor | None = None):
        """
        dt           : seconds
        force_field  : viewer-side force tensor (Nx,3) – optional.
                       If provided, pressure() can use it.
        """
        if not self._active.any():
            return

        aidx = torch.nonzero(self._active).flatten()        # active rows
        verts = self.boundary.verts
        occ   = self.boundary.occ
        qpos  = self.queue.pos
        vidx  = self.boundary.assigned_vidx[aidx]

        # ----------- 1) local swap to minimise stretch  -------------------
        for i, agent in enumerate(aidx):   # small slip set → tiny loop OK
            v_cur = int(vidx[i])
            neigh = self._adj[v_cur] + [v_cur]
            d2 = ((qpos[agent] - verts[neigh])**2).sum(dim=1)
            best = neigh[int(d2.argmin())]
            if best != v_cur and occ[best] == 0:
                occ[v_cur] -= 1
                occ[best]  += 1
                vidx[i]     = best

        self.boundary.assigned_vidx[aidx] = vidx  # keep global mapping

        # ----------- 2) stretch energy & pressure  ------------------------
        diff   = qpos[aidx] - verts[vidx]
        E_bar  = 0.5 * self.k_spring * (diff.norm(dim=1)**2).mean()

        P  = self.pressure(force_field, aidx, E_bar)  # customisable
        Pe = self.escape_probability(E_bar, P)        # customisable

        # ----------- 3) probabilistic expulsion  --------------------------
        mask_expel = torch.rand_like(self._active, dtype=torch.float32) < Pe
        if mask_expel.any():
            expelled = torch.nonzero(mask_expel).flatten()
            self._active[expelled] = False

            # small nudge to leave slip annulus so capture code doesn’t
            # immediately re-enrol on next frame
            nrm = verts[self.boundary.assigned_vidx[expelled]]
            nrm = nrm / (nrm.norm(dim=1, keepdim=True) + 1e-12)
            qpos[expelled] += 0.2 * nrm * self.queue.state[expelled].float().unsqueeze(1)

    # ================================ overridable hooks ===================
    def pressure(self, force_field, aidx, E_bar) -> float:
        """
        Default: synthetic pressure proportional to mean inward/outward
        force magnitude on active agents.
        """
        if force_field is None:
            return 0.0
        f_mag = force_field[self._idx_buffer[aidx]].norm(dim=1).mean()
        return f_mag.item()

    def escape_probability(self, E_bar: float, pressure: float) -> float:
        """
        Default: logistic that rises when   E_bar  falls below comfort_E0.
        Pressure can modulate α or shift the curve if desired; for now we
        simply ignore it (you can override).
        """
        return torch.sigmoid(self.alpha * (self.E0 - E_bar)).item()

    # ---------------------------------------------------------------------
    def active_indices(self) -> torch.Tensor:
        "Return viewer indices currently in slip layer."
        return self._idx_buffer[self._active]
class OsmoticSlip(SlipLayer):
    def pressure(self, force_field, aidx, E_bar):
        # example: mean dot(n, force) where n is membrane normal
        n = self.boundary.verts[self.boundary.assigned_vidx[aidx]]
        n = n / (n.norm(dim=1, keepdim=True) + 1e-12)
        if force_field is None: return 0.
        return (force_field[self._idx_buffer[aidx]] * n).sum(dim=1).mean().abs().item()

    def escape_probability(self, E_bar, P):
        # stronger inward pressure lowers pop-through chance
        return torch.sigmoid(self.alpha * (self.E0 - E_bar) - 0.3 * P)

if __name__ == "__main__":
    random.seed(0)
    torch.manual_seed(0)
    demo_membrane_instance()
    demo_symbolic_graphs(recomb=0)
    demo_surface_evolution(steps=12)
    demo_diffusion(steps=20, kernel="boltzmann")

    # NEW demo call --------------------------------------------------------
    demo_embedding_manager(steps=30)
