from turtle import width
import sympy
import numpy as np
from sympy import Sum, IndexedBase, Idx, symbols, Function
import networkx as nx
from colorama import Fore, Style, init
from solver_types import Operation, NodeSet, Node, READWRITE, DomainNode, Edge
from operator_defs import default_funcs, operator_signatures
from ilpscheduler import ILPScheduler
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import colorsys
SIMD_DEFAULT_CONCURRENCY = 4  # default concurrency for SIMD operations

import sympy
import numpy as np
from sympy import Sum, IndexedBase, Idx, symbols, Function

from physics import (
    vis_viva_eq,
    sch_eq,
    efe,
    laplace_de_rham,
    fourier_transform,
    A, B,
    maxwell_eq,
    field_tensor,
    Tμν_em,
    wave_eq,
    dirac_eq,
)
from dec import DEC
from orbital import Orbit
# Assume mesh_dual_edge and mesh_voronoi_cell are defined in physics.py as well

M_val, N_val, K_val = 3, 6, 9
M, N, K = symbols('M N K', integer=True)
i, j, k = Idx('i', M), Idx('j', N), Idx('k', K)
A_t, B_t, C_t = IndexedBase('A'), IndexedBase('B'), IndexedBase('C')
x, y, z = symbols('x y z', real=True)

test_suite = [
    {
        'name': "1 + 1",
        'expr_fn': lambda: sympy.Integer(1) + sympy.Integer(1),
        'dims': (1,),
        'data_sources': lambda: {},
        'expected_fn': lambda ds: 2
    },
    {
        'name': "(x + y) * z (irreducible)",
        'expr_fn': lambda: (sympy.Symbol('x') + sympy.Symbol('y')) * sympy.Symbol('z'),
        'dims': (1,),
        'data_sources': lambda: {'x': 2, 'y': 3, 'z': 4},
        'expected_fn': lambda ds: (ds['x'] + ds['y']) * ds['z']
    },
    {
        'name': "Elementwise A + C",
        'expr_fn': lambda ii, jj: A_t[ii, jj] + C_t[ii, jj],
        'dims': (M_val, N_val),
        'data_sources': lambda: {
            'A': np.random.rand(M_val, N_val),
            'C': np.random.rand(M_val, N_val)
        },
        'expected_fn': lambda ds: ds['A'] + ds['C']
    },
    {
        'name': "Dot product across K",
        'expr_fn': lambda ii, jj: Sum(A_t[ii, k] * B_t[k, jj], (k, 0, K_val - 1)),
        'dims': (M_val, N_val),
        'data_sources': lambda: {
            'A': np.random.rand(M_val, K_val),
            'B': np.random.rand(K_val, N_val)
        },
        'expected_fn': lambda ds: np.dot(ds['A'], ds['B'])
    },
    {
        'name': "Elementwise A * C",
        'expr_fn': lambda ii, jj: A_t[ii, jj] * C_t[ii, jj],
        'dims': (M_val, N_val),
        'data_sources': lambda: {
            'A': np.random.rand(M_val, N_val),
            'C': np.random.rand(M_val, N_val)
        },
        'expected_fn': lambda ds: ds['A'] * ds['C']
    },
    {
        'name': "(A + C) ** 2",
        'expr_fn': lambda ii, jj: (A_t[ii, jj] + C_t[ii, jj])**2,
        'dims': (M_val, N_val),
        'data_sources': lambda: {
            'A': np.random.rand(M_val, N_val),
            'C': np.random.rand(M_val, N_val)
        },
        'expected_fn': lambda ds: (ds['A'] + ds['C']) ** 2
    },
    {
        'name': "3D Tensor Outer Product",
        'expr_fn': lambda ii, jj, kk: A_t[ii, jj] * B_t[kk, jj] + C_t[ii, kk],
        'dims': (M_val, N_val, K_val),
        'data_sources': lambda: {
            'A': np.random.rand(M_val, N_val),
            'B': np.random.rand(K_val, N_val),
            'C': np.random.rand(M_val, K_val)
        },
        'expected_fn': lambda ds: np.einsum('ij,kj->ikj', ds['A'], ds['B']) + np.broadcast_to(ds['C'][:, None, :], (M_val, N_val, K_val))
    },
    {
        'name': "Elementwise Sin + Exp",
        'expr_fn': lambda ii, jj: sympy.sin(A_t[ii, jj]) + sympy.exp(C_t[ii, jj]),
        'dims': (M_val, N_val),
        'data_sources': lambda: {
            'A': np.random.rand(M_val, N_val),
            'C': np.random.rand(M_val, N_val)
        },
        'expected_fn': lambda ds: np.sin(ds['A']) + np.exp(ds['C'])
    },
    {
        'name': "DEC Laplace-de Rham (symbolic, generic)",
        'expr_fn': lambda: DEC.laplace_de_rham(Function('φ'), Function('vol_dual_cell'), Function('vol_dual_edge')),
        'dims': (1,),
        'data_sources': lambda: {},
        'expected_fn': lambda ds: None
    },
    {
        'name': "DEC Laplace-de Rham (mesh dual functions)",
        'expr_fn': lambda: DEC.laplace_de_rham(Function('φ'), DEC.mesh_voronoi_cell, DEC.mesh_dual_edge),
        'dims': (1,),
        'data_sources': lambda: {},
        'expected_fn': lambda ds: None
    },
    {
        'name': "DEC Voronoi Dual Cell Area (vertex 0, tris)",
        'expr_fn': lambda: DEC.mesh_voronoi_cell(0, [(0,1,2),(0,2,3),(0,3,1)], sympy.IndexedBase('V')),
        'dims': (1,),
        'data_sources': lambda: {},
        'expected_fn': lambda ds: None
    },
    {
        'name': "DEC Dual Edge Length (0,2,1,3)",
        'expr_fn': lambda: DEC.mesh_dual_edge(0, 2, 1, 3, sympy.IndexedBase('V')),
        'dims': (1,),
        'data_sources': lambda: {},
        'expected_fn': lambda ds: None
    },
    {
        'name': "DEC Circumcenter2D (0,1,2)",
        'expr_fn': lambda: DEC.circumcenter2D(0, 1, 2, sympy.IndexedBase('V')),
        'dims': (1,),
        'data_sources': lambda: {},
        'expected_fn': lambda ds: None
    },
    {
        'name': "Symbolic Fourier Transform (from physics.py)",
        'expr_fn': lambda: fourier_transform,
        'dims': (1,),
        'data_sources': lambda: {},
        'expected_fn': lambda ds: None
    },
    {
        'name': "Vis Viva Equation (Orbital Mechanics)",
        'expr_fn': lambda: vis_viva_eq,
        'dims': (1,),
        'data_sources': lambda: {},
        'expected_fn': lambda ds: None
    },
    {
        'name': "Schrödinger Equation (physics.py)",
        'expr_fn': lambda: sch_eq,
        'dims': (1,),
        'data_sources': lambda: {},
        'expected_fn': lambda ds: None
    },
    {
        'name': "Einstein Field Equation (symbolic, 4x4)",
        'expr_fn': lambda: efe,
        'dims': (4, 4),
        'data_sources': lambda: {},
        'expected_fn': lambda ds: None
    },
    {
        'name': "DEC Laplace-de Rham (physics.py symbol)",
        'expr_fn': lambda: laplace_de_rham,
        'dims': (1,),
        'data_sources': lambda: {},
        'expected_fn': lambda ds: None
    },
    {
        'name': "Magnetic Vector Potential (A)",
        'expr_fn': lambda: A,
        'dims': (3,),  # 3-vector
        'data_sources': lambda: {},
        'expected_fn': lambda ds: None
    },
    {
        'name': "Magnetic Field (B = curl A)",
        'expr_fn': lambda: B,
        'dims': (3,),
        'data_sources': lambda: {},
        'expected_fn': lambda ds: None
    },
    {
        'name': "Maxwell Equation (covariant symbolic)",
        'expr_fn': lambda: maxwell_eq,
        'dims': (4, 4),
        'data_sources': lambda: {},
        'expected_fn': lambda ds: None
    },
    {
        'name': "Field Tensor F^{μν}=∂^μA^ν-∂^νA^μ",
        'expr_fn': lambda: field_tensor,
        'dims': (4, 4),
        'data_sources': lambda: {},
        'expected_fn': lambda ds: None
    },
    {
        'name': "Stress-Energy Tensor T^{μν}",
        'expr_fn': lambda: Tμν_em,
        'dims': (4, 4),
        'data_sources': lambda: {},
        'expected_fn': lambda ds: None
    },
    {
        'name': "Wave Equation (physics.py)",
        'expr_fn': lambda: wave_eq,
        'dims': (1,),
        'data_sources': lambda: {},
        'expected_fn': lambda ds: None
    },
    {
        'name': "Dirac Equation (physics.py)",
        'expr_fn': lambda: dirac_eq,
        'dims': (1,),
        'data_sources': lambda: {},
        'expected_fn': lambda ds: None
    },
    {
        'name': "Orbital symbolic transfer EOM",
        'expr_fn': lambda: Orbit.stable_orbit_transfer_solution(
            Orbit.symbolic_orbit('1'),
            Orbit.symbolic_orbit('2')
        )['equation_of_motion'],
        'dims': (1,),
        'data_sources': lambda: {},
        'expected_fn': lambda ds: None
    },
]


init(autoreset=True)
import colorsys

MAX_HUES = 12  # maximum distinct hues before wrapping
def multi_sort(collection, key_funcs):
    compound_keys = [
        tuple(func(item) for func in key_funcs)
        for item in collection
    ]
    items_with_keys = list(zip(collection, compound_keys))
    items_with_keys.sort(key=lambda x: x[1])
    return [item for item, _ in items_with_keys]
def get_color(level, num_levels, max_hues=MAX_HUES):
    """Generate ANSI TrueColor escape for a level normalized to num_levels (capped at max_hues)."""
    # Determine effective hue count
    hue_count = min(num_levels, max_hues)
    # Normalize level into [0,1)
    h = (level % hue_count) / hue_count
    r, g, b = colorsys.hsv_to_rgb(h, 1, 1)
    return f"\033[38;2;{int(r*255)};{int(g*255)};{int(b*255)}m"

# fallback static list (unused)
colors = []
import torch

class ExpressionTensor:
    def __init__(self, data, contexts=None, sequence_length=1, domain_shape=None, function_index=None):
        self.data = data
        self.contexts = contexts or [0]
        self.sequence_length = sequence_length
        self.domain_shape = domain_shape or self._infer_shape(data)
        self.function_index = function_index

    def _infer_shape(self, data):
        shape = []
        while isinstance(data, list):
            shape.append(len(data))
            data = data[0] if data else []
        return tuple(shape)

    @property
    def shape(self):
        return self.domain_shape

    def __getitem__(self, idx):
        return self.data[idx]

    def __setitem__(self, idx, value):
        self.data[idx] = value

    def to_numpy(self):

        # Helper to walk the nested structure and collect types
        def collect_types(x, depth=0):
            #print(f"Collecting types from: {x} at depth {depth}")
            types = []
            if isinstance(x, list):
                for item in x:
                    types.extend(collect_types(item, depth + 1))
            else:
                types.append(type(x))
            return types

        try:
            arr = np.array(self.data)
            # if object dtype, fallback to advanced conversion
            if arr.dtype == object:
                raise ValueError("object-dtype array, falling back to advanced conversion")
            return arr
        except Exception as e:
            # Fallback: recursively stack nested lists of arrays/scalars into one ndarray
            import numpy as _np
            def _recurse_stack(d):
                if isinstance(d, list):
                    subs = [_recurse_stack(x) for x in d]
                    if not subs:
                        raise ValueError("Empty list cannot be stacked")
                    shapes = [s.shape for s in subs]
                    dtypes = [s.dtype for s in subs]
                    if len(set(shapes)) == 1 and len(set(dtypes)) == 1:
                        return _np.stack(subs, axis=0)
                    else:
                        raise ValueError("Inconsistent shapes or dtypes in nested data")
                # leaf: convert scalar or array
                arr = _np.array(d)
                return arr
            try:
                return _recurse_stack(self.data)
            except Exception:
                pass

            # On failure, report types in the nested structure
            print("⚠️ Failed to convert ExpressionTensor to numpy array.")
            print(f"Error: {e}")
            all_types = collect_types(self.data)
            from collections import Counter
            type_counts = Counter(all_types)
            print("Types found in tensor data:")
            for t, count in type_counts.items():
                print(f"  {t}: {count}")
            raise ValueError("Mixed types detected in ExpressionTensor data, cannot safely convert to numpy.") from e


    def __array__(self):
        return self.to_numpy()

    def __repr__(self):
        return f"ExpressionTensor(shape={self.shape}, data={self.data})"


class ProcessGraph:
    def __init__(self, recombinatorics_level=0, expand_complex=False):
        self.G = nx.DiGraph()
        self.levels = {}
        self.node_map = {}
        # integer level for recombinatorics aggressiveness: 0=no, higher unlock more transforms
        self.recombinatorics_level = recombinatorics_level
        self.expand_complex = expand_complex
        self.domain_shape = ()
        self.roots = []
        self.role_schemas = {
            'IndexedBase': {'up':{'base':1}, 'down':{}},
            'Indexed': {'up':{'base':1, 'index':'many'},'down':{}},
            'Idx': {'up':{'limits': 'many'}, 'down':{}},
            'Sum': {'up':{'body': 1, 'limits': 'many'}, 'down':{}},
            # etc - you can expand this for functions, FFTs, etc.
        }
        self.scheduler = ILPScheduler(self)
        

    def full_recombinatorics(self, expr, level=1):
        """
        Apply symbolic transforms with increasing aggressiveness based on level:
        level>=1: doit
        level>=2: expand
        level>=3: expand_mul, expand_power_exp
        level>=4: expand_log, trigsimp
        level>=5: cancel, apart
        level>=6: factor, simplify
        """
        if level >= 1:
            expr = expr.doit()
        if level >= 2:
            expr = sympy.expand(expr, power_exp=True, log=True,
                                 multinomial=True, complex=self.expand_complex, trig=True)
        if level >= 3:
            expr = sympy.expand_mul(expr)
            expr = sympy.expand_power_exp(expr)
        if level >= 4:
            expr = sympy.expand_log(expr)
            expr = sympy.trigsimp(expr)
        if level >= 5:
            expr = sympy.cancel(expr)
            try:
                expr = sympy.apart(expr)
            except Exception:
                pass
        if level >= 6:
            expr = sympy.factor(expr)
            expr = sympy.simplify(expr)
        return expr
    def ensure_node(self, node, store_id=None):
        nid = id(node)
        if nid not in self.G:
            node_type = type(node).__name__
            #print(f"Building graph node: type={type(node).__name__}, repr={repr(node)}")

            sig = operator_signatures.get(node_type, operator_signatures['Default'])
            extra_args = {}
            for param in sig.get('parameters', []):
                value = getattr(node, param, None)
                if value is not None:
                    extra_args[param] = value
            self.G.add_node(nid,
                label=str(node),
                type=node_type,
                expr_obj=node,
                extra_args=extra_args,
                domain_node=DomainNode(
                    shape=(1,1,1), #default will be function pointer
                    unit_size=1,  # default unit size for function pointers
                ),
                store_id=store_id,
                parents=[],
                children=[])
            self.node_map[nid] = node
            return nid, False
        else:
            return nid, True  # return nid and flag if already defined

    def connect(self, src_id, tgt_id, producer_role, consumer_role, store_id=None):
        src_label = self.G.nodes[src_id]['label'] if src_id in self.G.nodes else "??"
        tgt_label = self.G.nodes[tgt_id]['label'] if tgt_id in self.G.nodes else "??"

        edge = Edge(
            id=(src_id, tgt_id, producer_role, consumer_role),
            operation=None,
            source=src_id,
            target=tgt_id,
            store_id=store_id
        )
        if not self.G.has_edge(src_id, tgt_id):
            self.G.add_edge(src_id, tgt_id, extra=set())
        if 'extra' not in self.G[src_id][tgt_id]:
            self.G[src_id][tgt_id]['extra'] = set()
        self.G[src_id][tgt_id]['extra'].add(edge)

        if 'children' not in self.G.nodes[src_id]:
            self.G.nodes[src_id]['children'] = []
        if 'parents' not in self.G.nodes[tgt_id]:
            self.G.nodes[tgt_id]['parents'] = []
        if tgt_id not in [p for p, _ in self.G.nodes[src_id]['children']]:
            self.G.nodes[src_id]['children'].append((tgt_id, producer_role))
        if src_id not in [p for p, _ in self.G.nodes[tgt_id]['parents']]:
            self.G.nodes[tgt_id]['parents'].append((src_id, consumer_role))

    def _recurse_spec(self, nid, args, spec, direction, store_id=None):
        i = 0
        for role, param in spec.items():
            if param == 1:
                if direction == 'down':
                    self.build_graph(args[i], producer_id=nid, producer_role=role, consumer_role=f"arg{i}", store_id=store_id)
                else:
                    self.build_graph(args[i], consumer_id=nid, producer_role="output", consumer_role=role, store_id=store_id)
                i += 1
            elif param == 'many':
                while i < len(args):
                    if direction == 'down':
                        self.build_graph(args[i], producer_id=nid, producer_role=role, consumer_role=f"arg{i}", store_id=store_id)
                    else:
                        self.build_graph(args[i], consumer_id=nid, producer_role="output", consumer_role=role, store_id=store_id)
                    i += 1
            elif isinstance(param, tuple):
                num = param[1] if len(param) == 2 else param[0]
                for _ in range(num):
                    if i < len(args):
                        if direction == 'down':
                            self.build_graph(args[i], producer_id=nid, producer_role=role, consumer_role=f"arg{i}", store_id=store_id)
                        else:
                            self.build_graph(args[i], consumer_id=nid, producer_role="output", consumer_role=role, store_id=store_id)
                        i += 1


    def build_graph(self, node, producer_id=None, consumer_id=None, producer_role=None, consumer_role=None, store_id=None):
        if not self.domain_shape:
            self.domain_shape = (1,)

        nid, already_defined = self.ensure_node(node, store_id)

        if already_defined:
            # just hook up to parents or consumers and exit
            if producer_id is not None:
                self.connect(producer_id, nid, producer_role, consumer_role, store_id)
            if consumer_id is not None:
                self.connect(nid, consumer_id, producer_role, consumer_role, store_id)
            return nid

        node_type = type(node).__name__
        schema = self.role_schemas.get(node_type, None)
        args = list(getattr(node, 'args', []))

        if schema:
            self._recurse_spec(nid, args, schema.get('up', {}), direction='up', store_id=store_id)
            self._recurse_spec(nid, args, schema.get('down', {}), direction='down', store_id=store_id)
        else:
            for idx, arg in enumerate(args):
                self.build_graph(arg, consumer_id=nid, producer_role="output", consumer_role=f'arg{idx}', store_id=store_id)

        # now that we've fully resolved, connect this node in the context given
        if producer_id is not None:
            self.connect(producer_id, nid, producer_role, consumer_role, store_id)
        if consumer_id is not None:
            self.connect(nid, consumer_id, producer_role, consumer_role, store_id)
        if producer_id is None and consumer_id is None:
            self.roots.append(nid)

        return nid



    def finalize_graph_with_outputs(self):
        """
        Ensure every node satisfies its min_outputs.
        If missing, generate explicit Store nodes to fulfill output slots.
        """
        for nid in list(self.G.nodes):
            node_data = self.G.nodes[nid]
            op_type = node_data['type']
            sig = operator_signatures.get(op_type, operator_signatures['Default'])
            min_outputs = sig.get('min_outputs', 1)
            current_outputs = len(node_data['children'])
            store_id = node_data.get('store_id', None)
            while current_outputs < min_outputs:
                store_label = f"Store[{nid}:{current_outputs}]"
                store_node_id = id(store_label)
                self.G.add_node(
                    store_node_id,
                    label=store_label,
                    type="Store",
                    domain_node=DomainNode(
                        id=store_node_id,
                        shape=(1, 1, 1),  # default shape for store nodes
                        unit_size=1,  # default unit size for store nodes
                    ),#when you get back fix datasetexclusivity in data graph

                    store_id=store_id,
                    expr_obj=store_label,
                    parents=[(nid, 'value')],
                    children=[]
                )
                node_data['children'].append((store_node_id, 'value'))

                edge = Edge(
                    id = (nid, store_node_id, 'output', 'result'),
                    operation = None,
                    source = nid,
                    store_id = store_id,
                    target = store_node_id,
                )
                self.G.add_edge(nid, store_node_id, extra=[edge])

                current_outputs += 1
    def group_edges_by_dataset(self, dataG):
        """
        Returns a nested dict grouping each edge by the (role, level, type) tuples found in its 'extras'.
        Structure: { role: { level: { typ: [ (src, tgt), ... ] } } }
        """
        grouped = {}
        for src, tgt, attrs in dataG.edges(data=True):
            for ds in attrs.get('extras', []):
                role, level, typ = ds
                # initialize nested dicts if needed
                if level not in grouped:
                    grouped[level] = {}
                if typ not in grouped[level]:
                    grouped[level][typ] = {}
                if role not in grouped[level][typ]:
                    grouped[level][typ][role] = []
                # append the edge tuple
                grouped[level][typ][role].append((src, tgt))
        return grouped

    def check_set_involvement(self, node, nodeset):
        """
        Check if a node is involved in a nodeset.
        Returns True if the node is part of the nodeset, False otherwise.
        """
        for (role, lvl, typ), candidate_node in nodeset:
            if candidate_node == node:
                return (role, lvl, typ)  # return the role, level, type if involved
        return None  # not involved in this nodeset

    def create_data_flow_dag(self, nodesets, uG):
        dataG = nx.DiGraph()
        datasets = {}   # will map dataset_id -> set of DomainNode.id
        for dataset_id, ns in nodesets.items():         # unpack the dict item
            datasets[dataset_id] = set()
            # for each process node in this nodeset
            for member in ns.member_nodes:              # Node objects
                proc_nid = member.id                     # matches uG’s node IDs
                if proc_nid not in uG:
                    continue
                dom_node = uG.nodes[proc_nid]['domain_node']
                datasets[dataset_id].add(dom_node.id)

                # add the domain node as a vertex in the new DAG
                dataG.add_node(
                    dom_node.id,
                    label=uG.nodes[proc_nid]['label'],
                    type=uG.nodes[proc_nid]['type'],
                    original_node=proc_nid,
                    domain_node=dom_node,
                    dataset_id=dataset_id,
                )

            import itertools

            # … after you’ve added all nodes …

            # now add every uG edge (in or out) for each member
            for member in ns.member_nodes:
                n = member.id
                for src, tgt in itertools.chain(uG.in_edges(n), uG.out_edges(n)):
                    dom_src = uG.nodes[src]['domain_node'].id
                    dom_tgt = uG.nodes[tgt]['domain_node'].id

                    if dataG.has_edge(dom_src, dom_tgt):
                        dataG.edges[dom_src, dom_tgt].setdefault('extras', []).append(dataset_id)
                    else:
                        dataG.add_edge(dom_src, dom_tgt, extras=[dataset_id])

        return dataG


    def compute_levels(self, method='asap', order='processing', interference_mode='asap-maxslack'):
        """
        Compute levels using ILPScheduler.
        - method='asap' for earliest
        - method='alap' for latest
        """
        self.finalize_graph_with_outputs()  # ensure min_outputs satisfied

        self.levels = self.scheduler.compute_levels(method, order)
        self.proc_interference_graph, self.proc_lifespans = self.compute_asap_maxslack_interference(interference_mode)
        self.produce_proc_and_mem_bins(self.proc_lifespans)
        self.universal_graph_interference_bins = self.merge_proc_and_mem_graphs(self.G, self.mG, self.process_bins, self.memory_bins, self.proc_interference_graph)
        self.nodesets = self.condense_to_nodesets()
        self.dataG = self.create_data_flow_dag(self.nodesets, self.uG)
        #print exauhstive summary of items produced
        print(f"Levels computed: {len(self.levels)} nodes")
        print(f"Process interference graph: {len(self.proc_interference_graph.nodes)} nodes, {len(self.proc_interference_graph.edges)} edges")
        print(f"Memory interference graph: {len(self.mG.nodes)} nodes, {len(self.mG.edges)} edges")
        print(f"Process bins: {len(self.process_bins)} bins")
        print(f"Memory bins: {len(self.memory_bins)} bins")
        print(f"Nodesets: {len(self.nodesets)} sets")
        print(f"Recombinatorics level: {self.recombinatorics_level}")
        print(f"Domain shape: {self.domain_shape}")
        print(f"Universal graph: {len(self.uG.nodes)} nodes, {len(self.uG.edges)} edges")
        print(f"Universal interference bins: {len(self.uGI.nodes)} nodes, {len(self.uGI.edges)} edges")
        print(f"Universal interference graph: {len(self.uGI.nodes)} nodes, {len(self.uGI.edges)} edges")

    def extract_full_process_graph(self):
        nodes = {}
        for nid, data in self.G.nodes(data=True):
            nodes[nid] = {
                'type': data['type'],
                'label': data['label'],
                'expr_obj': data['expr_obj'],
                'parents': list(data['parents']),
                'children': list(data['children']),
                'level': self.levels.get(nid),
            }
        levels_map = {}
        for nid, lvl in self.levels.items():
            levels_map.setdefault(lvl, []).append(nid)
        # include roots list so consumer knows final outputs
        return {'nodes': nodes, 'levels': levels_map, 'roots': list(self.roots)}

    def build_from_expression(self, expr_or_tensor, *domain_dims):
        if isinstance(expr_or_tensor, tuple) and isinstance(expr_or_tensor[1], ExpressionTensor):
            registry, et = expr_or_tensor
            print(registry)
            
            self.domain_shape = et.domain_shape
            self.roots = []
            def expr_fn(*indices):
                idx = et.data[0, -1][indices].item()
                return registry[idx]
            self.build_lateral_graph_across_domain(*self.domain_shape, expr_fn)
        elif callable(expr_or_tensor):
            self.build_lateral_graph_across_domain(*domain_dims, expr_or_tensor)
        else:
            # treat as single scalar SymPy expression (or trivial 1D shape)
            self.domain_shape = (1,)
            self.roots = []
            self.build_graph(expr_or_tensor)


    def to_sympy(self):
        meta = self.extract_full_process_graph()
        nodes_meta = meta['nodes']
        cache = {}

        def emit(nid):
            if nid in cache:
                return cache[nid]
            m = nodes_meta[nid]
            typ = m['type']

            role_map = {}
            for p, role in m['parents']:
                value = emit(p)
                role_map.setdefault(role, []).append(value)

            if typ in ('Store', 'Output'):
                out = emit(m['parents'][0][0])
            elif typ == 'ImaginaryUnit':
                out = sympy.I
            elif typ == 'Symbol':
                out = sympy.Symbol(m['label'])
            elif typ == 'Integer':
                out = sympy.Integer(int(m['label']))
            elif typ in ('One','Zero','NegativeOne'):
                val = {'One':1,'Zero':0,'NegativeOne':-1}[typ]
                out = sympy.Integer(val)
            elif typ == 'IndexedBase':
                out = sympy.IndexedBase(m['label'])
            elif typ == 'Indexed':
                base = role_map.get("base", [])[0]
                indices = role_map.get("index", [])
                out = sympy.Indexed(base, *indices)
            elif typ == 'Idx':
                if "limit" in role_map and len(role_map["limit"]) == 2:
                    out = sympy.Idx(m['label'], (role_map["limit"][0], role_map["limit"][1]))
                elif "limit" in role_map and len(role_map["limit"]) == 1:
                    out = sympy.Idx(m['label'], role_map["limit"][0])
                else:
                    out = sympy.Idx(m['label'])
            elif typ in ('Mul','Add','Pow','Tuple'):
                cls = {'Mul': sympy.Mul, 'Add': sympy.Add, 'Pow': sympy.Pow, 'Tuple': sympy.Tuple}[typ]
                all_args = sum(role_map.values(), [])
                out = cls(*all_args, evaluate=False)
            elif typ == 'Sum':
                expr_obj = m['expr_obj']
                out = sympy.Sum(expr_obj.args[0], expr_obj.limits)
            else:
                expr_obj = m['expr_obj']
                all_args = sum(role_map.values(), [])
                if isinstance(expr_obj, sympy.Function):
                    out = expr_obj.func(*all_args)
                else:
                    raise ValueError(f"Unhandled type: {typ}")
            
            cache[nid] = out
            return out

        # --- Build nested list of expressions from roots ---
        roots_copy = self.roots.copy()

        def build_nested_list(emit_fn, roots, shape):
            if len(shape) == 1:
                return [emit_fn(roots.pop(0)) for _ in range(shape[0])]
            return [build_nested_list(emit_fn, roots, shape[1:]) for _ in range(shape[0])]

        nested_list_exprs = build_nested_list(emit, roots_copy, self.domain_shape)

        # --- Flatten for simplification ---
        def flatten_nested_list(nested):
            flat = []
            for item in nested:
                if isinstance(item, list):
                    flat.extend(flatten_nested_list(item))
                else:
                    flat.append(item)
            return flat

        flat_exprs = flatten_nested_list(nested_list_exprs)

        # --- Simplify / CSE ---
        simplified = [self.full_recombinatorics(e, self.recombinatorics_level) for e in flat_exprs] \
                    if self.recombinatorics_level > 0 else flat_exprs

        replacements, reduced_exprs = sympy.cse(simplified)

        # --- Build registry (defs first, then main) ---
        expression_registry = []
        registry_defs_count = 0

        for sym, defn in replacements:
            expression_registry.append(sympy.Tuple(sym, defn))
            registry_defs_count += 1

        main_start = registry_defs_count
        for expr in reduced_exprs:
            expression_registry.append(expr)

        # --- Build nested list of indices matching domain shape ---
        flat_indices = list(range(main_start, main_start + len(reduced_exprs)))

        def rebuild_nested_list(shape, flat):
            if len(shape) == 1:
                return [flat.pop(0) for _ in range(shape[0])]
            return [rebuild_nested_list(shape[1:], flat) for _ in range(shape[0])]

        nested_list_indices = rebuild_nested_list(self.domain_shape, flat_indices.copy())

        # --- Convert to torch tensor ---
        indices_tensor = torch.tensor(nested_list_indices, dtype=torch.long)
        expr_tensor_data = indices_tensor.unsqueeze(0).unsqueeze(0)  # add context and sequence dims

        # --- Build ExpressionTensor ---
        et = ExpressionTensor(
            contexts=[0],
            sequence_length=1,
            domain_shape=self.domain_shape,
            function_index=None
        )
        et.data = expr_tensor_data

        return expression_registry, et

    def run(self, data_sources, operator_funcs=None):
        import numpy as np
        if operator_funcs is None:
            operator_funcs = {}

        results = {}

        # Compose final lookup
        op_dispatch = {**default_funcs, **operator_funcs}

        # Traverse levels in order
        for lvl in sorted(set(self.levels.values())):
            for nid, node_level in self.levels.items():
                if node_level != lvl:
                    continue

                node_data = self.G.nodes[nid]
                typ = node_data['type']
                parents = node_data['parents']

                if not parents:
                    results[nid] = data_sources.get(node_data['label'], node_data['expr_obj'])
                else:
                    role_map = {}
                    for parent_id, role in parents:
                        val = results[parent_id]
                        role_map.setdefault(role, []).append(val)
                    func = op_dispatch.get(typ)
                    if not func:
                        raise TypeError(f"No handler for node type '{typ}'")
                    results[nid] = func(role_map)



        # Build the nested structure according to store_id
        tensor_data = self._create_nested_data_container(self.domain_shape)

        for nid, data in self.G.nodes(data=True):
            break #diagnostic avoidance
            if data['type'] == 'Store':

                if isinstance(results[nid], np.ndarray):
                    # diagnostic dump
                    print(f"Node {nid} ({node_data['label']}): Result is numpy array with shape {results[nid].shape}")
                    
                

                store_idx = data.get('store_id')
                value = results.get(nid)
                if store_idx is not None:
                    self._insert_into_nested(tensor_data, store_idx, value)

        return ExpressionTensor(data=tensor_data, domain_shape=self.domain_shape)

    def consumer_at(self, src):
        return self.consumer_queues[src] if src in self.consumer_queues else None
    
    def run_process_node(self, src, data):
        producer = self.G.nodes[src].get('expr_obj', None)
        if producer is None:
            raise ValueError(f"No producer function found for node {src}")
        if isinstance(data, ExpressionTensor):
            data = data.to_numpy()
        return producer(data)


    def run_at(self, level, type, role):
        """
        Run the graph at a specific level, type, and role.
        Returns a list of results for that level/type/role.
        """
        results = []
        # 2) get nested grouping: { level → { type → { role → [ (src, tgt), … ] } } }
        grouped = self.group_edges_by_dataset(self.dataG)

        # 3) flatten that grouping into an ordered sequence (level→type→role)
        ordered_keys = []
        for lvl in sorted(grouped):
            for typ in sorted(grouped[lvl]):
                for role in sorted(grouped[lvl][typ]):
                    ordered_keys.append((role, lvl, typ))
        total = len(ordered_keys)
        for idx, (role, lvl, typ) in enumerate(ordered_keys):
            if role == "input":
                for src, tgt in grouped[lvl][typ][role]:
                    self.dataG.nodes[src].get('domain_node').put((tgt, self.consumer_at(src)))
            elif role == "intermediate":
                for src, tgt in grouped[lvl][typ][role]:
                    is_writing = tgt in self.mG.nodes
                    if is_writing:
                        next_tgts = self.dataG.edges[src, tgt].get('extras', [])
                        if next_tgts:
                            for next_tgt in next_tgts:
                                new_data = self.run_process_node(src, self.dataG.nodes[src].get('domain_node').get(tgt))
                                self.dataG.nodes[tgt].get('domain_node').put((next_tgt, new_data))
                    else:
                        self.dataG.nodes[tgt].get('domain_node').put(self.dataG.nodes[src].get('domain_node').get(tgt))
            elif role == "output":
                return self.dataG.nodes[src].get('domain_node').get()
        return results

    def merge_proc_and_mem_graphs(self, proc_graph, mem_graph, proc_bins, mem_bins, proc_interference_graph):
        """
        Merge process and memory graphs into a single graph.
        """
        self.uG = nx.DiGraph()
        self.uGI = nx.DiGraph()  # interference graph
        self.uG.add_nodes_from(proc_graph.nodes(data=True))
        self.uG.add_nodes_from(mem_graph.nodes(data=True))
        self.uGI.add_nodes_from(self.uG.nodes(data=True))
        self.uGI.add_edges_from(mem_graph.edges(data=False))
        self.uGI.add_edges_from(proc_interference_graph.edges(data=False))

        universal_graph_interference_bins = []

        for idx, (stage1, stage2) in enumerate(zip(mem_bins, proc_bins)):
            while len(universal_graph_interference_bins) <= idx:
                universal_graph_interference_bins.append([])

            for node in stage1:
                if node in mem_graph:
                    for src, dst, data in proc_graph.edges(data=True):
                        for extra_item in data.get('extra', []):
                            # our memory node ids come from the Edge subedge that defined them
                            if id(extra_item) == node and (src in stage2 or dst in stage2):
                                self.uG.add_edge(src, node, label=f"{self.G.nodes[src]['label']} -> {self.mG.nodes[node]['label']}")
                                self.uG.add_edge(node, dst, label=f"{self.mG.nodes[node]['label']} -> {self.G.nodes[dst]['label']}")
                                if not universal_graph_interference_bins[idx]:
                                    universal_graph_interference_bins[idx] = []
                                #for all permutations of src, node, and dst, add edges
                                for perm in self.tuple_perms((src, node, dst), 2):
                                    self.uGI.add_edge(*perm)
                                universal_graph_interference_bins[idx].append(node)
                                universal_graph_interference_bins[idx].append(src)
                                universal_graph_interference_bins[idx].append(dst)
        return universal_graph_interference_bins
    def tuple_perms(self, tup, r):
        """Generate all r-length permutations of the input tuple."""
        from itertools import permutations
        return list(permutations(tup, r))
    def _create_nested_data_container(self, shape):
        """Create an empty nested list structure of given shape."""
        if not shape:
            return None
        if len(shape) == 1:
            return [None] * shape[0]
        return [self._create_nested_data_container(shape[1:]) for _ in range(shape[0])]

    def _insert_into_nested(self, container, index_tuple, value):
        """Insert value into nested list structure at index_tuple."""
        sub = container
        for idx in index_tuple[:-1]:
            sub = sub[idx]
        sub[index_tuple[-1]] = value
    def produce_proc_and_mem_bins(self, lifespans):
        """
        Produce process and memory bins from lifespans.
        Returns process_bins, memory_bins, min_time, max_time.
        """
        process_bins, memory_bins, min_time, max_time = self.bin_lifespans_to_bins(lifespans)
        self.process_bins = process_bins
        self.memory_bins = memory_bins
        self.min_time = min_time
        self.max_time = max_time
        return process_bins, memory_bins, min_time, max_time
    def print_lifespans_ascii(self, width=50, sort_keys=None):
        """
        Prints an ASCII visualization of lifespans.

        :param width: width of timeline
        :param sort_keys: optional list of key functions for multi-level sort
                        Defaults to ascending start, then descending end.
        """


        for label, bins in [("process", self.process_bins),
                            ("memory", self.memory_bins),
                            ("universal", self.universal_graph_interference_bins)]:
            if not bins:
                print(f"No lifespans to visualize for {label}.")
                continue
            scale = width // (self.max_time - self.min_time + 1)

            # Build node lifespans
            node_lifespans = {}
            for idx, bin_nodes in enumerate(bins):
                for node in bin_nodes:
                    if node not in node_lifespans:
                        node_lifespans[node] = [idx, idx]
                    else:
                        node_lifespans[node][1] = idx

            # Convert to summary records
            node_summaries = [
                {'id': node, 'start': start, 'end': end, 'duration': end - start}
                for node, (start, end) in node_lifespans.items()
            ]

            # Determine sort
            if sort_keys is None:
                sort_keys = [
                    lambda x: x['start'],      # ascending start
                    lambda x: x['end']        # ascending end
                ]

            sorted_nodes = multi_sort(node_summaries, sort_keys)

            # Print
            print(f"\n=== Lifespan Timeline ({label}) ===")
            print(f"Time range: [{self.min_time}, {self.max_time}]")

            for node_info in sorted_nodes:
                node, start, end, duration = (node_info['id'], node_info['start'],
                                            node_info['end'], node_info['duration'])
                line = [' '] * width
                scaled_start = start * scale
                scaled_end = (end+1) * scale
                for i in range(scaled_start, min(scaled_end, width)):
                    line[i] = '#'
                print(f"Node {node}: |{''.join(line)}| start={start} end={end} duration={duration}")

    def bin_lifespans_to_bins(self, lifespans):
        """
        Converts lifespans into bins where each bin contains a list of node IDs.
        """
        # Determine global min/max time
        min_time = min(start for start, end in lifespans.values())
        max_time = max(end for start, end in lifespans.values())
        total_span = max_time - min_time
        start_time = min_time
        offset = 0

        if start_time < 0:
            # If start time is negative, adjust min_time to 0
            min_time = 0
            max_time += -start_time
            offset = -start_time
            total_span += offset

        
        bins = [[] for _ in range(total_span + 1)]
        memory_bins = [[] for _ in range(total_span + 1)]
        
        self.mG = nx.DiGraph()  # memory graph for edges
        for node, (start, end) in lifespans.items():
            start += offset
            end += offset

            start_idx = (start - min_time)
            end_idx = (end - min_time)

            

            for i in range(start_idx, end_idx + 1):
                bins[i].append(node)

        for idx, bin in enumerate(bins):
            # for each bin,  establish output and input edges as concurrent memory need nodes in the memory bins to make a concurrency graph of storage demands
            if bin:
                for node in bin:
                    for (src, dst, extra) in self.G.edges(node, data='extra'):
                        if extra:
                            for edge in extra:
                                if isinstance(edge, Edge):
                                    # check the schema of the edge for domain node shape hints
                                    # in the event of "many" count, we need to obtain the true shape
                                    # at the present moment
                                    source_node = self.G.nodes[src]
                                    target_node = self.G.nodes[dst]
                                    target_type = target_node['type']
                                    shape = (1,)  # default shape for domain nodes
                                    if target_type in self.role_schemas:
                                        if 'base' in self.role_schemas[target_type]['up']:
                                            # all items get domain nodes but base items
                                            # will have a size associated with them

                                            shape = self.role_schemas[target_type]['up']['base']
                                            
                                            if shape == 'many':
                                                symbolic_engine_object = source_node.get('expr_obj', None)
                                                if symbolic_engine_object is not None:
                                                    print(f"Symbolic engine object for source node: {symbolic_engine_object}")
                                                shape = symbolic_engine_object.shape if hasattr(symbolic_engine_object, 'shape') else (1,)
                                    

                                    domain_node = DomainNode(
                                        id=id(edge),
                                        shape=shape
                                    )
                                    self.mG.add_node(
                                        domain_node.id,
                                        label=f"Memory for: {source_node['label']} -> {target_node['label']}",
                                        domain_node=domain_node,
                                        type='Memory',
                                        store_id=source_node.get('store_id', None),
                                    )
                                    memory_bins[idx].append(domain_node.id)
                                    # we don't extend the domain node over an additional idx
                                    # because it's tracking the process nodes that already
                                    # extend their lifespan over the same idx
        for bin in memory_bins:
            if bin:
                nodes_in_bin = set(bin)
                # Create edges between all nodes in the bin
                for i, src in enumerate(nodes_in_bin):
                    for j, dst in enumerate(nodes_in_bin):
                        if src != dst:
                            self.mG.add_edge(src, dst)

        return bins, memory_bins, min_time, max_time

    

    def compute_asap_maxslack_interference(self, mode='asap-maxslack'):
        interference_graph, lifespans = self.scheduler.compute_asap_maxslack_interference(mode)

        
        return interference_graph, lifespans

    def lateral_graph_merge(self, graphs_meta):
        offset = max(self.G.nodes)+1 if self.G.nodes else 0
        for G_loc, lv_loc, nm_loc in graphs_meta:
            id_map = {old:offset+i for i,old in enumerate(G_loc.nodes)}
            for old,new in id_map.items():
                d=G_loc.nodes[old]
                self.G.add_node(new, **{k:d[k] for k in ['label','type','expr_obj']}, parents=set(), children=set())
                self.node_map[new]=nm_loc[old]
                self.levels[new]=lv_loc[old]
            for u,v in G_loc.edges:
                uu, vv = id_map[u], id_map[v]
                self.G.add_edge(uu,vv)
                self.G.nodes[uu]['children'].add(vv)
                self.G.nodes[vv]['parents'].add(uu)
            offset+=len(G_loc.nodes)

    def group_by_level_and_type(self):
        grouping={}
        for nid in self.G.nodes:
            lvl=self.levels[nid]; tp=self.G.nodes[nid]['type']
            grouping.setdefault(lvl,{}).setdefault(tp,[]).append(nid)
        return grouping

    def build_lateral_graph_across_domain(self, *dims_and_expr):
        *dims, expr_fn = dims_and_expr

        self.domain_shape = dims
        self.roots = []

        def recurse_build(index_prefix, remaining_dims):
            if not remaining_dims:
                try:
                    base_expr = expr_fn(*index_prefix)
                except TypeError:
                    base_expr = expr_fn()
                expr = self.full_recombinatorics(base_expr, self.recombinatorics_level) if self.recombinatorics_level > 0 else base_expr
                self.build_graph(expr, store_id=index_prefix)
                
            else:
                for i in range(remaining_dims[0]):
                    recurse_build(index_prefix + (i,), remaining_dims[1:])

        recurse_build((), dims)
        

    def parse_requirements(self, proc_graph):
        nodes = proc_graph['nodes']
        levels_map = proc_graph['levels']
        # map node id to its level
        id2lvl = {nid: lvl for lvl, ids in levels_map.items() for nid in ids}
        # classified node collections
        input_nodes = {}
        intermediate_nodes = {}
        output_nodes = {}
        operations = {}
        # build Operation objects and classify parent/child roles
        for nid, data in nodes.items():
            lvl = id2lvl[nid]
            sig = operator_signatures.get(data['type'], operator_signatures['Default'])
            op = Operation(
                id=nid,
                inputs=data['parents'],
                max_inputs=sig['max_inputs'],
                outputs=data['children'],
                max_outputs=sig['max_outputs'],
                string=data['label'],
                type=data['type'],
                sequence_order=lvl,
                time_penalty=0.0
            )
            operations[nid] = op
            # classify inputs vs intermediates by examining grandparents
            for parent_id, _ in data['parents']:
                grandparents = nodes[parent_id]['parents']
                if grandparents:
                    intermediate_nodes.setdefault(lvl, {}).setdefault(data['type'], {}).setdefault(nid, set()).add(parent_id)
                else:
                    input_nodes.setdefault(lvl, {}).setdefault(data['type'], {}).setdefault(nid, set()).add(parent_id)
            # classify outputs vs intermediates by examining grandchildren
            for child_id, _ in data['children']:
                grandchildren = nodes[child_id]['children']
                if grandchildren:
                    intermediate_nodes.setdefault(lvl, {}).setdefault(data['type'], {}).setdefault(nid, set()).add(child_id)
                else:
                    output_nodes.setdefault(lvl, {}).setdefault(data['type'], {}).setdefault(nid, set()).add(child_id)
        # convert all sets to lists
        for grouping in (input_nodes, intermediate_nodes, output_nodes):
            for lvl, type_dict in grouping.items():
                for typ, nid_dict in type_dict.items():
                    for op_id in list(nid_dict):
                        nid_dict[op_id] = list(nid_dict[op_id])
        return input_nodes, intermediate_nodes, output_nodes, operations
    
    def condense_to_nodesets(self, proc_graph=None):
        """
        After building graph, optionally condense inputs, intermediates and outputs into NodeSets,
        grouped by (type, level). Returns a dict of NodeSets keyed by (role, level, type).
        """
        if proc_graph is None:
            proc_graph = self.extract_full_process_graph()
        nodes = proc_graph['nodes']
        levels_map = proc_graph['levels']

        # classify by role (input/inter/output)
        inputs, intermediates, outputs, operations = self.parse_requirements(proc_graph)

        nodesets = {}
            

        def create_nodesets(node_group, role):

            for lvl, type_dict in node_group.items():
                for typ, nid_dict in type_dict.items():
                    ids = list(nid_dict)
                    # Determine trivial shape
                    shape = (len(ids), 1, 1)
                    ns = NodeSet(*shape)
                    ns.member_nodes = [Node(id=nid,
                                            location_in_set=ns.nd_from_flat(i),
                                            location_in_memory=self.uG.nodes[nid].get('domain_node', None),
                                            readwrite=READWRITE)
                                       for i, nid in enumerate(ids)]
                    nodesets[(role, lvl, typ)] = ns

        create_nodesets(inputs, "input")
        create_nodesets(intermediates, "intermediate")
        create_nodesets(outputs, "output")
        

        return nodesets
    
    def serialize_bands(self):
        bands={}
        for nid,lvl in self.levels.items():
            tp=self.G.nodes[nid]['type']; lbl=self.G.nodes[nid]['label']
            bands.setdefault(lvl,{}).setdefault(tp,[]).append(lbl)
        return bands

    def print_parallel_bands(self):
        """
        Colorized print of parallel execution bands.
        """
        bands = self.serialize_bands()
        num_levels = len(bands)
        print("\n=== Parallel execution bands ===")
        for lvl in sorted(bands):
            color = get_color(lvl, num_levels)
            print(f"{color}Level {lvl}:{Style.RESET_ALL}")
            for tp, labels in bands[lvl].items():
                print(f"{color}  {tp}:{Style.RESET_ALL}")
                for lbl in labels:
                    print(f"{color}    - {lbl}{Style.RESET_ALL}")

    def print_colorized_operations(self):
        """
        Colorized print of operations sorted by dependencies.
        """
        proc = self.extract_full_process_graph()
        _, _, _, ops = self.parse_requirements(proc)
        print("\n=== Operations ===")
        # normalize hues to sequence levels count
        seqs = [op.sequence_order for op in ops.values()]
        num_levels = max(seqs) + 1 if seqs else 1
        for op in Operation.default_sort(ops.values()):
            lvl = op.sequence_order
            color = get_color(lvl, num_levels)
            print(f"{color}{op}{Style.RESET_ALL}")

    def print_colorized_expressions(self):
        """
        Inline serial colorization: show each node's full symbolic expression with component levels colored.
        """
        proc = self.extract_full_process_graph()
        nodes_meta = proc['nodes']
        num_levels = len(proc['levels'])  # for hue normalization
        print("\n=== Dependency fabric with color-coded parent inclusions ===")
        for nid, data in nodes_meta.items():
            text = data['label']
            for pid, _ in data['parents']:
                parent_label = nodes_meta[pid]['label']
                parent_level = nodes_meta[pid]['level']
                color = get_color(parent_level, num_levels)
                if parent_label in text:
                    import re
                    pattern = re.compile(re.escape(parent_label))
                    text = pattern.sub(f"{color}[{parent_label}]{Style.RESET_ALL}", text, count=1)


            print(f"Expr: {text}")

    def print_bands_and_ops(self):
        # Colorized parallel bands, expressions, and operations
        self.print_parallel_bands()
        self.print_colorized_expressions()
        self.print_colorized_operations()


    def plot_simple_graph(self, graph, layout='spring'):
        """
        Plots a simple flowchart-like view of the graph without edge labels.
        """
        plt.figure(figsize=(12, 8))
        if layout == 'spring':
            pos = nx.spring_layout(graph, seed=42)
        elif layout == 'shell':
            pos = nx.shell_layout(graph)
        else:
            pos = nx.spring_layout(graph, seed=42)
        labels = nx.get_node_attributes(graph, 'label')
        nx.draw(graph, pos, with_labels=True, labels=labels, node_size=800,
                node_color='lightblue', edge_color='gray', font_size=10)
        plt.show()

    def plot_graph_with_roles(self, layout='spring'):
        """
        Plots the graph with edge labels showing producer->consumer roles from Edge.extra.
        """
        plt.figure(figsize=(12, 8))
        if layout == 'spring':
            pos = nx.spring_layout(self.G, seed=42)
        elif layout == 'shell':
            pos = nx.shell_layout(self.G)
        else:
            pos = nx.spring_layout(self.G, seed=42)
        
        nx.draw(self.G, pos, with_labels=True, node_size=800, node_color='lightblue',
                edge_color='gray', font_size=10)
        
        # Extract edge roles
        edge_labels = {}
        for u, v, data in self.G.edges(data=True):
            # Build label from your Edge extras
            extras = data.get('extra', [])
            label_parts = []
            for e in extras:
                if hasattr(e, 'id') and len(e.id) >= 4:
                    label_parts.append(f"{e.id[2]}→{e.id[3]}")
            edge_labels[(u, v)] = ", ".join(label_parts)
        
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels, font_size=8)
        plt.show()


    def get_rgb_color(level, num_levels):
        """Convert a level to an RGB tuple."""
        hue = (level % num_levels) / num_levels
        r, g, b = colorsys.hsv_to_rgb(hue, 1, 1)
        return (r, g, b)

    def animate_data_flow(self, dataG, duration=5, fps=30):
        """
        Animate the data flow graph by cycling through datasets present in edge 'extras'.
        Colors nodes and edges belonging to the current dataset.
        """
        # 1) layout
        pos = nx.shell_layout(dataG)

        # 2) get nested grouping: { level → { type → { role → [ (src, tgt), … ] } } }
        grouped = self.group_edges_by_dataset(dataG)

        # 3) flatten that grouping into an ordered sequence (level→type→role)
        ordered_keys = []
        for lvl in sorted(grouped):
            for typ in sorted(grouped[lvl]):
                for role in sorted(grouped[lvl][typ]):
                    ordered_keys.append((role, lvl, typ))
        total = len(ordered_keys)

        # 4) draw static background
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title("Data Flow Animation")
        ax.axis('off')
        nx.draw_networkx_edges(dataG, pos, ax=ax, edge_color='lightgray')
        node_collection = nx.draw_networkx_nodes(dataG, pos, ax=ax, node_color='lightgray')
        nx.draw_networkx_labels(dataG, pos, ax=ax, font_size=8)

        # 5) keep handles to each edge line + its extras
        edge_lines = {}
        for (u, v, attrs) in dataG.edges(data=True):
            line = ax.plot(
                [pos[u][0], pos[v][0]],
                [pos[u][1], pos[v][1]],
                color='lightgray',
                linewidth=2
            )[0]
            edge_lines[(u, v)] = (line, attrs.get('extras', []))

        # 6) animation update uses only ordered_keys
        def update(frame):
            current = ordered_keys[frame % total]
            ax.set_title(f"Dataset: {current}")

            # highlight edges in this dataset
            for (u, v), (line, extras) in edge_lines.items():
                if current in extras:
                    line.set_color('red')
                    line.set_linewidth(3)
                else:
                    line.set_color('lightgray')
                    line.set_linewidth(1)

            # highlight connected nodes
            highlights = {
                u for (u, v), (_, extras) in edge_lines.items() if current in extras
            } | {
                v for (u, v), (_, extras) in edge_lines.items() if current in extras
            }
            node_colors = [
                'blue' if n in highlights else 'lightgray'
                for n in dataG.nodes()
            ]
            node_collection.set_color(node_colors)
            return list(edge_lines.keys()) + [node_collection]

        frames = total * fps
        anim = FuncAnimation(fig, update, frames=frames, interval=1000/fps, blit=False, repeat=True)
        plt.show()

# Example usage:
# animate_data_flow(pg.dataG)

# ----------------------------
# Demo execution (compartmentalized to main)
# ----------------------------
# ----------------------------
def main():

    # ----------------------------
    # Unified runner
    # ----------------------------
    def run(process_graph, data_sources, expected_fn):
        try:
            result = process_graph.run(data_sources, default_funcs)
            expected = expected_fn(data_sources)
            if isinstance(result, sympy.Basic):
                # If symbolic, turn into numeric function
                symbols = sorted(result.free_symbols, key=lambda s: s.name)
                func = sympy.lambdify(symbols, result, modules='numpy')
                values = [data_sources[str(s)] for s in symbols]
                numeric_result = func(*values)
                assert np.allclose(numeric_result, expected), \
                    f"Graph symbolic did not match expected: {numeric_result} vs {expected}"
            elif isinstance(result, torch.Tensor):
                # If tensor, convert to numpy and compare
                result_np = result.numpy()
                expected_np = np.array(expected)
                assert np.allclose(result_np, expected_np), \
                    f"Graph tensor did not match expected: {result_np} vs {expected_np}"
            else:
                assert np.allclose(result, expected), \
                    f"Graph numeric did not match expected: {result} vs {expected}"
            print("✅ Test passed. Graph matches expected value.")
        except Exception as e:
            print(f"❌ Test failed: {e}")
            
        return result



        


    # ----------------------------
    # Execute all tests
    # ----------------------------
    for idx, test in enumerate(test_suite):
        print(f"\n=== Running test {idx+1}: {test['name']} ===")
        pg = ProcessGraph(5, False)
        pg.build_from_expression(test['expr_fn'], *test['dims'])
        
        
        
        #print("\n--- ASAP schedule ---")
        #pg.compute_levels(method='asap')
        #pg.print_parallel_bands()
        
        #print("\n--- ALAP schedule ---")
        #pg.compute_levels(method='alap')
        #pg.print_parallel_bands()

        #print("\n--- Maxmimum Slack Schedule ---")
        #pg.compute_levels(method='max_slack') 
        #pg.print_parallel_bands()

        # run the original data correctness
        pg.compute_levels(method='alap')  # use ASAP for correct run to match tests
        #pg.print_lifespans_ascii()
        data_sources = test['data_sources']()
        pg.animate_data_flow(pg.dataG, duration=5, fps=2)
        #pg.plot_simple_graph(pg.dataG, layout='shell')
        #print(run(pg, data_sources, test['expected_fn']))


if __name__ == "__main__":
    main()
