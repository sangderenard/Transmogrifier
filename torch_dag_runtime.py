# ── imports ───────────────────────────────────────────────────────────────
import torch, threading

from collections import defaultdict, deque
import networkx as nx
import re
from typing import Dict, Mapping, Any
from graph_deep_compiler import GraphDeepCompiler
from operator_defs import default_funcs, operator_signatures, numpy_funcs, torch_funcs, numpy_sigs, torch_sigs
# global device for all torch tensors
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")  # for testing purposes
class RingFull(Exception):
    """Raised when `_alloc_row` makes a full lap around the ring."""
    pass

_rand_re = re.compile(r"rand\((.*?)\)$")   # captures '3,4' etc.
# ─────────────────────────────────────────────────────────────────────────────

# ─── utils --------------------------------------------------------------
def _full_mask(R: int) -> int:                    # all-roles-present constant
    return (1 << R) - 1

# ─── base class with the mask logic ------------------------------------
class _PinnedNodeBufferBase:
    __slots__ = ("roles", "role_idx", "Q", "C",
                 "_buf", "_row_mask", "_row_jid", "_write", "_outbox",
                 "_full")

    def _init_core(self, in_roles, Q, C, dtype, device):
        self.roles    = tuple(in_roles)
        self.role_idx = {r:i for i,r in enumerate(self.roles)}
        self.Q, self.C = Q, C
        self._full    = _full_mask(len(self.roles))      # e.g. 0b111 for 3 roles

        self._row_mask = torch.zeros(Q, dtype=torch.int64, pin_memory=True)
        self._row_jid  = torch.full( (Q,), -1, dtype=torch.int64, pin_memory=True)
        self._write    = 0
        self._outbox   = {}

    # ------ helpers shared by torch / numpy impls ----------------------
    def _row_ready(self, row: int) -> bool:
        return self._row_mask[row] == self._full
    @torch.inference_mode()
    def num_free_rows(self):
        # torch / numpy both expose _row_mask & _row_jid
        return int((self._row_jid == -1).sum())     # free ⇔ jid == -1
    def store_output(self, jid: int, value):
        """Called by executors to cache the node’s result for one job."""
        self._outbox[jid] = value

    def fetch_output(self, jid: int):
        """Fetch (but keep) a previously stored output tensor/array."""
        return self._outbox[jid]
class PinnedNodeBufferTorch(_PinnedNodeBufferBase):
    def __init__(self, in_roles, col_cap, queue_cap=4096,
                 dtype=torch.float32, device=DEVICE):          # ← use DEVICE by default
        self._init_core(in_roles, queue_cap, col_cap, dtype, device)
        self._jid2row = {}
        # allocate buffer on DEVICE
        cpu_pinned = (device.type == "cpu")          # helper flag
        self._buf = torch.empty(
            (self.Q, len(self.roles), self.C),
            dtype=dtype,
            pin_memory=cpu_pinned,                   # <<< only on CPU
            device=device,
        )


    @torch.inference_mode()
    def add_input(self, jid:int, role:str, tensor:torch.Tensor):
        # fast‐path: lookup existing row
        row = self._jid2row.get(jid)
        if row is None:
            row = self._alloc_row(jid)
            self._jid2row[jid] = row
        ridx = self.role_idx[role]
        flat = tensor.reshape(-1).to(self._buf.dtype, copy=False)
        self._grow_cols_if_needed(flat.numel())
        self._buf[row, ridx, :flat.numel()] = flat
        self._row_mask[row] |= (1 << ridx)

    
    def _alloc_row(self, jid:int)->int:
        start = self._write
        while self._row_jid[self._write] != -1:
            self._write = (self._write + 1) & (self.Q - 1)
            if self._write == start:
                raise RingFull
        row, self._write = self._write, (self._write + 1) & (self.Q - 1)
        self._row_jid[row]  = jid
        self._row_mask[row] = 0
        return row

    def _grow_cols_if_needed(self, n:int):
        if n <= self.C: return
        newC = 1 << (n-1).bit_length()
        new = torch.empty(
            (self.Q, len(self.roles), newC),
            dtype=self._buf.dtype,
            pin_memory=(self._buf.device.type == "cpu"),   # same rule
            device=self._buf.device,
        )
        new[..., :self.C].copy_(self._buf)
        self._buf, self.C = new, newC

    @torch.inference_mode()
    def ready_jobs(self):
        ready_idx = (self._row_mask == self._full).nonzero(as_tuple=False).squeeze(1)
        if not ready_idx.numel():
            return ([], [])

        rows = ready_idx.tolist()                       # pure Python ints
        jids = self._row_jid[ready_idx].tolist()        # ditto

        for jid in jids:                                # safe to hash now
            self._jid2row.pop(jid, None)

        # mark slots free
        self._row_jid[ready_idx]  = -1
        self._row_mask[ready_idx] = 0
        return jids, rows


    def pop_inputs(self, row:int):
        return self._buf[row]
import numpy as np
class PinnedNodeBufferNpy(_PinnedNodeBufferBase):
    def __init__(self, in_roles, col_cap, queue_cap=4096, dtype=np.float32):
        # init core… override torch arrays with NumPy
        self.roles    = tuple(in_roles)
        self.role_idx = {r:i for i,r in enumerate(self.roles)}
        self.Q, self.C = queue_cap, col_cap
        self._full    = _full_mask(len(self.roles))
        self._row_mask = np.zeros(self.Q, dtype=np.int64)
        self._row_jid  = np.full(self.Q, -1, dtype=np.int64)
        self._write    = 0
        self._outbox   = {}
        self._buf = np.empty((self.Q, len(self.roles), self.C), dtype=dtype)
        self._jid2row = {}                # ← add missing map

    # pure‐NumPy free‐rows
    def num_free_rows(self):
        return int((self._row_jid == -1).sum())

    def add_input(self, jid: int, role: str, arr: np.ndarray):
        row = self._jid2row.get(jid)
        if row is None:
            row = self._alloc_row(jid)
            self._jid2row[jid] = row
        ridx = self.role_idx[role]
        flat = np.asarray(arr, dtype=self._buf.dtype).reshape(-1)
        self._grow_cols(flat.size)
        self._buf[row, ridx, :flat.size] = flat
        self._row_mask[row] |= (1 << ridx)
    # ————————————————————————————————————————————————————————
    # implement missing NumPy methods
    def _alloc_row(self, jid:int)->int:
        start = self._write
        while self._row_jid[self._write] != -1:
            self._write = (self._write + 1) & (self.Q - 1)
            if self._write == start:
                raise RuntimeError("Pinned ring full")
        row, self._write = self._write, (self._write + 1) & (self.Q - 1)
        self._row_jid[row]  = jid
        self._row_mask[row] = 0
        return row

    def _grow_cols(self, n:int):
        if n <= self.C: return
        newC = 1 << (n-1).bit_length()
        new  = np.empty((self.Q, len(self.roles), newC), dtype=self._buf.dtype)
        new[..., :self.C] = self._buf
        self._buf, self.C = new, newC

    def ready_jobs(self):
        rows = np.where(self._row_mask == self._full)[0]
        if rows.size == 0:
            return ([], [])

        jids = self._row_jid[rows].tolist()        # already 1-D
        for jid in jids:          # ➋ remove from map
            self._jid2row.pop(jid, None)
        self._row_jid[rows]  = -1
        self._row_mask[rows] = 0
        return jids, rows.tolist()

    def pop_inputs(self, row:int):
        return self._buf[row]
import time

# ─── Pump thread adjustments ────────────────────────────────────────────────
class PinnedBufferFeeder(threading.Thread):
    def __init__(self,
                 buffer   : "PinnedNodeBufferBase",
                 roles     : Mapping[str, int],
                 queue_cap : int   = 10,
                 poll_delay: float = 0.0005):
        super().__init__(daemon=True)
        self.buf         = buffer
        self.role_idx    = roles
        self.queue_cap   = queue_cap
        self.poll_delay  = poll_delay
        self.idle_delay  = poll_delay * 10          # longer sleep when idle
        self.pending : deque = deque()              # FIFO queue
        self._stop = threading.Event()

    def enqueue(self, job_id: int, data: Dict[str, torch.Tensor]):
        # just append, no random-access needed
        self.pending.append((job_id, data))

    def stop(self):
        self._stop.set()

    # ---------- main loop ---------------------------------------------------
    def run(self):
        while not self._stop.is_set():
            if not self.pending:
                time.sleep(self.idle_delay)
                continue

            free = self.buf.num_free_rows()
            if free == 0:
                time.sleep(self.poll_delay)
                continue

            # consume up to `free` jobs
            for _ in range(min(free, len(self.pending))):
                jid, payload = self.pending.popleft()
                again = False
                for role, tensor in payload.items():
                    flat = tensor.reshape(-1)
                    C    = self.buf.C
                    if flat.numel() > C:
                        self.enqueue(jid, {role: flat[C:]})
                        flat = flat[:C]
                        again = True
                    self.buf.add_input(jid, role, flat)
                # head is enqueued; tail if split has been re-queued

            # adaptive sleep
            time.sleep(self.poll_delay if self.pending else self.idle_delay)


# ─────────────────────────────────────────────────────────────────────────────
class FlatNodeSpec:
    """Lightweight execution record."""
    def __init__(self, nid, handler, in_roles, out_edges, is_symbol, const, is_output=False):
        self.nid      = nid
        self.handler  = handler
        self.buffer   = PinnedNodeBuffer(in_roles, col_cap=2)  # pre-allocated
        self.out_edges = out_edges          # [(dst_nid, param_name), ...] for routing
        self.is_symbol = is_symbol              # ↳ special handler
        self.const     = const                  # job-agnostic tensor
        self.is_output = is_output              # is this an output node?
        

    def attach_feeder(self, queue_cap=1024, poll=5e-4):
        """Spawn a helper thread that streams dict-style jobs into `buffer`."""
        if hasattr(self, "feeder"):
            return self.feeder                       # already attached
        self.feeder = PinnedBufferFeeder(self.buffer,
                                         self.buffer.role_idx,
                                         queue_cap   = queue_cap,
                                         poll_delay  = poll)
        self.feeder.start()
        return self.feeder
# ─────────────────────────────────────────────────────────────────────────────
class TorchDAGRuntimeSimple:
    """
    DAG executor with per-node hash buffers.
    Assumes the DAG is already topologically level-sorted for correctness.
    """
    def __init__(self, level_list, node_specs):
        self.levels = level_list            # [[nid, ...], ...]
        self.nodes  = node_specs            # {nid: FlatNodeSpec}
        self.active_jobs: set[int] = set()  
    # ---------- job submission ------------------------------------------------
    def submit(self, job_id: int, inputs: dict):
        self.active_jobs.add(job_id)
        for key, tensor in inputs.items():
            nid, param = self._root_param_to_node(key)
            node = self.nodes[nid]
            if hasattr(node, "feeder"):
                node.feeder.enqueue(job_id, {param: tensor})
            else:
                node.buffer.add_input(job_id, param, tensor)


    # ---------- execution -----------------------------------------------------

    def run_level(self, level_idx: int = -1) -> None:
        """
        Execute **all** ready jobs.

        By design we treat three disjoint *classes* of nodes:

        1.  **Symbol / Constant**  – value already known, just scatter.
        2.  **Operator**          – ordinary compute nodes; handler(tensor).
        3.  **Store / Output**    – same op semantics but require
                                handler(job_ids, tensor).

        Within each class the nodes are *further* grouped by `handler`
        so every handler runs exactly once on a single, padded batch-tensor.
        """
        from collections import defaultdict
        import torch

        # ─────── Node selection ────────────────────────────────────────────────
        nids_to_process = (
            [nid for lvl in self.levels for nid in lvl]
            if level_idx < 0 else list(self.levels[level_idx])
        )

        # ─────── Categorise by class → handler → [nid,…] ───────────────────────
        symbol_groups:  dict[callable, list[int]] = defaultdict(list)
        operator_groups: dict[callable, list[int]] = defaultdict(list)
        store_groups:    dict[callable, list[int]] = defaultdict(list)

        for nid in nids_to_process:
            node = self.nodes[nid]
            if node.is_symbol:
                symbol_groups[node.handler].append(nid)
            elif node.is_output:
                store_groups[node.handler].append(nid)
            else:
                operator_groups[node.handler].append(nid)

        # ───────────────────────────────────────────────────────────────────────
        # 1) SYMBOL / CONST  – scatter directly, no handler call
        # ───────────────────────────────────────────────────────────────────────
        for nids in symbol_groups.values():
            for nid in nids:
                node   = self.nodes[nid]
                role   = node.buffer.roles[0]
                ridx   = node.buffer.role_idx[role]
                if node.const is not None:
                    ready = list(self.active_jobs)
                    rows  = node.const(len(ready))
                else:
                    ready, row_ids = node.buffer.ready_jobs()
                    rows = [ node.buffer.pop_inputs(r)[ridx] for r in row_ids ]

                if not ready:
                    continue
                for j, tensor in zip(ready, rows):
                    node.buffer.store_output(j, tensor)
                    for dst, param in node.out_edges:
                        self.nodes[dst].buffer.add_input(j, param, tensor)

        # ───────────────────────────────────────────────────────────────────────
        # helper: batch-execute many nodes sharing a handler
        # ───────────────────────────────────────────────────────────────────────
        def _run_group(nids, call_requires_ids=False):
            rows, metas = [], []
            for nid in nids:
                node = self.nodes[nid]
                (jids, rws) = node.buffer.ready_jobs()
                for jid, r in zip(jids, rws):
                    rows.append(node.buffer.pop_inputs(r).reshape(-1))  # flatten R×C→RC
                    metas.append((nid, jid))
            if not rows:
                return
            # move batch to DEVICE

            batch = torch.stack(rows)                    # build on whichever device rows are
            if batch.device.type == "cpu":
                batch = batch.pin_memory()   
            handler = self.nodes[nids[0]].handler
            outs = handler([m[1] for m in metas], batch) if call_requires_ids else handler(batch)
            for (nid, jid), out in zip(metas, outs):
                node = self.nodes[nid]
                node.buffer.store_output(jid, out)
                for dst, param in node.out_edges:
                    self.nodes[dst].buffer.add_input(jid, param, out)


        # ───────────────────────────────────────────────────────────────────────
        # 2) OPERATORS (no job_id arg)  then  3) STORE/OUTPUT (needs job_ids)
        # ───────────────────────────────────────────────────────────────────────
        for handler, nids in operator_groups.items():
            _run_group(nids, call_requires_ids=False)

        for handler, nids in store_groups.items():
            _run_group(nids, call_requires_ids=True)


    def run_all(self, saturate=True):
        if saturate:
            for i in range(len(self.levels)):
                self.run_level()
        
        for i in range(len(self.levels)):
            self.run_level(i)

    # ---------- trivial helper (now wired) -----------------------------------
    def _root_param_to_node(self, name):
        """Fast label → (nid, role) lookup injected by TorchDAGRuntime."""
        return self._root_map[name]          # _root_map will be supplied

BACKEND = "torch"          # or "numpy"

if BACKEND == "torch":
    PinnedNodeBuffer = PinnedNodeBufferTorch
    op_table          = torch_funcs       # ← use torch_funcs
else:
    PinnedNodeBuffer = PinnedNodeBufferNpy
    op_table          = numpy_funcs    # ← use upstream numpy table

operator_signatures = torch_sigs if BACKEND == "torch" else numpy_sigs

class TorchDAGRuntime(TorchDAGRuntimeSimple):
    def __init__(self, pg, op_funcs: Mapping[str, Any] = None, *, use_compiled: bool = False, backend = BACKEND):
        """
        If use_compiled=True, we will lower the DAG to a single
        Python function via DeepCompiler, and thereafter dispatch
        all .submit() calls directly to that flat function.
        """
        global BACKEND
        if backend not in ("torch", "numpy"):
            raise ValueError(f"Invalid backend '{backend}', expected 'torch' or 'numpy'")
        self.backend = backend
        BACKEND = backend #temporary
        self.pg       = pg
        self.op_funcs = {**op_table, **(op_funcs or {})}
        self.use_compiled = use_compiled

        if self.use_compiled:
            # build the compiled, stepless function once:
            compiler = GraphDeepCompiler(
                pg,
                op_table=self.op_funcs,
                signatures=operator_signatures
            )
            # this returns something like:
            #   def graph_fn(x, y, z, ...):  ...
            self.graph_fn = compiler.build_function(DEVICE)
            self._compiled_fn = self.graph_fn
            # nothing more to build: we never need buffers
            return
    

        # root-ID registry
        self._root_name_to_int = {}
        self._root_nids_by_id  = []
        self._root_roles_by_id = []

        # restore store helpers (used by _make_store_handler)
        self.store_config      = {}
        self.store_logs        = defaultdict(lambda: open("store_log.txt","a"))
        self.store_queues      = defaultdict(list)
        self.store_ring_buffers= defaultdict(list)

        self.recompile()       # initial compile
        self._compiler = GraphDeepCompiler(pg, self.op_funcs, numpy_sigs if BACKEND == "numpy" else torch_sigs)
        self._compiled_fn = self._compiler.build_function()
        
        super().__init__(self.levels, self.nodes)
    def dump_all_outputs(self):
        for store_id, q in self.store_queues.items():
            print(f"[Queue {store_id}] {list(q)}")
        for store_id, ring in self.store_ring_buffers.items():
            print(f"[Ring {store_id}] {list(ring)}")
    def compiled(self):
        """Return the pre-built fast callable (no recompilation)."""
        return self._compiled_fn

    # ───── compilation ────────────────────────────────────────────────────
    def recompile(self):
        """
        Build `self.levels`   = [[nid,…], …]  (ordered by pg.levels)
             `self.nodes`    = {nid: FlatNodeSpec}
             `_root_map`     = {symbol_name: (nid, 'x')}
        """
        G   = self.pg.G
        lvl = self.pg.levels

        # 1) level list ---------------------------------------------------
        max_lvl = max(lvl.values()) if lvl else -1
        self.levels = [[] for _ in range(max_lvl+1)]
        for nid, lv in lvl.items():
            self.levels[lv].append(nid)

        # 2) build specs --------------------------------------------------
        self.nodes = {}
        for nid in G.nodes:
            in_roles  = set()
            out_edges = []
            for src, tgt, attrs in G.in_edges(nid, data=True):
                for e in attrs.get("extra", []):
                    if e.target == nid:
                        in_roles.add(e.id[3])                  # consumer_role
            
            for src, tgt, attrs in G.out_edges(nid, data=True):
                for e in attrs.get("extra", []):
                    out_edges.append((tgt, e.id[3]))           # (dst, producer_role)

            ntype  = G.nodes[nid]["type"]
            is_output = ntype in ("Store", "Output")
            if is_output:
                store_id = G.nodes[nid].get("store_id", f"store_{nid}")
                mode = self.store_config.get(store_id, "queue")
                handler = self._make_store_handler(store_id, mode)
                self.nodes[nid] = FlatNodeSpec(nid, handler, in_roles, out_edges,
                                            is_symbol=False, const=None, is_output=True)

            else:
                # detect symbol
                is_symbol = ntype in ("Symbol", "Input", "Var", "Integer", "Float", "NegativeOne", "Zero", "One")
                is_special_symbol = ntype in ("Half", "BooleanTrue", "BooleanFalse")
                is_symbol = is_symbol or is_special_symbol
                if is_symbol:
                    label = G.nodes[nid]["label"]
                        
                    if is_special_symbol:
                        if ntype == "Half":
                            label = .5
                        elif ntype == "BooleanTrue":
                            label = True
                        elif ntype == "BooleanFalse":
                            label = False

                    if label not in in_roles:
                        in_roles.add(label)
                handler = _symbol_handler if is_symbol else self.op_funcs.get(ntype)
                if handler is None:
                    print(f"Warning: No handler for node {nid} of type '{ntype}'")
                    print(f"The handler must support the following object:")
                    print(f" self.pg.G.nodes[{nid}] = {G.nodes[nid]}")
                    raise KeyError(f"No Torch handler for node-type '{ntype}'")

                self.nodes[nid] = FlatNodeSpec(nid, handler, list(in_roles), out_edges,
                                            is_symbol, None, is_output=False)

                # 3) root mapping (inputs) → assign an int id instead of str key
                if not G.nodes[nid]["parents"] or is_symbol:        # no incoming edges ⇒ source
                    lbl = G.nodes[nid]["label"]
                    role = next(iter(in_roles)) if in_roles else "value"
                    rid = len(self._root_nids_by_id)
                    self._root_name_to_int[lbl] = rid
                    self._root_nids_by_id .append(nid)
                    self._root_roles_by_id.append(role)
                    self.nodes[nid].attach_feeder(queue_cap=1024)

    # ───── external API — unchanged except for _root_param_to_node() ─────
    def _root_param_to_node(self, key):
        # accept either int id or legacy str label
        if isinstance(key, int):
            return (self._root_nids_by_id[key],
                    self._root_roles_by_id[key])
        # fallback from string → int → nid,role
        rid = self._root_name_to_int[key]
        return (self._root_nids_by_id[rid],
                self._root_roles_by_id[rid])

    def flush_outputs(self, force=False):
        """
        Checks all Store nodes for available outputs.
        Returns a dict {job_id: {store_id: tensor, ...}} for completed jobs.

        • If force=True, will emit whatever is present without requiring all stores.
        • Otherwise, only jobs appearing in all stores are returned.
        • Outputs are removed from the queues/rings upon return.
        """
        # Collect all job_ids seen in any store
        jobs_per_store = {}
        for nid, node in self.nodes.items():
            if not node.is_output:
                continue
            
            store_id = self.pg.G.nodes[nid].get("store_id", f"store_{nid}")
            parent_id = self.pg.G.nodes[nid]['parents'][0][0]
            parent_outbox = self.nodes[parent_id].buffer._outbox
            jobs_per_store[store_id] = set(parent_outbox.keys())

        if not jobs_per_store:
            return {}

        # Determine jobs ready to flush
        if force:
            all_jobs = set.union(*jobs_per_store.values())
        else:
            all_jobs = set.intersection(*jobs_per_store.values())

        output_results = {}

        for job_id in all_jobs:
            output_results[job_id] = {}
            for store_id, job_set in jobs_per_store.items():
                if job_id in job_set:
                    # find which Store node this store_id refers to
                    for nid, node in self.nodes.items():
                        if node.is_output and self.pg.G.nodes[nid].get("store_id", f"store_{nid}") == store_id:
                            parent_id = self.pg.G.nodes[nid]['parents'][0][0]
                            tensor = self.nodes[parent_id].buffer.fetch_output(job_id)
                            output_results[job_id][store_id] = tensor
                            break

        return output_results


    def bind_constant(self, symbol_label, value):
        nid, _ = self._root_param_to_node(symbol_label)

        if isinstance(value, str):
            m = _rand_re.match(value.replace(" ", ""))
            if m:
                shape = tuple(int(x) for x in m.group(1).split(',') if x)
                # random on DEVICE
                self.nodes[nid].const = lambda bsz=1: torch.rand((bsz, *shape), device=DEVICE)
                return

        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, device=DEVICE)
        else:
            value = value.to(DEVICE)
        # broadcast on DEVICE
        self.nodes[nid].const = lambda bsz=1, val=value: (
            val.expand(bsz, *val.shape) if val.ndim else val.repeat(bsz))

    def enqueue_input(self, symbol_label, job_id, tensor):
        nid, _ = self._root_param_to_node(symbol_label)
        role = self.nodes[nid].buffer.roles[0]
        self.nodes[nid].buffer.add_input(job_id, role, tensor)
        self.active_jobs.add(job_id)
    def submit(self, job_id: int, inputs: Dict[Any, torch.Tensor]):
        if self.use_compiled:
            # wrap inputs based on operator_signatures.param schemas
            kwargs = {}
            for name, tensor in inputs.items():
                sig = operator_signatures.get(self.pg.G.nodes[self._root_name_to_int.get(name, name)]["type"], {})
                # for example: if sig expects 'many', convert tensor to list
                param_schema = sig.get("parameters", [])
                if name in param_schema and isinstance(tensor, torch.Tensor):
                    kwargs[name] = tensor.tolist()
                else:
                    kwargs[name] = tensor
            # direct, in‐line invocation
            return self.graph_fn(**kwargs)

        # otherwise, the old buffer‐based path
        try:
            return super().submit(job_id, inputs)
        except RuntimeError as e:
            if "Pinned ring full" in str(e):
                # forced serial: drop back to compiled impl
                return self.graph_fn(**{k: v for k,v in inputs.items()})
            raise

    def _make_store_handler(self, store_id, mode):
        def handler(job_id, tensor):
            if mode == "print":
                print(f"[Store {store_id}] job {job_id}: {tensor}")
            elif mode == "log":
                self.store_logs[store_id].write(f"{job_id}: {tensor}\n")
            elif mode == "queue":
                self.store_queues[store_id].append((job_id, tensor))
            elif mode == "ring":
                self.store_ring_buffers[store_id].append((job_id, tensor))
            elif mode == "trash":
                pass
            else:
                raise ValueError(f"Unknown store mode: {mode}")
            return tensor  # return value is ignored in this case
        return handler
    def run_all(self):
        return_val = super().run_all()
        #self.flush_outputs()  # flush outputs after running all levels
        return return_val

# ─────────────────────────────────────────────────────────────────────────────
def _symbol_handler(node: FlatNodeSpec, job_ids):
    # dequeue actual ready rows, ignore passed job_ids
    role = node.buffer.roles[0]
    ridx = node.buffer.role_idx[role]
    jids, rows = node.buffer.ready_jobs()
    if not rows:
        return torch.empty(0)
    output = [ node.buffer.pop_inputs(r)[ridx] for r in rows ]
    return torch.stack(output)
def benchmark_torture():
    
    import sympy as sp
    import numpy as np
    import torch
    import time
    from graph_express2 import ProcessGraph
    from operator_defs import default_funcs


    print("\n===== Preparing DAG vs lambdify =====")
    # ---------------- DAG prep ----------------
    t0 = time.time()
    pg = ProcessGraph(3)
    x, y, z = sp.symbols('x y z')

    def make_torture_expr(base, depth=10):
        """
        Starting from `base`, repeatedly wrap it in a mix of Add, Mul, Sub, Div, Pow,
        generating an expression of roughly O(depth) size in each iteration.
        """
        expr = base
        for i in range(depth):
            # Mix in all five ops + a couple of func calls
            # – add a scaled copy of itself
            a = expr + (i+1)*y
            # – multiply by another shifted copy
            m = a * (expr - (i+2)*z)
            # – subtract a power
            s = m - (x + expr)**2
            # – divide by a shifted sum (avoid zero by adding 1)
            d = s / (y*z + expr + 1)
            # – finally power it again (keeps tree growing)
            expr = d**(1 + (i % 3))
        return expr

    # build your torture‐course expression
    base = x + y*z
    expr = make_torture_expr(base, depth=5)

    # inspect size
    pg.build_from_expression(expr)
    pg.compute_levels("asap")
    rt = TorchDAGRuntime(pg)
    t1 = time.time()

    # ---------------- lambdify prep ----------------
    t2 = time.time()
    f_lambdify = sp.lambdify(("x","y","z"), expr, modules="numpy")
    t3 = time.time()

    print(f"Graph build time:   {t1 - t0:.5f} sec")
    print(f"lambdify build time:{t3 - t2:.5f} sec")

    # =====================================================================
    print("\n===== Batch vs Batch =====")
    N = 1_000_000
    x_vals = np.random.rand(N)
    y_vals = np.random.rand(N)
    z_vals = np.random.rand(N)
    # **pre-allocate once**
    x_blk = torch.from_numpy(x_vals).view(-1, 1)
    y_blk = torch.from_numpy(y_vals).view(-1, 1)
    z_blk = torch.from_numpy(z_vals).view(-1, 1)
    # --------- lambdify numpy batch ----------
    t4 = time.time()
    out_lambdify_batch = f_lambdify(x_vals, y_vals, z_vals)
    t5 = time.time()
    print(f"lambdify batch: {t5 - t4:.5f} sec")

    # --------- torch dag batch ----------
    # we'll use torch tensors in single call to mimic batch op
    t6 = time.time()
    rt_batch = TorchDAGRuntime(pg)
    job_id = 0
    rt_batch.submit(job_id,
        {0: x_blk, 1: y_blk, 2: z_blk})
    rt_batch.run_all()
    t7 = time.time()
    print(f"torch dag batch: {t7 - t6:.5f} sec")

    # =====================================================================
    print("\n===== Serial vs Serial =====")
    # --------- lambdify serial ----------
    t8 = time.time()
    output_lambdify_serial = []
    for xi, yi, zi in zip(x_vals, y_vals, z_vals):
        output_lambdify_serial.append(f_lambdify(xi, yi, zi))
    t9 = time.time()
    print(f"lambdify serial: {t9 - t8:.5f} sec")

    # --------- torch dag serial ----------
    rt_serial = TorchDAGRuntime(pg)
    rt_serial_fn = rt_serial.compiled()  # pre-compiled function
    # pre‐slice into tensors once
    x_iter, y_iter, z_iter = x_blk.unbind(), y_blk.unbind(), z_blk.unbind()
    output = []
    t10 = time.time()
    for i in range(N):
        output.append(rt_serial_fn(x=x_iter[i], y=y_iter[i], z=z_iter[i]))
    print(f"torch dag serial: {len(output)} outputs")
    t11 = time.time()
    print(f"torch dag serial: {t11 - t10:.5f} sec")

    rt_serial = TorchDAGRuntime(pg, use_compiled=True, backend="numpy")
    rt_serial_fn = rt_serial.compiled()  # pre-compiled function
    output = []
    t14 = time.time()
    for xi, yi, zi in zip(x_vals, y_vals, z_vals):
        output.append(rt_serial_fn(x=xi, y=yi, z=zi))
    print(f"torch dag serial: {len(output)} outputs")
    t16 = time.time()
    print(f"torch dag serial: {t16 - t14:.5f} sec")