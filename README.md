# Transmogrifier

**A symbolic mathematical DAG engine with concurrency analysis and minimal-energy geometric embedding.**

---

## 📐 Mathematical Domain

The `Transmogrifier` software suite constructs and manipulates **Directed Acyclic Graphs (DAGs)** derived from symbolic mathematical expressions. This system is rigorously grounded in:

- **Symbolic Algebra**  
  Using the `sympy` library, we parse expressions into their minimal decomposed forms, supporting sums, indexed tensors, reductions, and arbitrary algebraic operators.

- **Graph Theory**  
  Each expression is represented as a dependency graph (DAG), where nodes represent atomic operations or symbolic entities, and edges encode explicit producer-consumer relationships.

- **Concurrency Scheduling**  
  Utilizing integer-level scheduling (ASAP, ALAP, maximum slack), we determine the minimal depth parallel execution bands. We further analyze temporal concurrency and create interference graphs to identify memory overlap and concurrent read/write conflicts.

- **Memory Lifespan Tracking**  
  The system computes precise temporal lifespans for each intermediate computation, enabling binning into process and memory concurrency stages.

- **Geometric Transmogrification**  
  We introduce a novel paradigm: *the transmogrification of symbolic temporal process into spatial geometry.*  
  By applying equal-tension spring physics to the process graph, with inputs and outputs geometrically pinned, we allow the mathematical computation to solve itself into its minimal geometric embodiment.

---

## 🧮 Mathematical Layers

1. **Symbolic Expression Ingestion**  
   - Constructs a symbolic tensor domain from arbitrary dimensional expressions, supporting indexed bases and summations.

2. **Process Graph Construction (ProcessGraph)**  
   - Builds a DAG from symbolic expressions, maintaining strict producer-consumer edge roles.
   - Uses a role schema system to classify arguments and outputs for correct data flow.

3. **Concurrency Scheduling (ILPScheduler)**  
   - Computes ASAP (as soon as possible), ALAP (as late as possible), and maximum slack schedules.
   - Builds concurrency bands reflecting integer-level minimal parallel execution requirements.

4. **Interference and Lifespan Analysis**  
   - Tracks node lifespans across temporal bins.
   - Constructs interference graphs to model concurrent memory demands.

5. **Memory Domain Graph (DomainNodes)**  
   - Encodes explicit data domain lifespans into nodes, preparing for memory interference resolution.

6. **Geometric Embedding (Transmogrification)**  
   - Uses a spring physics layout to interpolate the spatial shape of the process. 
   - Pins inputs and outputs in fixed positions, letting the entire computation interpolate into the minimal energy shape — creating a **geometric realization of the transformation**.

---

## 🚀 Usage Example

```python
from graph_express2 import ProcessGraph
from graph_spring_viewer import spring_graph_viewer

pg = ProcessGraph()
pg.build_from_expression(lambda i, j: Sum(A[i,k]*B[k,j], (k,0,K-1)) + C[i,j], M_val, N_val)

# Fix inputs on left, outputs on right
fixed_nodes = {}
for nid, data in pg.G.nodes(data=True):
    if data['type'] in ('IndexedBase', 'Idx'):
        fixed_nodes[nid] = np.array([-8, np.random.uniform(-4,4)])
    if data['type'] == 'Store':
        fixed_nodes[nid] = np.array([8, np.random.uniform(-4,4)])

spring_graph_viewer(pg.G, fixed_positions=fixed_nodes)
