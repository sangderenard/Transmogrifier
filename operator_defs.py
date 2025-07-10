import numpy as np
import sympy

SIMD_DEFAULT_CONCURRENCY = 4

# -------------------------------------------------
#  Anonymous signature definitions (shared objects)
# -------------------------------------------------
sig_binary_elementwise = {
    'min_inputs': 1, 'max_inputs': 2,
    'min_outputs': 1, 'max_outputs': 1,
    'concurrency': SIMD_DEFAULT_CONCURRENCY,
    'allows_inplace': True
}

sig_unary_elementwise = {
    'min_inputs': 1, 'max_inputs': 1,
    'min_outputs': 1, 'max_outputs': 1,
    'concurrency': SIMD_DEFAULT_CONCURRENCY,
    'allows_inplace': True
}

sig_sum_like = {
    'min_inputs': 1, 'max_inputs': None,
    'min_outputs': 1, 'max_outputs': 1,
    'concurrency': SIMD_DEFAULT_CONCURRENCY,
    'allows_inplace': True,
    'parameters': ['limits']
}

sig_idx_like = {
    'min_inputs': 1, 'max_inputs': None,
    'min_outputs': 1, 'max_outputs': 1,
    'concurrency': SIMD_DEFAULT_CONCURRENCY,
    'allows_inplace': True,
    'parameters': ['range']
}

sig_indexed = {
    'min_inputs': 1, 'max_inputs': None,
    'min_outputs': 1, 'max_outputs': None,
    'concurrency': SIMD_DEFAULT_CONCURRENCY,
    'allows_inplace': True
}

sig_indexed_base = {
    'min_inputs': 0, 'max_inputs': 0,
    'min_outputs': 1, 'max_outputs': 1,
    'concurrency': SIMD_DEFAULT_CONCURRENCY,
    'allows_inplace': True
}

sig_store = {
    'min_inputs': 1, 'max_inputs': 1,
    'min_outputs': 0, 'max_outputs': 0,
    'concurrency': SIMD_DEFAULT_CONCURRENCY,
    'allows_inplace': True
}

sig_default = {
    'min_inputs': 1, 'max_inputs': 1,
    'min_outputs': 0, 'max_outputs': 0,
    'concurrency': SIMD_DEFAULT_CONCURRENCY,
    'allows_inplace': True
}

# -------------------------------------------------
# Operation name -> signature mapping
# -------------------------------------------------
operator_signatures = {
    'Add': sig_binary_elementwise,
    'Mul': sig_binary_elementwise,
    'Pow': sig_binary_elementwise,

    'Sum': sig_sum_like,
    'Idx': sig_idx_like,
    'Indexed': sig_indexed,
    'IndexedBase': sig_indexed_base,
    'Tuple': sig_unary_elementwise,
    'Store': sig_store,
    'Default': sig_default,

    # Trigonometric, log, exp, sqrt etc
    'Sin': sig_unary_elementwise,
    'Cos': sig_unary_elementwise,
    'Tan': sig_unary_elementwise,
    'Exp': sig_unary_elementwise,
    'Log': sig_unary_elementwise,
    'Sqrt': sig_unary_elementwise
}

# -------------------------------------------------
# Operator function mappings (default execution impls)
# -------------------------------------------------
def add_op(role_map):
    return sum(vals[0] for vals in role_map.values())

def mul_op(role_map):
    iter_vals = iter(role_map.values())
    result = next(iter_vals)[0]
    for vals in iter_vals:
        result *= vals[0]
    return result

def pow_op(role_map):
    base = role_map.get('arg0', [None])[0]
    exp = role_map.get('arg1', [1])[0]
    return np.power(base, exp)

def indexed_op(role_map):
    base = role_map.get('base', [[]])[0]
    indices = tuple(role_map.get('index', []))


    if not indices:
        raise ValueError("No indices provided for Indexed operation.")
    
    if isinstance(indices, tuple) and len(indices) == 1:
        indices = indices[0]
    elif isinstance(indices, tuple):
        ndim_desired = len(indices)
        ndim_base = len(base.shape) if isinstance(base, np.ndarray) else 1
        if ndim_desired > ndim_base and isinstance(base, np.ndarray):
            base = base.reshape((1,) * (ndim_desired - ndim_base) + base.shape)
        if ndim_desired > ndim_base and isinstance(base, list):
            for i in enumerate(indices):
                base = [base]
    indices = slice(*indices) if isinstance(indices, tuple) else indices
    return base[indices]

def indexedbase_op(role_map):
    #print(f"Role map for IndexedBase operation: {role_map}")
    return role_map.get('base', [[]])[0]

def sum_op(role_map):
    return sum(role_map.get('body', [0])[0])

# Scientific / trig functions
def sin_op(role_map):
    return np.sin(role_map.get('arg0', [0])[0])

def cos_op(role_map):
    return np.cos(role_map.get('arg0', [0])[0])

def tan_op(role_map):
    return np.tan(role_map.get('arg0', [0])[0])

def exp_op(role_map):
    return np.exp(role_map.get('arg0', [0])[0])

def log_op(role_map):
    return np.log(role_map.get('arg0', [0])[0])

def sqrt_op(role_map):
    return np.sqrt(role_map.get('arg0', [0])[0])

def store_op(role_map):
    value = role_map.get('value', [None])[0]
    #print(f"Store operation completed. Produced value: {value}")
    return value

# -------------------------------------------------
# Complete operator function dispatch
# -------------------------------------------------
default_funcs = {
    'Add': add_op,
    'Mul': mul_op,
    'Pow': pow_op,
    'Indexed': indexed_op,
    'IndexedBase': indexedbase_op,
    'Sum': sum_op,

    'Sin': sin_op,
    'Cos': cos_op,
    'Tan': tan_op,
    'Exp': exp_op,
    'Log': log_op,
    'Sqrt': sqrt_op,
    'Store': store_op,
}
