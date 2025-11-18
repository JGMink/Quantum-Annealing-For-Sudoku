"""
Sudoku QUBO Matrix Reduction
Variable elimination for puzzles with given cells.
"""

import numpy as np
from qubo_generation import build_sudoku_qubo


def build_reduced_qubo(N, box_size, givens, L1=1.0, L2=1.0, L3=1.0, L4=1.0):
    """
    Build reduced QUBO by extracting submatrix from full QUBO.
    
    Args:
        N: Sudoku size
        box_size: Box size
        givens: Dict {(i,j): digit} (REQUIRED)
        L1, L2, L3, L4: Lagrange multipliers
    
    Returns:
        Q_reduced: Reduced QUBO matrix
        var_to_idx: Mapping (i,j,k) → reduced index
        idx_to_var: Mapping reduced index → (i,j,k)
        offset: Constant offset
        info: Statistics dictionary
    """
    if givens is None:
        raise ValueError("givens required")
    
    Q_full, var_to_idx_full, idx_to_var_full, offset = build_sudoku_qubo(
        N, box_size, givens, L1, L2, L3, L4
    )
    
    free_vars = []
    for i in range(N):
        for j in range(N):
            if (i, j) not in givens:
                for k in range(N):
                    free_vars.append((i, j, k))
    
    n_free = len(free_vars)
    
    var_to_idx_reduced = {}
    idx_to_var_reduced = {}
    old_to_new = {}
    
    for new_idx, var in enumerate(free_vars):
        var_to_idx_reduced[var] = new_idx
        idx_to_var_reduced[new_idx] = var
        old_idx = var_to_idx_full[var]
        old_to_new[old_idx] = new_idx
    
    Q_reduced = np.zeros((n_free, n_free))
    
    for var_i in free_vars:
        old_i = var_to_idx_full[var_i]
        new_i = old_to_new[old_i]
        
        for var_j in free_vars:
            old_j = var_to_idx_full[var_j]
            new_j = old_to_new[old_j]
            
            Q_reduced[new_i, new_j] = Q_full[old_i, old_j]
    
    info = {
        'n_total_vars': N * N * N,
        'n_free_vars': n_free,
        'n_eliminated_vars': N * N * N - n_free,
        'n_given_cells': len(givens),
        'reduction_pct': 100 * (1 - n_free / (N * N * N)),
        'matrix_size_reduction_pct': 100 * (1 - (n_free ** 2) / ((N * N * N) ** 2))
    }
    
    return Q_reduced, var_to_idx_reduced, idx_to_var_reduced, offset, info


def build_reduced_qubo_direct(N, box_size, givens, L1=1.0, L2=1.0, L3=1.0, L4=1.0):
    """
    Build reduced QUBO directly (more efficient).
    
    Args:
        N: Sudoku size
        box_size: Box size
        givens: Dict {(i,j): digit} (REQUIRED)
        L1, L2, L3, L4: Lagrange multipliers
    
    Returns:
        Q: Reduced QUBO matrix
        var_to_idx: Mapping (i,j,k) → index
        idx_to_var: Mapping index → (i,j,k)
        offset: Constant offset
        info: Statistics dictionary
    """
    if givens is None:
        raise ValueError("givens required")
    
    free_vars = []
    for i in range(N):
        for j in range(N):
            if (i, j) not in givens:
                for k in range(N):
                    free_vars.append((i, j, k))
    
    n_free = len(free_vars)
    
    var_to_idx = {}
    idx_to_var = {}
    
    for idx, var in enumerate(free_vars):
        var_to_idx[var] = idx
        idx_to_var[idx] = var
    
    Q = np.zeros((n_free, n_free))
    constant_offset = 0.0
    
    def add_quadratic(var1, var2, coeff):
        if var1 in var_to_idx and var2 in var_to_idx:
            idx1 = var_to_idx[var1]
            idx2 = var_to_idx[var2]
            if idx1 <= idx2:
                Q[idx1, idx2] += coeff
            else:
                Q[idx2, idx1] += coeff
    
    def add_linear(var, coeff):
        if var in var_to_idx:
            idx = var_to_idx[var]
            Q[idx, idx] += coeff
    
    # E1: Each cell has exactly one digit
    for i in range(N):
        for j in range(N):
            if (i, j) in givens:
                continue
            
            for k in range(N):
                add_linear((i, j, k), -L1)
            
            for k in range(N):
                for kp in range(k + 1, N):
                    add_quadratic((i, j, k), (i, j, kp), 2 * L1)
            
            constant_offset += L1
    
    # E2: Each row has each digit exactly once
    for i in range(N):
        for k in range(N):
            given_count = 0
            free_cells = []
            
            for j in range(N):
                if (i, j) in givens:
                    if givens[(i, j)] == k + 1:
                        given_count += 1
                else:
                    free_cells.append(j)
            
            if given_count == 1 and len(free_cells) == 0:
                continue
            
            target_adjustment = 1 - given_count
            
            for j in free_cells:
                add_linear((i, j, k), L2 * (1 - 2 * target_adjustment))
            
            for j_idx, j in enumerate(free_cells):
                for jp in free_cells[j_idx + 1:]:
                    add_quadratic((i, j, k), (i, jp, k), 2 * L2)
            
            constant_offset += L2 * (target_adjustment ** 2)
    
    # E3: Each column has each digit exactly once
    for j in range(N):
        for k in range(N):
            given_count = 0
            free_cells = []
            
            for i in range(N):
                if (i, j) in givens:
                    if givens[(i, j)] == k + 1:
                        given_count += 1
                else:
                    free_cells.append(i)
            
            if given_count == 1 and len(free_cells) == 0:
                continue
            
            target_adjustment = 1 - given_count
            
            for i in free_cells:
                add_linear((i, j, k), L3 * (1 - 2 * target_adjustment))
            
            for i_idx, i in enumerate(free_cells):
                for ip in free_cells[i_idx + 1:]:
                    add_quadratic((i, j, k), (ip, j, k), 2 * L3)
            
            constant_offset += L3 * (target_adjustment ** 2)
    
    # E4: Each box has each digit exactly once
    boxes_per_side = N // box_size
    
    for box_row in range(boxes_per_side):
        for box_col in range(boxes_per_side):
            for k in range(N):
                given_count = 0
                free_cells = []
                
                for i in range(box_row * box_size, (box_row + 1) * box_size):
                    for j in range(box_col * box_size, (box_col + 1) * box_size):
                        if (i, j) in givens:
                            if givens[(i, j)] == k + 1:
                                given_count += 1
                        else:
                            free_cells.append((i, j))
                
                if given_count == 1 and len(free_cells) == 0:
                    continue
                
                target_adjustment = 1 - given_count
                
                for (i, j) in free_cells:
                    add_linear((i, j, k), L4 * (1 - 2 * target_adjustment))
                
                for cell_idx, (i, j) in enumerate(free_cells):
                    for (ip, jp) in free_cells[cell_idx + 1:]:
                        add_quadratic((i, j, k), (ip, jp, k), 2 * L4)
                
                constant_offset += L4 * (target_adjustment ** 2)
    
    info = {
        'n_total_vars': N * N * N,
        'n_free_vars': n_free,
        'n_eliminated_vars': N * N * N - n_free,
        'n_given_cells': len(givens),
        'reduction_pct': 100 * (1 - n_free / (N * N * N)),
        'matrix_size_reduction_pct': 100 * (1 - (n_free ** 2) / ((N * N * N) ** 2))
    }
    
    return Q, var_to_idx, idx_to_var, constant_offset, info


def evaluate_reduced_qubo(Q_reduced, bitstring_full, var_to_idx, idx_to_var, constant_offset=0):
    """
    Evaluate reduced QUBO energy given full bitstring.
    
    Args:
        Q_reduced: Reduced QUBO matrix
        bitstring_full: Full bitstring (includes given cells)
        var_to_idx: Mapping (i,j,k) → index
        idx_to_var: Mapping index → (i,j,k)
        constant_offset: Constant term
    
    Returns:
        Energy value
    """
    n_free = len(idx_to_var)
    x_reduced = np.zeros(n_free, dtype=int)
    
    bitstring_clean = bitstring_full.replace(' ', '')
    N = int(round(len(bitstring_clean) ** (1/3)))
    
    idx_full = 0
    for i in range(N):
        for j in range(N):
            for k in range(N):
                var = (i, j, k)
                if idx_full < len(bitstring_clean):
                    bit_val = int(bitstring_clean[idx_full])
                    
                    if var in var_to_idx:
                        reduced_idx = var_to_idx[var]
                        x_reduced[reduced_idx] = bit_val
                
                idx_full += 1
    
    energy = 0.0
    
    for i in range(n_free):
        energy += Q_reduced[i, i] * x_reduced[i]
    
    for i in range(n_free):
        for j in range(i + 1, n_free):
            energy += 2 * Q_reduced[i, j] * x_reduced[i] * x_reduced[j]
    
    energy += constant_offset
    
    return energy


def reconstruct_full_solution(reduced_bitstring, var_to_idx, idx_to_var, givens, N):
    """
    Convert reduced solution to full solution.
    
    Args:
        reduced_bitstring: Reduced bitstring (free variables only)
        var_to_idx: Mapping (i,j,k) → index
        idx_to_var: Mapping index → (i,j,k)
        givens: Dict {(i,j): digit}
        N: Sudoku size
    
    Returns:
        Full bitstring
    """
    full_solution = ['0'] * (N * N * N)
    
    for reduced_idx, bit in enumerate(reduced_bitstring):
        i, j, k = idx_to_var[reduced_idx]
        full_idx = i * (N * N) + j * N + k
        full_solution[full_idx] = bit
    
    for (i, j), digit in givens.items():
        for k in range(N):
            full_idx = i * (N * N) + j * N + k
            if k == (digit - 1):
                full_solution[full_idx] = '1'
            else:
                full_solution[full_idx] = '0'
    
    return ''.join(full_solution)


def print_reduction_stats(info, N, box_size):
    """Print variable elimination statistics"""
    print("\nVariable Elimination Statistics:")
    print(f"  Total variables: {info['n_total_vars']}")
    print(f"  Given cells: {info['n_given_cells']}")
    print(f"  Eliminated variables: {info['n_eliminated_vars']}")
    print(f"  Free variables: {info['n_free_vars']}")
    print(f"  Variable reduction: {info['reduction_pct']:.1f}%")
    print(f"  Matrix size reduction: {info['matrix_size_reduction_pct']:.1f}%")
    
    full_size = info['n_total_vars']
    reduced_size = info['n_free_vars']
    
    print(f"\nMatrix Dimensions:")
    print(f"  Full QUBO: {full_size}×{full_size} = {full_size**2:,} entries")
    print(f"  Reduced QUBO: {reduced_size}×{reduced_size} = {reduced_size**2:,} entries")
    print(f"  Savings: {full_size**2 - reduced_size**2:,} entries")