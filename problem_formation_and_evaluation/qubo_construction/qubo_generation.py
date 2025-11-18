"""
Sudoku QUBO Construction
Builds full QUBO matrices for Sudoku puzzles.
"""

import numpy as np


def build_sudoku_qubo(N, box_size, givens=None, L1=1.0, L2=1.0, L3=1.0, L4=1.0):
    """
    Build complete QUBO matrix for Sudoku.
    
    Args:
        N: Sudoku size (4 for 4x4, 9 for 9x9)
        box_size: Box size (2 for 4x4, 3 for 9x9)
        givens: Dict {(i,j): digit} for known cells (1-indexed digits)
        L1, L2, L3, L4: Lagrange multipliers
    
    Returns:
        Q: QUBO matrix (N³ × N³)
        var_to_idx: Dict mapping (i,j,k) → index
        idx_to_var: Dict mapping index → (i,j,k)
        constant_offset: Constant term
    """
    n_vars = N * N * N
    Q = np.zeros((n_vars, n_vars))
    constant_offset = 0.0
    
    var_to_idx = {}
    idx_to_var = {}
    idx = 0
    for i in range(N):
        for j in range(N):
            for k in range(N):
                var_to_idx[(i, j, k)] = idx
                idx_to_var[idx] = (i, j, k)
                idx += 1
    
    def add_quadratic(var1, var2, coeff):
        idx1 = var_to_idx[var1]
        idx2 = var_to_idx[var2]
        if idx1 <= idx2:
            Q[idx1, idx2] += coeff
        else:
            Q[idx2, idx1] += coeff
    
    def add_linear(var, coeff):
        idx = var_to_idx[var]
        Q[idx, idx] += coeff
    
    # E1: Each cell has exactly one digit
    for i in range(N):
        for j in range(N):
            if givens and (i, j) in givens:
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
                if givens and (i, j) in givens:
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
                if givens and (i, j) in givens:
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
                        if givens and (i, j) in givens:
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
    
    return Q, var_to_idx, idx_to_var, constant_offset


def evaluate_qubo(Q, bitstring, constant_offset=0):
    """
    Evaluate QUBO energy: E = x^T Q x + constant
    
    Args:
        Q: QUBO matrix
        bitstring: Binary string (spaces allowed)
        constant_offset: Constant term
    
    Returns:
        Energy value
    """
    x = np.array([int(b) for b in bitstring.replace(' ', '')])
    
    energy = constant_offset
    
    for i in range(len(x)):
        energy += Q[i, i] * x[i]
    
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            energy += 2 * Q[i, j] * x[i] * x[j]
    
    return energy


def build_E1(N, givens=None, L1=1.0):
    """Build E1: Each cell has exactly one digit"""
    n_vars = N * N * N
    Q = np.zeros((n_vars, n_vars))
    constant = 0.0
    
    var_to_idx = {}
    idx = 0
    for i in range(N):
        for j in range(N):
            for k in range(N):
                var_to_idx[(i, j, k)] = idx
                idx += 1
    
    for i in range(N):
        for j in range(N):
            if givens and (i, j) in givens:
                continue
            
            for k in range(N):
                idx = var_to_idx[(i, j, k)]
                Q[idx, idx] += -L1
            
            for k in range(N):
                for kp in range(k + 1, N):
                    idx1 = var_to_idx[(i, j, k)]
                    idx2 = var_to_idx[(i, j, kp)]
                    Q[idx1, idx2] += 2 * L1
            
            constant += L1
    
    return Q, None, constant


def build_E2(N, givens=None, L2=1.0):
    """Build E2: Each row has each digit exactly once"""
    n_vars = N * N * N
    Q = np.zeros((n_vars, n_vars))
    constant = 0.0
    
    var_to_idx = {}
    idx = 0
    for i in range(N):
        for j in range(N):
            for k in range(N):
                var_to_idx[(i, j, k)] = idx
                idx += 1
    
    for i in range(N):
        for k in range(N):
            given_count = 0
            free_cells = []
            
            for j in range(N):
                if givens and (i, j) in givens:
                    if givens[(i, j)] == k + 1:
                        given_count += 1
                else:
                    free_cells.append(j)
            
            if given_count == 1 and len(free_cells) == 0:
                continue
            
            target_adjustment = 1 - given_count
            
            for j in free_cells:
                idx = var_to_idx[(i, j, k)]
                Q[idx, idx] += L2 * (1 - 2 * target_adjustment)
            
            for j_idx, j in enumerate(free_cells):
                for jp in free_cells[j_idx + 1:]:
                    idx1 = var_to_idx[(i, j, k)]
                    idx2 = var_to_idx[(i, jp, k)]
                    Q[idx1, idx2] += 2 * L2
            
            constant += L2 * (target_adjustment ** 2)
    
    return Q, None, constant


def build_E3(N, givens=None, L3=1.0):
    """Build E3: Each column has each digit exactly once"""
    n_vars = N * N * N
    Q = np.zeros((n_vars, n_vars))
    constant = 0.0
    
    var_to_idx = {}
    idx = 0
    for i in range(N):
        for j in range(N):
            for k in range(N):
                var_to_idx[(i, j, k)] = idx
                idx += 1
    
    for j in range(N):
        for k in range(N):
            given_count = 0
            free_cells = []
            
            for i in range(N):
                if givens and (i, j) in givens:
                    if givens[(i, j)] == k + 1:
                        given_count += 1
                else:
                    free_cells.append(i)
            
            if given_count == 1 and len(free_cells) == 0:
                continue
            
            target_adjustment = 1 - given_count
            
            for i in free_cells:
                idx = var_to_idx[(i, j, k)]
                Q[idx, idx] += L3 * (1 - 2 * target_adjustment)
            
            for i_idx, i in enumerate(free_cells):
                for ip in free_cells[i_idx + 1:]:
                    idx1 = var_to_idx[(i, j, k)]
                    idx2 = var_to_idx[(ip, j, k)]
                    Q[idx1, idx2] += 2 * L3
            
            constant += L3 * (target_adjustment ** 2)
    
    return Q, None, constant


def build_E4(N, box_size, givens=None, L4=1.0):
    """Build E4: Each box has each digit exactly once"""
    n_vars = N * N * N
    Q = np.zeros((n_vars, n_vars))
    constant = 0.0
    
    var_to_idx = {}
    idx = 0
    for i in range(N):
        for j in range(N):
            for k in range(N):
                var_to_idx[(i, j, k)] = idx
                idx += 1
    
    boxes_per_side = N // box_size
    
    for box_row in range(boxes_per_side):
        for box_col in range(boxes_per_side):
            for k in range(N):
                given_count = 0
                free_cells = []
                
                for i in range(box_row * box_size, (box_row + 1) * box_size):
                    for j in range(box_col * box_size, (box_col + 1) * box_size):
                        if givens and (i, j) in givens:
                            if givens[(i, j)] == k + 1:
                                given_count += 1
                        else:
                            free_cells.append((i, j))
                
                if given_count == 1 and len(free_cells) == 0:
                    continue
                
                target_adjustment = 1 - given_count
                
                for (i, j) in free_cells:
                    idx = var_to_idx[(i, j, k)]
                    Q[idx, idx] += L4 * (1 - 2 * target_adjustment)
                
                for cell_idx, (i, j) in enumerate(free_cells):
                    for (ip, jp) in free_cells[cell_idx + 1:]:
                        idx1 = var_to_idx[(i, j, k)]
                        idx2 = var_to_idx[(ip, jp, k)]
                        Q[idx1, idx2] += 2 * L4
                
                constant += L4 * (target_adjustment ** 2)
    
    return Q, None, constant


def print_E1_details(N, givens=None):
    """Print E1 construction details"""
    Q, _, const = build_E1(N, givens)
    print("E1: Each cell has exactly one digit")
    print(f"Constant: {const}")
    return Q, None, const


def print_E2_details(N, givens=None):
    """Print E2 construction details"""
    Q, _, const = build_E2(N, givens)
    print("E2: Each row has each digit exactly once")
    print(f"Constant: {const}")
    return Q, None, const


def print_E3_details(N, givens=None):
    """Print E3 construction details"""
    Q, _, const = build_E3(N, givens)
    print("E3: Each column has each digit exactly once")
    print(f"Constant: {const}")
    return Q, None, const


def print_E4_details(N, box_size, givens=None):
    """Print E4 construction details"""
    Q, _, const = build_E4(N, box_size, givens)
    print("E4: Each box has each digit exactly once")
    print(f"Constant: {const}")
    return Q, None, const