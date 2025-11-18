"""
Sudoku QUBO Reduction - Variable Elimination

This module takes a full QUBO matrix (e.g., 64×64 for 4×4 Sudoku) and
reduces it to only include free variables (e.g., 32×32 if 8 cells are given).

TWO METHODS PROVIDED:
1. build_reduced_qubo() - Extraction approach (pedagogical)
2. build_reduced_qubo_direct() - Direct construction (efficient)

Both produce identical results.
"""

import numpy as np
from qubo_generation import build_sudoku_qubo


def build_reduced_qubo(N, box_size, givens, L1=1.0, L2=1.0, L3=1.0, L4=1.0):
    """
    Build reduced QUBO by EXTRACTING submatrix from full QUBO.
    
    This is the pedagogical approach:
    1. Build full 64×64 matrix
    2. Identify free variables
    3. Extract 32×32 submatrix
    
    Args:
        N: Sudoku size (4 for 4x4, 9 for 9x9)
        box_size: Box size (2 for 4x4, 3 for 9x9)
        givens: Dictionary {(i,j): digit} for known cells (REQUIRED)
        L1, L2, L3, L4: Lagrange multipliers
    
    Returns:
        Q_reduced: Reduced QUBO matrix (n_free × n_free)
        var_to_idx_reduced: Mapping (i,j,k) → reduced index
        idx_to_var_reduced: Mapping reduced index → (i,j,k)
        offset: Constant offset
        elimination_info: Statistics about the reduction
    """
    if givens is None:
        raise ValueError("build_reduced_qubo requires givens dictionary")
    
    # STEP 1: Build full QUBO matrix
    Q_full, var_to_idx_full, idx_to_var_full, offset = build_sudoku_qubo(
        N, box_size, givens, L1, L2, L3, L4
    )
    
    # STEP 2: Identify free variables (cells not in givens)
    free_vars = []
    eliminated_vars = []
    
    for i in range(N):
        for j in range(N):
            if (i, j) not in givens:  # This cell is free
                for k in range(N):
                    free_vars.append((i, j, k))
            else:  # This cell is given
                for k in range(N):
                    eliminated_vars.append((i, j, k))
    
    n_free = len(free_vars)
    
    # STEP 3: Create new index mappings for reduced variables
    var_to_idx_reduced = {}
    idx_to_var_reduced = {}
    old_to_new = {}  # Map old index → new index
    
    for new_idx, var in enumerate(free_vars):
        var_to_idx_reduced[var] = new_idx
        idx_to_var_reduced[new_idx] = var
        old_idx = var_to_idx_full[var]
        old_to_new[old_idx] = new_idx
    
    # STEP 4: Extract submatrix
    Q_reduced = np.zeros((n_free, n_free))
    
    # Copy relevant entries from full Q to reduced Q
    for var_i in free_vars:
        old_i = var_to_idx_full[var_i]
        new_i = old_to_new[old_i]
        
        for var_j in free_vars:
            old_j = var_to_idx_full[var_j]
            new_j = old_to_new[old_j]
            
            Q_reduced[new_i, new_j] = Q_full[old_i, old_j]
    
    # STEP 5: Gather statistics
    elimination_info = {
        'n_total_vars': N * N * N,
        'n_free_vars': n_free,
        'n_eliminated_vars': len(eliminated_vars),
        'n_given_cells': len(givens),
        'reduction_pct': 100 * (1 - n_free / (N * N * N)),
        'matrix_size_reduction_pct': 100 * (1 - (n_free ** 2) / ((N * N * N) ** 2))
    }
    
    return Q_reduced, var_to_idx_reduced, idx_to_var_reduced, offset, elimination_info


def build_reduced_qubo_direct(N, box_size, givens, L1=1.0, L2=1.0, L3=1.0, L4=1.0):
    """
    Build reduced QUBO DIRECTLY (more efficient).
    
    This is the production approach:
    1. Identify free variables upfront
    2. Build 32×32 matrix directly
    3. Never allocate space for eliminated variables
    
    Args:
        N: Sudoku size (4 for 4x4, 9 for 9x9)
        box_size: Box size (2 for 4x4, 3 for 9x9)
        givens: Dictionary {(i,j): digit} for known cells (REQUIRED)
        L1, L2, L3, L4: Lagrange multipliers
    
    Returns:
        Q_reduced: Reduced QUBO matrix (n_free × n_free)
        var_to_idx: Mapping (i,j,k) → reduced index
        idx_to_var: Mapping reduced index → (i,j,k)
        offset: Constant offset
        elimination_info: Statistics about the reduction
    """
    if givens is None:
        raise ValueError("build_reduced_qubo_direct requires givens dictionary")
    
    # STEP 1: Identify free variables only
    free_vars = []
    for i in range(N):
        for j in range(N):
            if (i, j) not in givens:
                for k in range(N):
                    free_vars.append((i, j, k))
    
    n_free = len(free_vars)
    
    # STEP 2: Create index mappings for free variables
    var_to_idx = {}
    idx_to_var = {}
    
    for idx, var in enumerate(free_vars):
        var_to_idx[var] = idx
        idx_to_var[idx] = var
    
    # STEP 3: Initialize reduced QUBO matrix
    Q = np.zeros((n_free, n_free))
    constant_offset = 0.0
    
    # Helper functions
    def add_quadratic(var1, var2, coeff):
        """Add quadratic term to Q (only if both vars are free)"""
        if var1 in var_to_idx and var2 in var_to_idx:
            idx1 = var_to_idx[var1]
            idx2 = var_to_idx[var2]
            if idx1 <= idx2:
                Q[idx1, idx2] += coeff
            else:
                Q[idx2, idx1] += coeff
    
    def add_linear(var, coeff):
        """Add linear term to Q (only if var is free)"""
        if var in var_to_idx:
            idx = var_to_idx[var]
            Q[idx, idx] += coeff
    
    # STEP 4: Build constraints (same logic as full QUBO)
    
    # ===== E1: Each cell has exactly one digit =====
    for i in range(N):
        for j in range(N):
            if (i, j) in givens:
                continue  # Skip given cells
            
            # Linear terms
            for k in range(N):
                add_linear((i, j, k), -L1)
            
            # Quadratic terms
            for k in range(N):
                for kp in range(k + 1, N):
                    add_quadratic((i, j, k), (i, j, kp), 2 * L1)
            
            # Constant
            constant_offset += L1
    
    # ===== E2: Each row has each digit exactly once =====
    for i in range(N):
        for k in range(N):
            # Count givens and find free cells
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
            
            # Linear terms
            for j in free_cells:
                add_linear((i, j, k), L2 * (1 - 2 * target_adjustment))
            
            # Quadratic terms
            for j_idx, j in enumerate(free_cells):
                for jp in free_cells[j_idx + 1:]:
                    add_quadratic((i, j, k), (i, jp, k), 2 * L2)
            
            # Constant
            constant_offset += L2 * (target_adjustment ** 2)
    
    # ===== E3: Each column has each digit exactly once =====
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
            
            # Linear terms
            for i in free_cells:
                add_linear((i, j, k), L3 * (1 - 2 * target_adjustment))
            
            # Quadratic terms
            for i_idx, i in enumerate(free_cells):
                for ip in free_cells[i_idx + 1:]:
                    add_quadratic((i, j, k), (ip, j, k), 2 * L3)
            
            # Constant
            constant_offset += L3 * (target_adjustment ** 2)
    
    # ===== E4: Each box has each digit exactly once =====
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
                
                # Linear terms
                for (i, j) in free_cells:
                    add_linear((i, j, k), L4 * (1 - 2 * target_adjustment))
                
                # Quadratic terms
                for cell_idx, (i, j) in enumerate(free_cells):
                    for (ip, jp) in free_cells[cell_idx + 1:]:
                        add_quadratic((i, j, k), (ip, jp, k), 2 * L4)
                
                # Constant
                constant_offset += L4 * (target_adjustment ** 2)
    
    # STEP 5: Gather statistics
    elimination_info = {
        'n_total_vars': N * N * N,
        'n_free_vars': n_free,
        'n_eliminated_vars': N * N * N - n_free,
        'n_given_cells': len(givens),
        'reduction_pct': 100 * (1 - n_free / (N * N * N)),
        'matrix_size_reduction_pct': 100 * (1 - (n_free ** 2) / ((N * N * N) ** 2))
    }
    
    return Q, var_to_idx, idx_to_var, constant_offset, elimination_info


def evaluate_reduced_qubo(Q_reduced, bitstring_full, var_to_idx, idx_to_var, constant_offset=0):
    """
    Evaluate reduced QUBO energy given a FULL bitstring.
    
    This extracts only the free variable values from the full bitstring
    and evaluates the reduced QUBO.
    
    Args:
        Q_reduced: Reduced QUBO matrix
        bitstring_full: Full bitstring (includes bits for given cells)
        var_to_idx: Mapping from (i,j,k) to reduced index
        idx_to_var: Mapping from reduced index to (i,j,k)
        constant_offset: Constant to add to energy
    
    Returns:
        Energy value
    """
    n_free = len(idx_to_var)
    
    # Extract free variable values
    x_reduced = np.zeros(n_free, dtype=int)
    
    # Parse full bitstring
    bitstring_clean = bitstring_full.replace(' ', '')
    N = int(round(len(bitstring_clean) ** (1/3)))  # N³ = len
    
    idx_full = 0
    for i in range(N):
        for j in range(N):
            for k in range(N):
                var = (i, j, k)
                if idx_full < len(bitstring_clean):
                    bit_val = int(bitstring_clean[idx_full])
                    
                    # If this variable is in the reduced set, store it
                    if var in var_to_idx:
                        reduced_idx = var_to_idx[var]
                        x_reduced[reduced_idx] = bit_val
                
                idx_full += 1
    
    # Evaluate QUBO: E = x^T Q x + constant
    energy = 0.0
    
    # Diagonal terms
    for i in range(n_free):
        energy += Q_reduced[i, i] * x_reduced[i]
    
    # Off-diagonal terms (Q is upper triangular)
    for i in range(n_free):
        for j in range(i + 1, n_free):
            energy += 2 * Q_reduced[i, j] * x_reduced[i] * x_reduced[j]
    
    # Add the constant offset
    energy += constant_offset
    
    return energy


def reconstruct_full_solution(reduced_bitstring, var_to_idx, idx_to_var, givens, N):
    """
    Convert reduced bitstring (32-bit) back to full bitstring (64-bit).
    
    Args:
        reduced_bitstring: String of 0s and 1s for free variables only
        var_to_idx: Mapping (i,j,k) → reduced index
        idx_to_var: Mapping reduced index → (i,j,k)
        givens: Dictionary {(i,j): digit} of given cells
        N: Sudoku size
    
    Returns:
        Full bitstring (64-bit for 4×4)
    """
    # Initialize full solution
    full_solution = ['0'] * (N * N * N)
    
    # Fill in FREE variables from reduced solution
    for reduced_idx, bit in enumerate(reduced_bitstring):
        i, j, k = idx_to_var[reduced_idx]
        
        # Calculate original index
        full_idx = i * (N * N) + j * N + k
        full_solution[full_idx] = bit
    
    # Fill in GIVEN variables
    for (i, j), digit in givens.items():
        for k in range(N):
            full_idx = i * (N * N) + j * N + k
            if k == (digit - 1):  # This is the given digit
                full_solution[full_idx] = '1'
            else:
                full_solution[full_idx] = '0'
    
    return ''.join(full_solution)


def print_reduction_stats(elimination_info, N, box_size):
    """
    Print statistics about variable elimination.
    
    Args:
        elimination_info: Dictionary from build_reduced_qubo or build_reduced_qubo_direct
        N: Sudoku size
        box_size: Box size
    """
    print("\nVariable Elimination Statistics:")
    print(f"  Total variables (full): {elimination_info['n_total_vars']}")
    print(f"  Given cells: {elimination_info['n_given_cells']}")
    print(f"  Eliminated variables: {elimination_info['n_eliminated_vars']}")
    print(f"  Free variables: {elimination_info['n_free_vars']}")
    print(f"  Variable reduction: {elimination_info['reduction_pct']:.1f}%")
    print(f"  Matrix size reduction: {elimination_info['matrix_size_reduction_pct']:.1f}%")
    
    full_size = elimination_info['n_total_vars']
    reduced_size = elimination_info['n_free_vars']
    
    print(f"\nMatrix Dimensions:")
    print(f"  Full QUBO: {full_size} × {full_size} = {full_size**2:,} entries")
    print(f"  Reduced QUBO: {reduced_size} × {reduced_size} = {reduced_size**2:,} entries")
    print(f"  Savings: {full_size**2 - reduced_size**2:,} entries eliminated")


# ============================================================================
# Documentation: Two Approaches Explained
# ============================================================================

"""
TWO METHODS FOR BUILDING REDUCED QUBO MATRICES:

Method 1: build_reduced_qubo() - EXTRACTION APPROACH
────────────────────────────────────────────────────
Process:
  1. Build full QUBO matrix (N³ × N³)
  2. Identify free variables
  3. Extract submatrix for free variables only
  4. Return reduced matrix (n_free × n_free)

Advantages:
  + Clear pedagogical demonstration
  + Shows relationship between full and reduced formulations
  + Easy to verify correctness

Disadvantages:
  - Builds unnecessary full matrix first
  - Memory inefficient for large puzzles
  - More computational overhead

Example for 4×4 with 8 givens:
  1. Build 64×64 matrix (4,096 entries)
  2. Identify 32 free variables
  3. Extract 32×32 submatrix (1,024 entries)
  4. Discard the remaining 3,072 entries


Method 2: build_reduced_qubo_direct() - DIRECT CONSTRUCTION
────────────────────────────────────────────────────────────
Process:
  1. Identify free variables upfront
  2. Create reduced index mappings directly
  3. Build only the reduced matrix (n_free × n_free)
  4. Add terms only for free variable interactions

Advantages:
  + More memory efficient
  + Faster for large problems
  + Production-ready implementation

Disadvantages:
  - Less obvious how it relates to full formulation
  - Requires careful index management

Example for 4×4 with 8 givens:
  1. Identify 32 free variables
  2. Build 32×32 matrix directly (1,024 entries)
  3. Never allocate space for the other 3,072 entries


USAGE RECOMMENDATIONS:
─────────────────────
- For learning/teaching: Use build_reduced_qubo() (extraction)
  → Shows how givens eliminate variables from the full problem
  → Easier to understand the transformation

- For production/large problems: Use build_reduced_qubo_direct()
  → More efficient memory usage
  → Better performance for 9×9 Sudoku and larger

- Both methods produce IDENTICAL results (verified in tests)


VARIABLE INDEXING EXAMPLE (4×4 Sudoku with 8 givens):
─────────────────────────────────────────────────────
Given cells: (0,0), (0,2), (1,1), (1,3), (2,0), (2,2), (3,1), (3,3)

Full QUBO indexing (64 variables):
  Index 0-3:   x(0,0,0-3) ← GIVEN CELL (eliminated)
  Index 4-7:   x(0,1,0-3) ← FREE CELL
  Index 8-11:  x(0,2,0-3) ← GIVEN CELL (eliminated)
  Index 12-15: x(0,3,0-3) ← FREE CELL
  ...

Reduced QUBO indexing (32 variables):
  Index 0-3:   x(0,1,0-3) ← Was index 4-7
  Index 4-7:   x(0,3,0-3) ← Was index 12-15
  Index 8-11:  x(1,0,0-3) ← Was index 16-19
  ...

The mappings (var_to_idx and idx_to_var) handle this remapping automatically.
"""