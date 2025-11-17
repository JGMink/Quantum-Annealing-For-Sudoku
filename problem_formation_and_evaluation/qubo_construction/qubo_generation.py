import numpy as np

def build_sudoku_qubo(N, box_size, givens=None, L1=1.0, L2=1.0, L3=1.0, L4=1.0):
    """
    Build QUBO matrix for Sudoku puzzle
    
    Args:
        N: Size of Sudoku (4 for 4x4, 9 for 9x9)
        box_size: Size of each box (2 for 4x4, 3 for 9x9)
        givens: Dictionary {(i,j): digit} for known cells (1-indexed digits)
                If None, builds unconstrained QUBO
        L1, L2, L3, L4: Lagrange multipliers for constraints
    
    Returns:
        Q: QUBO matrix of shape (N³, N³)
        var_to_idx: Dictionary mapping (i,j,k) to bitstring index
        idx_to_var: Dictionary mapping bitstring index to (i,j,k)
        constant_offset: Constant term to add to energy
    """
    # Total number of binary variables
    n_vars = N * N * N
    
    # Initialize QUBO matrix
    Q = np.zeros((n_vars, n_vars))
    
    # Initialize constant offset
    constant_offset = 0.0
    
    # Create index mappings
    var_to_idx = {}
    idx_to_var = {}
    idx = 0
    for i in range(N):
        for j in range(N):
            for k in range(N):
                var_to_idx[(i, j, k)] = idx
                idx_to_var[idx] = (i, j, k)
                idx += 1
    
    # Helper function to add quadratic term to Q
    def add_quadratic(var1, var2, coeff):
        idx1 = var_to_idx[var1]
        idx2 = var_to_idx[var2]
        if idx1 <= idx2:
            Q[idx1, idx2] += coeff
        else:
            Q[idx2, idx1] += coeff
    
    # Helper function to add linear term to Q (diagonal)
    def add_linear(var, coeff):
        idx = var_to_idx[var]
        Q[idx, idx] += coeff
    
    # ===== E1: Each cell has exactly one digit =====
    # E1 = Σ(i,j) [Σk x[i,j,k] - 1]²
    #    = Σ(i,j) [-Σk x[i,j,k] + 2Σ(k<k') x[i,j,k]x[i,j,k'] + 1]
    
    for i in range(N):
        for j in range(N):
            # If this cell is given, add constant penalty of 0 (constraint satisfied)
            if givens and (i, j) in givens:
                constant_offset += 0  # Constraint automatically satisfied
                continue
            
            # Linear terms: -x[i,j,k]
            for k in range(N):
                add_linear((i, j, k), -L1)
            
            # Quadratic terms: 2*x[i,j,k]*x[i,j,k']
            for k in range(N):
                for kp in range(k + 1, N):
                    add_quadratic((i, j, k), (i, j, kp), 2 * L1)
            
            # Constant: +1
            constant_offset += L1
    
    # ===== E2: Each row has each digit exactly once =====
    # E2 = Σ(i,k) [Σj x[i,j,k] - 1]²
    # When some cells are given, we need: [Σj(free) x[i,j,k] + count_given - 1]²
    
    for i in range(N):
        for k in range(N):
            # Count how many cells in this row already have digit k (from givens)
            given_count = 0
            free_cells = []
            
            for j in range(N):
                if givens and (i, j) in givens:
                    if givens[(i, j)] == k + 1:  # k is 0-indexed, givens are 1-indexed
                        given_count += 1
                else:
                    free_cells.append(j)
            
            # If already satisfied by givens, skip
            if given_count == 1 and len(free_cells) == 0:
                constant_offset += 0
                continue
            
            # Expand: [Σj(free) x[i,j,k] + given_count - 1]²
            # = [Σj(free) x[i,j,k] - (1 - given_count)]²
            # = Σj x² - 2(1-given_count)Σj x + 2Σ(j<j') x*x' + (1-given_count)²
            
            target_adjustment = 1 - given_count
            
            # Linear terms: (1 - 2*target_adjustment)*x[i,j,k]
            for j in free_cells:
                add_linear((i, j, k), L2 * (1 - 2 * target_adjustment))
            
            # Quadratic terms: 2*x[i,j,k]*x[i,j',k]
            for j_idx, j in enumerate(free_cells):
                for jp in free_cells[j_idx + 1:]:
                    add_quadratic((i, j, k), (i, jp, k), 2 * L2)
            
            # Constant: target_adjustment²
            constant_offset += L2 * (target_adjustment ** 2)
    
    # ===== E3: Each column has each digit exactly once =====
    # E3 = Σ(j,k) [Σi x[i,j,k] - 1]²
    
    for j in range(N):
        for k in range(N):
            # Count how many cells in this column already have digit k (from givens)
            given_count = 0
            free_cells = []
            
            for i in range(N):
                if givens and (i, j) in givens:
                    if givens[(i, j)] == k + 1:
                        given_count += 1
                else:
                    free_cells.append(i)
            
            # If already satisfied by givens, skip
            if given_count == 1 and len(free_cells) == 0:
                constant_offset += 0
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
    # E4 = Σ(box,k) [Σ(i,j in box) x[i,j,k] - 1]²
    
    boxes_per_side = N // box_size
    
    for box_row in range(boxes_per_side):
        for box_col in range(boxes_per_side):
            for k in range(N):
                # Count how many cells in this box already have digit k (from givens)
                given_count = 0
                free_cells = []
                
                for i in range(box_row * box_size, (box_row + 1) * box_size):
                    for j in range(box_col * box_size, (box_col + 1) * box_size):
                        if givens and (i, j) in givens:
                            if givens[(i, j)] == k + 1:
                                given_count += 1
                        else:
                            free_cells.append((i, j))
                
                # If already satisfied by givens, skip
                if given_count == 1 and len(free_cells) == 0:
                    constant_offset += 0
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
    
    return Q, var_to_idx, idx_to_var, constant_offset


def build_E1(N, givens=None, L1=1.0):
    """
    Build E1 component: Each cell has exactly one digit
    Returns Q matrix, polynomial terms, and constant
    """
    n_vars = N * N * N
    Q = np.zeros((n_vars, n_vars))
    polynomial_terms = []
    constant = 0.0
    
    # Create index mapping
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
            
            # Linear terms
            for k in range(N):
                idx = var_to_idx[(i, j, k)]
                Q[idx, idx] += -L1
                polynomial_terms.append(f"-{L1}*x[{i},{j},{k}]")
            
            # Quadratic terms
            for k in range(N):
                for kp in range(k + 1, N):
                    idx1 = var_to_idx[(i, j, k)]
                    idx2 = var_to_idx[(i, j, kp)]
                    Q[idx1, idx2] += 2 * L1
                    polynomial_terms.append(f"+{2*L1}*x[{i},{j},{k}]*x[{i},{j},{kp}]")
            
            constant += L1
    
    return Q, polynomial_terms, constant


def build_E2(N, givens=None, L2=1.0):
    """
    Build E2 component: Each row has each digit exactly once
    Returns Q matrix, polynomial terms, and constant
    """
    n_vars = N * N * N
    Q = np.zeros((n_vars, n_vars))
    polynomial_terms = []
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
            
            # Linear terms
            for j in free_cells:
                idx = var_to_idx[(i, j, k)]
                Q[idx, idx] += L2 * (1 - 2 * target_adjustment)
                polynomial_terms.append(f"{L2 * (1 - 2 * target_adjustment)}*x[{i},{j},{k}]")
            
            # Quadratic terms
            for j_idx, j in enumerate(free_cells):
                for jp in free_cells[j_idx + 1:]:
                    idx1 = var_to_idx[(i, j, k)]
                    idx2 = var_to_idx[(i, jp, k)]
                    Q[idx1, idx2] += 2 * L2
                    polynomial_terms.append(f"+{2*L2}*x[{i},{j},{k}]*x[{i},{jp},{k}]")
            
            constant += L2 * (target_adjustment ** 2)
    
    return Q, polynomial_terms, constant


def build_E3(N, givens=None, L3=1.0):
    """
    Build E3 component: Each column has each digit exactly once
    Returns Q matrix, polynomial terms, and constant
    """
    n_vars = N * N * N
    Q = np.zeros((n_vars, n_vars))
    polynomial_terms = []
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
            
            # Linear terms
            for i in free_cells:
                idx = var_to_idx[(i, j, k)]
                Q[idx, idx] += L3 * (1 - 2 * target_adjustment)
                polynomial_terms.append(f"{L3 * (1 - 2 * target_adjustment)}*x[{i},{j},{k}]")
            
            # Quadratic terms
            for i_idx, i in enumerate(free_cells):
                for ip in free_cells[i_idx + 1:]:
                    idx1 = var_to_idx[(i, j, k)]
                    idx2 = var_to_idx[(ip, j, k)]
                    Q[idx1, idx2] += 2 * L3
                    polynomial_terms.append(f"+{2*L3}*x[{i},{j},{k}]*x[{ip},{j},{k}]")
            
            constant += L3 * (target_adjustment ** 2)
    
    return Q, polynomial_terms, constant


def build_E4(N, box_size, givens=None, L4=1.0):
    """
    Build E4 component: Each box has each digit exactly once
    Returns Q matrix, polynomial terms, and constant
    """
    n_vars = N * N * N
    Q = np.zeros((n_vars, n_vars))
    polynomial_terms = []
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
                
                # Linear terms
                for (i, j) in free_cells:
                    idx = var_to_idx[(i, j, k)]
                    Q[idx, idx] += L4 * (1 - 2 * target_adjustment)
                    polynomial_terms.append(f"{L4 * (1 - 2 * target_adjustment)}*x[{i},{j},{k}]")
                
                # Quadratic terms
                for cell_idx, (i, j) in enumerate(free_cells):
                    for (ip, jp) in free_cells[cell_idx + 1:]:
                        idx1 = var_to_idx[(i, j, k)]
                        idx2 = var_to_idx[(ip, jp, k)]
                        Q[idx1, idx2] += 2 * L4
                        polynomial_terms.append(f"+{2*L4}*x[{i},{j},{k}]*x[{ip},{jp},{k}]")
                
                constant += L4 * (target_adjustment ** 2)
    
    return Q, polynomial_terms, constant


def print_E1_details(N, givens=None, L1=1.0):
    """Print detailed construction of E1 component"""
    Q, terms, const = build_E1(N, givens, L1)
    print(f"E1: Each cell has exactly one digit")
    print(f"  Number of terms: {len(terms)}")
    print(f"  Constant: {const}")
    print(f"  Sample terms: {terms[:5] if len(terms) > 5 else terms}")
    return Q, terms, const


def print_E2_details(N, givens=None, L2=1.0):
    """Print detailed construction of E2 component"""
    Q, terms, const = build_E2(N, givens, L2)
    print(f"E2: Each row has each digit exactly once")
    print(f"  Number of terms: {len(terms)}")
    print(f"  Constant: {const}")
    print(f"  Sample terms: {terms[:5] if len(terms) > 5 else terms}")
    return Q, terms, const


def print_E3_details(N, givens=None, L3=1.0):
    """Print detailed construction of E3 component"""
    Q, terms, const = build_E3(N, givens, L3)
    print(f"E3: Each column has each digit exactly once")
    print(f"  Number of terms: {len(terms)}")
    print(f"  Constant: {const}")
    print(f"  Sample terms: {terms[:5] if len(terms) > 5 else terms}")
    return Q, terms, const


def print_E4_details(N, box_size, givens=None, L4=1.0):
    """Print detailed construction of E4 component"""
    Q, terms, const = build_E4(N, box_size, givens, L4)
    print(f"E4: Each box has each digit exactly once")
    print(f"  Number of terms: {len(terms)}")
    print(f"  Constant: {const}")
    print(f"  Sample terms: {terms[:5] if len(terms) > 5 else terms}")
    return Q, terms, const


def evaluate_qubo(Q, bitstring, constant_offset=0):
    """
    Evaluate QUBO energy for a given bitstring
    
    Args:
        Q: QUBO matrix
        bitstring: Binary string or list of 0/1 values
        constant_offset: Constant to add to energy
    
    Returns:
        Energy value
    """
    x = np.array([int(b) for b in bitstring])
    
    # E = x^T Q x + constant
    # For upper triangular Q: E = Σi Q[i,i]*x[i]² + 2*Σ(i<j) Q[i,j]*x[i]*x[j]
    energy = 0.0
    
    # Diagonal terms
    for i in range(len(x)):
        energy += Q[i, i] * x[i]
    
    # Off-diagonal terms (Q is upper triangular)
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            energy += 2 * Q[i, j] * x[i] * x[j]
    
    return energy + constant_offset

def print_qubo_stats(Q, N, givens=None):
    """Print statistics about the QUBO matrix"""
    n_vars = N * N * N
    
    print(f"\nQUBO Matrix Statistics:")
    print(f"  Size: {n_vars} × {n_vars}")
    print(f"  Total possible entries: {n_vars * n_vars:,}")
    
    # Count non-zero entries
    nonzero_diag = np.count_nonzero(np.diag(Q))
    nonzero_upper = np.count_nonzero(np.triu(Q, k=1))
    total_nonzero = nonzero_diag + nonzero_upper
    
    print(f"  Non-zero diagonal entries: {nonzero_diag}")
    print(f"  Non-zero off-diagonal entries: {nonzero_upper}")
    print(f"  Total non-zero entries: {total_nonzero}")
    print(f"  Sparsity: {100 * (1 - total_nonzero / (n_vars * n_vars)):.2f}%")
    
    if givens:
        print(f"\nGiven cells: {len(givens)}")
        print(f"Free variables: {n_vars - len(givens) * N}")


def build_reduced_qubo_direct(N, box_size, givens, L1=1.0, L2=1.0, L3=1.0, L4=1.0):
    """
    Build a reduced QUBO matrix directly, only including free variables.
    
    This is more efficient than build_reduced_qubo() because it doesn't build
    the full QUBO matrix first. It directly constructs the smaller matrix.
    
    Args:
        N: Size of Sudoku (4 for 4x4, 9 for 9x9)
        box_size: Size of each box (2 for 4x4, 3 for 9x9)
        givens: Dictionary {(i,j): digit} for known cells (1-indexed digits)
                Must not be None
        L1, L2, L3, L4: Lagrange multipliers for constraints
    
    Returns:
        Q_reduced: Reduced QUBO matrix of shape (n_free, n_free)
        var_to_idx_reduced: Mapping from (i,j,k) to reduced index
        idx_to_var_reduced: Mapping from reduced index to (i,j,k)
        constant_offset: Constant term to add to energy
        elimination_info: Dict with info about eliminated variables
    """
    if givens is None:
        raise ValueError("givens cannot be None for reduced QUBO. Use build_sudoku_qubo for blank puzzles.")
    
    # Identify free variables upfront
    free_vars = []
    eliminated_count = 0
    
    for i in range(N):
        for j in range(N):
            for k in range(N):
                if (i, j) in givens:
                    eliminated_count += 1
                else:
                    free_vars.append((i, j, k))
    
    n_free = len(free_vars)
    
    # Create mappings for reduced matrix
    var_to_idx_reduced = {}
    idx_to_var_reduced = {}
    
    for idx, var in enumerate(free_vars):
        var_to_idx_reduced[var] = idx
        idx_to_var_reduced[idx] = var
    
    # Initialize reduced QUBO matrix
    Q_reduced = np.zeros((n_free, n_free))
    constant_offset = 0.0
    
    # Helper functions to add terms (only for free variables)
    def add_quadratic(var1, var2, coeff):
        if var1 in var_to_idx_reduced and var2 in var_to_idx_reduced:
            idx1 = var_to_idx_reduced[var1]
            idx2 = var_to_idx_reduced[var2]
            if idx1 <= idx2:
                Q_reduced[idx1, idx2] += coeff
            else:
                Q_reduced[idx2, idx1] += coeff
    
    def add_linear(var, coeff):
        if var in var_to_idx_reduced:
            idx = var_to_idx_reduced[var]
            Q_reduced[idx, idx] += coeff
    
    # ===== E1: Each cell has exactly one digit =====
    for i in range(N):
        for j in range(N):
            if givens and (i, j) in givens:
                constant_offset += 0  # Given cells don't contribute
                continue
            
            # Linear terms: -x[i,j,k]
            for k in range(N):
                add_linear((i, j, k), -L1)
            
            # Quadratic terms: 2*x[i,j,k]*x[i,j,k']
            for k in range(N):
                for kp in range(k + 1, N):
                    add_quadratic((i, j, k), (i, j, kp), 2 * L1)
            
            constant_offset += L1
    
    # ===== E2: Each row has each digit exactly once =====
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
                constant_offset += 0
                continue
            
            target_adjustment = 1 - given_count
            
            # Linear terms
            for j in free_cells:
                add_linear((i, j, k), L2 * (1 - 2 * target_adjustment))
            
            # Quadratic terms
            for j_idx, j in enumerate(free_cells):
                for jp in free_cells[j_idx + 1:]:
                    add_quadratic((i, j, k), (i, jp, k), 2 * L2)
            
            constant_offset += L2 * (target_adjustment ** 2)
    
    # ===== E3: Each column has each digit exactly once =====
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
                constant_offset += 0
                continue
            
            target_adjustment = 1 - given_count
            
            # Linear terms
            for i in free_cells:
                add_linear((i, j, k), L3 * (1 - 2 * target_adjustment))
            
            # Quadratic terms
            for i_idx, i in enumerate(free_cells):
                for ip in free_cells[i_idx + 1:]:
                    add_quadratic((i, j, k), (ip, j, k), 2 * L3)
            
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
                        if givens and (i, j) in givens:
                            if givens[(i, j)] == k + 1:
                                given_count += 1
                        else:
                            free_cells.append((i, j))
                
                if given_count == 1 and len(free_cells) == 0:
                    constant_offset += 0
                    continue
                
                target_adjustment = 1 - given_count
                
                # Linear terms
                for (i, j) in free_cells:
                    add_linear((i, j, k), L4 * (1 - 2 * target_adjustment))
                
                # Quadratic terms
                for cell_idx, (i, j) in enumerate(free_cells):
                    for (ip, jp) in free_cells[cell_idx + 1:]:
                        add_quadratic((i, j, k), (ip, jp, k), 2 * L4)
                
                constant_offset += L4 * (target_adjustment ** 2)
    
    elimination_info = {
        'n_total_vars': N * N * N,
        'n_free_vars': n_free,
        'n_eliminated_vars': eliminated_count,
        'n_given_cells': len(givens),
        'reduction_pct': 100 * (1 - n_free / (N * N * N)),
        'matrix_size_reduction_pct': 100 * (1 - (n_free ** 2) / ((N * N * N) ** 2))
    }
    
    return Q_reduced, var_to_idx_reduced, idx_to_var_reduced, constant_offset, elimination_info


def build_reduced_qubo(N, box_size, givens, L1=1.0, L2=1.0, L3=1.0, L4=1.0):
    """
    Build a reduced QUBO matrix that only includes free variables.
    
    This method works by:
    1. Building the full QUBO matrix (which internally handles givens)
    2. Identifying free variables (those not in given cells)
    3. Extracting the submatrix corresponding to free variables
    
    This is less efficient than build_reduced_qubo_direct() but demonstrates
    the relationship between full and reduced formulations clearly.
    
    Args:
        N: Size of Sudoku (4 for 4x4, 9 for 9x9)
        box_size: Size of each box (2 for 4x4, 3 for 9x9)
        givens: Dictionary {(i,j): digit} for known cells (1-indexed digits)
                Must not be None
        L1, L2, L3, L4: Lagrange multipliers for constraints
    
    Returns:
        Q_reduced: Reduced QUBO matrix of shape (n_free, n_free)
        var_to_idx_reduced: Mapping from (i,j,k) to reduced index
        idx_to_var_reduced: Mapping from reduced index to (i,j,k)
        constant_offset: Constant term to add to energy
        elimination_info: Dict with info about eliminated variables
    
    Example:
        For a 4×4 Sudoku with 8 givens:
        - Full QUBO: 64×64 matrix (many zeros for given cells)
        - Reduced QUBO: 32×32 matrix (only free variables)
        - Variable remapping: x(0,1,0) at index 4 in full → index 0 in reduced
    """
    if givens is None:
        raise ValueError("givens cannot be None for reduced QUBO. Use build_sudoku_qubo for blank puzzles.")
    
    # STEP 1: Build full QUBO (which already handles givens internally)
    # The full QUBO has dimension (N³ × N³) but has zeros/small values for given variables
    Q_full, var_to_idx_full, idx_to_var_full, offset = build_sudoku_qubo(
        N, box_size, givens, L1, L2, L3, L4
    )
    
    # STEP 2: Identify free variables
    # Free variables are those where the cell (i,j) is not in givens
    free_vars = []
    eliminated_vars = []
    
    for idx in range(N * N * N):
        var = idx_to_var_full[idx]
        i, j, k = var
        
        if (i, j) in givens:
            eliminated_vars.append(var)
        else:
            free_vars.append(var)
    
    n_free = len(free_vars)
    
    # STEP 3: Create new index mappings for reduced matrix
    # Old indexing: x(i,j,k) → index in [0, 63] for 4×4
    # New indexing: x(i,j,k) → index in [0, 31] for 4×4 with 8 givens
    var_to_idx_reduced = {}
    idx_to_var_reduced = {}
    
    for new_idx, var in enumerate(free_vars):
        var_to_idx_reduced[var] = new_idx
        idx_to_var_reduced[new_idx] = var
    
    # STEP 4: Build reduced matrix by extracting submatrix
    # This extracts only the rows/columns corresponding to free variables
    Q_reduced = np.zeros((n_free, n_free))
    
    # Create mapping from old indices to new indices
    old_to_new = {}
    for new_idx, var in enumerate(free_vars):
        old_idx = var_to_idx_full[var]
        old_to_new[old_idx] = new_idx
    
    # Copy relevant entries from full Q to reduced Q
    # Example: Q_full[4,5] (interaction between x(0,1,0) and x(0,1,1))
    #       → Q_reduced[0,1] (same variables, new indices)
    for var_i in free_vars:
        old_i = var_to_idx_full[var_i]
        new_i = old_to_new[old_i]
        
        for var_j in free_vars:
            old_j = var_to_idx_full[var_j]
            new_j = old_to_new[old_j]
            
            Q_reduced[new_i, new_j] = Q_full[old_i, old_j]
    
    # STEP 5: Gather elimination statistics
    elimination_info = {
        'n_total_vars': N * N * N,
        'n_free_vars': n_free,
        'n_eliminated_vars': len(eliminated_vars),
        'n_given_cells': len(givens),
        'reduction_pct': 100 * (1 - n_free / (N * N * N)),
        'matrix_size_reduction_pct': 100 * (1 - (n_free ** 2) / ((N * N * N) ** 2))
    }
    
    return Q_reduced, var_to_idx_reduced, idx_to_var_reduced, offset, elimination_info


def evaluate_reduced_qubo(Q_reduced, bitstring_full, var_to_idx_reduced, idx_to_var_reduced, constant_offset=0):
    """
    Evaluate reduced QUBO energy given a full bitstring.
    
    This extracts only the free variable values from the full bitstring
    and evaluates the reduced QUBO.
    
    Args:
        Q_reduced: Reduced QUBO matrix
        bitstring_full: Full bitstring (including bits for given cells)
        var_to_idx_reduced: Mapping from (i,j,k) to reduced index
        idx_to_var_reduced: Mapping from reduced index to (i,j,k)
        constant_offset: Constant to add to energy
    
    Returns:
        Energy value
    """
    n_free = len(idx_to_var_reduced)
    
    # Extract free variable values
    x_reduced = np.zeros(n_free, dtype=int)
    
    # Parse full bitstring to get variable values
    bitstring_clean = bitstring_full.replace(' ', '')
    N = int(round(len(bitstring_clean) ** (1/3)))  # N³ = len, so N = len^(1/3)
    
    idx_full = 0
    for i in range(N):
        for j in range(N):
            for k in range(N):
                var = (i, j, k)
                if idx_full < len(bitstring_clean):
                    bit_val = int(bitstring_clean[idx_full])
                    
                    # If this variable is in the reduced set, store it
                    if var in var_to_idx_reduced:
                        reduced_idx = var_to_idx_reduced[var]
                        x_reduced[reduced_idx] = bit_val
                
                idx_full += 1
    
    # Evaluate QUBO: E = x^T Q x + constant
    energy = 0.0  # Don't add constant here - it's already baked into Q_reduced
    
    # Diagonal terms
    for i in range(n_free):
        energy += Q_reduced[i, i] * x_reduced[i]
    
    # Off-diagonal terms (Q is upper triangular)
    for i in range(n_free):
        for j in range(i + 1, n_free):
            energy += 2 * Q_reduced[i, j] * x_reduced[i] * x_reduced[j]
    
    # NOW add the constant offset
    energy += constant_offset
    
    return energy


def print_reduction_stats(elimination_info, N, box_size):
    """
    Print statistics about variable elimination.
    
    Args:
        elimination_info: Dictionary from build_reduced_qubo
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
# COMPARISON: Two Approaches to Building Reduced QUBO
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
  Index 0-3:   x(0,0,0), x(0,0,1), x(0,0,2), x(0,0,3)  ← GIVEN CELL (eliminated)
  Index 4-7:   x(0,1,0), x(0,1,1), x(0,1,2), x(0,1,3)  ← FREE CELL
  Index 8-11:  x(0,2,0), x(0,2,1), x(0,2,2), x(0,2,3)  ← GIVEN CELL (eliminated)
  Index 12-15: x(0,3,0), x(0,3,1), x(0,3,2), x(0,3,3)  ← FREE CELL
  ...

Reduced QUBO indexing (32 variables):
  Index 0-3:   x(0,1,0), x(0,1,1), x(0,1,2), x(0,1,3)  ← Was index 4-7
  Index 4-7:   x(0,3,0), x(0,3,1), x(0,3,2), x(0,3,3)  ← Was index 12-15
  Index 8-11:  x(1,0,0), x(1,0,1), x(1,0,2), x(1,0,3)  ← Was index 16-19
  ...

Note how indices are remapped: old index 4 → new index 0, etc.


MATRIX STRUCTURE COMPARISON:
────────────────────────────
Full QUBO Q_full (64×64):
  ┌─────────────────────────────┐
  │ 0  0  0  0  .  .  .  .  ... │  ← Given cell (0,0): all zeros
  │ 0  0  0  0  .  .  .  .  ... │
  │ 0  0  0  0  .  .  .  .  ... │
  │ 0  0  0  0  .  .  .  .  ... │
  │ . -4  .  .  2  .  .  .  ... │  ← Free cell (0,1): active coefficients
  │ .  .  .  .  .  .  .  .  ... │
  │ .  .  .  .  .  .  .  .  ... │
  │ .  .  .  .  .  .  .  .  ... │
  │ ...................... ... │
  └─────────────────────────────┘
  
Reduced QUBO Q_reduced (32×32):
  ┌──────────────────┐
  │-4  .  .  .  2  . │  ← Free cell (0,1) is now at index 0
  │ .  .  .  .  .  . │
  │ .  .  .  .  .  . │
  │ .  .  .  .  .  . │
  │ 2  .  .  . -4  . │
  │ .  .  .  .  .  . │
  └──────────────────┘

All the zeros from given cells are gone! Only meaningful entries remain.
"""