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