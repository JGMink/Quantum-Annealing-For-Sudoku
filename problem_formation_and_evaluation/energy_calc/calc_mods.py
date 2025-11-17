import numpy as np

# === Lagrange Penalties (Globals) ===
L1, L2, L3, L4 = 1.0, 1.0, 1.0, 1.0  # All set to 1.0

def compute_E1(x, N):
    """
    Ensures each cell has exactly one digit
    E1 = Σ(i,j) [Σk x[i,j,k] - 1]²
    """
    E1 = 0
    for i in range(N):
        for j in range(N):
            cell_sum = np.sum(x[i, j, :])
            E1 += (cell_sum - 1) ** 2
    return int(E1)

def compute_E2(x, N):
    """
    Ensures each row has each digit exactly once
    E2 = Σ(i,k) [Σj x[i,j,k] - 1]²
    """
    E2 = 0
    for i in range(N):
        for k in range(N):
            row_digit_sum = np.sum(x[i, :, k])
            E2 += (row_digit_sum - 1) ** 2
    return int(E2)

def compute_E3(x, N):
    """
    Ensures each column has each digit exactly once
    E3 = Σ(j,k) [Σi x[i,j,k] - 1]²
    """
    E3 = 0
    for j in range(N):
        for k in range(N):
            col_digit_sum = np.sum(x[:, j, k])
            E3 += (col_digit_sum - 1) ** 2
    return int(E3)

def compute_E4(x, N, box_size):
    """
    Ensures each box has each digit exactly once
    E4 = Σ(box,k) [Σ(i,j in box) x[i,j,k] - 1]²
    
    For N=4, box_size=2, we have 4 boxes (2x2 grid of boxes)
    For N=9, box_size=3, we have 9 boxes (3x3 grid of boxes)
    """
    E4 = 0
    boxes_per_side = N // box_size
    
    for box_row in range(boxes_per_side):
        for box_col in range(boxes_per_side):
            for k in range(N):
                box_digit_sum = 0
                for i in range(box_row * box_size, (box_row + 1) * box_size):
                    for j in range(box_col * box_size, (box_col + 1) * box_size):
                        box_digit_sum += x[i, j, k]
                E4 += (box_digit_sum - 1) ** 2
    
    return int(E4)

def bitstring_to_tensor(bitstring, N):
    """
    Convert bitstring to 3D tensor x[i,j,k]
    bitstring has length N³, ordered as (i, j, k)
    """
    x = np.zeros((N, N, N), dtype=int)
    idx = 0
    for i in range(N):
        for j in range(N):
            for k in range(N):
                x[i, j, k] = int(bitstring[idx])
                idx += 1
    return x

def tensor_to_grid(x, N):
    """
    Convert 3D tensor x[i,j,k] to 2D Sudoku grid
    Returns None if a cell doesn't have exactly one digit
    """
    grid = np.zeros((N, N), dtype=int)
    for i in range(N):
        for j in range(N):
            digits = np.where(x[i, j, :] == 1)[0]
            if len(digits) == 1:
                grid[i, j] = digits[0] + 1  # Convert from 0-indexed to 1-indexed
            else:
                return None  # Invalid: cell has 0 or multiple digits
    return grid

def total_energy(bitstring, N, box_size, L1=1.0, L2=1.0, L3=1.0, L4=1.0, verbose=False):
    """
    Compute total QUBO energy for Sudoku with optional debug output
    
    Args:
        bitstring: Binary string of length N³
        N: Size of Sudoku (4 for 4x4, 9 for 9x9)
        box_size: Size of each box (2 for 4x4, 3 for 9x9)
        L1, L2, L3, L4: Lagrange multipliers
        verbose: Whether to print detailed output
    """
    x = bitstring_to_tensor(bitstring, N)
    
    if verbose:
        print("\nTensor x[i,j,k] shape:", x.shape)
        grid = tensor_to_grid(x, N)
        if grid is not None:
            print("\nSudoku grid:")
            print_grid(grid, N, box_size)
        else:
            print("\nInvalid grid (cells with 0 or multiple digits)")
        print("\nComputing energy components:")
    
    E_1 = compute_E1(x, N)
    E_2 = compute_E2(x, N)
    E_3 = compute_E3(x, N)
    E_4 = compute_E4(x, N, box_size)
    
    total = L1*E_1 + L2*E_2 + L3*E_3 + L4*E_4
    
    if verbose:
        print(f"  E1 (cell constraint)   = {E_1}")
        print(f"  E2 (row constraint)    = {E_2}")
        print(f"  E3 (column constraint) = {E_3}")
        print(f"  E4 (box constraint)    = {E_4}")
        print(f"\nTotal energy calculation:")
        print(f"  E = {L1}×{E_1} + {L2}×{E_2} + {L3}×{E_3} + {L4}×{E_4}")
        print(f"  E = {L1*E_1} + {L2*E_2} + {L3*E_3} + {L4*E_4}")
        print(f"  E = {total}")
    
    return total, (E_1, E_2, E_3, E_4)

def is_valid_solution(breakdown):
    """Check if configuration satisfies all constraints"""
    E1, E2, E3, E4 = breakdown
    return E1 == 0 and E2 == 0 and E3 == 0 and E4 == 0

def print_grid(grid, N, box_size):
    """Pretty print Sudoku grid with box separators"""
    for i in range(N):
        if i > 0 and i % box_size == 0:
            print("-" * (2*N + box_size - 1))
        row_str = ""
        for j in range(N):
            if j > 0 and j % box_size == 0:
                row_str += "| "
            row_str += str(grid[i, j]) + " "
        print(row_str.rstrip())

def grid_to_bitstring(grid, N):
    """
    Convert 2D Sudoku grid to bitstring
    grid[i,j] should contain digit 1 to N
    """
    bitstring = ""
    for i in range(N):
        for j in range(N):
            for k in range(N):
                # k is 0-indexed, grid values are 1-indexed
                if grid[i, j] == k + 1:
                    bitstring += "1"
                else:
                    bitstring += "0"
    return bitstring

if __name__ == "__main__":
    N = 4  # 4x4 Sudoku
    box_size = 2
    
    # Example 1: Valid complete solution
    print("="*60)
    print("=== Example 1: Valid complete solution ===")
    print("="*60)
    
    # Grid representation:
    # 2 1 | 4 3
    # 4 3 | 2 1
    # -----+----
    # 1 2 | 3 4
    # 3 4 | 1 2
    
    bitstring1 = (
        "0100" + "1000" + "0001" + "0010" +  # Row 0: 2,1,4,3
        "0001" + "0010" + "0100" + "1000" +  # Row 1: 4,3,2,1
        "1000" + "0100" + "0010" + "0001" +  # Row 2: 1,2,3,4
        "0010" + "0001" + "1000" + "0100"    # Row 3: 3,4,1,2
    )
    total_E, breakdown = total_energy(bitstring1, N, box_size, L1, L2, L3, L4, verbose=True)
    print(f"\nTotal Energy: {total_E}")
    print(f"Energy Breakdown: E1={breakdown[0]}, E2={breakdown[1]}, E3={breakdown[2]}, E4={breakdown[3]}")
    print(f"Valid solution: {is_valid_solution(breakdown)}")
    
    # Example 2: Column and box violations (no row violation)
    print("\n" + "="*60)
    print("=== Example 2: Column and box violations ===")
    print("="*60)
    
    # Grid representation:
    # 2 1 | 4 3
    # 2 3 | 4 1    <- Swapped (1,0) and (1,2): 4->2 and 2->4
    # -----+----
    # 1 2 | 3 4
    # 3 4 | 1 2
    
    bitstring2 = (
        "0100" + "1000" + "0001" + "0010" +  # Row 0: 2,1,4,3
        "0100" + "0010" + "0001" + "1000" +  # Row 1: 2,3,4,1 (violations!)
        "1000" + "0100" + "0010" + "0001" +  # Row 2: 1,2,3,4
        "0010" + "0001" + "1000" + "0100"    # Row 3: 3,4,1,2
    )
    total_E, breakdown = total_energy(bitstring2, N, box_size, L1, L2, L3, L4, verbose=True)
    print(f"\nTotal Energy: {total_E}")
    print(f"Energy Breakdown: E1={breakdown[0]}, E2={breakdown[1]}, E3={breakdown[2]}, E4={breakdown[3]}")
    print(f"Valid solution: {is_valid_solution(breakdown)}")
    print("\nNote: Row 1 has all digits 1-4 once (no row violation)")
    print("      Column 0 has two 2's, Column 2 has two 4's")
    print("      Box 0 (top-left) has two 2's, Box 1 (top-right) has two 4's")
    
    # Example 3: Multiple violations
    print("\n" + "="*60)
    print("=== Example 3: Multiple violations ===")
    print("="*60)
    
    # Grid representation:
    # 2 2 | 4 3    <- Row 0 has two 2's, missing 1
    # 4 3 | 2 1
    # -----+----
    # 1 2 | 3 4
    # 3 4 | 1 2
    
    bitstring3 = (
        "0100" + "0100" + "0001" + "0010" +  # Row 0: 2,2,4,3 (violations!)
        "0001" + "0010" + "0100" + "1000" +  # Row 1: 4,3,2,1
        "1000" + "0100" + "0010" + "0001" +  # Row 2: 1,2,3,4
        "0010" + "0001" + "1000" + "0100"    # Row 3: 3,4,1,2
    )
    total_E, breakdown = total_energy(bitstring3, N, box_size, L1, L2, L3, L4, verbose=True)
    print(f"\nTotal Energy: {total_E}")
    print(f"Energy Breakdown: E1={breakdown[0]}, E2={breakdown[1]}, E3={breakdown[2]}, E4={breakdown[3]}")
    print(f"Valid solution: {is_valid_solution(breakdown)}")
    print("\nNote: Row 0 has two 2's (no 1)")
    print("      Column 1 has two 2's (no 1)")
    print("      Box 0 (top-left) has two 2's (no 1)")
    
    # Example 4: E1 violation - cell with multiple digits
    print("\n" + "="*60)
    print("=== Example 4: E1 violation (raw bitstring input) ===")
    print("="*60)
    
    # This example shows why we need raw bitstring input from QPU:
    # Some cells can have 0 digits or multiple digits assigned - impossible
    # to represent as a simple grid, but possible as QPU output.
    #
    # Grid representation (invalid):
    # {1,2} 3 | 4 _    <- Cell (0,0) has TWO digits, cell (0,3) has NONE
    # 4     3 | 2 1
    # ---------+------
    # 1     2 | 3 4
    # 3     4 | 1 2
    
    bitstring4 = (
        "1100" + "0010" + "0001" + "0000" +  # Row 0: {1,2},3,4,_ (E1 violations!)
        "0001" + "0010" + "0100" + "1000" +  # Row 1: 4,3,2,1
        "1000" + "0100" + "0010" + "0001" +  # Row 2: 1,2,3,4
        "0010" + "0001" + "1000" + "0100"    # Row 3: 3,4,1,2
    )
    
    total_E, breakdown = total_energy(bitstring4, N, box_size, L1, L2, L3, L4, verbose=True)
    print(f"\nTotal Energy: {total_E}")
    print(f"Energy Breakdown: E1={breakdown[0]}, E2={breakdown[1]}, E3={breakdown[2]}, E4={breakdown[3]}")
    print(f"Valid solution: {is_valid_solution(breakdown)}")
    print("\nNote: Cell (0,0) has 2 digits assigned → E1 += (2-1)² = 1")
    print("      Cell (0,3) has 0 digits assigned → E1 += (0-1)² = 1")
    print("      Total E1 = 2")