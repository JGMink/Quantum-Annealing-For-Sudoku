"""
DETAILED WALKTHROUGH: Reducing a 4×4 Sudoku with Givens

Puzzle:
2 _ | 4 _
_ 3 | _ 1
----+----
1 _ | 3 _
_ 4 | _ 2

Givens: (0,0)=2, (0,2)=4, (1,1)=3, (1,3)=1, (2,0)=1, (2,2)=3, (3,1)=4, (3,3)=2
"""

import numpy as np

# ============================================================================
# STEP 1: Map out ALL 64 variables
# ============================================================================

print("="*80)
print("STEP 1: Full Variable List (64 variables)")
print("="*80)
print()

givens = {
    (0, 0): 2,           (0, 2): 4,
              (1, 1): 3,           (1, 3): 1,
    (2, 0): 1,           (2, 2): 3,
              (3, 1): 4,           (3, 3): 2
}

# Create full variable list
all_vars = []
for i in range(4):
    for j in range(4):
        for k in range(4):
            all_vars.append((i, j, k))

print("Variable Index | Cell | Digit | Status | Value")
print("-" * 65)

for idx, (i, j, k) in enumerate(all_vars):
    digit = k + 1
    is_given = (i, j) in givens
    
    if is_given:
        given_digit = givens[(i, j)]
        value = 1 if (k + 1) == given_digit else 0
        status = "GIVEN (FIXED)"
    else:
        value = "?"
        status = "FREE"
    
    # Highlight free variables
    marker = "  ◄──" if status == "FREE" else ""
    
    print(f"     {idx:2d}        ({i},{j})    {digit}     {status:15s}  {value}{marker}")

print()
print("Summary:")
print(f"  Given cells: {len(givens)} × 4 digits = {len(givens) * 4} FIXED variables")
print(f"  Free cells:  {16 - len(givens)} × 4 digits = {(16 - len(givens)) * 4} FREE variables")
print()

# ============================================================================
# STEP 2: Extract Free Variables
# ============================================================================

print("="*80)
print("STEP 2: Extract Only Free Variables")
print("="*80)
print()

free_vars = []
eliminated_vars = []

for idx, (i, j, k) in enumerate(all_vars):
    if (i, j) in givens:
        eliminated_vars.append((idx, i, j, k))
    else:
        free_vars.append((idx, i, j, k))

print(f"Eliminated {len(eliminated_vars)} variables (from given cells)")
print(f"Keeping {len(free_vars)} variables (from free cells)")
print()

print("Free Variables Only:")
print("New Index | Old Index | Cell | Digit | Note")
print("-" * 70)

for new_idx, (old_idx, i, j, k) in enumerate(free_vars[:20]):  # Show first 20
    digit = k + 1
    print(f"    {new_idx:2d}    |     {old_idx:2d}    | ({i},{j})  |   {digit}   | "
          f"Old idx {old_idx} → New idx {new_idx}")

if len(free_vars) > 20:
    print(f"    ...       ...       ...     ...       ({len(free_vars) - 20} more)")

print()

# ============================================================================
# STEP 3: Show Index Remapping
# ============================================================================

print("="*80)
print("STEP 3: Index Remapping Details")
print("="*80)
print()

print("Example remappings:")
print()

# Cell (0,1) - first free cell
print("Cell (0,1) is FREE:")
for k in range(4):
    old_idx = 0*16 + 1*4 + k  # i*16 + j*4 + k
    new_idx = None
    for n_idx, (o_idx, i, j, kk) in enumerate(free_vars):
        if o_idx == old_idx:
            new_idx = n_idx
            break
    
    print(f"  x(0,1,{k}) [digit {k+1}]: Old index {old_idx:2d} → New index {new_idx:2d}")

print()

# Cell (0,3) - another free cell
print("Cell (0,3) is FREE:")
for k in range(4):
    old_idx = 0*16 + 3*4 + k
    new_idx = None
    for n_idx, (o_idx, i, j, kk) in enumerate(free_vars):
        if o_idx == old_idx:
            new_idx = n_idx
            break
    
    print(f"  x(0,3,{k}) [digit {k+1}]: Old index {old_idx:2d} → New index {new_idx:2d}")

print()

# Cell (0,0) - given cell
print("Cell (0,0) is GIVEN (value=2):")
for k in range(4):
    old_idx = 0*16 + 0*4 + k
    print(f"  x(0,0,{k}) [digit {k+1}]: Old index {old_idx:2d} → ELIMINATED")

print()

# ============================================================================
# STEP 4: Matrix Size Comparison
# ============================================================================

print("="*80)
print("STEP 4: Matrix Dimensions")
print("="*80)
print()

full_size = 64
reduced_size = len(free_vars)

print(f"Full QUBO Matrix:    {full_size} × {full_size} = {full_size**2:,} entries")
print(f"Reduced QUBO Matrix: {reduced_size} × {reduced_size} = {reduced_size**2:,} entries")
print()
print(f"Reduction:")
print(f"  Variables: {full_size} → {reduced_size} ({100*(1-reduced_size/full_size):.1f}% reduction)")
print(f"  Matrix size: {full_size**2:,} → {reduced_size**2:,} entries ({100*(1-(reduced_size**2)/(full_size**2)):.1f}% reduction)")
print()

# ============================================================================
# STEP 5: Show Matrix Structure
# ============================================================================

print("="*80)
print("STEP 5: Matrix Structure (Conceptual)")
print("="*80)
print()

print("Full QUBO Matrix (64×64) - showing first 16×16 block:")
print("(G = given cell variable, F = free cell variable)")
print()

print("     ", end="")
for j in range(16):
    print(f"{j:2d} ", end="")
print()

for i in range(16):
    print(f"{i:2d} | ", end="")
    for j in range(16):
        # Determine if this is a given or free variable
        cell_i = i // 4
        cell_j = i % 4
        is_given_i = (cell_i // 4, cell_i % 4) in givens
        
        cell_j_row = j // 4
        cell_j_col = j % 4
        is_given_j = (cell_j_row // 4, cell_j_col % 4) in givens
        
        if i == j:
            if is_given_i:
                print(" 0 ", end="")  # Given variables have zero diagonal
            else:
                print("-4 ", end="")  # Free variables have -4 diagonal
        elif is_given_i or is_given_j:
            print(" 0 ", end="")  # Interactions with given variables are zero
        else:
            print(" · ", end="")  # Possible non-zero interaction
    print()

print()
print("Reduced QUBO Matrix (32×32) - showing first 8×8 block:")
print("(All entries are meaningful)")
print()

print("     ", end="")
for j in range(8):
    print(f"{j:2d} ", end="")
print()

for i in range(8):
    print(f"{i:2d} | ", end="")
    for j in range(8):
        if i == j:
            # Check if this diagonal should be -4 or 0
            # Variables 0-3 are cell (0,1), vars 4-7 are cell (0,3)
            # Within each cell, only one variable should be active
            var_info = free_vars[i]
            cell = (var_info[1], var_info[2])
            
            print("-4 ", end="")  # Diagonal for free variable
        else:
            print(" · ", end="")  # Possible interaction
    print()

print()

# ============================================================================
# STEP 6: Show Actual Code Usage
# ============================================================================

print("="*80)
print("STEP 6: How to Use the Reduction in Code")
print("="*80)
print()

print("""
from qubo_generation import build_reduced_qubo_direct

# Define givens
givens = {
    (0, 0): 2,           (0, 2): 4,
              (1, 1): 3,           (1, 3): 1,
    (2, 0): 1,           (2, 2): 3,
              (3, 1): 4,           (3, 3): 2
}

# Build reduced QUBO directly (efficient method)
Q_reduced, var_to_idx, idx_to_var, offset, info = build_reduced_qubo_direct(
    N=4, 
    box_size=2, 
    givens=givens
)

print(f"Built {Q_reduced.shape[0]}×{Q_reduced.shape[1]} matrix")
print(f"Reduced from 64 to {info['n_free_vars']} variables")

# The mapping tells you: variable (i,j,k) → index in reduced matrix
# Example: x(0,1,0) is at index var_to_idx[(0,1,0)]

# To evaluate a solution:
bitstring = "0100100000010010..."  # Your 64-bit solution
energy = evaluate_reduced_qubo(Q_reduced, bitstring, var_to_idx, idx_to_var, offset)
""")

print()

# ============================================================================
# STEP 7: Key Insights
# ============================================================================

print("="*80)
print("KEY INSIGHTS")
print("="*80)
print()

print("1. VARIABLE ELIMINATION:")
print("   - Given cells are FIXED → their variables are known")
print("   - No need to optimize over fixed variables")
print("   - Reduce: 64 variables → 32 variables")
print()

print("2. INDEX REMAPPING:")
print("   - Old indices: 0-63 (all variables)")
print("   - New indices: 0-31 (only free variables)")
print("   - Example: x(0,1,0) at old index 4 → new index 0")
print()

print("3. MATRIX SIZE:")
print("   - Full: 64×64 = 4,096 entries")
print("   - Reduced: 32×32 = 1,024 entries")
print("   - Savings: 3,072 entries (75%)")
print()

print("4. CONSTRAINT ADJUSTMENT:")
print("   - Row constraint for row 0:")
print("     * Full: \"sum of 4 variables for each digit = 1\"")
print("     * Reduced: \"sum of 2 free variables + 2 given = 1\"")
print("   - The given contribution is absorbed into the constant offset")
print()

print("5. ENERGY EVALUATION:")
print("   - Full bitstring: 64 bits (includes fixed givens)")
print("   - Extract only 32 bits corresponding to free variables")
print("   - Evaluate reduced QUBO on those 32 bits")
print("   - Get same energy as full QUBO!")
print()

print("="*80)