# Sudoku QUBO Construction and Reduction

This package provides tools for building QUBO (Quadratic Unconstrained Binary Optimization) formulations of Sudoku puzzles, with support for variable elimination when given cells are present.

## File Structure

### 1. `qubo_construction.py` - Full QUBO Matrix Builder
**Purpose:** Builds the FULL QUBO matrix including all variables.

**Key Functions:**
- `build_sudoku_qubo(N, box_size, givens=None, ...)` - Build complete QUBO matrix
- `build_E1()`, `build_E2()`, `build_E3()`, `build_E4()` - Individual constraint components
- `evaluate_qubo()` - Evaluate energy of a solution
- `print_E1_details()`, etc. - Educational printing functions

**Matrix Size:**
- 4×4 Sudoku: 64×64 matrix (4³ variables)
- 9×9 Sudoku: 729×729 matrix (9³ variables)

### 2. `qubo_reduction.py` - Variable Elimination & Reduction
**Purpose:** Reduces QUBO by eliminating variables from given cells.

**Key Functions:**
- `build_reduced_qubo()` - Extract reduced matrix from full matrix (pedagogical)
- `build_reduced_qubo_direct()` - Build reduced matrix directly (efficient)
- `evaluate_reduced_qubo()` - Evaluate energy using reduced matrix
- `reconstruct_full_solution()` - Convert reduced solution back to full
- `print_reduction_stats()` - Display reduction statistics

**Matrix Size Example (4×4 with 8 givens):**
- Full: 64×64 = 4,096 entries
- Reduced: 32×32 = 1,024 entries
- Savings: 75% reduction

### 3. `construction_test.py` - Comprehensive Test Suite
**Purpose:** Validates both construction and reduction methods.

**Tests:**
1. Blank 4×4 Sudoku (all free variables)
2. Partially filled 4×4 (8 givens, 8 free cells)
3. 9×9 Sudoku construction only
4. Reduced QUBO validation (compares extraction vs direct methods)

## Usage Examples

### Building a Full QUBO

```python
from qubo_construction import build_sudoku_qubo

# Define givens (1-indexed digits)
givens = {
    (0, 0): 2,
    (0, 2): 4,
    (1, 1): 3,
    # ... more givens
}

# Build full QUBO (64×64 for 4×4 Sudoku)
Q, var_to_idx, idx_to_var, offset = build_sudoku_qubo(
    N=4, 
    box_size=2, 
    givens=givens
)

print(f"Matrix size: {Q.shape}")  # (64, 64)
```

### Building a Reduced QUBO

```python
from qubo_reduction import build_reduced_qubo_direct

# Build reduced QUBO directly (32×32 for 4×4 with 8 givens)
Q_reduced, var_to_idx, idx_to_var, offset, info = build_reduced_qubo_direct(
    N=4,
    box_size=2,
    givens=givens
)

print(f"Matrix size: {Q_reduced.shape}")  # (32, 32)
print(f"Variable reduction: {info['reduction_pct']:.1f}%")
```

### Evaluating Solutions

```python
from qubo_construction import evaluate_qubo
from qubo_reduction import evaluate_reduced_qubo

# Full solution (64 bits)
bitstring_full = "0100100000010010..."

# Evaluate with full QUBO
energy_full = evaluate_qubo(Q_full, bitstring_full, offset_full)

# Evaluate with reduced QUBO (extracts only free variable bits)
energy_reduced = evaluate_reduced_qubo(
    Q_reduced, 
    bitstring_full,  # Still pass full bitstring
    var_to_idx, 
    idx_to_var, 
    offset_reduced
)

# Both give the same result!
assert abs(energy_full - energy_reduced) < 1e-10
```

### Reconstructing Solutions

```python
from qubo_reduction import reconstruct_full_solution

# If you have a reduced solution (32 bits for free variables only)
reduced_bitstring = "01001000000100100010010000100100"

# Reconstruct to full 64-bit solution
full_bitstring = reconstruct_full_solution(
    reduced_bitstring,
    var_to_idx,
    idx_to_var,
    givens,
    N=4
)

print(len(full_bitstring))  # 64
```

## Key Concepts

### Variable Indexing
Variables are indexed as `x(i,j,k)`:
- `i` = row (0-indexed)
- `j` = column (0-indexed)
- `k` = digit - 1 (0-indexed, so k=0 means digit 1)
- `x(i,j,k) = 1` means "digit k+1 is placed in cell (i,j)"

### Index Mappings
The functions return two dictionaries:
- `var_to_idx`: Maps `(i,j,k)` → matrix index
- `idx_to_var`: Maps matrix index → `(i,j,k)`

These handle the remapping when variables are eliminated.

### Two Reduction Methods

**Method 1: Extraction (`build_reduced_qubo`)**
1. Build full 64×64 matrix
2. Extract 32×32 submatrix for free variables
3. Good for understanding the process

**Method 2: Direct (`build_reduced_qubo_direct`)**
1. Build 32×32 matrix directly
2. Never allocate memory for eliminated variables
3. More efficient for production use

Both produce **identical** results!

## Running Tests

```bash
python construction_test.py
```

This will:
- Validate all QUBO construction functions
- Compare full vs reduced formulations
- Verify both reduction methods produce identical results
- Show detailed statistics and visualizations

## Performance Notes

For 4×4 Sudoku with 8 givens:
- **Full QUBO:** 64 variables, 4,096 matrix entries
- **Reduced QUBO:** 32 variables, 1,024 matrix entries
- **Savings:** 50% fewer variables, 75% smaller matrix

For 9×9 Sudoku with typical ~30 givens:
- **Full QUBO:** 729 variables, 531,441 matrix entries
- **Reduced QUBO:** ~459 variables, ~210,681 matrix entries
- **Savings:** ~37% fewer variables, ~60% smaller matrix

## Dependencies

```python
import numpy as np
```

That's it! No other dependencies required.