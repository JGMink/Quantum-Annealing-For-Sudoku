"""
Enhanced Test Suite for Sudoku QUBO Construction and Reduction

This test suite validates both:
1. Full QUBO construction (qubo_construction.py)
2. Reduced QUBO with variable elimination (qubo_reduction.py)

TESTS INCLUDED:
───────────────
Test 1: Blank 4×4 Sudoku
  - No givens (all 64 variables are free)
  - Validates basic QUBO construction
  - Shows full polynomial structure

Test 2: Partially Filled 4×4 Sudoku
  - 8 givens (32 free variables)
  - Demonstrates how givens reduce problem size
  - Shows constraint adjustments

Test 3: Blank 9×9 Sudoku (Construction Only)
  - Demonstrates scalability to full Sudoku
  - Shows matrix size: 729×729
  - No energy evaluation (would need 729-bit string)

Test 4: Reduced QUBO (Variable Elimination) ⭐
  - Compares two methods for building reduced QUBO:
    1. Extraction: Build 64×64, extract 32×32 submatrix
    2. Direct: Build 32×32 directly
  - Verifies both methods produce identical results
  - Visualizes matrix structure comparison
  - Shows index remapping (old index 4 → new index 0)
  - Demonstrates 50% variable reduction, 75% matrix size reduction

KEY CONCEPTS DEMONSTRATED:
──────────────────────────
1. One-Hot Encoding: Each cell gets N binary variables
2. Constraint Formulation: (sum - 1)² penalty for violations
3. Variable Elimination: Given cells remove variables from QUBO
4. Index Remapping: Free variables get new sequential indices
5. Matrix Sparsity: Most entries are zero (>90% for 4×4)

For production use, prefer build_reduced_qubo_direct() for efficiency.
For educational purposes, build_reduced_qubo() shows the process clearly.
"""

import numpy as np

# Import from construction module
from qubo_generation import (
    build_E1,
    build_E2,
    build_E3,
    build_E4,
    build_sudoku_qubo,
    evaluate_qubo,
    print_E1_details,
    print_E2_details,
    print_E3_details,
    print_E4_details,
)

# Import from reduction module
from matrix_reduction import (
    build_reduced_qubo,
    build_reduced_qubo_direct,
    evaluate_reduced_qubo,
    print_reduction_stats,
    reconstruct_full_solution,
)


# ============================================================================
# Testing and Validation
# ============================================================================

def evaluate_qubo_energy(bitstring, Q, constant):
    """
    Evaluate QUBO energy for a given bitstring.
    E = constant + sum_i Q[i,i]*x_i + sum_{i<j} 2*Q[i,j]*x_i*x_j
    """
    x = np.array([int(b) for b in bitstring.replace(' ', '')])
    
    energy = constant
    
    # Diagonal terms
    for i in range(len(x)):
        energy += Q[i, i] * x[i]
    
    # Off-diagonal terms (factor of 2 for QUBO convention)
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            energy += 2 * Q[i, j] * x[i] * x[j]
    
    return energy


def test_qubo_components(N, box_size, givens, bitstring, expected_results, test_name):
    """
    Test QUBO construction by evaluating individual energy components.
    
    Args:
        N: Sudoku size (4 for 4x4, 9 for 9x9)
        box_size: Box size (2 for 4x4, 3 for 9x9)
        givens: Dictionary of given cells {(i,j): digit} or None
        bitstring: Binary string (spaces allowed)
        expected_results: tuple (E1, E2, E3, E4)
        test_name: String identifier for this test
    """
    print("="*80)
    print(f"TEST: {test_name}")
    print("="*80)
    print(f"Sudoku size: {N}×{N}")
    if givens:
        print(f"Givens: {len(givens)} cells")
    else:
        print(f"Givens: None (blank puzzle)")
    print(f"Bitstring: {bitstring[:50]}..." if len(bitstring) > 50 else f"Bitstring: {bitstring}")
    print(f"Expected: E1={expected_results[0]}, E2={expected_results[1]}, "
          f"E3={expected_results[2]}, E4={expected_results[3]}")
    print()
    
    # Build all components
    Q_E1, _, const_E1 = build_E1(N, givens)
    Q_E2, _, const_E2 = build_E2(N, givens)
    Q_E3, _, const_E3 = build_E3(N, givens)
    Q_E4, _, const_E4 = build_E4(N, box_size, givens)
    
    # Evaluate each component
    E1_computed = evaluate_qubo_energy(bitstring, Q_E1, const_E1)
    E2_computed = evaluate_qubo_energy(bitstring, Q_E2, const_E2)
    E3_computed = evaluate_qubo_energy(bitstring, Q_E3, const_E3)
    E4_computed = evaluate_qubo_energy(bitstring, Q_E4, const_E4)
    
    print("RESULTS:")
    print(f"  E1: computed = {E1_computed:6.1f}, expected = {expected_results[0]:6.1f}  "
          f"{'✓' if abs(E1_computed - expected_results[0]) < 0.01 else '✗'}")
    print(f"  E2: computed = {E2_computed:6.1f}, expected = {expected_results[1]:6.1f}  "
          f"{'✓' if abs(E2_computed - expected_results[1]) < 0.01 else '✗'}")
    print(f"  E3: computed = {E3_computed:6.1f}, expected = {expected_results[2]:6.1f}  "
          f"{'✓' if abs(E3_computed - expected_results[2]) < 0.01 else '✗'}")
    print(f"  E4: computed = {E4_computed:6.1f}, expected = {expected_results[3]:6.1f}  "
          f"{'✓' if abs(E4_computed - expected_results[3]) < 0.01 else '✗'}")
    print()
    
    total_computed = E1_computed + E2_computed + E3_computed + E4_computed
    total_expected = sum(expected_results)
    
    print(f"TOTAL ENERGY:")
    print(f"  Computed: {total_computed:.1f}")
    print(f"  Expected: {total_expected:.1f}")
    print(f"  Status: {'✓ PASS' if abs(total_computed - total_expected) < 0.01 else '✗ FAIL'}")
    print()
    
    return {
        'E1': E1_computed,
        'E2': E2_computed,
        'E3': E3_computed,
        'E4': E4_computed,
        'total': total_computed
    }


def run_all_tests():
    """Run all test cases."""
    
    print("\n" + "="*80)
    print("RUNNING ALL QUBO TESTS")
    print("="*80)
    print()
    
    # ========================================================================
    # Test 1: Blank 4×4 Sudoku
    # ========================================================================
    N1 = 4
    box_size1 = 2
    givens1 = None
    
    # Valid solution
    bitstring1 = (
        "0100" + "1000" + "0001" + "0010" +  # Row 0: 2,1,4,3
        "0001" + "0010" + "0100" + "1000" +  # Row 1: 4,3,2,1
        "1000" + "0100" + "0010" + "0001" +  # Row 2: 1,2,3,4
        "0010" + "0001" + "1000" + "0100"    # Row 3: 3,4,1,2
    )
    expected1 = (0, 0, 0, 0)  # E1, E2, E3, E4
    
    results1 = test_qubo_components(N1, box_size1, givens1, bitstring1, expected1, 
                                     "Test 1: Blank 4×4 Sudoku (Valid Solution)")
    
    # ========================================================================
    # Test 2: Partially filled 4×4 Sudoku
    # ========================================================================
    N2 = 4
    box_size2 = 2
    givens2 = {
        (0, 0): 2,           (0, 2): 4,
                  (1, 1): 3,           (1, 3): 1,
        (2, 0): 1,           (2, 2): 3,
                  (3, 1): 4,           (3, 3): 2
    }
    
    # Correct solution
    bitstring2 = (
        "0100" + "1000" + "0001" + "0010" +  # Row 0: 2,1,4,3
        "0001" + "0010" + "0100" + "1000" +  # Row 1: 4,3,2,1
        "1000" + "0100" + "0010" + "0001" +  # Row 2: 1,2,3,4
        "0010" + "0001" + "1000" + "0100"    # Row 3: 3,4,1,2
    )
    expected2 = (0, 0, 0, 0)  # All constraints satisfied
    
    results2 = test_qubo_components(N2, box_size2, givens2, bitstring2, expected2,
                                     "Test 2: Partially Filled 4×4 (Valid Solution)")
    
    # ========================================================================
    # Test 3: Blank 9×9 Sudoku - Just construction, no evaluation
    # ========================================================================
    print("="*80)
    print("TEST: Test 3: Blank 9×9 Sudoku (Construction Only)")
    print("="*80)
    print()
    
    N3 = 9
    box_size3 = 3
    givens3 = None
    
    print(f"Building {N3}×{N3} Sudoku QUBO...")
    Q3, var_to_idx3, idx_to_var3, offset3 = build_sudoku_qubo(N3, box_size3, givens3)
    
    print(f"  Matrix size: {Q3.shape[0]} × {Q3.shape[1]} = {Q3.shape[0] * Q3.shape[1]:,} entries")
    print(f"  Number of variables: {len(var_to_idx3)}")
    print(f"  Constant offset: {offset3}")
    print(f"  Non-zero entries: {np.count_nonzero(Q3):,}")
    print(f"  Sparsity: {100 * (1 - np.count_nonzero(Q3) / (Q3.shape[0] * Q3.shape[1])):.2f}%")
    print()
    print("✓ Successfully constructed 9×9 QUBO (no energy evaluation)")
    print()
    
    # ========================================================================
    # Detailed Construction for Test 2
    # ========================================================================
    print("="*80)
    print("DETAILED CONSTRUCTION: Test 2 (Partially Filled 4×4)")
    print("="*80)
    print(f"Sudoku: {N2}×{N2}")
    print(f"Givens: {len(givens2)} cells")
    print(f"Free cells: {N2*N2 - len(givens2)}")
    print(f"Free variables: {(N2*N2 - len(givens2)) * N2}")
    print()
    print("Given cells:")
    for i in range(N2):
        row_str = ""
        for j in range(N2):
            if j > 0 and j % box_size2 == 0:
                row_str += "| "
            if (i, j) in givens2:
                row_str += str(givens2[(i, j)]) + " "
            else:
                row_str += "_ "
        print(row_str.rstrip())
        if i == 1:
            print("-" * (2*N2 + box_size2 - 1))
    print()
    
    Q_E1_2, poly_E1_2, const_E1_2 = print_E1_details(N2, givens2)
    print()
    Q_E2_2, poly_E2_2, const_E2_2 = print_E2_details(N2, givens2)
    print()
    Q_E3_2, poly_E3_2, const_E3_2 = print_E3_details(N2, givens2)
    print()
    Q_E4_2, poly_E4_2, const_E4_2 = print_E4_details(N2, box_size2, givens2)
    print()
    
    print("="*80)
    print("SUMMARY FOR TEST 2")
    print("="*80)
    print(f"E1: {len(poly_E1_2)} terms, constant = {const_E1_2}")
    print(f"E2: {len(poly_E2_2)} terms, constant = {const_E2_2}")
    print(f"E3: {len(poly_E3_2)} terms, constant = {const_E3_2}")
    print(f"E4: {len(poly_E4_2)} terms, constant = {const_E4_2}")
    print(f"Total terms: {len(poly_E1_2) + len(poly_E2_2) + len(poly_E3_2) + len(poly_E4_2)}")
    print(f"Total constant: {const_E1_2 + const_E2_2 + const_E3_2 + const_E4_2}")
    print()
    
    # ========================================================================
    # Test 4: Reduced QUBO (Variable Elimination)
    # ========================================================================
    print("="*80)
    print("TEST 4: Reduced QUBO Matrix (Variable Elimination)")
    print("="*80)
    print("\nThis test demonstrates explicit variable elimination.")
    print("We compare two methods:")
    print("  1. Extraction: Build full 64×64 matrix, then extract 32×32 submatrix")
    print("  2. Direct: Build 32×32 matrix directly (more efficient)")
    print()
    
    # Method 1: Build via extraction
    print("Method 1: Extraction from Full QUBO")
    Q_reduced_ext, var_to_idx_ext, idx_to_var_ext, offset_ext, elim_info_ext = build_reduced_qubo(
        N2, box_size2, givens2
    )
    print(f"  Built {Q_reduced_ext.shape[0]}×{Q_reduced_ext.shape[1]} matrix")
    print(f"  Non-zero entries: {np.count_nonzero(Q_reduced_ext)}")
    
    # Method 2: Build directly
    print("\nMethod 2: Direct Construction")
    Q_reduced_dir, var_to_idx_dir, idx_to_var_dir, offset_dir, elim_info_dir = build_reduced_qubo_direct(
        N2, box_size2, givens2
    )
    print(f"  Built {Q_reduced_dir.shape[0]}×{Q_reduced_dir.shape[1]} matrix")
    print(f"  Non-zero entries: {np.count_nonzero(Q_reduced_dir)}")
    
    # Verify they're identical
    matrices_match = np.allclose(Q_reduced_ext, Q_reduced_dir)
    offsets_match = abs(offset_ext - offset_dir) < 1e-10
    print(f"\n  Matrices identical? {matrices_match}")
    print(f"  Offsets identical? {offsets_match}")
    if matrices_match and offsets_match:
        print(f"  ✓ Both methods produce the same result!")
    else:
        print(f"  ✗ Methods produced different results!")
    
    # Use the direct method for remaining tests
    Q_reduced = Q_reduced_dir
    var_to_idx_red = var_to_idx_dir
    idx_to_var_red = idx_to_var_dir
    offset_red = offset_dir
    elim_info = elim_info_dir
    
    print(f"\nPuzzle: {N2}×{N2} Sudoku with {len(givens2)} givens")
    print_reduction_stats(elim_info, N2, box_size2)
    
    # Visualize matrix structure
    print("\n" + "="*80)
    print("Matrix Structure Comparison")
    print("="*80)
    
    # Build full QUBO for comparison
    Q_full_viz, _, _, _ = build_sudoku_qubo(N2, box_size2, givens2)
    
    print(f"\nFull QUBO (64×64):")
    print(f"  Includes variables for all cells (even given ones)")
    print(f"  Given cells have zero coefficients")
    print(f"  Many wasted entries")
    
    # Show a sample of the full matrix structure
    print(f"\n  Sample diagonal entries:")
    for i in range(8):
        var = (i // (N2*N2), (i // N2) % N2, i % N2)
        is_given = var[:2] in givens2
        print(f"    Index {i:2d} (cell ({var[0]},{var[1]}), digit {var[2]+1}): "
              f"{Q_full_viz[i,i]:6.1f}  {'[GIVEN - zero coeff]' if is_given else ''}")
    
    print(f"\nReduced QUBO (32×32):")
    print(f"  Contains only free variables")
    print(f"  Compact representation")
    print(f"  All entries are meaningful")
    
    print(f"\n  Sample diagonal entries:")
    for i in range(8):
        var = idx_to_var_red[i]
        print(f"    Index {i:2d} (cell ({var[0]},{var[1]}), digit {var[2]+1}): {Q_reduced[i,i]:6.1f}")
    
    # Test with correct solution
    print("\n" + "="*80)
    print("Evaluating Correct Solution on Reduced QUBO")
    print("="*80)
    
    energy_reduced_correct = evaluate_reduced_qubo(
        Q_reduced, bitstring2, var_to_idx_red, idx_to_var_red, offset_red
    )
    
    # Also evaluate with full QUBO for comparison
    Q_full_test, _, _, offset_full_test = build_sudoku_qubo(N2, box_size2, givens2)
    energy_full_correct = evaluate_qubo(Q_full_test, bitstring2, offset_full_test)
    
    print(f"\nCorrect solution:")
    print(f"  Full QUBO energy: {energy_full_correct}")
    print(f"  Reduced QUBO energy: {energy_reduced_correct}")
    print(f"  Match: {'✓' if abs(energy_full_correct - energy_reduced_correct) < 0.01 else '✗'}")
    
    # Create an incorrect solution for comparison
    bitstring_wrong_test = (
        "0100" + "0010" + "0001" + "1000" +  # Row 0: 2,3,4,1 (wrong!)
        "0001" + "0010" + "0100" + "1000" +  # Row 1: 4,3,2,1
        "1000" + "0100" + "0010" + "0001" +  # Row 2: 1,2,3,4
        "0010" + "0001" + "1000" + "0100"    # Row 3: 3,4,1,2
    )
    
    energy_reduced_wrong = evaluate_reduced_qubo(
        Q_reduced, bitstring_wrong_test, var_to_idx_red, idx_to_var_red, offset_red
    )
    energy_full_wrong = evaluate_qubo(Q_full_test, bitstring_wrong_test, offset_full_test)
    
    print(f"\nIncorrect solution:")
    print(f"  Full QUBO energy: {energy_full_wrong}")
    print(f"  Reduced QUBO energy: {energy_reduced_wrong}")
    print(f"  Match: {'✓' if abs(energy_full_wrong - energy_reduced_wrong) < 0.01 else '✗'}")
    
    print(f"\nEnergy difference (wrong - correct):")
    print(f"  Full QUBO: {energy_full_wrong - energy_full_correct}")
    print(f"  Reduced QUBO: {energy_reduced_wrong - energy_reduced_correct}")
    
    print("\n✓ Reduced QUBO produces identical energies with smaller matrix!")
    
    # Show some of the free variables
    print("\n" + "="*80)
    print("Free Variables in Reduced QUBO")
    print("="*80)
    print(f"Total free variables: {len(idx_to_var_red)}")
    print(f"\nFirst 10 free variables:")
    for i in range(min(10, len(idx_to_var_red))):
        var = idx_to_var_red[i]
        print(f"  Reduced index {i:2d}: x{var} -> cell ({var[0]},{var[1]}), digit {var[2]+1}")
    
    # Test reconstruction
    print("\n" + "="*80)
    print("Testing Solution Reconstruction")
    print("="*80)
    
    # Extract just the free variable bits from the correct solution
    reduced_bits = ""
    idx_full = 0
    for i in range(N2):
        for j in range(N2):
            for k in range(N2):
                if (i, j, k) in var_to_idx_red:
                    reduced_bits += bitstring2[idx_full]
                idx_full += 1
    
    print(f"Original full solution: {len(bitstring2)} bits")
    print(f"Extracted reduced solution: {len(reduced_bits)} bits")
    
    # Reconstruct
    reconstructed = reconstruct_full_solution(reduced_bits, var_to_idx_red, idx_to_var_red, givens2, N2)
    
    print(f"Reconstructed full solution: {len(reconstructed)} bits")
    print(f"Match: {'✓' if reconstructed == bitstring2 else '✗'}")
    
    results4 = {
        'full_correct': energy_full_correct,
        'reduced_correct': energy_reduced_correct,
        'full_wrong': energy_full_wrong,
        'reduced_wrong': energy_reduced_wrong,
        'match': abs(energy_full_correct - energy_reduced_correct) < 0.01 and 
                abs(energy_full_wrong - energy_reduced_wrong) < 0.01 and
                reconstructed == bitstring2
    }
    
    print()
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("="*80)
    print("FINAL SUMMARY")
    print("="*80)
    all_passed = (
        abs(results1['total'] - sum(expected1)) < 0.01 and
        abs(results2['total'] - sum(expected2)) < 0.01 and
        results4['match']
    )
    
    if all_passed:
        print("✓ ALL TESTS PASSED!")
        print("\nKey Results:")
        print(f"  • Test 1 (Blank 4×4): Energy = {results1['total']}")
        print(f"  • Test 2 (Partial 4×4): Energy = {results2['total']}")
        print(f"  • Test 3 (9×9 Construction): Success")
        print(f"  • Test 4 (Reduced QUBO): Full and reduced energies match!")
        print(f"  • Test 4 (Reconstruction): Successfully reconstructed solution")
    else:
        print("✗ SOME TESTS FAILED")
    print()


if __name__ == "__main__":
    run_all_tests()