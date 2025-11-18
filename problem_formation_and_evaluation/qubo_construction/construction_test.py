"""
Test Suite for Sudoku QUBO Construction and Reduction
"""

import numpy as np
from qubo_generation import build_E1, build_E2, build_E3, build_E4, build_sudoku_qubo, evaluate_qubo
from matrix_reduction import build_reduced_qubo, build_reduced_qubo_direct, evaluate_reduced_qubo


def test_qubo_components(N, box_size, givens, bitstring, expected, test_name):
    """Test QUBO construction by evaluating energy components"""
    print("="*80)
    print(f"TEST: {test_name}")
    print("="*80)
    
    Q_E1, _, const_E1 = build_E1(N, givens)
    Q_E2, _, const_E2 = build_E2(N, givens)
    Q_E3, _, const_E3 = build_E3(N, givens)
    Q_E4, _, const_E4 = build_E4(N, box_size, givens)
    
    def eval_energy(Q, const):
        x = np.array([int(b) for b in bitstring.replace(' ', '')])
        energy = const
        for i in range(len(x)):
            energy += Q[i, i] * x[i]
        for i in range(len(x)):
            for j in range(i + 1, len(x)):
                energy += 2 * Q[i, j] * x[i] * x[j]
        return energy
    
    E1 = eval_energy(Q_E1, const_E1)
    E2 = eval_energy(Q_E2, const_E2)
    E3 = eval_energy(Q_E3, const_E3)
    E4 = eval_energy(Q_E4, const_E4)
    
    print(f"E1: {E1:.1f} (expected {expected[0]:.1f}) {'✓' if abs(E1-expected[0])<0.01 else '✗'}")
    print(f"E2: {E2:.1f} (expected {expected[1]:.1f}) {'✓' if abs(E2-expected[1])<0.01 else '✗'}")
    print(f"E3: {E3:.1f} (expected {expected[2]:.1f}) {'✓' if abs(E3-expected[2])<0.01 else '✗'}")
    print(f"E4: {E4:.1f} (expected {expected[3]:.1f}) {'✓' if abs(E4-expected[3])<0.01 else '✗'}")
    
    total = E1 + E2 + E3 + E4
    expected_total = sum(expected)
    
    print(f"\nTotal: {total:.1f} (expected {expected_total:.1f})")
    print(f"Status: {'✓ PASS' if abs(total-expected_total)<0.01 else '✗ FAIL'}\n")
    
    return {'E1': E1, 'E2': E2, 'E3': E3, 'E4': E4, 'total': total}


def run_all_tests():
    """Run all test cases"""
    
    print("\n" + "="*80)
    print("SUDOKU QUBO TEST SUITE")
    print("="*80 + "\n")
    
    # Test 1: Blank 4×4 Sudoku
    N1 = 4
    box_size1 = 2
    givens1 = None
    
    bitstring1 = (
        "0100" + "1000" + "0001" + "0010" +
        "0001" + "0010" + "0100" + "1000" +
        "1000" + "0100" + "0010" + "0001" +
        "0010" + "0001" + "1000" + "0100"
    )
    expected1 = (0, 0, 0, 0)
    
    results1 = test_qubo_components(N1, box_size1, givens1, bitstring1, expected1,
                                     "Test 1: Blank 4×4 (Valid Solution)")
    
    # Test 2: Partially filled 4×4 Sudoku
    N2 = 4
    box_size2 = 2
    givens2 = {
        (0, 0): 2, (0, 2): 4,
        (1, 1): 3, (1, 3): 1,
        (2, 0): 1, (2, 2): 3,
        (3, 1): 4, (3, 3): 2
    }
    
    bitstring2 = (
        "0100" + "1000" + "0001" + "0010" +
        "0001" + "0010" + "0100" + "1000" +
        "1000" + "0100" + "0010" + "0001" +
        "0010" + "0001" + "1000" + "0100"
    )
    expected2 = (0, 0, 0, 0)
    
    results2 = test_qubo_components(N2, box_size2, givens2, bitstring2, expected2,
                                     "Test 2: Partial 4×4 (Valid Solution)")
    
    # Test 3: 9×9 Construction
    print("="*80)
    print("TEST: Test 3: 9×9 Construction")
    print("="*80)
    
    N3 = 9
    box_size3 = 3
    givens3 = None
    
    Q3, _, _, offset3 = build_sudoku_qubo(N3, box_size3, givens3)
    
    print(f"Matrix size: {Q3.shape[0]}×{Q3.shape[1]} = {Q3.shape[0]*Q3.shape[1]:,} entries")
    print(f"Variables: {Q3.shape[0]}")
    print(f"Non-zero entries: {np.count_nonzero(Q3):,}")
    print("✓ Construction successful\n")
    
    # Test 4: Reduced QUBO
    print("="*80)
    print("TEST: Test 4: Reduced QUBO")
    print("="*80)
    
    print("\nMethod 1: Extraction")
    Q_ext, var_idx_ext, idx_var_ext, off_ext, info_ext = build_reduced_qubo(N2, box_size2, givens2)
    print(f"  Matrix: {Q_ext.shape[0]}×{Q_ext.shape[1]}")
    
    print("\nMethod 2: Direct")
    Q_dir, var_idx_dir, idx_var_dir, off_dir, info_dir = build_reduced_qubo_direct(N2, box_size2, givens2)
    print(f"  Matrix: {Q_dir.shape[0]}×{Q_dir.shape[1]}")
    
    matrices_match = np.allclose(Q_ext, Q_dir)
    offsets_match = abs(off_ext - off_dir) < 1e-10
    
    print(f"\nMatrices identical: {matrices_match} {'✓' if matrices_match else '✗'}")
    print(f"Offsets identical: {offsets_match} {'✓' if offsets_match else '✗'}")
    
    Q_reduced = Q_dir
    var_idx_red = var_idx_dir
    idx_var_red = idx_var_dir
    off_red = off_dir
    
    print(f"\nReduction statistics:")
    print(f"  Variables: {info_dir['n_total_vars']} → {info_dir['n_free_vars']} ({info_dir['reduction_pct']:.1f}%)")
    print(f"  Matrix: {info_dir['n_total_vars']**2:,} → {info_dir['n_free_vars']**2:,} entries ({info_dir['matrix_size_reduction_pct']:.1f}%)")
    
    # Energy comparison
    Q_full, _, _, off_full = build_sudoku_qubo(N2, box_size2, givens2)
    
    energy_full = evaluate_qubo(Q_full, bitstring2, off_full)
    energy_red = evaluate_reduced_qubo(Q_reduced, bitstring2, var_idx_red, idx_var_red, off_red)
    
    print(f"\nEnergy comparison (correct solution):")
    print(f"  Full QUBO:    {energy_full:.6f}")
    print(f"  Reduced QUBO: {energy_red:.6f}")
    print(f"  Match: {'✓' if abs(energy_full-energy_red)<1e-10 else '✗'}")
    
    bitstring_wrong = (
        "0100" + "0010" + "0001" + "1000" +
        "0001" + "0010" + "0100" + "1000" +
        "1000" + "0100" + "0010" + "0001" +
        "0010" + "0001" + "1000" + "0100"
    )
    
    energy_full_wrong = evaluate_qubo(Q_full, bitstring_wrong, off_full)
    energy_red_wrong = evaluate_reduced_qubo(Q_reduced, bitstring_wrong, var_idx_red, idx_var_red, off_red)
    
    print(f"\nEnergy comparison (incorrect solution):")
    print(f"  Full QUBO:    {energy_full_wrong:.6f}")
    print(f"  Reduced QUBO: {energy_red_wrong:.6f}")
    print(f"  Match: {'✓' if abs(energy_full_wrong-energy_red_wrong)<1e-10 else '✗'}")
    
    results4 = {
        'match': matrices_match and offsets_match and 
                abs(energy_full-energy_red)<1e-10 and 
                abs(energy_full_wrong-energy_red_wrong)<1e-10
    }
    
    print()
    
    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    
    all_passed = (
        abs(results1['total']) < 0.01 and
        abs(results2['total']) < 0.01 and
        results4['match']
    )
    
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print(f"\n  Test 1 (Blank 4×4): {results1['total']:.1f}")
        print(f"  Test 2 (Partial 4×4): {results2['total']:.1f}")
        print(f"  Test 3 (9×9): Construction successful")
        print(f"  Test 4 (Reduced QUBO): All checks passed")
    else:
        print("✗ SOME TESTS FAILED")
    
    print()


if __name__ == "__main__":
    run_all_tests()