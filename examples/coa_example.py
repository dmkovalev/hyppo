"""
Example demonstrating the use of hyppo's Causal Ordering Analysis (COA) module.
This example models a complex chemical reaction system with multiple interconnected
reactions, concentrations, and rate equations.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from hyppo.coa._base import Equation, Structure
from sympy import Symbol

def create_test_system(n_equations):
    """Create a test system with n equations and approximately n variables."""
    equations = []
    # Create chain reactions A -> B -> C -> D -> ...
    for i in range(n_equations):
        if i == 0:
            # First reaction depends on initial concentration
            tex = f"k_{i}*A_{i} - r_{i} = 0"
        else:
            # Each reaction depends on previous product
            tex = f"k_{i}*A_{i-1} - r_{i} = 0"
        equations.append(Equation(formula=tex))
    return equations

def measure_performance(max_size=50, step=5):
    """Measure performance for different system sizes."""
    sizes = range(step, max_size + step, step)
    times = []
    
    for size in sizes:
        equations = create_test_system(size)
        structure = Structure(equations=equations)
        
        start_time = time.time()
        # Perform key operations
        structure.build_full_causal_mapping()
        structure.build_matrix()
        structure.build_transitive_closure()
        elapsed = time.time() - start_time
        
        times.append(elapsed)
    
    return sizes, times

def plot_complexity(sizes, times):
    """Plot execution time vs problem size."""
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, times, 'bo-', label='Measured time')
    
    # Fit O(n^2) curve
    coeff = np.polyfit(sizes, times, 2)
    fit_times = np.polyval(coeff, sizes)
    plt.plot(sizes, fit_times, 'r--', label='O(n²) fit')
    
    plt.xlabel('Number of equations/variables (n)')
    plt.ylabel('Execution time (seconds)')
    plt.title('COA Algorithm Complexity Analysis')
    plt.legend()
    plt.grid(True)
    plt.savefig('complexity_analysis.png')
    plt.close()

def main():
    # Example 1: Complex Chemical Reaction System
    print("Chemical Reaction System Analysis")
    print("-" * 50)
    
    # Define equations for a complex chemical reaction network
    equations = [
        Equation(formula=r"k_1*[A] - r_1 = 0"),                    # Rate of first reaction
        Equation(formula=r"k_2*[B] - r_2 = 0"),                    # Rate of second reaction
        Equation(formula=r"r_1 - \frac{d[B]}{dt} = 0"),           # B formation
        Equation(formula=r"r_2 - \frac{d[C]}{dt} = 0"),           # C formation
        Equation(formula=r"[A]_0 - [A] - [B] - [C] = 0"),         # Mass conservation
        Equation(formula=r"k_1 - A_0*e^{-E_1/RT} = 0"),           # Arrhenius equation 1
        Equation(formula=r"k_2 - A_0*e^{-E_2/RT} = 0"),          # Arrhenius equation 2
        Equation(formula=r"T - T_0 - \alpha t = 0"),              # Temperature profile
        Equation(formula=r"[A]_0 - 100 = 0"),                     # Initial concentration
        Equation(formula=r"T_0 - 298 = 0")                        # Initial temperature
    ]
    
    structure = Structure(equations=equations)
    
    print("\nSystem Analysis:")
    print("Exogenous variables:", structure.exogenous())
    print("Endogenous variables:", structure.endogenous())
    
    print("\nFull Causal Mapping:")
    fcm = structure.build_full_causal_mapping()
    for eq, var in fcm.items():
        print(f"Equation '{eq}' determines variable: {var}")
    
    print("\nDirected Causal Graph (in DOT format):")
    dcg = structure.build_dcg()
    print(dcg.source)
    
    print("\nTransitive Closure of Dependencies:")
    tc = structure.build_transitive_closure()
    for var, deps in tc.items():
        print(f"Variable {var} depends on: {deps}")
    
    # Example 2: Complexity Analysis
    print("\nPerforming Complexity Analysis...")
    sizes, times = measure_performance()
    plot_complexity(sizes, times)
    print("Complexity analysis plot saved as 'complexity_analysis.png'")

if __name__ == "__main__":
    main()