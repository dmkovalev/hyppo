"""
Example demonstrating the use of hyppo's Causal Ordering Analysis (COA) module.
This example models a simple physical system with mass, force, and acceleration
relationships using Newton's laws.
"""

from hyppo.coa._base import Equation, Structure
from sympy import Symbol

def main():
    # Define equations in LaTeX format representing F = ma system
    # f1: Force equation F = 10 (constant force)
    # f2: Newton's second law F = ma
    # f3: Acceleration definition a = dv/dt
    tex1 = r"F - 10 = 0"
    tex2 = r"F - m*a = 0"
    tex3 = r"a - \frac{dv}{dt} = 0"

    # Create Equation objects
    e1 = Equation(formula=tex1)
    e2 = Equation(formula=tex2)
    e3 = Equation(formula=tex3)

    # Create a Structure from these equations
    equations = [e1, e2, e3]
    structure = Structure(equations=equations)

    print("System Analysis:")
    print("-" * 50)
    
    # Display exogenous and endogenous variables
    print("\nExogenous variables:", structure.exogenous())
    print("Endogenous variables:", structure.endogenous())

    # Build and display the full causal mapping
    print("\nFull Causal Mapping:")
    fcm = structure.build_full_causal_mapping()
    for eq, var in fcm.items():
        print(f"Equation '{eq}' determines variable: {var}")

    # Generate and display the directed causal graph
    print("\nDirected Causal Graph (in DOT format):")
    dcg = structure.build_dcg()
    print(dcg.source)

    # Show transitive closure of dependencies
    print("\nTransitive Closure of Dependencies:")
    tc = structure.build_transitive_closure()
    for var, deps in tc.items():
        print(f"Variable {var} depends on: {deps}")

if __name__ == "__main__":
    main()