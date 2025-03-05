"""
Example script demonstrating how to create random complete structures.
"""

from hyppo.generator import create_random_complete_structure, create_random_complete_structures
from hyppo.coa._base import Structure, Equation

def main():
    # Create a single random complete structure with default parameters
    print("Creating a single random complete structure with default parameters:")
    structure = create_random_complete_structure()
    print(f"Number of equations: {len(structure.equations)}")
    print(f"Number of variables: {len(structure.vars)}")
    print("Equations:")
    for eq in structure.equations:
        print(f"  {eq.formula}")
    print(f"Is complete: {structure.is_complete()}")
    print()

    # Create a single random complete structure with custom parameters
    print("Creating a single random complete structure with custom parameters:")
    structure = create_random_complete_structure(
        num_vars=7,
        min_vars_per_eq=2,
        max_vars_per_eq=4,
        exogenous_ratio=0.3
    )
    print(f"Number of equations: {len(structure.equations)}")
    print(f"Number of variables: {len(structure.vars)}")
    print("Equations:")
    for eq in structure.equations:
        print(f"  {eq.formula}")
    print(f"Is complete: {structure.is_complete()}")
    print()

    # Create multiple random complete structures
    print("Creating multiple random complete structures:")
    structures = create_random_complete_structures(
        num_structures=3,
        min_vars=4,
        max_vars=8,
        min_vars_per_eq=1,
        max_vars_per_eq=3,
        exogenous_ratio=0.25
    )
    for i, structure in enumerate(structures):
        print(f"Structure {i+1}:")
        print(f"  Number of equations: {len(structure.equations)}")
        print(f"  Number of variables: {len(structure.vars)}")
        print(f"  Is complete: {structure.is_complete()}")
    print()

    # Demonstrate additional functionality
    print("Demonstrating additional functionality:")
    structure = structures[0]
    
    # Check if the structure is complete
    print(f"Is complete: {structure.is_complete()}")
    
    # Get exogenous variables
    exogenous = structure.exogenous()
    print(f"Exogenous variables: {[str(v) for v in exogenous]}")
    
    # Get endogenous variables
    endogenous = structure.endogenous()
    print(f"Endogenous variables: {[str(v) for v in endogenous]}")
    
    # Build full causal mapping
    try:
        fcm = structure.build_full_causal_mapping()
        print("Full causal mapping:")
        for formula, var in fcm.items():
            print(f"  {formula} -> {var}")
    except Exception as e:
        print(f"Error building full causal mapping: {e}")
    
    # Build transitive closure
    try:
        tc = structure.build_transitive_closure()
        print("Transitive closure:")
        for var, deps in tc.items():
            print(f"  {var} -> {[str(d) for d in deps]}")
    except Exception as e:
        print(f"Error building transitive closure: {e}")

if __name__ == "__main__":
    main()