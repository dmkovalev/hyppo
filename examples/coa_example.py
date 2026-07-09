"""
Example demonstrating the use of hyppo's Causal Ordering Analysis (COA) module.
This example models a simple chain of relationships between variables, showing
how causal ordering can be determined from a system of equations.
"""

from hyppo.coa._base import Equation, Structure


def main():
    # Example 1: Simple Chain System
    print("Simple Chain System Analysis")
    print("-" * 50)

    # Define equations for a simple chain system
    equations = [
        Equation(formula=r"x_1 - 10 = 0"),          # Initial value
        Equation(formula=r"x_2 - 2*x_1 = 0"),        # x2 depends on x1
        Equation(formula=r"x_3 - 3*x_2 = 0"),        # x3 depends on x2
        Equation(formula=r"x_4 - x_3 - 5 = 0"),      # x4 depends on x3
        Equation(formula=r"y_1 - x_1*x_2 = 0"),      # y1 depends on x1 and x2
        Equation(formula=r"y_2 - x_3*x_4 = 0"),      # y2 depends on x3 and x4
        Equation(formula=r"z - y_1 - y_2 = 0"),      # z depends on y1 and y2
    ]

    structure = Structure(equations=equations)

    print("\nStructure validity checks:")
    print(f"  is_structure : {structure.is_structure()}")
    print(f"  is_complete  : {structure.is_complete()}")
    print(f"  is_minimal   : {structure.is_minimal()}")

    print("\nVariables:")
    print(f"  All        : {structure.vars}")
    print(f"  Exogenous  : {structure.exogenous()}")
    print(f"  Endogenous : {structure.endogenous()}")

    print("\nFull Causal Mapping (equation -> determined variable):")
    fcm = structure.build_full_causal_mapping()
    for eq, var in fcm.items():
        print(f"  '{eq}'  ->  {var}")

    print("\nTransitive Closure of Dependencies (variable -> all downstream variables):")
    tc = structure.build_transitive_closure()
    for var, deps in sorted(tc.items(), key=lambda kv: str(kv[0])):
        print(f"  {var} -> {deps if deps else '{}'}")

    print("\nMinimal Sub-Structures:")
    minimals = structure.find_minimal_structures()
    for i, ms in enumerate(minimals, 1):
        print(f"  [{i}] vars={ms.vars}  is_minimal={ms.is_minimal()}")

    # Example 2: Isolated / Disconnected System
    print("\n" + "=" * 50)
    print("Two Independent Sub-Systems")
    print("-" * 50)

    eq_a = [
        Equation(formula=r"a_1 - 5 = 0"),
        Equation(formula=r"a_2 - 2*a_1 = 0"),
    ]
    eq_b = [
        Equation(formula=r"b_1 - 7 = 0"),
        Equation(formula=r"b_2 - 3*b_1 = 0"),
    ]

    sa = Structure(equations=eq_a)
    sb = Structure(equations=eq_b)
    combined = sa.union(sb)

    print(f"\nSub-system A  vars={sa.vars}")
    print(f"Sub-system B  vars={sb.vars}")
    print(f"Union         vars={combined.vars}")
    print(f"  is_structure : {combined.is_structure()}")
    print(f"  is_complete  : {combined.is_complete()}")

    print("\nUnion Causal Mapping:")
    for eq, var in combined.build_full_causal_mapping().items():
        print(f"  '{eq}'  ->  {var}")


if __name__ == "__main__":
    main()
