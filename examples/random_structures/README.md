# Random Complete Structures Example

This example demonstrates how to create random complete structures using the `hyppo` package.

## What are Complete Structures?

In the context of the `hyppo` package, a structure is a set of equations and variables. A structure is considered "complete" if the number of equations equals the number of variables.

## Usage

The `hyppo.generator` module provides two functions for creating random complete structures:

1. `create_random_complete_structure`: Creates a single random complete structure
2. `create_random_complete_structures`: Creates multiple random complete structures

### Creating a Single Random Complete Structure

```python
from hyppo.generator import create_random_complete_structure

# Create a structure with default parameters
structure = create_random_complete_structure()

# Create a structure with custom parameters
structure = create_random_complete_structure(
    num_vars=7,              # Number of variables in the structure
    min_vars_per_eq=2,       # Minimum number of variables per equation
    max_vars_per_eq=4,       # Maximum number of variables per equation
    exogenous_ratio=0.3      # Ratio of exogenous variables (equations with only one variable)
)
```

### Creating Multiple Random Complete Structures

```python
from hyppo.generator import create_random_complete_structures

# Create multiple structures with custom parameters
structures = create_random_complete_structures(
    num_structures=3,        # Number of structures to create
    min_vars=4,              # Minimum number of variables per structure
    max_vars=8,              # Maximum number of variables per structure
    min_vars_per_eq=1,       # Minimum number of variables per equation
    max_vars_per_eq=3,       # Maximum number of variables per equation
    exogenous_ratio=0.25     # Ratio of exogenous variables
)
```

## Running the Example

To run this example, execute the `main.py` script:

```bash
python examples/random_structures/main.py
```

This will create several random complete structures and demonstrate their properties.