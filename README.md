# IPyM
## IPM Solver for BCP in LP

### Description

This project implements an Interior Point Method (IPM) solver for Branch-Cut-Price (BCP) in Linear Programming (LP). It is designed to efficiently solve large-scale LP problems using advanced numerical techniques. The solver is crafted as a library for use in Python, leveraging the capabilities of pybind11.

### Dependencies

    - Eigen: A C++ template library for linear algebra.
    - Cholmod: A library for solving sparse linear systems.
    - pybind11: A lightweight header-only library that exposes C++ types in Python and vice versa.

To ensure full functionality, make sure these dependencies are properly installed and configured in your environment.

### Solving Method

The *AUGMENTED* definition changes the compilation to use an augmented matrix form.

### Installation

To install this project, follow these steps:

    - Install CMake, which will be used to build the project.
    - Clone the repository: git clone [repository-url].
    - Navigate to the project directory: cd [project-directory].
    - Create a build directory and navigate into it: mkdir build && cd build.
    - Run CMake to configure the project: cmake ...
    - Build the project: cmake --build ..

### Usage

First, ensure that the library is properly imported in your Python script:

```
import ipy_selfdual as ipy
```

Then, you can use the run_optimization function as follows:

```
# Define your problem parameters
A = ... # Coefficient matrix
b = ... # Right-hand side vector
c = ... # Cost vector
lo = ... # Lower bounds for variables
hi = ... # Upper bounds for variables
sense = ... # Constraint sense
tol = ... # Tolerance

# Run the optimization
x0, lambdas, slacks, obj = ipy.run_optimization(A, b, -c, lo, hi, sense_ipm, tol)

# x0: Solution vector
# lambdas: Dual variables
# slacks: Slack variables
# obj: Objective
```

### Contact

For any queries or collaborations, please contact Laio Oriel Seman at laio@ieee.org.