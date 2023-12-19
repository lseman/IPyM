# IPyM
## IPM Solver for BCP in LP

### Description
IPyM is a Python-based Interior Point Method (IPM) solver tailored for Branch-Cut-Price (BCP) in Linear Programming (LP). It efficiently tackles large-scale LP problems using advanced numerical techniques and integrates seamlessly with Python through pybind11.

### Dependencies
- **Eigen**: A versatile C++ template library for linear algebra.
- **Cholmod**: Specialized in solving sparse linear systems.
- **pybind11**: Facilitates C++ and Python interoperability.

*Ensure these dependencies are installed and configured in your environment for optimal functionality.*

### Solving Method
Utilize the *AUGMENTED* definition to compile with an augmented matrix form.

### Installation
1. Install CMake for building the project.
2. Clone the repository: `git clone [repository-url]`.
3. Navigate to the project directory: `cd [project-directory]`.
4. Create and enter the build directory: `mkdir build && cd build`.
5. Configure with CMake: `cmake ...`.
6. Build the project: `cmake --build ..`.

### Usage
Import the library in your Python script:
```python
import ipy_selfdual as ipy
```
Optimize using `run_optimization`:
```python
# Define problem parameters
A, b, c = ... # Coefficient matrix, RHS vector, Cost vector
lo, hi = ... # Variable bounds
sense, tol = ... # Constraint sense, Tolerance

# Execute optimization
x0, lambdas, slacks, obj = ipy.run_optimization(A, b, -c, lo, hi, sense_ipm, tol)

# Outputs: Solution vector, Dual variables, Slack variables, Objective value
```

### Contact
For inquiries or collaborations, reach out to Laio Oriel Seman at laio@ieee.org.
