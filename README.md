# IPyM - Interior Po'y'nt Method

### Description
IPyM is a C++-based Interior Point Method (IPM) Python Library tailored for Linear Programming (LP). It efficiently tackles large-scale LP problems using advanced numerical techniques and integrates seamlessly with Python through pybind11. It is based on the self-dual formulation:

  ```
  \text{minimize/maximize} \quad & c^T x - b^T y \\
  \text{subject to} \quad & Ax + s = b, \quad A^T y + z = c, \\
  & x \geq 0, \quad s \geq 0, \quad y \text{ unrestricted}, \quad z \geq 0
  ```

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
For inquiries or collaborations, reach out to Laio Oriel Seman at laio [at] ieee.org.

### References

- Andersen, E.D., Andersen, K.D. (2000). The Mosek Interior Point Optimizer for Linear Programming: An Implementation of the Homogeneous Algorithm. In: Frenk, H., Roos, K., Terlaky, T., Zhang, S. (eds) High Performance Optimization. Applied Optimization, vol 33. Springer, Boston, MA. https://doi.org/10.1007/978-1-4757-3216-0_8

- Tanneau, M., Anjos, M.F. & Lodi, A. Design and implementation of a modular interior-point solver for linear optimization. Math. Prog. Comp. 13, 509–551 (2021). https://doi.org/10.1007/s12532-020-00200-8

- https://github.com/sinha-abhi/PDIPS.jl