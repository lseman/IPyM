/*
 * Project: IPM for BCP
 * Author: Laio Oriel Seman
 * Email: laio@ieee.org
 */

#include "cholmod.h"

// #define EIGEN_USE_MKL_ALL

// create define to decide between augmented and normal
#define AUGMENTED

#define EIGEN_INITIALIZE_MATRICES_BY_ZERO
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <algorithm>
#include <limits>
#include <omp.h>
#include <stdexcept>
#include <tuple>
#include <vector>

using namespace Eigen;

/**
 * @brief Converts a dense vector to a sparse diagonal matrix.
 * 
 * This function takes a dense vector and converts it into a sparse diagonal matrix.
 * The resulting sparse matrix has non-zero values only on its diagonal, where the
 * values are taken from the input vector.
 * 
 * @param vec The input dense vector.
 * @return The resulting sparse diagonal matrix.
 */
Eigen::SparseMatrix<double>
convertToSparseDiagonal(const Eigen::VectorXd &vec) {
    Eigen::SparseMatrix<double> mat(vec.size(), vec.size());

    // Reserve space for diagonal elements
    mat.reserve(Eigen::VectorXd::Constant(vec.size(), 1));

    for (int i = 0; i < vec.size(); ++i) {
        mat.insert(i, i) = vec(i);
    }

    // Make the sparse matrix compressed
    // mat.makeCompressed();

    return mat;
}

/**
 * Converts the given linear programming problem to standard form.
 *
 * @param A The coefficient matrix of the linear constraints.
 * @param b The right-hand side vector of the linear constraints.
 * @param c The objective function coefficients.
 * @param lb The lower bounds on the variables.
 * @param ub The upper bounds on the variables.
 * @param sense The sense of the linear constraints (1 for <=, -1 for >=, 0 for =).
 * @param As The output sparse coefficient matrix in standard form.
 * @param bs The output right-hand side vector in standard form.
 * @param cs The output objective function coefficients in standard form.
 */
void convert_to_standard_form(
    const Eigen::SparseMatrix<double> &A, const Eigen::VectorXd &b,
    const Eigen::VectorXd &c, const Eigen::VectorXd &lb,
    const Eigen::VectorXd &ub, const Eigen::VectorXd &sense,
    Eigen::SparseMatrix<double> &As, Eigen::VectorXd &bs, Eigen::VectorXd &cs) {
    double infty = std::numeric_limits<double>::infinity();
    int n = A.rows();
    int m = A.cols();

    Eigen::VectorXd lo = lb;
    Eigen::VectorXd hi = ub;

    int n_free = 0, n_ubounds = 0, nzv = 0;
    int nv = A.cols();
    // count number of upper bounds
    for (int i = 0; i < lo.size(); ++i) {
        double l = lo[i];
        double h = hi[i];

        if (l == -infty && h == infty) {
            ++n_free;
        } else if (std::isfinite(l) && std::isfinite(h)) {
            ++n_ubounds;
        } else if (l == -infty && std::isfinite(h)) {
            // To be dealt with later
        } else if (std::isfinite(l) && h == infty) {
            // To be dealt with later
        } else {
            throw std::runtime_error("unexpected bounds");
        }
    }

    std::vector<int> I(nzv), J(nzv); // row and column indices
    std::vector<double> V(nzv);      // replace double with the actual type
    std::vector<int> ind_ub(n_ubounds);
    std::vector<double> val_ub(
        n_ubounds); // replace double with the actual type
    int num_slacks = n - sense.sum();

    cs.conservativeResize(c.size() + n_free + num_slacks);
    cs.setZero();
    cs.head(m) = c;

    bs.conservativeResize(b.size());
    bs.head(n) = b;

    int free = 0, ubi = 0;
    nzv = 0;
    for (int j = 0; j < lo.size(); ++j) {
        double l = lo[j];
        double h = hi[j];

        for (Eigen::SparseMatrix<double>::InnerIterator it(A, j); it; ++it) {
            int i = it.row();      // Row index
            double v = it.value(); // Value at A(i, j)

            if (l == -infty && h == infty) {
                // free variable
                cs[j] = c[j];
                cs[nv + free] = -c[j];

                bs[i] -= (v * 0);

                ++nzv;
                I.push_back(i);
                J.push_back(j);
                V.push_back(v);

                ++nzv;
                I.push_back(i);
                J.push_back(nv + free);
                V.push_back(-v);

                ++free;
            } else if (std::isfinite(l) && std::isfinite(h)) {
                // l <= x <= h
                cs[j + free] = c[j];

                bs[i] -= (v * l);
                ++nzv;
                I.push_back(i);
                J.push_back(j);
                V.push_back(v);

                ++ubi;
                ind_ub.push_back(j);
                val_ub.push_back(h - l);
            } else if (l == -infty && std::isfinite(h)) {
                // x <= h
                cs[j] = -c[j];

                bs[i] -= (-v * h);
                ++nzv;
                I.push_back(i);
                J.push_back(j);
                V.push_back(-v);
            } else if (std::isfinite(l) && h == infty) {
                // l <= x
                cs[j] = c[j];

                bs[i] -= (v * l);
                ++nzv;
                I.push_back(i);
                J.push_back(j);
                V.push_back(v);
            } else {
                throw std::runtime_error("Unexpected bounds");
            }
        }
    }

    // Adding slack variables
    int slack_counter = 0;
    for (int i = 0; i < sense.size(); ++i) {
        if (sense(i) == 0) {
            ++nzv;
            I.push_back(i);
            J.push_back(nv + n_free + slack_counter);
            V.push_back(1.0);

            ++slack_counter;
        }
    }

    std::vector<Eigen::Triplet<double>> triplets;

    for (int k = 0; k < nzv; ++k) {
        if (I[k] >= bs.size() || J[k] >= cs.size()) {
            std::cout << "Out-of-bounds triplet: (" << I[k] << ", " << J[k]
                      << ", " << V[k] << ")" << std::endl;
        }
        triplets.push_back(Eigen::Triplet<double>(I[k], J[k], V[k]));
    }
    As.resize(bs.size(), cs.size());
    As.setFromTriplets(triplets.begin(), triplets.end());
    As.makeCompressed();

}

class SparseSolver {
public:
    int n;
    int m;
    Eigen::VectorXd theta;
    Eigen::VectorXd regP;
    Eigen::VectorXd regD;
    Eigen::SparseMatrix<double> A;
    Eigen::SparseMatrix<double> S;
    Eigen::SparseMatrix<double> AD;
    Eigen::SparseMatrix<double> D;
    bool firstFactorization = true;

    cholmod_common c;
    cholmod_factor *L;
    cholmod_dense *X = NULL, *Y = NULL, *E = NULL;

    SparseSolver() {
        cholmod_start(&c);
        c.useGPU = 0; // Use this line instead of &c->useGPU = 1;

        c.nmethods = 1;
        c.method[0].ordering = CHOLMOD_METIS;
        c.postorder = true;

// only if define augmented = 1
#ifdef AUGMENTED
        c.supernodal =
            CHOLMOD_SIMPLICIAL; // Use the supernodal factorization method
#else
        c.supernodal =
            CHOLMOD_SUPERNODAL; // Use the supernodal factorization method
#endif

        // c.maxGpuMemBytes = 10000000000;
        // c.method = CHOLMOD_SUPERNODAL;

        L = nullptr;
    }

    ~SparseSolver() {

        if (L) {
            cholmod_free_factor(&L, &c);
        }
        cholmod_finish(&c);
    }

    /**
     * Factorizes the given sparse matrix using Cholesky decomposition.
     * 
     * @param matrix The sparse matrix to be factorized.
     */
    void factorizeMatrix(
        const Eigen::SparseMatrix<double, Eigen::RowMajor, int> &matrix) {
        cholmod_sparse *A = viewAsCholmod(matrix);

        if (firstFactorization) {
            if (L) {
                cholmod_free_factor(&L, &c);
            }
            L = cholmod_analyze(A, &c);
            firstFactorization = false;
        }

        cholmod_factorize(A, L, &c);
    }

    /**
     * Solves a linear system of equations using the Cholesky decomposition method.
     * 
     * @param rhs The right-hand side vector of the linear system.
     * @return The solution vector.
     */
    Eigen::VectorXd solve(const Eigen::VectorXd &rhs) {
        cholmod_dense *b = viewAsCholmod(rhs);

        cholmod_solve2(CHOLMOD_A, L, b, NULL, &X, NULL, &Y, &E, &c);
        Eigen::VectorXd result = viewAsEigen(X);
        return result;
    }

private:
    /**
     * @brief Converts an Eigen sparse matrix to a cholmod_sparse matrix.
     * 
     * This function takes an Eigen sparse matrix and converts it to a cholmod_sparse matrix,
     * which is a sparse matrix representation used by the CHOLMOD library.
     * 
     * @param matrix The Eigen sparse matrix to be converted.
     * @return A pointer to the converted cholmod_sparse matrix.
     */
    static cholmod_sparse *viewAsCholmod(
        const Eigen::SparseMatrix<double, Eigen::RowMajor, int> &matrix) {
        cholmod_sparse *result = new cholmod_sparse;
        result->nrow = matrix.rows();
        result->ncol = matrix.cols();
        result->p = const_cast<int *>(matrix.outerIndexPtr());
        result->i = const_cast<int *>(matrix.innerIndexPtr());

        result->x = const_cast<double *>(matrix.valuePtr());
        result->z = nullptr;
        result->stype = -1;
        result->itype = CHOLMOD_INT;
        result->xtype = CHOLMOD_REAL;
        result->dtype = CHOLMOD_DOUBLE;
        result->sorted = 1;
        result->packed = 1;
        return result;
    }

    /**
     * @brief Converts an Eigen::VectorXd to a cholmod_dense object.
     * 
     * This function takes an Eigen::VectorXd object and converts it to a cholmod_dense object.
     * The resulting cholmod_dense object has the same dimensions and data as the input vector.
     * 
     * @param vector The Eigen::VectorXd object to be converted.
     * @return A pointer to the resulting cholmod_dense object.
     */
    static cholmod_dense *viewAsCholmod(const Eigen::VectorXd &vector) {
        cholmod_dense *result = new cholmod_dense;
        result->nrow = vector.size();
        result->ncol = 1;
        result->nzmax = vector.size();
        result->d = vector.size();
        result->x = const_cast<double *>(vector.data());
        result->z = nullptr;
        result->xtype = CHOLMOD_REAL;
        result->dtype = CHOLMOD_DOUBLE;
        return result;
    }

    /**
     * @brief Converts a cholmod_dense vector to an Eigen::VectorXd.
     * 
     * This function takes a pointer to a cholmod_dense vector and returns an Eigen::VectorXd
     * by mapping the data of the cholmod_dense vector to an Eigen::VectorXd object.
     * 
     * @param vectorPtr A pointer to the cholmod_dense vector.
     * @return An Eigen::VectorXd object containing the data of the cholmod_dense vector.
     */
    static Eigen::VectorXd viewAsEigen(cholmod_dense *vectorPtr) {
        return Eigen::VectorXd::Map(reinterpret_cast<double *>(vectorPtr->x),
                                    vectorPtr->nrow);
    }
};

/**
 * Starts the linear solver by initializing the necessary data structures and performing factorization.
 * 
 * @param ls The SparseSolver object to be used for solving the linear system.
 * @param A The sparse matrix representing the system of linear equations.
 */
void start_linear_solver(SparseSolver &ls,
                         const Eigen::SparseMatrix<double> A) {
    ls.A = A;
    ls.m = A.rows();
    ls.n = A.cols();

#ifdef AUGMENTED
    ls.theta = Eigen::VectorXd::Ones(ls.n);
    ls.regP = Eigen::VectorXd::Ones(ls.n);
    ls.regD = Eigen::VectorXd::Ones(ls.m);

    Eigen::SparseMatrix<double> topRight = ls.A.transpose();
    Eigen::SparseMatrix<double> bottomLeft = ls.A;
    Eigen::SparseMatrix<double> topLeft =
        convertToSparseDiagonal(-ls.theta - ls.regP);
    Eigen::SparseMatrix<double> bottomRight = convertToSparseDiagonal(ls.regD);

    // S_ is known, reserve space for it
    Eigen::SparseMatrix<double> S_(ls.n + ls.m, ls.n + ls.m);

    // Reserving space for tripletList
    int estimated_nonzeros =
        topLeft.nonZeros() + 2 * topRight.nonZeros() + bottomRight.nonZeros();
    std::vector<Eigen::Triplet<double>> tripletList;
    tripletList.reserve(estimated_nonzeros);

    // Insert topLeft, topRight, bottomLeft, bottomRight matrices
    auto insertBlock = [&](const Eigen::SparseMatrix<double> &block,
                           int startRow, int startCol) {
        for (int k = 0; k < block.outerSize(); ++k) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(block, k); it;
                 ++it) {
                tripletList.emplace_back(it.row() + startRow,
                                         it.col() + startCol, it.value());
            }
        }
    };

    insertBlock(topLeft, 0, 0);
    insertBlock(topRight, 0, ls.n);
    insertBlock(bottomLeft, ls.n, 0);
    insertBlock(bottomRight, ls.n, ls.n);

    // Finally, set the values from the triplets
    S_.setFromTriplets(tripletList.begin(), tripletList.end());
    // S_.makeCompressed();

    ls.S = S_;
    // Factorize
    ls.factorizeMatrix(ls.S);
#endif
}

/**
 * Updates the linear solver with new values for theta, regP, and regD.
 * If the AUGMENTED flag is defined, it efficiently updates the diagonal elements of the matrix S and refactors it.
 * If the AUGMENTED flag is not defined, it constructs the left-hand side (lhs) matrix for the normal equations using the given values of theta, regP, and regD, and factorizes it.
 *
 * @param ls The SparseSolver object representing the linear solver.
 * @param theta The vector of theta values.
 * @param regP The vector of regP values.
 * @param regD The vector of regD values.
 */
void update_linear_solver(SparseSolver &ls, const Eigen::VectorXd &theta,
                          const Eigen::VectorXd &regP,
                          const Eigen::VectorXd &regD) {
    // Update internal data
    ls.theta = theta;
    ls.regP = regP;
    ls.regD = regD;

#ifdef AUGMENTED
    // Update S. S is stored as upper-triangular and only its diagonal changes.
    Eigen::VectorXd combinedValues(ls.n + ls.m);
    combinedValues.head(ls.n) = -theta - regP;
    combinedValues.tail(ls.m) = regD;

    // Efficiently update diagonal elements
    for (int i = 0; i < combinedValues.size(); i++) {
        ls.S.coeffRef(i, i) = combinedValues[i];
    }

    // Refactorize
    ls.factorizeMatrix(ls.S);
#else

    // define lhs for normal equations
    Eigen::SparseMatrix<double> lhs(ls.n + ls.m, ls.n + ls.m);
    // define lhs as   A (\Theta^{-1} + R_{p})^{-1} A^{\top} + R_{d}
    Eigen::VectorXd d = 1.0 / (ls.theta.array() + ls.regP.array());
    // set rhs as \xi_{p} + A (Θ^{-1} + R_{p})^{-1} \xi_{d}
    Eigen::MatrixXd dDense = d.asDiagonal();
    Eigen::SparseMatrix<double> dSparse = dDense.sparseView();

    Eigen::MatrixXd regDDense = regD.asDiagonal();
    Eigen::SparseMatrix<double> regDSparse = regDDense.sparseView();
    Eigen::SparseMatrix<double> AD = ls.A * dSparse;
    Eigen::SparseMatrix<double> ADA = AD * ls.A.transpose();
    lhs = ADA + regDSparse;
    ls.factorizeMatrix(lhs);
#endif
}

struct Residuals {
    VectorXd rp, ru, rd, rl;
    double rpn, run, rdn, rgn, rg, rln;
};

/**
 * @brief Updates the residuals of a self-dual interior point method.
 *
 * This function calculates and updates the primal, upper bound, dual, and gap residuals
 * based on the given variables and parameters.
 *
 * @param res The Residuals object to store the updated residuals.
 * @param x The primal variable vector.
 * @param lambda The dual variable vector.
 * @param s The slack variable vector.
 * @param v The upper bound variable vector.
 * @param w The dual slack variable vector.
 * @param A The sparse matrix representing the linear constraints.
 * @param b The right-hand side vector of the linear constraints.
 * @param c The cost vector.
 * @param ubv The upper bound vector for the primal variables.
 * @param ubi The indices of the primal variables with upper bounds.
 * @param vbv The upper bound vector for the dual variables.
 * @param vbi The indices of the dual variables with upper bounds.
 * @param tau The scaling factor for the primal and upper bound residuals.
 * @param kappa The constant term in the gap residual.
 */
void update_residuals(Residuals &res, const VectorXd &x, const VectorXd &lambda,
                      const VectorXd &s, const VectorXd &v, const VectorXd &w,
                      const SparseMatrix<double> &A, const VectorXd &b,
                      const VectorXd &c, const VectorXd &ubv,
                      const VectorXi &ubi, const VectorXd &vbv,
                      const VectorXi &vbi, double tau, double kappa) {
    // Calculate rp and its norm
    // primal residual
    res.rp.noalias() = tau * b - A * x;
    // res.rpn = res.rp.norm();

    // Calculate ru and its norm
    // uper bound residual
    res.ru.noalias() = -v;
    for (int i = 0; i < ubi.size(); ++i) {
        res.ru(ubi(i)) -= x(ubi(i));
    }
    res.ru.array() += tau * ubv.array();
    // res.run = res.ru.norm();

    // Calculate rd and its norm
    // dual residual
    res.rd.noalias() = tau * c - (A.transpose() * lambda + s);
    for (int i = 0; i < ubi.size(); ++i) {
        res.rd(ubi(i)) += x(ubi(i));
    }
    // Calculate rg and its norm
    // gap residual
    res.rg = kappa + c.dot(x) - b.dot(lambda) + ubv.dot(w);
    res.rgn = std::sqrt(
        res.rg *
        res.rg); // Since rg is a scalar, its norm is the absolute value
}

/**
 * Solves the augmented system of equations to obtain the solution vectors dx and dy.
 * The augmented system is solved using a SparseSolver object.
 *
 * @param dx The solution vector for dx.
 * @param dy The solution vector for dy.
 * @param ls The SparseSolver object used to solve the system.
 * @param xi_p The vector xi_p.
 * @param xi_d The vector xi_d.
 */
void solve_augmented_system(Eigen::VectorXd &dx, Eigen::VectorXd &dy,
                            SparseSolver &ls, const Eigen::VectorXd &xi_p,
                            const Eigen::VectorXd &xi_d) {
#ifdef AUGMENTED
    // Set-up right-hand side
    Eigen::VectorXd xi(xi_d.size() + xi_p.size());
    xi << xi_d, xi_p;

    // Solve augmented system
    Eigen::VectorXd d = ls.solve(xi);

    // Recover dx, dy
    dx = d.head(xi_d.size()); // Gets the first n elements
    dy = d.tail(xi_p.size()); // Gets the last m elements
                              // Recover dx
    // dx = d.asDiagonal() * (ls.A.transpose() * dy - xi_d);
#else
    Eigen::VectorXd d = 1.0 / (ls.theta.array() + ls.regP.array());
    Eigen::VectorXd xi_ = xi_p + ls.A * (d.asDiagonal() * xi_d);

    // Solve augmented system
    dy = ls.solve(xi_);

    // Recover dx
    dx = d.asDiagonal() * (ls.A.transpose() * dy - xi_d);
#endif
}

/**
 * Solves the augmented system of equations to compute the values of delta_x, delta_y, and delta_z.
 *
 * @param delta_x The vector to store the computed values of delta_x.
 * @param delta_y The vector to store the computed values of delta_y.
 * @param delta_z The vector to store the computed values of delta_z.
 * @param ls The SparseSolver object used to solve the augmented system.
 * @param theta_vw The vector containing the values of theta_vw.
 * @param ubi The vector containing the indices of ubi.
 * @param xi_p The vector containing the values of xi_p.
 * @param xi_d The vector containing the values of xi_d.
 * @param xi_u The vector containing the values of xi_u.
 */
void solve_augsys(Eigen::VectorXd &delta_x, Eigen::VectorXd &delta_y,
                  Eigen::VectorXd &delta_z, SparseSolver &ls,
                  const Eigen::VectorXd &theta_vw, const Eigen::VectorXi &ubi,
                  const Eigen::VectorXd &xi_p, const Eigen::VectorXd &xi_d,
                  const Eigen::VectorXd &xi_u) {
    // Efficiently initialize delta_z with the right size and set to zero
    delta_z = Eigen::VectorXd::Zero(ubi.size());

    // Efficient modification of xi_d using sparse operations
    Eigen::SparseVector<double> _xi_d = xi_d.sparseView();
    Eigen::SparseVector<double> xi_u_theta =
        (xi_u.cwiseProduct(theta_vw)).sparseView();
    for (int i = 0; i < ubi.size(); ++i) {
        _xi_d.coeffRef(ubi(i)) -= xi_u_theta.coeff(i);
    }

    // Call the function to solve the augmented system
    solve_augmented_system(delta_x, delta_y, ls, xi_p, _xi_d);

    // Efficient update of delta_z
    for (int i = 0; i < ubi.size(); ++i) {
        delta_z.coeffRef(i) =
            (delta_x.coeff(ubi(i)) - xi_u.coeff(i)) * theta_vw.coeff(i);
    }
}

/**
 * Solves the Newton system of equations to update the variables Delta_x, Delta_lambda, Delta_w,
 * Delta_s, Delta_v, Delta_tau, and Delta_kappa.
 *
 * @param Delta_x The update for the variable x.
 * @param Delta_lambda The update for the variable lambda.
 * @param Delta_w The update for the variable w.
 * @param Delta_s The update for the variable s.
 * @param Delta_v The update for the variable v.
 * @param Delta_tau The update for the variable tau.
 * @param Delta_kappa The update for the variable kappa.
 * @param ls The sparse solver used to solve the augmented system.
 * @param theta_vw The vector theta_vw.
 * @param b The vector b.
 * @param c The vector c.
 * @param ubi The vector ubi.
 * @param ubv The vector ubv.
 * @param delta_x The vector delta_x.
 * @param delta_y The vector delta_y.
 * @param delta_w The vector delta_w.
 * @param delta_0 The value delta_0.
 * @param iter_x The vector iter_x.
 * @param iter_lambda The vector iter_lambda.
 * @param iter_w The vector iter_w.
 * @param iter_s The vector iter_s.
 * @param iter_v The vector iter_v.
 * @param iter_tau The value iter_tau.
 * @param iter_kappa The value iter_kappa.
 * @param xi_p The vector xi_p.
 * @param xi_u The vector xi_u.
 * @param xi_d The vector xi_d.
 * @param xi_g The value xi_g.
 * @param xi_xs The vector xi_xs.
 * @param xi_vw The vector xi_vw.
 * @param xi_tau_kappa The value xi_tau_kappa.
 */
void solve_newton_system(
    VectorXd &Delta_x, VectorXd &Delta_lambda, VectorXd &Delta_w,
    VectorXd &Delta_s, VectorXd &Delta_v, double &Delta_tau,
    double &Delta_kappa, SparseSolver &ls, const VectorXd &theta_vw,
    const VectorXd &b, const VectorXd &c, const VectorXi &ubi,
    const VectorXd &ubv, const VectorXd &delta_x, const VectorXd &delta_y,
    const VectorXd &delta_w, double delta_0, const VectorXd &iter_x,
    const VectorXd &iter_lambda, const VectorXd &iter_w, const VectorXd &iter_s,
    const VectorXd &iter_v, double iter_tau, double iter_kappa,
    const VectorXd &xi_p, const VectorXd &xi_u, const VectorXd &xi_d,
    double xi_g, const VectorXd &xi_xs, const VectorXd &xi_vw,
    double xi_tau_kappa) {
    VectorXd xi_d_copy = xi_d - (xi_xs.array() / iter_x.array()).matrix();
    VectorXd xi_u_copy = xi_u - (xi_vw.array() / iter_w.array()).matrix();

    // Call solve_augsys function here to update Delta_x, Delta_lambda, and
    // Delta_w
    solve_augsys(Delta_x, Delta_lambda, Delta_w, ls, theta_vw, ubi, xi_p,
                 xi_d_copy, xi_u_copy);

    Delta_tau = (xi_g + (xi_tau_kappa / iter_tau) + c.dot(Delta_x) -
                 b.dot(Delta_lambda) + ubv.dot(Delta_w)) /
                delta_0;
    Delta_kappa = (xi_tau_kappa - iter_kappa * Delta_tau) / iter_tau;

    Delta_x.array() += Delta_tau * delta_x.array();
    Delta_lambda.array() += Delta_tau * delta_y.array();
    Delta_w.array() += Delta_tau * delta_w.array();

    Delta_s = (xi_xs - iter_s.cwiseProduct(Delta_x)).cwiseQuotient(iter_x);
    Delta_v = (xi_vw - iter_v.cwiseProduct(Delta_w)).cwiseQuotient(iter_w);
}

/**
 * Calculates the maximum value of alpha based on the given vectors v and dv.
 * 
 * @param v The input vector.
 * @param dv The derivative vector.
 * @return The maximum value of alpha.
 */
double max_alpha_single(const VectorXd &v, const VectorXd &dv) {

    double alpha = std::numeric_limits<double>::infinity();
    // #pragma omp parallel for reduction(min : alpha)
    for (int i = 0; i < v.size(); ++i) {
        if (dv(i) < 0) {
            double potential_alpha = -v(i) / dv(i);
            alpha = std::min(alpha, potential_alpha);
        }
    }

    return alpha;
}

/**
 * Calculates the maximum alpha value for a given set of parameters.
 *
 * @param x     The input vector x.
 * @param dx    The input vector dx.
 * @param v     The input vector v.
 * @param dv    The input vector dv.
 * @param s     The input vector s.
 * @param ds    The input vector ds.
 * @param w     The input vector w.
 * @param dw    The input vector dw.
 * @param tau   The value of tau.
 * @param dtau  The value of dtau.
 * @param kappa The value of kappa.
 * @param dkappa The value of dkappa.
 * @return      The maximum alpha value.
 */
double max_alpha(const VectorXd &x, const VectorXd &dx, const VectorXd &v,
                 const VectorXd &dv, const VectorXd &s, const VectorXd &ds,
                 const VectorXd &w, const VectorXd &dw, double tau, double dtau,
                 double kappa, double dkappa) {
    double alpha_tau = (dtau < 0) ? (-tau / dtau) : 1.0;
    double alpha_kappa = (dkappa < 0) ? (-kappa / dkappa) : 1.0;

    double alpha = std::min({1.0, max_alpha_single(x, dx),
                             max_alpha_single(v, dv), max_alpha_single(s, ds),
                             max_alpha_single(w, dw), alpha_tau, alpha_kappa});

    return alpha;
}

std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, double>

run_optimization(const Eigen::SparseMatrix<double> &As,
                 const Eigen::VectorXd &bs, const Eigen::VectorXd &cs,
                 const Eigen::VectorXd &lo, const Eigen::VectorXd &hi,
                 const Eigen::VectorXd &sense, const double tol) {

    // omp_set_num_threads(32);
    // Eigen::setNbThreads(32);
    //  Convert to standard form
    Eigen::SparseMatrix<double> A;
    Eigen::VectorXd b;
    Eigen::VectorXd c;

    int nv_orig = cs.size();

    convert_to_standard_form(As, bs, cs, lo, hi, sense, A, b, c);

    int n = A.cols();
    int m = A.rows();

    // Output the initial results
    // std::cout << "Initial x_0: " << std::endl << x_k << std::endl;
    // std::cout << "Initial lambda_0: " << std::endl << lambda_k << std::endl;
    // std::cout << "Initial s_0: " << std::endl << s_k << std::endl;

    // Tolerance and maximum iterations
    int max_iter = 100;

    // Initialize vectors and scalars
    Eigen::VectorXd x = Eigen::VectorXd::Ones(n);
    Eigen::VectorXd lambda = Eigen::VectorXd::Zero(m);
    Eigen::VectorXd s = Eigen::VectorXd::Ones(n);

    // initialize ubi and ubv as empty vectors
    Eigen::VectorXi ubi;
    Eigen::VectorXd ubv;

    int count = 0; // Count of non-zero entries
    for (int i = 0; i < hi.size(); i++) {
        if (hi[i] != std::numeric_limits<double>::infinity()) {
            count++;
        }
    }

    Eigen::VectorXi tempUbi(count);
    Eigen::VectorXd tempUbv(count);

    double infty = std::numeric_limits<double>::infinity();
    count = 0;
    for (int i = 0; i < hi.size(); i++) {
        if (hi[i] != infty) {
            tempUbi[count] = i;
            tempUbv[count] = hi[i];
            count++;
        }
    }

    ubi = tempUbi;
    ubv = tempUbv;

    Eigen::VectorXd v = Eigen::VectorXd::Ones(ubv.size());
    Eigen::VectorXd w = Eigen::VectorXd::Ones(ubv.size());

    // initialize vbi and vbv as empty vectors
    Eigen::VectorXi vbi;
    Eigen::VectorXd vbv;

    double tau = 1.0;
    double kappa = 1.0;

    // Assuming lp.nv and lp.nc are the dimensions you need
    Eigen::VectorXd regP = Eigen::VectorXd::Ones(n);
    Eigen::VectorXd regD = Eigen::VectorXd::Ones(m);
    double regG = 1.0;

    SparseSolver ls;
    start_linear_solver(ls, A);

    int nc = A.rows(); // Assuming ls is the sparse matrix
    int nv = A.cols();
    int nu = ubi.size();

    Eigen::VectorXd delta_x(nv), delta_y(nc), delta_z(nu);
    // Residuals
    Residuals res;

    // Dimensions and constants
    double r_min =
        std::sqrt(std::numeric_limits<double>::epsilon()); // approx 1e-8
    int attempt = 0;
    // Residual related variables
    double _p, _d, _g;
    double mu;
    // Step length and corrections
    double alpha, alpha_c, alpha_;
    int ncor;
    double beta;
    // Damping factors
    double gamma, damping;
    double oneMinusAlpha;
    // Cross products and thresholds
    double mu_l, mu_u;
    double taukappa, t0;
    // Theta values
    Eigen::VectorXd theta_vw, theta_xs;
    // Xi values
    Eigen::VectorXd xi_p, xi_d, xi_u, xi_xs, xi_vw;
    // Delta values
    Eigen::VectorXd Delta_x(x.size()), Delta_lambda(lambda.size()),
        Delta_w(w.size()), Delta_s(s.size()), Delta_v(v.size());
    double Delta_tau, Delta_kappa;
    // Corrected Delta values
    Eigen::VectorXd Delta_x_c(x.size()), Delta_lambda_c(lambda.size()),
        Delta_w_c(w.size()), Delta_s_c(s.size()), Delta_v_c(v.size());
    double Delta_tau_c, Delta_kappa_c;
    // Temporary values for corrections
    Eigen::VectorXd xs, vw;
    Eigen::ArrayXd t_xs_lower, t_xs_upper, t_vw_lower, t_vw_upper;
    Eigen::VectorXd t_xs, t_vw;
    // Delta calculations
    double delta_0, bl_dot_lambda, correction;
    
    for (int k = 0; k < max_iter; ++k) {

        // zero the necessary variables
        ncor = 0;
        beta = 0.1;
        // Zero out the predictor search direction variables
        delta_x.setZero();
        delta_y.setZero();
        delta_z.setZero();

        Delta_x.setZero();
        Delta_lambda.setZero();
        Delta_w.setZero();
        Delta_s.setZero();
        Delta_v.setZero();
        Delta_tau = 0.0;
        Delta_kappa = 0.0;

        // Call the function
        update_residuals(res, x, lambda, s, v, w, A, b, c, ubv, ubi, vbv, vbi,
                         tau, kappa);
        mu = (tau * kappa + x.dot(s) + v.dot(w)) / (n + ubi.size() + 1);

        // calculate _p = max(|rp| / (τ * (1 + |b|)), |ru| / (τ * (1 + |u|)))
        bl_dot_lambda = b.dot(lambda);

        _p = std::fmax(res.rp.lpNorm<Eigen::Infinity>() /
                           (tau * (1.0 + b.lpNorm<Eigen::Infinity>())),
                       res.ru.lpNorm<Eigen::Infinity>() /
                           (tau * (1.0 + ubv.lpNorm<Eigen::Infinity>())));
        // calculate _d = |rd| / (τ * (1 + |c|))
        _d = res.rd.lpNorm<Eigen::Infinity>() /
             (tau * (1.0 + c.lpNorm<Eigen::Infinity>()));

        _g = std::abs(c.dot(x) - bl_dot_lambda) /
             (tau + std::abs(bl_dot_lambda));

        //   check optimality
        if (_d < tol) {
            break;
        }
        // check infesibility
        if ((mu < 1e-6) || (tau / kappa < 1e-6)) {
            break;
        }

        // Vectorized scaling factors computation
        theta_vw = w.cwiseQuotient(v);
        theta_xs = s.cwiseQuotient(x);

        for (int i = 0; i < ubi.size(); ++i) {
            theta_xs[ubi[i]] += theta_vw[i];
        }

        // update regularizations
        // Element-wise operations for regP and regD
        regP /= 10.0;
        regP = regP.cwiseMax(r_min);
        regD /= 10.0;
        regD = regD.cwiseMax(r_min);

        // Scalar operation for regG
        regG = std::max(r_min, regG / 10.0);

        // factorization
        // make three attempts, after increasing regularization
        while (attempt < 2) {
            try {
                // std::cout << "attempt: " << attempt << std::endl;
                update_linear_solver(ls, theta_xs, regP, regD);
                break;
            } catch (std::runtime_error &) {
                regP *= 100.0;
                regD *= 100.0;
                regG *= 100.0;
                attempt++;
            }
        }

        // Call the solve_augsys function
        solve_augsys(delta_x, delta_y, delta_z, ls, theta_vw, ubi, b, c, ubv);
        // Calculate delta_0
        delta_0 = regG + kappa / tau - delta_x.dot(c) + delta_y.dot(b) -
                  delta_z.dot(ubv);

        // Call the function
        solve_newton_system(Delta_x, Delta_lambda, Delta_w, Delta_s, Delta_v,
                            Delta_tau, Delta_kappa, ls, theta_vw, b, c, ubi,
                            ubv, delta_x, delta_y, delta_z, delta_0, x, lambda,
                            w, s, v, tau, kappa, res.rp, res.ru, res.rd, res.rg,
                            -x.cwiseProduct(s), // xi_xs
                            -v.cwiseProduct(w), // xi_vw
                            -tau * kappa        // xi_tau_kappa
        );

        // Calculate new step length
        alpha = max_alpha(x, Delta_x, v, Delta_v, s, Delta_s, w, Delta_w, tau,
                          Delta_tau, kappa, Delta_kappa);

        // Calculate gamma and bound it to 0.1 in a single line
        oneMinusAlpha = 1.0 - alpha;
        gamma = std::fmax(oneMinusAlpha * oneMinusAlpha *
                              std::fmin(beta, oneMinusAlpha),
                          0.1);
        damping = 1.0 - gamma;

        solve_newton_system(
            Delta_x, Delta_lambda, Delta_w, Delta_s, Delta_v, Delta_tau,
            Delta_kappa, ls, theta_vw, b, c, ubi, ubv, delta_x, delta_y,
            delta_z, delta_0, x, lambda, w, s, v, tau, kappa, damping * res.rp,
            damping * res.ru, damping * res.rd, damping * res.rg,
            (-x.cwiseProduct(s)).array() + (gamma * mu) -
                Delta_x.cwiseProduct(Delta_s).array(),
            (-v.cwiseProduct(w)).array() + (gamma * mu) -
                Delta_v.cwiseProduct(Delta_w).array(),
            (-tau * kappa) + (gamma * mu) - Delta_tau * Delta_kappa);

        alpha = max_alpha(x, Delta_x, v, Delta_v, s, Delta_s, w, Delta_w, tau,
                          Delta_tau, kappa, Delta_kappa);

        // compute high order corrections like Tulip
        while ((ncor <= 3) && (alpha < 0.9995)) {
            // Tentative step length
            ncor += 1;
            alpha_ = std::min(1.0, 2.0 * alpha);

            // Compute target cross-products
            mu_l = beta * mu * gamma;
            mu_u = gamma * mu / beta;

            // Temporary variables for xs and vw
            xs = x + alpha_ * Delta_x;
            xs.array() *= (s + alpha_ * Delta_s).array();
            vw = v + alpha_ * Delta_v;
            vw.array() *= (w + alpha_ * Delta_w).array();


            t_vw_lower = (vw.array() < mu_l).select(mu_l - vw.array(), 0);
            t_vw_upper = (vw.array() > mu_u).select(mu_u - vw.array(), 0);

            t_xs = (t_xs_lower + t_xs_upper).matrix();
            t_vw = (t_vw_lower + t_vw_upper).matrix();

            // define t0  as tau * kappa if tau * kappa ar between mu_l and mu_u
            taukappa =
                (tau + alpha_ * Delta_tau) * (kappa + alpha_ * Delta_kappa);

            /*
            if (taukappa < mu_l) {
                t0 = mu_l - taukappa;
            } else if (taukappa > mu_u) {
                t0 = mu_u - taukappa;
            } else {
                t0 = 0;
            }
            */
            t0 = std::clamp(taukappa, mu_l, mu_u) - taukappa;

            // correct xs, vw and t0
            correction = (t_xs.sum() + t_vw.sum() + t0) / (nv + nu + 1);
            t_xs.array() -= correction;
            t_vw.array() -= correction;
            t0 = t0 - (t_xs.sum() + t_vw.sum() + t0) / (nv + nu + 1);

            // create temporary Deltas to store the values of Delta_x, Delta_y,
            // Delta_z, Delta_tau, Delta_kappa
            Delta_x_c = Delta_x;
            Delta_lambda_c = Delta_lambda;
            Delta_w_c = Delta_w;
            Delta_s_c = Delta_s;
            Delta_v_c = Delta_v;
            Delta_tau_c = Delta_tau;
            Delta_kappa_c = Delta_kappa;
            solve_newton_system(
                Delta_x_c, Delta_lambda_c, Delta_w_c, Delta_s_c, Delta_v_c,
                Delta_tau_c, Delta_kappa_c, ls, theta_vw, b, c, ubi, ubv,
                delta_x, delta_y, delta_z, delta_0, x, lambda, w, s, v, tau,
                kappa, Eigen::VectorXd::Zero(res.rp.size()),
                Eigen::VectorXd::Zero(res.ru.size()),
                Eigen::VectorXd::Zero(res.rd.size()), 0, -t_xs, -t_vw, -t0);

            // compute max step length
            alpha_c =
                max_alpha(x, Delta_x_c, v, Delta_v_c, s, Delta_s_c, w,
                          Delta_w_c, tau, Delta_tau_c, kappa, Delta_kappa_c);

            if (alpha_c > alpha_) {
                Delta_x = Delta_x_c;
                Delta_lambda = Delta_lambda_c;
                Delta_w = Delta_w_c;
                Delta_s = Delta_s_c;
                Delta_v = Delta_v_c;
                Delta_tau = Delta_tau_c;
                Delta_kappa = Delta_kappa_c;
                alpha = alpha_c;
            }

            if (alpha_c < 1.1 * alpha_) {
                break;
            }
        }

        // multiply alpha by 0.99
        alpha *= 0.9995;

        // Update iterates
        x += alpha * Delta_x;
        lambda += alpha * Delta_lambda;
        s += alpha * Delta_s;
        v += alpha * Delta_v;
        w += alpha * Delta_w;
        tau += alpha * Delta_tau;
        kappa += alpha * Delta_kappa;
    }

    int free_var = 0;
    double inv_tau = 1.0 / tau;

    Eigen::VectorXd original_x(As.cols());
    for (int j = 0; j < lo.size(); ++j) {
        double l = lo[j];
        double h = hi[j];

        if (l == -infty && h == infty) {
            // For free variables, we had split them into x+ and x-.
            original_x[j] = (x[j + free_var] - x[nv_orig + free_var]) * inv_tau;
            free_var++;
        } else if (std::isfinite(l) && std::isfinite(h)) {
            // For variables with both lower and upper bounds.
            original_x[j] = l + x[j] * inv_tau;
        } else if (l == -infty && std::isfinite(h)) {
            // For variables with only upper bounds.
            original_x[j] = h - x[j] * inv_tau;
        } else if (std::isfinite(l) && h == infty) {
            // For variables with only lower bounds.
            original_x[j] = l + x[j] * inv_tau;
        }
    }

    double objetivo = cs.dot(original_x);
    lambda = lambda * inv_tau;

    // remove SparseSolver
    // ls.~SparseSolver();

    // std::cout << objetivo << std::endl;

    // dual objective
    // double dual_obj = b.dot(lambda);
    // std::cout << dual_obj << std::endl;

    return std::make_tuple(x, lambda, s, objetivo);
}

PYBIND11_MODULE(ipy_selfdual, m) {
    m.def("run_optimization", &run_optimization,
          "A function to run the optimization");
}
