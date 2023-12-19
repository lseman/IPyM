/*
 * Project: IPM for BCP
 * Author: Laio Oriel Seman
 * Email: laio@ieee.org
 */


#include "cholmod.h"

//#define EIGEN_USE_MKL_ALL

// create define to decide between augmented and normal
#define AUGMENTED

#define EIGEN_INITIALIZE_MATRICES_BY_ZERO
#include <iostream>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
// #include <Eigen/CholmodSupport>
// #include <Eigen/IterativeLinearSolvers>
#include <algorithm>
#include <limits>
#include <tuple>

// reader libraries
// #include <omp.h>

#include <stdexcept>
//#include <tuple>

// reader libraries
//#include <fstream>
//#include <sstream>
//#include <string>

#include <vector>

// #include <cusolverDn.h>
// #include <cusolverRf.h>

using namespace Eigen;

Eigen::SparseMatrix<double>
convertToSparseDiagonal(const Eigen::VectorXd &vec) {
    Eigen::SparseMatrix<double> mat(vec.size(), vec.size());
    mat.setIdentity();
    for (int i = 0; i < vec.size(); ++i) {
        mat.coeffRef(i, i) = vec(i);
    }
    return mat;
}

// Define a function to convert a linear programming problem to standard form
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

    //int csize = cs.size();
    //int bsize = bs.size();

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

    // As.setFromTriplets(triplets.begin(), triplets.end());
    As.makeCompressed();
    // cs.conservativeResize(m + n_ubounds + num_slacks);
    // cs.tail(num_slacks) = Eigen::VectorXd::Zero(num_slacks);
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
    //Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> F;
    bool firstFactorization = true;

    cholmod_common c;
    cholmod_factor *L;

    SparseSolver() {
        cholmod_l_start(&c);
        c.useGPU = 0; // Use this line instead of &c->useGPU = 1;

        c.nmethods = 1;
        c.method[0].ordering = CHOLMOD_METIS;

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
            cholmod_l_free_factor(&L, &c);
        }
        cholmod_l_finish(&c);
    }

    void factorizeMatrix(
        const Eigen::SparseMatrix<double, Eigen::RowMajor, long> &matrix) {
        cholmod_sparse A = viewAsCholmod(matrix);

        if (firstFactorization) {
            if (L) {
                cholmod_l_free_factor(&L, &c);
            }
            L = cholmod_l_analyze(&A, &c);
            firstFactorization = false;
        }

        cholmod_l_factorize(&A, L, &c);
    }

    Eigen::VectorXd solve(const Eigen::VectorXd &rhs) {
        cholmod_dense b = viewAsCholmod(rhs);
        cholmod_dense *x = cholmod_l_solve(CHOLMOD_A, L, &b, &c);

        Eigen::VectorXd result = viewAsEigen(x);
        cholmod_l_free_dense(&x, &c);
        return result;
    }

private:
    static cholmod_sparse viewAsCholmod(
        const Eigen::SparseMatrix<double, Eigen::RowMajor, long> &matrix) {
        cholmod_sparse result;
        result.nrow = matrix.rows();
        result.ncol = matrix.cols();
        result.p = const_cast<long *>(matrix.outerIndexPtr());
        result.i = const_cast<long *>(matrix.innerIndexPtr());

        result.x = const_cast<double *>(matrix.valuePtr());
        result.z = nullptr;
        result.stype = -1;
        result.itype = CHOLMOD_INT;
        result.xtype = CHOLMOD_REAL;
        result.dtype = CHOLMOD_DOUBLE;
        result.sorted = 1;
        result.packed = 1;
        return result;
    }

    static cholmod_dense viewAsCholmod(const Eigen::VectorXd &vector) {
        cholmod_dense result;
        result.nrow = vector.size();
        result.ncol = 1;
        result.nzmax = vector.size();
        result.d = vector.size();
        result.x = const_cast<double *>(vector.data());
        result.z = nullptr;
        result.xtype = CHOLMOD_REAL;
        result.dtype = CHOLMOD_DOUBLE;
        return result;
    }

    static Eigen::VectorXd viewAsEigen(cholmod_dense *vector) {
        return Eigen::VectorXd::Map(reinterpret_cast<double *>(vector->x),
                                    vector->nrow);
    }
};

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

    // Assuming the block matrix size is known, you can directly construct the
    // matrix `S`
    Eigen::SparseMatrix<double> S_(ls.n + ls.m, ls.n + ls.m);

    typedef Eigen::Triplet<double> Triplet;
    std::vector<Triplet> tripletList;

    // For topLeft matrix
    for (int k = 0; k < topLeft.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(topLeft, k); it;
             ++it) {
            tripletList.push_back(Triplet(it.row(), it.col(), it.value()));
        }
    }

    // For topRight matrix
    for (int k = 0; k < topRight.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(topRight, k); it;
             ++it) {
            tripletList.push_back(
                Triplet(it.row(), it.col() + topLeft.cols(), it.value()));
        }
    }

    // For bottomLeft matrix
    for (int k = 0; k < bottomLeft.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(bottomLeft, k); it;
             ++it) {
            tripletList.push_back(
                Triplet(it.row() + topLeft.rows(), it.col(), it.value()));
        }
    }

    // For bottomRight matrix
    for (int k = 0; k < bottomRight.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(bottomRight, k); it;
             ++it) {
            tripletList.push_back(Triplet(it.row() + topRight.rows(),
                                          it.col() + bottomLeft.cols(),
                                          it.value()));
        }
    }

    // Finally, set the values from the triplets
    S_.setFromTriplets(tripletList.begin(), tripletList.end());
    // S_.makeCompressed();

    ls.S = S_;
    // Factorize
    ls.factorizeMatrix(ls.S);
#endif
}

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

    int total = ls.n + ls.m;

    for (int i = 0; i < total; i++) {
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

void update_residuals(Residuals &res, const VectorXd &x, const VectorXd &lambda,
                      const VectorXd &s, const VectorXd &v, const VectorXd &w,
                      const MatrixXd &A, const VectorXd &b, const VectorXd &c,
                      const VectorXd &ubv, const VectorXi &ubi,
                      const VectorXd &vbv, const VectorXi &vbi, double tau,
                      double kappa) {
    // Calculate rp and its norm
    // primal residual
    res.rp.noalias() =  tau * b - A * x;
    res.rpn = res.rp.norm();

    // Calculate ru and its norm
    // uper bound residual
    res.ru.noalias() = -v;
    for (int i = 0; i < ubi.size(); ++i) {
        res.ru(i) -= x(ubi(i));
    }
    res.ru.array() += tau * ubv.array();
    res.run = res.ru.norm();

    // calculate rv and its norm
    // lower bound residuals
    // res.rl.noalias() = -l;
    // for (int i : vbi) {
    //    res.rl(i) += x(i);
    //}
    // res.rln = res.rl.norm();

    // Calculate rd and its norm
    // dual residual
    res.rd.noalias() = tau * c - (A.transpose() * lambda + s);
    for (int i = 0; i < ubi.size(); ++i) {
      res.rd(ubi(i)) += x(ubi(i));
    }

    res.rdn = res.rd.norm();

    // Calculate rg and its norm
    // gap residual
    res.rg = kappa + c.dot(x) - b.dot(lambda) + ubv.dot(w);
    res.rgn = std::sqrt(
        res.rg *
        res.rg); // Since rg is a scalar, its norm is the absolute value

    // l = xl
    // v = xu
    // w = zu
    // s = zl
}

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

// create function to see solving status

void solve_augsys(Eigen::VectorXd &delta_x, Eigen::VectorXd &delta_y,
                  Eigen::VectorXd &delta_z, SparseSolver &ls,
                  const Eigen::VectorXd &theta_vw, const Eigen::VectorXi &ubi,
                  const Eigen::VectorXd &xi_p, const Eigen::VectorXd &xi_d,
                  const Eigen::VectorXd &xi_u) {
    // Initialize delta_z to zero
    delta_z.setZero(ubi.size());

    // Create a copy of xi_d and modify it
    Eigen::VectorXd _xi_d = xi_d;
    Eigen::VectorXd xi_u_theta = xi_u.cwiseProduct(theta_vw);
    for (int i = 0; i < ubi.size(); ++i) {
        _xi_d(ubi(i)) -= xi_u_theta(i);
    }

    // Call the function to solve the augmented system
    solve_augmented_system(delta_x, delta_y, ls, xi_p, _xi_d);

    // Update delta_z
    delta_z -= xi_u;
    for (int i = 0; i < ubi.size(); ++i) {
        delta_z(i) += delta_x(ubi(i));
    }
    delta_z = delta_z.cwiseProduct(theta_vw);
}

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

    // Convert to standard form
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
    double delta_0;

    // keep old lambda and x for results
    // Eigen::VectorXd x_old = x;
    // Eigen::VectorXd lambda_old = lambda;

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
        //_p = std::fmax(res.rpn / (tau * (1.0 + b.norm())), res.run / (tau *
        //(1.0 +
        // ubv.norm())));

        _p = std::fmax(res.rp.lpNorm<Eigen::Infinity>() / (tau * (1.0 + b.lpNorm<Eigen::Infinity>())),
                       res.run / (tau * (1.0 + ubv.lpNorm<Eigen::Infinity>())));
        // calculate _d = |rd| / (τ * (1 + |c|))
        _d = res.rd.lpNorm<Eigen::Infinity>() / (tau * (1.0 + c.lpNorm<Eigen::Infinity>()));

        _g = std::abs(c.dot(x) - b.dot(lambda)) / (tau +
                                                   std::abs(b.dot(lambda)));

        // calculate _g = |cᵀx - bᵀλ| / (τ * (1 + |bᵀλ|))
        //_g = std::abs(c.dot(x) - b.dot(lambda)) / (tau +
        //std::abs(b.dot(lambda)));

        //std::cout << c.dot(x) << "             " << std::abs(b.dot(lambda)) << std::endl;
        //std::cout << "mu" << mu << "    tau/kappa   " << tau / kappa << std::endl;
        //std::cout << "p "<<  _p << "   -    " << "d " << _d << "           " << "g " << _g <<  "           " << tau << std::endl;
        //std::cout << "objetivo: " << c.dot(x) << std::endl;
        // std::cout << "p: " << _p << std::endl;
        // std::cout << "d: " << _d << std::endl;
        // std::cout << "g: " << _g << std::endl;
        //  check optimality
        if (_d < tol) {
            break;
        }
        // check infesibility
        if ((mu < 1e-6) || (tau/kappa < 1e-6)) {
            break;
        }

        // scaling factors
        theta_vw.array() = w.array() / v.array();
        theta_xs.array() = s.array() / x.array();

        for (int i = 0; i < ubi.size(); i++) {
            int index = ubi[i];
            theta_xs[index] += theta_vw[i];
        }

        // update regularizations
        // Element-wise operations for regP and regD
        regP = regP.array() / 10.0;
        regP = regP.array().max(r_min);
        regD = regD.array() / 10.0;
        regD = regD.array().max(r_min);
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
            alpha_ = alpha;
            ncor += 1;
            alpha_ = std::min(1.0, 2.0 * alpha);

            // Compute target cross-products
            mu_l = beta * mu * gamma;
            mu_u = gamma * mu / beta;

            // check if the targets are feasible
            Eigen::VectorXd xs = x;
            xs.noalias() += alpha_ * Delta_x;
            xs.array() *= (s + alpha_ * Delta_s).array();

            Eigen::VectorXd vw = v;
            vw.noalias() += alpha_ * Delta_v;
            vw.array() *= (w + alpha_ * Delta_w).array();

            t_xs_lower = (xs.array() < mu_l).select(mu_l - xs.array(), 0);
            t_xs_upper = (xs.array() > mu_u).select(mu_u - xs.array(), 0);

            t_vw_lower = (vw.array() < mu_l).select(mu_l - vw.array(), 0);
            t_vw_upper = (vw.array() > mu_u).select(mu_u - vw.array(), 0);

            t_xs = (t_xs_lower + t_xs_upper).matrix();
            t_vw = (t_vw_lower + t_vw_upper).matrix();

            // define t0  as tau * kappa if tau * kappa ar between mu_l and mu_u
            taukappa =
                (tau + alpha_ * Delta_tau) * (kappa + alpha_ * Delta_kappa);

            if (taukappa < mu_l) {
                t0 = mu_l - taukappa;
            } else if (taukappa > mu_u) {
                t0 = mu_u - taukappa;
            } else {
                t0 = 0;
            }

            // correct xs, vw and t0
            t_xs =
                t_xs.array() - (t_xs.sum() + t_vw.sum() + t0) / (nv + nu + 1);
            t_vw =
                t_vw.array() - (t_xs.sum() + t_vw.sum() + t0) / (nv + nu + 1);
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
                // std::cout << "corrected!" << std::endl;
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

        // x_old = x;
        // lambda_old = lambda;

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
            free_var += 1;
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

    //std::cout << objetivo << std::endl;

    // dual objective
    //double dual_obj = b.dot(lambda);
    //std::cout << dual_obj << std::endl;

    return std::make_tuple(x, lambda, s, objetivo);
}

PYBIND11_MODULE(ipy_selfdual, m) {
    m.def("run_optimization", &run_optimization,
          "A function to run the optimization");
}