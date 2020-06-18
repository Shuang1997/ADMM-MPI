#include <math.h>
#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <string>
#include "mpi.h"

void soft_threshold(Eigen::VectorXd& v, double k) {
    double vi;
    for (int i = 0; i < v.rows(); i++)
    {
        vi = v[i];
        if (vi > k)
            v[i] = vi - k;
        else if (vi < -k)
            v[i] = vi + k;
        else
            v[i] = 0;
    }
}

double objective(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> A, Eigen::VectorXd b,
                 double lambda, Eigen::VectorXd z)
{
    double obj = 0;
    Eigen::VectorXd Azb = A * z - b;
    double Azb_nrm2 = Azb.norm() * Azb.norm();
    obj = 0.5 * Azb_nrm2 + lambda * z.lpNorm<1>();
    return obj;
}

int main(int argc, char** argv)
{
    const int MAX_ITER = 50;
    const double RELTOL = 1e-2;
    const double ABSTOL = 1e-4;

    int rank;
    int size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    std::cout << "Processor " << rank << " out of " << size << std::endl;

    double N = (double)size;

    //int skinny;
    std::ifstream file_in;
    int m = 40, n = 500;
    int row, col;

    std::string s;
    double entry;

    // Read A.dat
    std::string file_name;
    file_name = "../data/A" + std::to_string(rank + 1) + ".dat";
    file_in.open(file_name);

    if (!file_in.is_open())
    {
        std::cerr << "[" << rank << "] ERROR: " << file_name
                  << " does not exist, exiting" << std::endl;
        exit(EXIT_FAILURE);
    }

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> A;
    A = Eigen::MatrixXd::Zero(40, 500);

    for (int i = 0; i < m*n; i++)
    {
        row = i % m;
        col = floor(i/m);

        file_in >> s;
        entry = std::stod(s);
        A(row,col) = entry;
    }
    file_in.close();

    // Read b.dat
    file_name = "../data/b" + std::to_string(rank + 1) + ".dat";
    file_in.open(file_name);
    if (!file_in.is_open())
    {
        std::cerr << "[" << rank << "] ERROR: " << file_name
                  << " does not exist, exiting" << std::endl;
        exit(EXIT_FAILURE);
    }
    Eigen::VectorXd b(40);
    for (int i = 0; i < m; i++)
    {
        file_in >> s;
        entry = std::stod(s);
        b(i) = entry;
    }
    file_in.close();

    double rho = 1.0;

    Eigen::VectorXd x(n);
    Eigen::VectorXd u(n);
    Eigen::VectorXd z(n);
    Eigen::VectorXd y(n);
    Eigen::VectorXd r(n);
    Eigen::VectorXd zprev(n);
    Eigen::VectorXd zdiff(n);

    x.setZero(n);
    u.setZero(n);
    z.setZero(n);
    y.setZero(n);
    r.setZero(n);
    zprev.setZero(n);
    zdiff.setZero(n);

    Eigen::VectorXd q(n);
    Eigen::VectorXd w(n);
    // Eigen::VectorXd Aq(m);
    Eigen::VectorXd p(m);

    q.setZero(n);
    w.setZero(n);
    // Aq.setZero(m);
    p.setZero(m);

    //Eigen::VectorXd Atb(n);

    double send[3];
    double recv[3];

    double nxstack = 0;
    double nystack = 0;
    double prires = 0;
    double dualres = 0;
    double eps_pri = 0;
    double eps_dual = 0;

    //Atb = A.transpose() * b;
    double lambda = 0.5;
    if (rank == 0)
        std::cout << "using lambda: " << lambda << std::endl;

    Eigen::Matrix<double, 40, 40> I;
    I.setIdentity(40, 40);
    Eigen::Matrix<double, 40, 40> L;

    L = I + 1/rho * A * A.transpose();

    int iter = 0;
    if (rank == 0)
        std::cout << "# r_norm    eps_pri"
                  << "    s_norm    eps_dual"
                  << "    objective" << std::endl;

    while (iter < MAX_ITER)
    {
        // u-update: u = u + x - z
        u = u + x -z;

        // x = (A^T * A + rho * I)^-1 * (A^T * b + rho * (z - u))
        q = A.transpose() * b + rho * (z - u);

        p = L.llt().solve(A * q);
        x = A.transpose() * p;
        x *= -1/(rho*rho);
        q *= 1/rho;
        x += q;

        w = x + u;

        send[0] = r.dot(r);
        send[1] = x.dot(x);
        send[2] = u.dot(u);
        send[2] /= (rho * rho);

        zprev = z;

        MPI_Allreduce(w.data(), z.data(), n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(send,     recv,     3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        prires = sqrt(recv[0]);
        nxstack = sqrt(recv[1]);
        nystack = sqrt(recv[2]);

        z *= 1/N;
        soft_threshold(z, lambda/(N*rho));

        // Termination checks

        // dual residual
        zdiff = z - zprev;
        // ||s^k||_2^2 = N * rho^2 * ||z - zprev||_2^2
        dualres = sqrt(N) * rho * zdiff.norm();

        // compute primal and dual feasibility tolerances
        eps_pri = sqrt(n*N) * ABSTOL + RELTOL * fmax(nxstack, sqrt(N)*z.norm());
        eps_dual = sqrt(n*N) * ABSTOL + RELTOL * nystack;

        if (rank == 0)
        {
            std::cout << iter << ",  " << prires << ",  " << eps_pri << ",  "
                      << dualres << ",  " << eps_dual << ",  "
                      << objective(A, b, lambda, z) << std::endl;
        }

        if (prires <= eps_pri && dualres <= eps_dual)
        {
            break;
        }

        // Compute residual: r = x - z
        r = x - z;

        iter++;
    }
    MPI_Finalize();

    return EXIT_SUCCESS;
}
