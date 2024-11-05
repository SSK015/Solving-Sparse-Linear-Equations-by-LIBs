#include <HYPRE.h>
#include <HYPRE_config.h>
#include <HYPRE_krylov.h>
#include <HYPRE_parcsr_ls.h>
#include <HYPRE_utilities.h>

HYPRE_Solver pcg_solver;

HYPRE_Int max_iter = 1000;
HYPRE_Real tol = 1.e-8;
HYPRE_Real atol1 = 0.;
HYPRE_Int ioutdat;

int main(int argc, char* argv[]) {
    int my_id, num_procs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    HYPRE_ParCSRBiCGSTABCreate(MPI_COMM_WORLD, &pcg_solver);
    HYPRE_BiCGSTABSetMaxIter(pcg_solver, max_iter);
    HYPRE_BiCGSTABSetTol(pcg_solver, tol);
    HYPRE_BiCGSTABSetAbsoluteTol(pcg_solver, atol1);
    HYPRE_BiCGSTABSetLogging(pcg_solver, ioutdat);
    HYPRE_BiCGSTABSetPrintLevel(pcg_solver, ioutdat);

    return 0;
}