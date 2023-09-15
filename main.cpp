#include "mpi.h"
#include <petsc.h>
#include <Eigen/Eigen>
#include "OptPara.h"
#include "InputMeshData.h"
#include "SolveFem.h"
#include "PDEFilter.h"
#include "MMA.h"
#include "Write2Vtu.h"

using namespace std;
using namespace Eigen;

/*
Authors: Yu Wang, Kun Wang, November 2022

Disclaimer:
The authors reserve all rights for the programs.The programs may be distributed and
used for academic and educational purposes.The authors do not guarantee that the
code is free from errors, and they shall not be liable in any event caused by the use
of the programs.

The Code of MMA.h\cpp
The code of MMA.h\cpp is copied from TopOpt_in_PETSc code. The original code can be found here: https://github.com/topopt/TopOpt_in_PETSc 
(N. Aage, E. Andreassen, B. S. Lazarov (2014), Topology optimization using PETSc: An easy-to-use, fully parallel, open source topology optimization framework, 
Structural and Multidisciplinary Optimization, DOI 10.1007/s00158-014-1157-0)

Acknowledgment:
The authors would like to thank Niels Aage for providing the C++ implementation of his
Method of Moving Asymptotes in parallel which was used in this work.
*/

static char help[] = "FEA based 3D U-Opt of minimum compliance objective using PETSc and Eigen\n";

int main(int argc, char** argv)
{
    PetscErrorCode ierr;
    PetscMPIInt rank,size;

    //Initialize Petsc and MPI
    ierr=PetscInitialize(&argc, &argv, PETSC_NULL, help); if(ierr) return ierr;

    double t_ini,t_fin;
    t_ini = MPI_Wtime();

    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &size );

    //Initialize optimized parameter values
    OptPara* opt = new OptPara();
    //Read the mesh data
    InputMeshData* mesh = new InputMeshData(opt->XPhys);
    //FEA, obj and sens
    SolveFem* fem = new SolveFem(mesh->elementEigen, mesh->eleXCoord, mesh->eleYCoord, mesh->eleZCoord, opt->Vol,opt->XPhys);
    //PDE filter
    PDEFilter* pde = new PDEFilter(mesh->elementEigen, mesh->eleXCoord, mesh->eleYCoord, mesh->eleZCoord, opt->rmin,opt->XPhys);
    //MMA
    MMA* mma;
    PetscInt itr = 0;
    opt->SetMMA(&itr, &mma);
    //Initialize vtu output class
    Write2Vtu* vtu;
    //Filtering the initial structure
    ierr = pde->Filter(opt->X, opt->XFiled,opt->Xmin);
    //Heaviside projection
    ierr = pde->HeavisideFilter(opt->XFiled, opt->XPhys, opt->x0, opt->beta0, opt->betamax, itr, opt->Xmin, opt->proj);CHKERRQ(ierr);
    PetscScalar ch = 1.0;
    double t1, t2;

    while (itr < opt->maxItr && ch>0.01) 
    {
        itr++;

        t1 = MPI_Wtime();

        //compute objective and sensitivity
        ierr = fem->ComputeObjectiveAndSensitivities(&(opt->fx), opt->gx, opt->dfdx, opt->m, opt->dgdx, opt->XPhys, opt->Emin, opt->Emax, opt->penal, opt->volfrac,
                                                       mesh->eleXCoord, mesh->eleYCoord, mesh->eleZCoord, opt->Vol, mesh->loadDofs, mesh->load, mesh->fixedDofs, mesh->vertex_loc);
        CHKERRQ(ierr);

        //output initial configuration
        if(itr==1)
        {
            vtu->WriteResults(mesh->vertex, mesh->element, opt->XPhys, fem->U, 0);
        }

        //sensitivity of the projected design variables
        if(opt->proj)
        {
            ierr = pde->ComputeProjectedSens(opt->XFiled, opt->dfdx, opt->dgdx, opt->m, opt->x0);CHKERRQ(ierr);
        }
        //sensitivity of the filtered design variables
        ierr = pde->ComputeFilteredSens(opt->dfdx,opt->dgdx,opt->m);CHKERRQ(ierr);

        //update design variables with MMA
        ierr = mma->SetOuterMovelimit(opt->Xmin, opt->Xmax, opt->movlim, opt->X, opt->xmin, opt->xmax);
        CHKERRQ(ierr);
        ierr = mma->Update(opt->X, opt->dfdx, opt->gx, opt->dgdx, opt->xmin, opt->xmax);
        CHKERRQ(ierr);

        ch = mma->DesignChange(opt->X, opt->Xold);

        //PDE filter
        ierr = pde->Filter(opt->X, opt->XFiled,opt->Xmin);CHKERRQ(ierr);

        //Heaviside projection
        ierr = pde->HeavisideFilter(opt->XFiled, opt->XPhys, opt->x0, opt->beta0, opt->betamax, itr, opt->Xmin, opt->proj);CHKERRQ(ierr);

        t2 = MPI_Wtime();

        //Print to screen
        PetscPrintf(PETSC_COMM_WORLD,"It.: %i, fx: %f, gx[0]: %f, ch.: %f, time: %f\n",itr,  opt->fx, opt->gx[0], ch,  t2 - t1);

        //output every 20 iterations
       if(itr%20==0)
       {
           vtu->WriteResults(mesh->vertex, mesh->element, opt->XPhys, fem->U, itr);
       }
    }

    //output final configuration
    vtu->WriteResults(mesh->vertex, mesh->element, opt->XPhys, fem->U, -1);

    delete opt;
    delete mesh;
    delete fem;
    delete pde;
    delete mma;
    delete vtu;

    t_fin = MPI_Wtime();
    PetscPrintf(PETSC_COMM_WORLD,"total time: %f hours\n",(t_fin - t_ini)/3600);

    //Finalize Petsc and MPI
    ierr = PetscFinalize();CHKERRQ(ierr);
    return 0;
}