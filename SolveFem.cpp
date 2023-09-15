#include "SolveFem.h"

using namespace Eigen;
using namespace std;

SolveFem::SolveFem(MatrixXi elementEigen, MatrixXd eleXCoord, MatrixXd eleYCoord, MatrixXd eleZCoord, Vec Vol, Vec X)
{
    nu = 0.3;

    PetscBool flg;
    PetscOptionsGetReal(NULL, NULL, "-nu", &nu, &flg);

    GetDofCoord(elementEigen,dofCoord,X);
    ComputeTotalVol(eleXCoord,eleYCoord,eleZCoord,Vol);
}

SolveFem::~SolveFem()
{
    //Free memory
    if(F!=NULL)
    {
       VecDestroy(&F);
    }
    if(U!=NULL)
    {
       VecDestroy(&U);
    }
    if(K!=NULL)
    {
       MatDestroy(&K);
    }
    if(ksp!=NULL)
    {
       KSPDestroy(&ksp);
    }
}

PetscErrorCode SolveFem::GetDofCoord(MatrixXi elementEigen, MatrixXi& dofCoord, Vec X)
{
    PetscErrorCode ierr;
    PetscInt ldim;
    ierr = VecGetLocalSize(X, &ldim); CHKERRQ(ierr);
    PetscInt low, high;
    ierr = VecGetOwnershipRange(X, &low, &high); CHKERRQ(ierr);
    //Coordinate matrix of degrees of freedom, the i-th row is the i-th element's degrees of freedom
    dofCoord.resize(high - low,COL_1 * COL_2);
    for (int i = 0; i < ldim; i++)
    {
         //The corresponding element displacement field is [u1 u2...u8; v1 v2...v8; w1 w2...w8]
        dofCoord.row(i) << 3*elementEigen(i,0),3*elementEigen(i,1),3*elementEigen(i,2),3*elementEigen(i,3),3*elementEigen(i,4),3*elementEigen(i,5),
                           3*elementEigen(i,6),3*elementEigen(i,7),3*elementEigen(i,0)+1,3*elementEigen(i,1)+1,3*elementEigen(i,2)+1,3*elementEigen(i,3)+1,
                           3*elementEigen(i,4)+1,3*elementEigen(i,5)+1,3*elementEigen(i,6)+1,3*elementEigen(i,7)+1,3*elementEigen(i,0)+2,3*elementEigen(i,1)+2,
                           3*elementEigen(i,2)+2,3*elementEigen(i,3)+2,3*elementEigen(i,4)+2,3*elementEigen(i,5)+2,3*elementEigen(i,6)+2,3*elementEigen(i,7)+2;
    }

    return ierr;
}

PetscErrorCode SolveFem::ComputeTotalVol(MatrixXd eleXCoord, MatrixXd eleYCoord, MatrixXd eleZCoord, Vec Vol)
{
    PetscErrorCode ierr;

    PetscScalar* vol;
    ierr = VecGetArray(Vol,&vol);CHKERRQ(ierr);

    eleCoord.resize(COL_2,3);

    PetscInt ldim;
    ierr = VecGetLocalSize(Vol,&ldim);CHKERRQ(ierr);

    TV=0.0;

    for(PetscInt ee=0;ee<ldim;ee++)
    {
        //Element coordinate matrix
        eleCoord << eleXCoord.row(ee).transpose(), eleYCoord.row(ee).transpose(), eleZCoord.row(ee).transpose();
        //Volume of each element
        vol[ee] = ComputeVolOfHex(eleCoord);
        //Total volume
        TV += vol[ee];
    }
    // Allreduce TV
    PetscScalar tmp = TV;
    TV = 0.0;
    MPI_Allreduce(&tmp, &TV, 1, MPIU_SCALAR, MPI_SUM, PETSC_COMM_WORLD); 

    VecRestoreArray(Vol,&vol);
    return ierr;
}

PetscErrorCode SolveFem::ComputeFem(Vec XPhys,PetscScalar Emin,PetscScalar Emax,PetscScalar penal,MatrixXd eleXCoord,MatrixXd eleYCoord,
                                    MatrixXd eleZCoord, int **loadDofs, double **load, int *fixedDofs, double *vertex_loc)
{
    PetscErrorCode ierr;

    PetscInt rank,size;
    MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
    MPI_Comm_size(PETSC_COMM_WORLD,&size);
    PetscInt rank_ne = ROW_1/size;
    if(rank==size-1) rank_ne = ROW_1-rank_ne*(size-1);
    //Create sparse matrix K
    MatCreate(PETSC_COMM_WORLD, &K);
    MatSetSizes(K, 3 * rank_ne, 3 * rank_ne, 3 * ROW_1, 3 * ROW_1);
    MatSetType(K,MATMPIAIJ);
    MatSetBlockSize(K,3);//set block size for GAMG preconditioner
    MatSetUp(K);
    MatZeroEntries(K);

    //Matrix pre-allocation£¬should be changed with mesh
    MatMPIAIJSetPreallocation(K,105,NULL,105,NULL);
   
    //Create vector
    VecCreate(PETSC_COMM_WORLD, &U);
    VecSetType(U,VECMPI);
    PetscObjectSetName((PetscObject)U, "Solution");
    VecSetSizes(U, 3 * rank_ne, 3 * ROW_1);//must have the same distribution with matrix K
    VecSetFromOptions(U);
    VecDuplicate(U, &F);

    ierr = AssembleFem(XPhys,Emin,Emax,penal,eleXCoord,eleYCoord,eleZCoord,loadDofs,load,fixedDofs);CHKERRQ(ierr);

    PC pc;
    //Create KSP solver,GAMG preconditioner and FGMRES solver
    KSPCreate(PETSC_COMM_WORLD, &ksp);
    KSPSetOperators(ksp, K, K);
    KSPSetType(ksp, KSPFGMRES);
    KSPSetInitialGuessNonzero(ksp, PETSC_TRUE);

    KSPGetPC(ksp,&pc);
    PCSetType(pc,PCGAMG);
    PCGAMGSetNlevels(pc,4);
    PCSetCoordinates(pc,3,rank_ne,vertex_loc);

    KSPSetTolerances(ksp,1.e-5,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);
    KSPSetFromOptions(ksp);
    KSPSetUp(ksp);

    //Solve
    KSPSolve(ksp, F, U);

    //norm of error
    Vec F1;
    VecDuplicate(U, &F1);
    PetscInt its;
    PetscReal norm1,norm2;
    VecNorm(F, NORM_2, &norm1);
    MatMult(K, U, F1);
    VecAXPY(F, -1.0, F1);
    VecNorm(F, NORM_2, &norm2);
    KSPGetIterationNumber(ksp,&its);

    PetscPrintf(PETSC_COMM_WORLD,"Norm of error %e, Iterations %d\n",norm2/norm1,its);

    VecDestroy(&F1);

    return ierr;
}

PetscErrorCode SolveFem::ComputeObjectiveAndSensitivities(PetscScalar* fx, PetscScalar* gx, Vec dfdx, PetscInt m, Vec* dgdx, Vec XPhys, PetscScalar Emin, PetscScalar Emax, PetscScalar penal, PetscScalar volfrac, MatrixXd eleXCoord, MatrixXd eleYCoord,
                                                          MatrixXd eleZCoord, Vec Vol, int **loadDofs, double **load, int *fixedDofs, double *vertex_loc)
{
    PetscErrorCode ierr;

  
    ierr = ComputeFem(XPhys,Emin,Emax,penal,eleXCoord,eleYCoord,eleZCoord,loadDofs,load,fixedDofs,vertex_loc);
    CHKERRQ(ierr);

    //Scatter U
    Vec V_SEQ;
    VecScatter ctx;
    VecScatterCreateToAll(U, &ctx, &V_SEQ);
    VecScatterBegin(ctx, U, V_SEQ, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(ctx, U, V_SEQ, INSERT_VALUES, SCATTER_FORWARD);
    //Pointer to the displacement vector
    PetscScalar* vp;
    VecGetArray(V_SEQ, &vp);

    //Pointer to the design variable vector
    PetscScalar* xp;
    ierr = VecGetArray(XPhys, &xp);
    CHKERRQ(ierr);

    //Pointer to the sens of obj vector
    PetscScalar* df;
    ierr = VecGetArray(dfdx, &df);
    CHKERRQ(ierr);

    //Pointer to the sens of constraint vector
    PetscScalar** dg;
    ierr = VecGetArrays(dgdx, m, &dg);
    CHKERRQ(ierr);

    //Pointer to the volume vector
    PetscScalar* vol;
    ierr = VecGetArray(Vol, &vol);
    CHKERRQ(ierr);

    PetscInt edof[COL_1 * COL_2] = { 0 };
    fx[0] = 0.0;
    for(PetscInt i = 0; i < m; i++)
    {
        gx[i] = 0.0;
    }

    PetscInt ldim;
    ierr = VecGetLocalSize(XPhys,&ldim);CHKERRQ(ierr);
 
    for (PetscInt ee = 0; ee < ldim; ee++)
    {
        //Element coordinate matrix
        eleCoord << eleXCoord.row(ee).transpose(), eleYCoord.row(ee).transpose(), eleZCoord.row(ee).transpose();

        //Calculate the eight-node isoparametric element stiffness matrix
        Hex8Isoparametric(eleCoord, ke, zero);
        ChangeEigenToPetsc(ke, KE);

        for (PetscInt j = 0; j < COL_1 * COL_2; j++)
        {
            edof[j] = dofCoord(ee, j);
        }       
        //Use SIMP interpolation method
        PetscScalar uKu = 0.0;
        for (PetscInt k = 0; k < COL_1 * COL_2; k++)
        {
            for (PetscInt h = 0; h < COL_1 * COL_2; h++)
            {
                uKu += vp[edof[k]] * KE[k * COL_1 * COL_2 + h] * vp[edof[h]];
            }
        }
        fx[0]+= (Emin + PetscPowScalar(xp[ee], penal) * (Emax - Emin)) * uKu;
        //sens of obj
        df[ee] = -1.0 * penal * PetscPowScalar(xp[ee], penal - 1) * (Emax - Emin) * uKu;
        //volume constraints
        gx[0] += xp[ee] * vol[ee];
        //sens of volume constraints
        dg[0][ee] = vol[ee] / TV;
    }

    // Allreduce fx[0]
    PetscScalar tmp = fx[0];
    fx[0] = 0.0;
    MPI_Allreduce(&tmp, &(fx[0]), 1, MPIU_SCALAR, MPI_SUM, PETSC_COMM_WORLD); 

    // Allreduce gx[0]
    PetscScalar tmp2 = gx[0];
    gx[0] = 0.0;
    MPI_Allreduce(&tmp2, &(gx[0]), 1, MPIU_SCALAR, MPI_SUM, PETSC_COMM_WORLD); 

    gx[0] = gx[0] / TV - volfrac;

    VecRestoreArray(XPhys, &xp);
    VecRestoreArray(dfdx, &df);
    VecRestoreArray(Vol,&vol);
    VecRestoreArrays(dgdx,m,&dg);

    VecScatterDestroy(&ctx);
    VecRestoreArray(V_SEQ, &vp);
    VecDestroy(&V_SEQ);

    VecDestroy(&F);
    MatDestroy(&K);
    KSPDestroy(&ksp);

    return ierr;
}

PetscErrorCode SolveFem::AssembleFem(Vec XPhys, PetscScalar Emin, PetscScalar Emax, PetscScalar penal, Eigen::MatrixXd eleXCoord, Eigen::MatrixXd eleYCoord,
                                     Eigen::MatrixXd eleZCoord, int **loadDofs, double **load, int *fixedDofs)
{
    PetscErrorCode ierr;

    MatZeroEntries(K);
    VecZeroEntries(U);
    VecZeroEntries(F);

    //Set the load
    for(int i = 0;i < NUM_LD; i++)
    {
        VecSetValues(F,Load,loadDofs[i],load[i],INSERT_VALUES);
    }
    VecAssemblyBegin(F);
    VecAssemblyEnd(F);

    eleCoord.resize(COL_2, 3), ke.resize(COL_1 * COL_2, COL_1 * COL_2), zero.resize(1, COL_2);
    zero.fill(0);

    PetscScalar* xp;
    ierr=VecGetArray(XPhys, &xp);
    CHKERRQ(ierr);

    //Element degrees of freedom
    PetscInt edof[COL_1 * COL_2] = { 0 };
  
    PetscInt ldim;
    ierr = VecGetLocalSize(XPhys,&ldim);CHKERRQ(ierr);

    for (PetscInt ee = 0; ee < ldim; ee++)
    {
        eleCoord << eleXCoord.row(ee).transpose(), eleYCoord.row(ee).transpose(), eleZCoord.row(ee).transpose();

        Hex8Isoparametric(eleCoord, ke, zero);
        ChangeEigenToPetsc(ke, KE);

        PetscScalar dens = Emin + PetscPowScalar(xp[ee], penal) * (Emax - Emin);
 
        for (PetscInt k = 0; k < COL_1 * COL_2 * COL_1 * COL_2; k++)
        {
            Ke[k] = KE[k] * dens;
        }

        for (int j = 0; j < COL_1 * COL_2; j++)
        {
            edof[j] = dofCoord(ee, j);         
        }
        //Assemble global stiffness matrix
        ierr = MatSetValues(K, COL_1 * COL_2, edof, COL_1 * COL_2, edof, Ke, ADD_VALUES);
        CHKERRQ(ierr);
    }

    MatAssemblyBegin(K, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(K, MAT_FINAL_ASSEMBLY);

    //Impose the Dirichlet boundary condition
    ierr = MatZeroRowsColumns(K, StrucDofs, fixedDofs, 1, 0, 0);
    CHKERRQ(ierr);

    PetscScalar edofToZero[StrucDofs] = { 0 };

    ierr = VecSetValues(F, StrucDofs, fixedDofs, edofToZero, INSERT_VALUES);
    CHKERRQ(ierr);

    VecAssemblyBegin(F);
    VecAssemblyEnd(F);

    VecRestoreArray(XPhys, &xp);
    return ierr;
}

int SolveFem::Hex8Isoparametric(MatrixXd eleCoord, MatrixXd& ke, MatrixXd zero)
{
    //Calculate the stiffness matrix of the eight-node isoparametric element with dianonal numbering element, not multiplied by E
    /*  Shape function for counterclockwise numbering element
        % N1 = (1 - s) * (1 - t) * (1 - n) / 8; % (-1, -1, -1)
        % N2 = (1 + s) * (1 - t) * (1 - n) / 8; % (1, -1, -1)
        % N3 = (1 + s) * (1 + t) * (1 - n) / 8; % (1, 1, -1)
        % N4 = (1 - s) * (1 + t) * (1 - n) / 8; % (-1, 1, -1)
        % N5 = (1 - s) * (1 - t) * (1 + n) / 8; % (-1, -1, 1)
        % N6 = (1 + s) * (1 - t) * (1 + n) / 8; % (1, -1, 1)
        % N7 = (1 + s) * (1 + t) * (1 + n) / 8; % (1, 1, 1)
        % N8 = (1 - s) * (1 + t) * (1 + n) / 8; % (-1, 1, 1)*/

    /*  Shape function for dianonal numbering element
        % N1 = (1 - s) * (1 - t) * (1 - n) / 8; % (-1, -1, -1)
        % N2 = (1 + s) * (1 - t) * (1 - n) / 8; % (1, -1, -1)
        % N3 = (1 - s) * (1 + t) * (1 - n) / 8; % (-1, 1, -1)
        % N4 = (1 + s) * (1 + t) * (1 - n) / 8; % (1, 1, -1)
        % N5 = (1 - s) * (1 - t) * (1 + n) / 8; % (-1, -1, 1)
        % N6 = (1 + s) * (1 - t) * (1 + n) / 8; % (1, -1, 1)
        % N7 = (1 - s) * (1 + t) * (1 + n) / 8; % (-1, 1, 1)
        % N8 = (1 + s) * (1 + t) * (1 + n) / 8; % (1, 1, 1)*/

    //Gaussian points and Gaussian weights
    MatrixXd G(1, 3), W(1, 3);
    G(0, 0) = -0.775, G(0, 1) = 0, G(0, 2) = 0.775;
    W(0, 0) = 0.556, W(0, 1) = 0.889, W(0, 2) = 0.556;

    ke.fill(0);

    //Elasticity Matrix D
    double lambda = nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    double mu = 1.0 / (2.0 * (1.0 + nu));
    MatrixXd D(6, 6);

    D << lambda + 2.0 * mu, lambda, lambda, 0.0, 0.0, 0.0, lambda, lambda + 2.0 * mu, lambda, 0.0, 0.0, 0.0,
        lambda, lambda, lambda + 2.0 * mu, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, mu, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, mu, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, mu;

    MatrixXd  Ndiff(3, COL_2), Jacob(3, 3), dN(3, COL_2), B(6, COL_1 * COL_2);

    //Calculte element stiffness matrix
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            for (int k = 0; k < 3; k++)
            {
                double xi = G(0, i);
                double eta = G(0, j);
                double zeta = G(0, k);
                Ndiff << -0.125 * (zeta - 1) * (eta - 1), 0.125 * (zeta - 1) * (eta - 1), 0.125 * (zeta - 1) * (eta + 1), -0.125 * (zeta - 1) * (eta + 1), 0.125 * (zeta + 1) * (eta - 1), -0.125 * (zeta + 1) * (eta - 1),
                    -0.125 * (zeta + 1) * (eta + 1), 0.125 * (zeta + 1) * (eta + 1),//row(0)
                    -0.125 * (zeta - 1) * (xi - 1), 0.125 * (zeta - 1) * (xi + 1), 0.125 * (zeta - 1) * (xi - 1), -0.125 * (zeta - 1) * (xi + 1), 0.125 * (zeta + 1) * (xi - 1), -0.125 * (zeta + 1) * (xi + 1),
                    -0.125 * (zeta + 1) * (xi - 1), 0.125 * (zeta + 1) * (xi + 1),//row(1)
                    -0.125 * (xi - 1) * (eta - 1), 0.125 * (xi + 1) * (eta - 1), 0.125 * (xi - 1) * (eta + 1), -0.125 * (xi + 1) * (eta + 1), 0.125 * (xi - 1) * (eta - 1), -0.125 * (xi + 1) * (eta - 1),
                    -0.125 * (xi - 1) * (eta + 1), 0.125 * (xi + 1) * (eta + 1);//row(2)
                //Jacob 3 * 3
                Jacob = Ndiff * eleCoord;
                //dN 3 * 8
                dN = Jacob.inverse() * Ndiff;
                //B 6 * 24
                B << dN.row(0), zero, zero, zero, dN.row(1), zero, zero, zero, dN.row(2), dN.row(1), dN.row(0), zero, zero, dN.row(2), dN.row(1), dN.row(2), zero, dN.row(0);
                //ke 24 * 24
                ke = ke + W(0, i) * W(0, j) * W(0, k) * B.transpose() * D * B * Jacob.determinant();
            }
        } 
    }
    return 0;
}


int SolveFem::ChangeEigenToPetsc(MatrixXd ke, PetscScalar* KE)
{
    for (int i = 0; i < COL_1 * COL_2; i++)
    {
        for (int j = 0; j < COL_1 * COL_2; j++)
        {
            KE[COL_1 * COL_2 * i +j ] = ke(i, j);
        }
    }
    return 0;
}

double SolveFem::ComputeVolOfHex(MatrixXd co)
{
    //     7------8      3------4
    //     |      |      |      |    Calculate the volume of the convex hexahedron with the diagonal numbering order
    //     5------6      1------2  
    //       upper        lower
    // 
    //Divide the hexahedron into five tetrahedra:
    //6-124£¬7-143£¬1-657£¬4-678£¬6-147, make the three prongs into a right-handed system
    //                                                                                         | a1  a2  a3 |
    // The formula for the volume of a tetrahedron is V = [ab ac bc]= 1/6 * a¡¤£¨b¡Ác£©= 1/6 * | b1  b2  b3 |   
    //                                                                                         | c1  c2  c3 | 

    //No.1
    Vector3d sixtoone = co.row(5) - co.row(0);
    Vector3d sixtotwo = co.row(5) - co.row(1);
    Vector3d sixtofour = co.row(5) - co.row(3);
    double p1 = 1.0/6.0 * sixtoone.dot(sixtotwo.cross(sixtofour));

    //No.2
    Vector3d seventoone = co.row(6) - co.row(0);
    Vector3d seventofour = co.row(6) - co.row(3);
    Vector3d seventothree = co.row(6) - co.row(2);
    double p2 = 1.0/6.0 * seventoone.dot(seventofour.cross(seventothree));

    //No.3
    Vector3d onetosix = co.row(0) - co.row(5);
    Vector3d onetofive = co.row(0) - co.row(4);
    Vector3d onetoseven = co.row(0) - co.row(6);
    double p3 = 1.0/6.0 * onetosix.dot(onetofive.cross(onetoseven));    

    //No.4
    Vector3d fourtosix = co.row(3) - co.row(5);
    Vector3d fourtoseven = co.row(3) - co.row(6);
    Vector3d fourtoeight = co.row(3) - co.row(7);
    double p4 = 1.0/6.0 * fourtosix.dot(fourtoseven.cross(fourtoeight));    

    //No.5
    Vector3d sixtoseven = co.row(5) - co.row(6);
    double p5 = 1.0/6.0 * sixtoone.dot(sixtofour.cross(sixtoseven));    
  
    double p = abs(p1) + abs(p2) + abs(p3) + abs(p4) + abs(p5);

    return p;
}