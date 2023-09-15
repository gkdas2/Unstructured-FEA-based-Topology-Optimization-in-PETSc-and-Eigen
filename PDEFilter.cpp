#include "PDEFilter.h"
#include "OptPara.h"

using namespace std;
using namespace Eigen;

PDEFilter::PDEFilter(MatrixXi elementEigen, MatrixXd eleXCoord, MatrixXd eleYCoord, MatrixXd eleZCoord, PetscScalar rmin, Vec X)
{
    PrepareFilter(elementEigen, eleXCoord, eleYCoord, eleZCoord, rmin, X);
}

PDEFilter::~PDEFilter()
{
    if(KF!=NULL)
    {
        MatDestroy(&KF);
    }
    if(TF!=NULL)
    {
        MatDestroy(&TF);
    }
    if(TFX!=NULL)
    {
        MatDestroy(&TFX);
    }
    if(TX!=NULL)
    {
        VecDestroy(&TX);
    }
    if(RX!=NULL)
    {
        VecDestroy(&RX);
    }
}

int PDEFilter::PrepareFilter(MatrixXi elementEigen, MatrixXd eleXCoord, MatrixXd eleYCoord, MatrixXd eleZCoord, PetscScalar rmin, Vec X)
{
    PetscErrorCode ierr;

    MatCreate(PETSC_COMM_WORLD, &KF);
    MatSetType(KF, MATMPIAIJ);
    MatSetSizes(KF, PETSC_DECIDE, PETSC_DECIDE, ROW_1, ROW_1);
    MatSetUp(KF);

    MatCreate(PETSC_COMM_WORLD, &TF);
    MatSetSizes(TF, PETSC_DECIDE, PETSC_DECIDE, ROW_1, ROW_2);
    MatSetType(TF, MATMPIAIJ);
    MatSetUp(TF);

    MatCreate(PETSC_COMM_WORLD, &TFX);
    MatSetSizes(TFX, PETSC_DECIDE, PETSC_DECIDE, ROW_1, ROW_2);
    MatSetType(TFX, MATMPIAIJ);
    MatSetUp(TFX);

    MatZeroEntries(KF);
    MatZeroEntries(TF);
    MatZeroEntries(TFX);

    //Matrix pre-allocation
    MatMPIAIJSetPreallocation(KF,100,NULL,100,NULL);
    MatMPIAIJSetPreallocation(TF,100,NULL,100,NULL);
    MatMPIAIJSetPreallocation(TFX,100,NULL,100,NULL);

    VecCreate(PETSC_COMM_WORLD,&TX);
    VecSetType(TX,VECMPI);
    VecSetSizes(TX,PETSC_DECIDE,ROW_1);
    VecDuplicate(TX,&RX);

    PetscInt edof[COL_2] = {0};

    PetscInt ldim;
    ierr = VecGetLocalSize(X, &ldim); CHKERRQ(ierr);
    PetscInt low, high;
    ierr = VecGetOwnershipRange(X, &low, &high); CHKERRQ(ierr);

    //Element stiffness matrix of Helmholtz filter
    PetscScalar Keh[COL_2 * COL_2];
	PetscScalar T[COL_2];
    PetscScalar TT[COL_2];
    MatrixXd keh(COL_2, COL_2);
    MatrixXd t(COL_2,1);
    MatrixXd eleCoord(COL_2, 3);

    for(PetscInt i = 0;i < ldim;i++)
    {
        eleCoord << eleXCoord.row(i).transpose(), eleYCoord.row(i).transpose(), eleZCoord.row(i).transpose();

        KeHel(eleCoord,keh,t,rmin);

        for (int ii = 0; ii < COL_2; ii++)
        {
            for (int jj = 0; jj < COL_2; jj++)
            {
                Keh[COL_2 * ii + jj] = keh(ii, jj);
            }
            T[ii] = t(ii,0);
            TT[ii] = 1.0 / (double)COL_2;
        }

        PetscInt eg = i + low;
        for(int j = 0;j < COL_2;j++)
        {
            edof[j] = elementEigen(i, j);
        }
        ierr = MatSetValues(KF,COL_2,edof,COL_2,edof,Keh,ADD_VALUES);CHKERRQ(ierr);
        ierr = MatSetValues(TF,COL_2,edof,1,&eg,T,ADD_VALUES);CHKERRQ(ierr);
        ierr = MatSetValues(TFX,COL_2,edof,1,&eg,TT,ADD_VALUES);CHKERRQ(ierr);
    }
    MatAssemblyBegin(KF,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(KF,MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(TF,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(TF,MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(TFX,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(TFX,MAT_FINAL_ASSEMBLY);

    //error settings
    PetscScalar rtol = 1.0e-5;
    PetscScalar atol = 1.0e-50;
    PetscScalar dtol = 1.0e5;
    PetscInt    maxitsGlobal = 10000;

    //Creatr KSP solver 
    ierr = KSPCreate(PETSC_COMM_WORLD, &ksp_f);
    CHKERRQ(ierr);
    ierr = KSPSetType(ksp_f, KSPCG);
    CHKERRQ(ierr);
    ierr = KSPSetInitialGuessNonzero(ksp_f, PETSC_TRUE);
    CHKERRQ(ierr);
    ierr = KSPSetTolerances(ksp_f, rtol, atol, dtol, maxitsGlobal);
    CHKERRQ(ierr);

    PC pc;
    KSPGetPC(ksp_f, &pc);
    PCSetType(pc, PCJACOBI);
    KSPSetFromOptions(ksp_f);
    KSPGetPC(ksp_f, &pc);
    KSPSetOperators(ksp_f, KF, KF);

    return ierr;
}

PetscErrorCode PDEFilter::Filter(Vec X,Vec XFiled,PetscScalar Xmin)
{
    PetscErrorCode ierr;

    //Filter design variables
    ierr = MatMult(TF,X,TX);CHKERRQ(ierr);
    ierr = KSPSolve(ksp_f,TX,RX);CHKERRQ(ierr);
    ierr = MatMultTranspose(TFX,RX,XFiled);CHKERRQ(ierr);

    //Remove negative values
    PetscInt ldim;
    ierr = VecGetLocalSize(XFiled,&ldim);CHKERRQ(ierr);
    PetscScalar* xf;
    ierr = VecGetArray(XFiled,&xf);CHKERRQ(ierr);

    for(PetscInt i = 0;i < ldim;i++)
    {
        if(xf[i] < Xmin) xf[i] = Xmin;
        else if(xf[i] > 1.0) xf[i] = 1.0;
    }
    
    VecRestoreArray(XFiled,&xf);
    return ierr;
}

PetscErrorCode PDEFilter::ComputeFilteredSens(Vec dfdx,Vec* dgdx,PetscInt m)
{
    PetscErrorCode ierr;
    //sens of filtered design variables
    ierr = MatMult(TF,dfdx,TX);CHKERRQ(ierr);
    ierr = KSPSolve(ksp_f,TX,RX);CHKERRQ(ierr);
    ierr = MatMultTranspose(TFX,RX,dfdx);CHKERRQ(ierr);
    //sens of filtered constraints
    for(PetscInt i = 0;i < m;i++)
    {
        ierr = MatMult(TF,dgdx[i],TX);CHKERRQ(ierr);
        ierr = KSPSolve(ksp_f,TX,RX);CHKERRQ(ierr);
        ierr = MatMultTranspose(TFX,RX,dgdx[i]);CHKERRQ(ierr);
    }

    return ierr;
}

int PDEFilter::KeHel(MatrixXd eleCoord, MatrixXd& KE, MatrixXd& T, PetscScalar rmin)
{
    //Helmholtz filter radius
    r = 1 / (2 * sqrt(3)) * rmin;

    KE.fill(0);
    T.fill(0);

    //Gauss points and weights
    MatrixXd G(1, 3), W(1, 3);
    G(0, 0) = -0.775, G(0, 1) = 0, G(0, 2) = 0.775;
    W(0, 0) = 0.556, W(0, 1) = 0.889, W(0, 2) = 0.556;

    MatrixXd N(1, COL_2), dN(3, COL_2), J(3, 3), Dn(3, COL_2);
    MatrixXd Kd(3,3);
    Kd.fill(0);
    //isotropic filter
    Kd(0,0) = r * r,Kd(1,1) = r * r,Kd(2,2) = r * r;

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            for (int k = 0; k < 3; k++)
            {
                double s = G(0, i);
                double t = G(0, j);
                double v = G(0, k);

                N << (1 - s) * (1 - t) * (1 - v) * 0.125,
                    (1 + s)* (1 - t)* (1 - v) * 0.125,
                    (1 - s)* (1 + t)* (1 - v) * 0.125,
                    (1 + s)* (1 + t)* (1 - v) * 0.125,
                    (1 - s)* (1 - t)* (1 + v) * 0.125,
                    (1 + s)* (1 - t)* (1 + v) * 0.125,                    
                    (1 - s)* (1 + t)* (1 + v) * 0.125,
                    (1 + s)* (1 + t)* (1 + v) * 0.125;

                dN << -(1 - t) * (1 - v) * 0.125, (1 - t)* (1 - v) * 0.125, -(1 + t)* (1 - v) * 0.125, (1 + t) * (1 - v) * 0.125,
                    -(1 - t) * (1 + v) * 0.125, (1 - t)* (1 + v) * 0.125, -(1 + t)* (1 + v) * 0.125, (1 + t) * (1 + v) * 0.125,//row0
                    -(1 - s) * (1 - v) * 0.125, -(1 + s) * (1 - v) * 0.125, (1 - s)* (1 - v) * 0.125, (1 + s)* (1 - v) * 0.125,
                    -(1 - s) * (1 + v) * 0.125, -(1 + s) * (1 + v) * 0.125, (1 - s)* (1 + v) * 0.125, (1 + s)* (1 + v) * 0.125,//row1
                    -(1 - s) * (1 - t) * 0.125, -(1 + s) * (1 - t) * 0.125, -(1 - s) * (1 + t) * 0.125, -(1 + s) * (1 + t) * 0.125,
                    (1 - s)* (1 - t) * 0.125, (1 + s)* (1 - t) * 0.125, (1 - s) * (1 + t) * 0.125, (1 + s) * (1 + t) * 0.125;//row2              

                J = dN * eleCoord;

                Dn = J.inverse() * dN;

                KE = KE + W(i) * W(j) * W(k) * (Dn.transpose() * Kd * Dn + N.transpose() * N) * J.determinant();
                T = T + W(i) * W(j) * W(k) * N.transpose() * J.determinant();
            }
        }
    }
    return 0;
}

PetscErrorCode PDEFilter::HeavisideFilter(Vec XFiled, Vec XPhys, PetscScalar x0, PetscScalar beta0, PetscScalar betamax, PetscInt itr, PetscScalar Xmin, PetscBool proj)
{
    PetscErrorCode ierr;

    if(proj)
    {
        PetscInt ldim;
        ierr=VecGetLocalSize(XPhys,&ldim);CHKERRQ(ierr);

        //Get pointer
        PetscScalar* xf;
        ierr = VecGetArray(XFiled, &xf);
        CHKERRQ(ierr);
        PetscScalar* xp;
        ierr = VecGetArray(XPhys, &xp);
        CHKERRQ(ierr);

        if (itr < 30)
        {
            beta = beta0;
        }
        else
        {  
            if(itr%10==0) 
            {
            beta = Min(betamax, temp + 1);
            }
        }

        for (int i = 0; i < ldim; i++)
        {
            xp[i] = (tanh(beta * x0) + tanh(beta * (xf[i] - x0))) / (tanh(beta * x0) + tanh(beta * (1 - x0)));

            if (xp[i] < Xmin)
            {
                xp[i] = Xmin;
            }
        }

        temp = beta;

        VecRestoreArray(XFiled, &xf);
        VecRestoreArray(XPhys, &xp);
        return ierr;
    }

    else
    {
        ierr = VecCopy(XFiled,XPhys);CHKERRQ(ierr);
        return ierr;
    }
}

PetscErrorCode PDEFilter::ComputeProjectedSens(Vec XFiled, Vec dfdx, Vec* dgdx, PetscInt m, PetscScalar x0)
{
    PetscErrorCode ierr;

    Vec DX;
    ierr = VecDuplicate(XFiled,&DX);CHKERRQ(ierr);
    VecSet(DX,1.0);

    PetscScalar *xf, *dx;
    PetscInt ldim;
    ierr = VecGetLocalSize(XFiled,&ldim);CHKERRQ(ierr);
    ierr = VecGetArray(XFiled,&xf);CHKERRQ(ierr);
    ierr = VecGetArray(DX,&dx);CHKERRQ(ierr);

    for(int i = 0; i < ldim; i++)
    {
        dx[i] = beta * (1.0 - pow(tanh(beta * (xf[i] - x0)), 2.0)) / (tanh(beta * x0) + tanh(beta * (1 - x0)));
    }
    VecRestoreArray(DX,&dx);

    //Get the updated vector value again
    ierr = VecGetArray(DX,&dx);CHKERRQ(ierr);
    PetscScalar* df;
    ierr = VecGetArray(dfdx,&df);CHKERRQ(ierr);
    for(int i = 0; i < ldim; i++)
    {
        df[i] = df[i] * dx[i];
    }

    PetscScalar* dg;
    for(int i = 0; i < m; i++)
    {
        ierr = VecGetArray(dgdx[i],&dg);CHKERRQ(ierr);
        for(int j = 0; j < ldim; j++)
        {
            dg[j] = dg[j] * dx[j];
        }
        VecRestoreArray(dgdx[i],&dg);
    }

    VecRestoreArray(dfdx,&df);
    VecRestoreArray(XFiled,&xf);
    VecRestoreArray(DX,&dx);
    VecDestroy(&DX);

    return ierr;
}

PetscScalar PDEFilter::Min(PetscScalar d1, PetscScalar d2) { return d1 < d2 ? d1 : d2; }
