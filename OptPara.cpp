#include "OptPara.h"

using namespace std;

const int ROW_1 = 410451;
const int COL_1 = 3;
const int ROW_2 = 392000;
const int COL_2 = 8;
const int StrucDofs = 8733;
const int Load = 41;
const int NUM_LD = 1;

OptPara::OptPara(PetscInt nconstraints)
{
	m = nconstraints;
	SetUp();
}

OptPara::OptPara()
{
	m = 1;
	SetUp();
}

OptPara::~OptPara()
{
	if (X != NULL)
	{
		VecDestroy(&X);
	}
	if (XFiled != NULL)
	{
		VecDestroy(&XFiled);
	}
	if (XPhys != NULL)
	{
		VecDestroy(&XPhys);
	}
    if (Xold != NULL)
    {
        VecDestroy(&Xold);
    }
	if (dfdx != NULL)
	{
		VecDestroy(&dfdx);
	}
	if (Vol != NULL)
	{
		VecDestroy(&Vol);
	}
	if (dgdx != NULL)
	{
		VecDestroyVecs(m, &dgdx);
	}
    if (gx != NULL)
    {
        delete[]gx;
    }
}

PetscErrorCode OptPara::SetUp()
{
	PetscErrorCode ierr;

	//the default settings
	maxItr = 200;
	rmin = 0.01;
	penal = 3.0;
	Emin = 1.0e-9;
	Emax = 1.0e6;
	movlim = 0.2;
	Xmin = 0.01;
	Xmax = 1.0;
    volfrac = 0.12;
	x0 = 0.5;
    beta0 = 1;
    betamax = 8;
	proj = PETSC_FALSE;

	ierr = SetUpOpt();
	CHKERRQ(ierr);

	return ierr;
}

PetscErrorCode OptPara::SetUpOpt()
{
	PetscErrorCode ierr;

	PetscBool flg;

	PetscOptionsGetReal(NULL, NULL, "-Emin", &Emin, &flg);
	PetscOptionsGetReal(NULL, NULL, "-Emax", &Emax, &flg);
	PetscOptionsGetReal(NULL, NULL, "-volfrac", &volfrac, &flg);
	PetscOptionsGetReal(NULL, NULL, "-penal", &penal, &flg);
	PetscOptionsGetReal(NULL, NULL, "-rmin", &rmin, &flg);
	PetscOptionsGetInt(NULL, NULL, "-maxItr", &maxItr, &flg);
	PetscOptionsGetReal(NULL, NULL, "-Xmin", &Xmin, &flg);
	PetscOptionsGetReal(NULL, NULL, "-Xmax", &Xmax, &flg);
	PetscOptionsGetReal(NULL, NULL, "-movlim", &movlim, &flg);
	PetscOptionsGetReal(NULL, NULL, "-x0", &x0, &flg);
    PetscOptionsGetReal(NULL, NULL, "-beta0", &beta0, &flg);
    PetscOptionsGetReal(NULL, NULL, "-betamax", &betamax, &flg);
	PetscOptionsGetBool(NULL, NULL, "-proj", &proj, &flg);

	PetscPrintf(PETSC_COMM_WORLD, "################### Optimization settings ####################\n");
	PetscPrintf(PETSC_COMM_WORLD, "# -rmin: %f\n", rmin);
	PetscPrintf(PETSC_COMM_WORLD, "# -volfrac: %f\n", volfrac);
	PetscPrintf(PETSC_COMM_WORLD, "# -penal: %f\n", penal);
	PetscPrintf(PETSC_COMM_WORLD, "# -Emin/-Emax: %e - %e \n", Emin, Emax);
	PetscPrintf(PETSC_COMM_WORLD, "# -Xmin/-Xmax: %e - %e \n", Xmin, Xmax);
	PetscPrintf(PETSC_COMM_WORLD, "# -maxItr: %i\n", maxItr);
	PetscPrintf(PETSC_COMM_WORLD, "# -movlim: %f\n", movlim);
	PetscPrintf(PETSC_COMM_WORLD, "# -x0: %f\n", x0);
    PetscPrintf(PETSC_COMM_WORLD, "# -beta0: %f\n", beta0);
    PetscPrintf(PETSC_COMM_WORLD, "# -betamax: %f\n", betamax);
	PetscPrintf(PETSC_COMM_WORLD, "# -proj: %i  (0/1)\n", proj);
	PetscPrintf(PETSC_COMM_WORLD, "##############################################################\n");

    gx = new PetscScalar[m];

    VecCreate(PETSC_COMM_WORLD, &X);
    VecSetType(X, VECMPI);
    VecSetSizes(X, PETSC_DECIDE, ROW_2);

	ierr = VecDuplicate(X, &XFiled);
    CHKERRQ(ierr);

    ierr = VecDuplicate(X, &XPhys);
    CHKERRQ(ierr);

    ierr = VecDuplicate(X, &xmin);
    CHKERRQ(ierr);

    ierr = VecDuplicate(X, &xmax);
    CHKERRQ(ierr);

    ierr = VecDuplicate(X, &Xold);
    CHKERRQ(ierr);

    ierr = VecDuplicate(X, &dfdx);
    CHKERRQ(ierr);

	ierr = VecDuplicate(X, &Vol);
    CHKERRQ(ierr);

    ierr = VecDuplicateVecs(X, m, &dgdx);
    CHKERRQ(ierr);

    VecSet(X, volfrac);
	VecSet(XFiled, volfrac);
    VecSet(XPhys, volfrac);
    VecSet(Xold, volfrac);

	return ierr;
}

PetscErrorCode OptPara::SetMMA(PetscInt* itr, MMA** mma)
{
    PetscErrorCode ierr = 0;

    // Set MMA parameters (for multiple load cases)
    PetscScalar aMMA[m];
    PetscScalar cMMA[m];
    PetscScalar dMMA[m];
    for (PetscInt i = 0; i < m; i++) {
        aMMA[i] = 0.0;
        dMMA[i] = 0.0;
        cMMA[i] = 1000.0;
    }

    PetscInt nGlobalDesignVar;
    VecGetSize(X,&nGlobalDesignVar); // ASSUMES THAT SIZE IS ALWAYS MATCHED TO CURRENT MESH
    *mma = new MMA(nGlobalDesignVar, m, X, aMMA, cMMA, dMMA);

    return ierr;
}
