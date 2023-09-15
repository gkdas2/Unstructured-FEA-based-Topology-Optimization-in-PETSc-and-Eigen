#pragma once

#include <Eigen/Eigen>
#include <math.h>
#include <petsc.h>
#include "OptPara.h"

/*
Copyright (C) 2022-2028, Yu Wang
*/

class SolveFem
{
public:

	SolveFem(Eigen::MatrixXi elementEigen, Eigen::MatrixXd eleXCoord, Eigen::MatrixXd eleYCoord, Eigen::MatrixXd eleZCoord, Vec Vol, Vec X);
	~SolveFem();

	//Coordinate matrix of degrees of freedom
	Eigen::MatrixXi dofCoord;
	//Poisson ratio
	PetscScalar nu;
	//Obtain coordinate matrix of degrees of freedom
	PetscErrorCode GetDofCoord(Eigen::MatrixXi elementEigen, Eigen::MatrixXi& dofCoord, Vec X);
	
	//Element coordinate matrix, element stiffness matrix, zero vector
	Eigen::MatrixXd eleCoord, ke, zero;

    //Assemble global stiffness matrix and load vector
    PetscErrorCode AssembleFem(Vec XPhys, PetscScalar Emin, PetscScalar Emax, PetscScalar penal, Eigen::MatrixXd eleXCoord, Eigen::MatrixXd eleYCoord, Eigen::MatrixXd eleZCoord, int **loadDofs, double **load, int *fixedDofs);
    //Solve linear equations system
	PetscErrorCode ComputeFem(Vec XPhys, PetscScalar Emin, PetscScalar Emax, PetscScalar penal, Eigen::MatrixXd eleXCoord, Eigen::MatrixXd eleYCoord, Eigen::MatrixXd eleZCoord, int **loadDofs, double **load, int *fixedDofs, double *vertex_loc);
	//Calculate the eight-node isoparametric element stiffness matrix
	int Hex8Isoparametric(Eigen::MatrixXd eleCoord, Eigen::MatrixXd& ke, Eigen::MatrixXd zero);

	int ChangeEigenToPetsc(Eigen::MatrixXd ke,PetscScalar* KE);

	//Compute obj and sens
	PetscErrorCode ComputeObjectiveAndSensitivities(PetscScalar* fx, PetscScalar* gx, Vec dfdx, PetscInt m, Vec* dgdx, Vec X, PetscScalar Emin, PetscScalar Emax, PetscScalar penal, PetscScalar volfrac,
	                                                Eigen::MatrixXd eleXCoord, Eigen::MatrixXd eleYCoord, Eigen::MatrixXd eleZCoord, Vec Vol, int **loadDofs, double **load, int *fixedDofs, double *vertex_loc);

	Vec   U;//Displacement vector

private:
	PetscScalar Ke[24 * 24];//Element stiffness matrix after interpolation
	PetscScalar KE[24 * 24];//Initial element stiffness matrix
	Mat   K;//global stiffness matrix
	Vec   F;//global load vector
	KSP   ksp;//KSP solver

	double ComputeVolOfHex(Eigen::MatrixXd co);//Calculate the volume of a convex hexahedron
	PetscErrorCode ComputeTotalVol(Eigen::MatrixXd eleXCoord, Eigen::MatrixXd eleYCoord, Eigen::MatrixXd eleZCoord, Vec Vol);//Calculate total volume
	PetscScalar TV;//Total volume

};



 

