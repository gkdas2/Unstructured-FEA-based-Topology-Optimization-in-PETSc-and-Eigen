#pragma once

#include <iostream>
#include <Eigen/Eigen>
#include <fstream>
#include <petsc.h>

/*
Copyright (C) 2022-2028, Yu Wang
*/

class PDEFilter
{
public:

	//Constructor and Destructor
	PDEFilter(Eigen::MatrixXi elementEigen, Eigen::MatrixXd eleXCoord, Eigen::MatrixXd eleYCoord, Eigen::MatrixXd eleZCoord, PetscScalar rmin, Vec X);
	~PDEFilter();

	int PrepareFilter(Eigen::MatrixXi elementEigen, Eigen::MatrixXd eleXCoord, Eigen::MatrixXd eleYCoord, Eigen::MatrixXd eleZCoord, PetscScalar rmin, Vec X);
	//PDE filter
	PetscErrorCode Filter(Vec X,Vec XFiled,PetscScalar Xmin);
	//Sens of filtered obj and cons
	PetscErrorCode ComputeFilteredSens(Vec dfdx,Vec* dgdx,PetscInt m);
	//Heaviside projection
	PetscErrorCode HeavisideFilter(Vec XFiled, Vec XPhys,PetscScalar x0,PetscScalar beta0,PetscScalar betamax,PetscInt itr, PetscScalar Xmin, PetscBool proj);
	//Sens of projected obj and cons
    PetscErrorCode ComputeProjectedSens(Vec XFiled, Vec dfdx, Vec* dgdx, PetscInt m, PetscScalar x0);
	 
private:

	Eigen::MatrixXi edofMat;
	//PDE stiffness matrix
	Mat KF;
	//PDE transfer matrix, element to node
	Mat TF;
	//PDE transfer matrix, node to element
	Mat TFX;

	//Vectors used for filter
	Vec TX;
	Vec RX;

	//KSP solver of filter
	KSP ksp_f;
	//Helmholtz filter radius
	double r;

	//arguments of projection
    PetscScalar beta;
	PetscScalar temp;
	//Calculate element stiffness matrix of Helmholtz filter
	int KeHel(Eigen::MatrixXd eleCoord, Eigen::MatrixXd& KE, Eigen::MatrixXd& T, PetscScalar rmin);

	PetscScalar Min(PetscScalar d1, PetscScalar d2);
};





