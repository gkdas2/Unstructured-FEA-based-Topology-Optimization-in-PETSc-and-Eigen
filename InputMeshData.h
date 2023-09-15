#pragma once

#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include "OptPara.h"
#include <stdio.h>

/*
Copyright (C) 2022-2028, Yu Wang
*/

class InputMeshData
{
public:
	//Constructors
	InputMeshData(Vec X);
	//Destructor
	~InputMeshData();
	//Get node and element information
	PetscErrorCode GetNodeAndElementData(Vec X);
	//Get boundary conditions
	int GetBcs();

	//Element Information
	Eigen::MatrixXi elementEigen;//Element Matrix
	//x,y,z coordinate matrix of element
	Eigen::MatrixXd eleXCoord, eleYCoord, eleZCoord;

	//vertex and element information of the mesh
	double* vertex;
	int* element;
	//Vector of load
	int **loadDofs;
	double **load;
	//Vector of fixed dofs
	int *fixedDofs;
	//local points info
	double *vertex_loc;
};

