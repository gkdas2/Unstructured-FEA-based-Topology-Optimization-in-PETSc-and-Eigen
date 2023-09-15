#pragma once

#include <iostream>
#include <petsc.h>
#include "MMA.h"
#include <fstream>

/*
Copyright (C) 2022-2028, Yu Wang
*/

extern const int ROW_1;//Total number of nodes
extern const int COL_1;//dimension
extern const int ROW_2;//Total number of elements
extern const int COL_2;//Number of nodes of each element
extern const int StrucDofs;//Number of fixed degrees of freedom
extern const int Load;//Number of degrees of load
extern const int NUM_LD;//Number of loads

class OptPara
{
public:

	OptPara(PetscInt nconstraints);
	OptPara();                     
	~OptPara();
	 
	Vec X;                  //the desgin variables
	Vec Xold;               //X from previous iteration
	Vec XFiled;             //the filtered desgin variables
	Vec XPhys;              //the projected desgin variables
	Vec dfdx;               //Sensitivities of objective
	Vec* dgdx;              //Sensitivities of constraints (vector array)
	Vec xmin, xmax;         //Vectors with max and min values of x
	Vec Vol;                //Vector with volume of each element

	PetscScalar fx;         //the objective
	PetscScalar Xmin, Xmax; //Min and Max value of design variables
	PetscScalar Emin, Emax; //Modified SIMP, max and min E
	PetscScalar volfrac;    //Volume fraction
	PetscScalar penal;      //Penalization parameter
	PetscScalar movlim;     //Max change of design variables
	PetscScalar rmin;       //filter radius
	PetscScalar* gx;        //Array with constraint values

	PetscInt maxItr;  //Maximum number of iterative steps
	PetscInt m;       //Number of constraints
	PetscScalar x0;         //x0 in Heaviside projection
	PetscScalar beta0;      //¦Â0 in Heaviside projection
	PetscScalar betamax;    //¦Âmax in Heaviside projection
    PetscScalar fscale;     //Scale factor of the objective function
	PetscBool proj;         //The criterion of projection or not

	//Initialize MMA
	PetscErrorCode SetMMA(PetscInt* itr, MMA** mma);

private:

	//Variable initialization
	PetscErrorCode SetUp();
	PetscErrorCode SetUpOpt();	 
};
