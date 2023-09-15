#pragma once

#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include "OptPara.h"

/*
Copyright (C) 2022-2028, Yu Wang
*/

class Write2Vtu
{
public:
    
    PetscErrorCode WriteResults(double* vertex, int* element, Vec X, Vec U, PetscInt itr);
    
};