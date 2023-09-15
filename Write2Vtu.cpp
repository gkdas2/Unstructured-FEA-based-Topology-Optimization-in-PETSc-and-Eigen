#include "Write2Vtu.h"

using namespace std;

PetscErrorCode Write2Vtu::WriteResults(double* vertex, int* element, Vec X, Vec U, PetscInt itr)
{
    PetscErrorCode ierr;
    //all_reduce displacement field
	Vec U_SEQ;
	VecScatter utx;
	ierr = VecScatterCreateToAll(U,&utx,&U_SEQ);CHKERRQ(ierr);
	ierr = VecScatterBegin(utx,U,U_SEQ,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
	ierr = VecScatterEnd(utx,U,U_SEQ,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
	PetscScalar* uout;
	ierr = VecGetArray(U_SEQ,&uout);
    //all_reduce design variable vector
	Vec X_SEQ;
	VecScatter xtx;
	ierr = VecScatterCreateToAll(X,&xtx,&X_SEQ);CHKERRQ(ierr);
	ierr = VecScatterBegin(xtx,X,X_SEQ,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
	ierr = VecScatterEnd(xtx,X,X_SEQ,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
	PetscScalar* xout;
	ierr = VecGetArray(X_SEQ,&xout);
   
	PetscMPIInt rank;
	MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

    if(rank==0)
	{
	string OutPutFileName;
	string VTUFileName;
	char name[PETSC_MAX_PATH_LEN];

	if(itr==0)
	{
		sprintf(name,"output_initial");
	}
	else if(itr==-1)
	{
		sprintf(name,"output_final");
	}
	else
	{
        sprintf(name,"output_%d",itr);
	}

	OutPutFileName = name;
	VTUFileName = OutPutFileName + ".vtu";

	ofstream out;
	out.open(VTUFileName,ios::out);
	if (!out.is_open())
	{
		string str = "can\'t create a new vtu file(" + VTUFileName + "), please make sure you have write permission";
		cout << str << endl;
		exit(0);
	}

	//vtu header
	out << "<?xml version=\"1.0\"?>\n";
	out << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\">\n";
	out << "<UnstructuredGrid>\n";
	out << "<Piece NumberOfPoints=\"" << ROW_1 << "\" NumberOfCells=\"" << ROW_2 << "\">\n";

	//setting precision
	out << scientific << setprecision(6);

    //**********output point information************
	out << "<Points>\n";
	out << "<DataArray type=\"Float64\" Name=\"nodes\" NumberOfComponents=\"3\" format=\"ascii\">\n";

	//output
	for (int i = 0; i < ROW_1; i++)
	{
		out << vertex[COL_1 * i] << " ";
		out << vertex[COL_1 * i + 1] << " ";
		out << vertex[COL_1 * i + 2] << "\n";
	}
	out << "</DataArray>\n";
	out << "</Points>\n";

	//***********output cell information**********
    out << "<Cells>\n";
	out << "<DataArray type=\"Int32\" Name=\"connectivity\" NumberOfComponents=\"1\" format=\"ascii\">\n";

	//need counterclockwise numbering element in vtk
	for(int i = 0;i < ROW_2;i++)
	{
        out << element[COL_2 * i] << " ";
		out << element[COL_2 * i + 1] << " ";
		out << element[COL_2 * i + 3] << " ";
		out << element[COL_2 * i + 2] << " ";
		out << element[COL_2 * i + 4] << " ";
		out << element[COL_2 * i + 5] << " ";
		out << element[COL_2 * i + 7] << " ";
		out << element[COL_2 * i + 6];
		out << "\n";
	}
	out << "</DataArray>\n";

	//***************output offset****************
	out << "<DataArray type=\"Int32\" Name=\"offsets\" NumberOfComponents=\"1\" format=\"ascii\">\n";
	int offset = 0;
	for(int i = 0;i < ROW_2;i++)
	{
		offset = offset + COL_2;
		out << offset <<" ";
	}
	out << "\n";
	out << "</DataArray>\n";

	//*************cell format (12 for hex8 10 for tet4)******************
	out << "<DataArray type=\"Int32\" Name=\"types\" NumberOfComponents=\"1\" format=\"ascii\">\n";
	for (int i = 0;i < ROW_2; i++)
	{
		out << 12 <<" ";
	}
	out << "\n";
	out << "</DataArray>\n";
	out << "</Cells>\n";

	//*************point variables***********************
	out << "<PointData>\n";
	out << "<DataArray type=\"Float64\" NumberOfComponents=\"1\" Name=\"ux\" format=\"ascii\">\n";
	for(int i = 0; i < ROW_1; i++)
	{
		out << uout[COL_1 * i] << " ";
	}
    out << "\n";
    out << "</DataArray>\n";
    out << "<DataArray type=\"Float64\" NumberOfComponents=\"1\" Name=\"uy\" format=\"ascii\">\n";
	for(int i = 0; i < ROW_1; i++)
	{
		out << uout[COL_1 * i + 1] << " ";
	}
	out << "\n";
	out << "</DataArray>\n";
	out << "<DataArray type=\"Float64\" NumberOfComponents=\"1\" Name=\"uz\" format=\"ascii\">\n";
	for(int i = 0; i < ROW_1; i++)
	{
		out << uout[COL_1 * i + 2] << " ";
	}
	out << "\n";
	out << "</DataArray>\n";
	out << "</PointData>\n";

	//**************element variables******************
	out << "<CellData>\n";
	out << "<DataArray type=\"Float64\" NumberOfComponents=\"1\" Name=\"xphys\" format=\"ascii\">\n";
	for(int i = 0; i < ROW_2; i++)
	{
		out << xout[i] <<" ";
	}
	out << "\n";
	out << "</DataArray>\n";
	out << "</CellData>\n";

	//end of vtu
    out << "</Piece>\n";
	out << "</UnstructuredGrid>\n";
	out << "</VTKFile>" << endl;

	out.close();
	}

    VecRestoreArray(U_SEQ,&uout);
	VecScatterDestroy(&utx);
	VecDestroy(&U_SEQ);
	VecRestoreArray(X_SEQ,&xout);
	VecScatterDestroy(&xtx);
	VecDestroy(&X_SEQ);

	return ierr;
}