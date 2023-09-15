#include "InputMeshData.h"

using namespace std;
using namespace Eigen;

InputMeshData::InputMeshData(Vec X)
{
    GetNodeAndElementData(X);
}

InputMeshData::~InputMeshData()
{
    delete []fixedDofs;
    delete []vertex_loc;

    for(int i = 0; i < NUM_LD; i++)
    {
        delete []loadDofs[i];
        delete []load[i];
    }
    delete []loadDofs;
    delete []load;

    PetscMPIInt rank;
    MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
    if(rank==0)
    {
        delete []vertex;
        delete []element;
    }
}

PetscErrorCode InputMeshData::GetNodeAndElementData(Vec X)
{
    PetscErrorCode ierr;

    PetscMPIInt rank,size;
    MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
    MPI_Comm_size(PETSC_COMM_WORLD,&size);

    //Get the length of storage required for each rank
    PetscInt ldim;
    ierr = VecGetLocalSize(X, &ldim); CHKERRQ(ierr);
    PetscInt low, high;
    ierr = VecGetOwnershipRange(X, &low, &high); CHKERRQ(ierr);

    elementEigen.resize(high-low,COL_2);
    eleXCoord.resize(high-low,COL_2),eleYCoord.resize(high-low,COL_2),eleZCoord.resize(high-low,COL_2);

    //The information received from other ranks of rank 0
    int send[2];
    int* recv;
    int* sendcounts;//The length each rank sends
    int* disp;//The offset of each rank
    int recvcounts = (high - low)*COL_2;
    send[0] = low,send[1] = high;

    if(rank==0)
    {
        recv = new int[2*size];
        sendcounts = new int[size];
        disp = new int[size];
    }

    MPI_Gather(send,2,MPI_INT,recv,2,MPI_INT,0,PETSC_COMM_WORLD);

    if(rank==0)
    {
        for(int i=0;i<size;i++)
        {
            sendcounts[i] = (recv[2*i+1]-recv[2*i])*COL_2;
            if(i==0)
            {
                disp[i]=0;
            }
            else
            {
                disp[i]=disp[i-1]+sendcounts[i-1];
            }
        }
    }
    MPI_Barrier(PETSC_COMM_WORLD);

    //The mesh information sent to each rank
    int* local_element = new int[(high-low)*COL_2];
    double* local_elex = new double[(high-low)*COL_2];
    double* local_eley = new double[(high-low)*COL_2];
    double* local_elez = new double[(high-low)*COL_2];
    //The whole mesh information
    double* ver_x;
    double* ver_y;
    double* ver_z;
    double* ele_x;
    double* ele_y;
    double* ele_z;

    if(rank==0)
    {
    //Read node information, specify the correct path
    ifstream readin1("/home/aiwanzhe/Desktop/mesh/vertex.txt");
    if (!readin1.is_open())
    {
        PetscPrintf(PETSC_COMM_WORLD,"Can't found document::vertex.txt\n");
        exit(0);
    }

    vertex = new double[ROW_1*COL_1];
    for (int i = 0; i < ROW_1*COL_1; i++)
    {
        readin1 >> vertex[i];
    }
    readin1.close();

    ver_x = new double[ROW_1];
    ver_y = new double[ROW_1];
    ver_z = new double[ROW_1];

    //Assign values to the Eigen node matrix
    for (int j = 0; j < ROW_1; j++)
    {
        ver_x[j] = vertex[3*j];
        ver_y[j] = vertex[3*j+1];
        ver_z[j] = vertex[3*j+2];
    }

    //Read the element information, specify the correct path
    ifstream readin2("/home/aiwanzhe/Desktop/mesh/element.txt");
    if (!readin2.is_open())
    {
        PetscPrintf(PETSC_COMM_WORLD,"Can't found document::element.txt\n");
        exit(0);
    }
    element = new int[ROW_2*COL_2];
    for (int i = 0; i < ROW_2*COL_2; i++)
    {
        readin2 >> element[i];
    }
    readin2.close();
    
    ele_x = new double[ROW_2*COL_2];
    ele_y = new double[ROW_2*COL_2];
    ele_z = new double[ROW_2*COL_2];
    //Element coordinates
    for (int j = 0; j < ROW_2*COL_2; j++)
    {
        ele_x[j] = ver_x[element[j]];
        ele_y[j] = ver_y[element[j]];
        ele_z[j] = ver_z[element[j]];
    }
    }

    MPI_Scatterv(element,sendcounts,disp,MPI_INT,local_element,recvcounts,MPI_INT,0,PETSC_COMM_WORLD);
    MPI_Scatterv(ele_x,sendcounts,disp,MPI_DOUBLE,local_elex,recvcounts,MPI_DOUBLE,0,PETSC_COMM_WORLD);
    MPI_Scatterv(ele_y,sendcounts,disp,MPI_DOUBLE,local_eley,recvcounts,MPI_DOUBLE,0,PETSC_COMM_WORLD);
    MPI_Scatterv(ele_z,sendcounts,disp,MPI_DOUBLE,local_elez,recvcounts,MPI_DOUBLE,0,PETSC_COMM_WORLD);

    for(int i=0;i<ldim;i++)
    {
        for(int j=0;j<COL_2;j++)
        {
            elementEigen(i,j)=local_element[j+COL_2*i];
            eleXCoord(i,j)=local_elex[j+COL_2*i];
            eleYCoord(i,j)=local_eley[j+COL_2*i];
            eleZCoord(i,j)=local_elez[j+COL_2*i];
        }
    }

    //assign vertex data to each rank
    PetscInt rank_ne = ROW_1/size;
    if(rank==size-1) rank_ne = ROW_1 - rank_ne * (size-1);

    if(rank!=size-1)
    {
        send[0] = rank * rank_ne;
        send[1] = rank * rank_ne + rank_ne;
    }
    else
    {
        send[0] = ROW_1 - 1 - rank_ne;
        send[1] = ROW_1 - 1;
    }

    MPI_Gather(send,2,MPI_INT,recv,2,MPI_INT,0,PETSC_COMM_WORLD);

    if(rank==0)
    {
        for(int i=0;i<size;i++)
        {
            sendcounts[i] = (recv[2*i+1]-recv[2*i])*COL_1;
            if(i==0)
            {
                disp[i] = 0;
            }
            else
            {
                disp[i] = disp[i-1]+sendcounts[i-1];
            }
        }
    }

    vertex_loc = new double[3 * rank_ne];
    recvcounts = 3 * rank_ne;
    MPI_Scatterv(vertex,sendcounts,disp,MPI_DOUBLE,vertex_loc,recvcounts,MPI_DOUBLE,0,PETSC_COMM_WORLD);

    delete []local_element;
    delete []local_elex;
    delete []local_eley;
    delete []local_elez;

    if(rank==0)
    {
    delete []recv;
    delete []sendcounts;
    delete []disp;
    delete []ver_x;
    delete []ver_y;
    delete []ver_z;
    delete []ele_x;
    delete []ele_y;
    delete []ele_z;
    }

    //Get boundary conditions
    GetBcs();
    return ierr;
}

int InputMeshData::GetBcs()
{
    //Get load dofs
    char name_dof[PETSC_MAX_PATH_LEN];
    char name_load[PETSC_MAX_PATH_LEN];
    loadDofs = new int*[NUM_LD];
    load = new double*[NUM_LD];

    for(int i=0;i < NUM_LD;i++)
    {
        sprintf(name_dof,"/home/aiwanzhe/Desktop/mesh/loaddof_%d.txt",i+1);
        loadDofs[i] = new int[Load];
        ifstream readin_lddof(name_dof);
        if (!readin_lddof.is_open())
        {
            PetscPrintf(PETSC_COMM_WORLD,"Can't found document::loaddof_%d.txt\n",i+1);
            exit(0);
        }

        for(int j = 0;j < Load;j++)
        {
            readin_lddof>>loadDofs[i][j];
        }

        sprintf(name_load,"/home/aiwanzhe/Desktop/mesh/load_%d.txt",i+1);
        load[i] = new double[Load];
        ifstream readin_ld(name_load);
        if (!readin_ld.is_open())
        {
            PetscPrintf(PETSC_COMM_WORLD,"Can't found document::load_%d.txt\n",i+1);
            exit(0);
        }

        for(int j = 0;j < Load;j++)
        {
            readin_ld>>load[i][j];
        }
        
        readin_lddof.close();
        readin_ld.close();
    }

    //Get fixed dofs
    ifstream readin_fixed("/home/aiwanzhe/Desktop/mesh/fixeddofs.txt");
    if (!readin_fixed.is_open())
    {
        PetscPrintf(PETSC_COMM_WORLD,"Can't found document::fixeddofs.txt\n");
        exit(0);
    }
    fixedDofs = new int[StrucDofs];
    for (int i = 0; i < StrucDofs; i++)
    {
        readin_fixed >> fixedDofs[i];
    }
    readin_fixed.close();

    return 0;
}