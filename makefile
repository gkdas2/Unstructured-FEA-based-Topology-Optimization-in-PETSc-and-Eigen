PETSC_DIR=/home/aiwanzhe/Downloads/petsc-3.16.4
PETSC_ARCH=arch-linux-c-opt
CFLAGS =
FFLAGS=
CXXFLAGS=
CPPFLAGS=
FPPFLAGS=
LOCDIR=
EXAMPLESC=
EXAMPLESF=
MANSEC=
CLEANFILES=
NP=

OBJECTS=main.o InputMeshData.o MMA.o PDEFilter.o SolveFem.o OptPara.o Write2Vtu.o

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

opt: $(OBJECTS)
	rm -rf opt
	-${CLINKER} -o opt $(OBJECTS) ${PETSC_SYS_LIB} 
	-${RM} ${OBJECTS}
myclean:
	rm -rf opt *.o 

