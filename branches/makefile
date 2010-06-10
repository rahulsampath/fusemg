
SHELL = /bin/sh

include ${PETSC_DIR}/${PETSC_ARCH}/conf/petscvariables
include ${PETSC_DIR}/conf/base

CEXT = c
CFLAGS = -DPETSC_USE_LOG

#-Wall -Wold-style-cast -Woverloaded-virtual -Weffc++ -Wp64

LIBS =  ${PETSC_LIB} 

all: fuseMg2d fuseMg3d

%.o: %.$(CEXT)
	$(PCC) -c $(CFLAGS) $(PETSC_INCLUDE) $< -o $@

fuseMg2d: ./fuseMg2d.o 
	$(PCC) $(CFLAGS) $^ -o $@ $(PETSC_LIB) 

fuseMg3d: ./fuseMg3d.o ./auxillary.o ./stiffness.o ./intergrid.o ./kspshell.o
	$(PCC) $(CFLAGS) $^ -o $@ $(PETSC_LIB) 

clobber:
	rm -rf ./*~ ./*.o
	rm -rf fuseMg2d fuseMg3d


