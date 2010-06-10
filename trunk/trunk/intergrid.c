
#include "fuseMg3d.h"
#include <stdlib.h>

extern PetscLogEvent createRPmatEvent;
extern PetscLogEvent buildPmatEvent;

PetscErrorCode createRPmats(Mat** _Rmat, Mat** _Pmat, DA* da, PetscInt nlevels) {
  Mat* Rmat;
  Mat  tmpRmat;
  Mat* Pmat;
  int lev;
  int npes;
  PetscInt xs, ys, zs, nxc, nyc, nzc, nxf, nyf, nzf;
  PetscInt useTmpToBuildRmat;

  PetscFunctionBegin;

  PetscLogEventBegin(createRPmatEvent, 0, 0, 0, 0);

  useTmpToBuildRmat = 0;
  PetscOptionsGetInt(PETSC_NULL, "-useTmpToBuildRmat", &useTmpToBuildRmat, PETSC_NULL);

  MPI_Comm_size(PETSC_COMM_WORLD, &npes);

  Rmat = (Mat*)(malloc( sizeof(Mat)*(nlevels - 1) ));
  Pmat = (Mat*)(malloc( sizeof(Mat)*(nlevels - 1) ));

  for(lev = 0; lev < (nlevels - 1); lev++) {
    nxc = nyc = nzc = 0;
    nxf = nyf = nzf = 0;

    if(da[lev]) {
      DAGetCorners(da[lev], &xs, &ys, &zs, &nxc, &nyc, &nzc);
    } 

    if(da[lev + 1]) {
      DAGetCorners(da[lev + 1], &xs, &ys, &zs, &nxf, &nyf, &nzf);
    }

    MatCreate(PETSC_COMM_WORLD, (Pmat + lev));
    MatSetSizes(Pmat[lev], (nxf*nyf*nzf), (nxc*nyc*nzc), PETSC_DETERMINE, PETSC_DETERMINE);
    if(npes > 1) {
      MatSetType(Pmat[lev], MATMPIAIJ);
      MatMPIAIJSetPreallocation(Pmat[lev], 8, PETSC_NULL, 8, PETSC_NULL);
    } else {
      MatSetType(Pmat[lev], MATSEQAIJ);
      MatSeqAIJSetPreallocation(Pmat[lev], 8, PETSC_NULL);
    }

    buildPmat(Pmat[lev], da[lev], da[lev + 1]);

    if(useTmpToBuildRmat) {
      MatDuplicate(Pmat[lev], MAT_COPY_VALUES, &tmpRmat);
      MatScale(tmpRmat, 0.125);
      MatCreateTranspose(tmpRmat, (Rmat + lev));
      MatDestroy(tmpRmat);
    } else {
      MatTranspose(Pmat[lev], MAT_INITIAL_MATRIX, (Rmat + lev));
      MatScale(Rmat[lev], 0.125);
    }
  }//end for lev

  *_Rmat = Rmat;
  *_Pmat = Pmat;

  PetscLogEventEnd(createRPmatEvent, 0, 0, 0, 0);

  PetscFunctionReturn(0);

}

PetscErrorCode buildPmat(Mat Pmat, DA dac, DA daf) {
  Vec fNall1;
  Vec fNall2[2];
  Vec fNall3[4];
  Vec fNall4[8];
  Vec fidNatural;
  int npes, i;
  PetscInt fLocalSize = 0;

  PetscFunctionBegin;

  PetscLogEventBegin(buildPmatEvent, 0, 0, 0, 0);

  MPI_Comm_size(PETSC_COMM_WORLD, &npes);

  if(daf) {
    Vec fidGlobal;
    PetscScalar* fidArr;
    PetscInt fStId, fEndId;

    DACreateGlobalVector(daf, &fidGlobal);
    DACreateNaturalVector(daf, &fidNatural);

    VecGetLocalSize(fidNatural, &fLocalSize);

    VecGetOwnershipRange(fidGlobal, &fStId, &fEndId);

    VecGetArray(fidGlobal, &fidArr);

    for(i = fStId; i < fEndId; i++) {
      fidArr[i - fStId] = i;
    }//end for i

    VecRestoreArray(fidGlobal, &fidArr);

    DAGlobalToNaturalBegin(daf, fidGlobal, INSERT_VALUES, fidNatural);
    DAGlobalToNaturalEnd(daf, fidGlobal, INSERT_VALUES, fidNatural);

    VecDestroy(fidGlobal);
  }//end if fine active

  VecCreate(PETSC_COMM_WORLD, &fNall1);
  VecSetSizes(fNall1, fLocalSize, PETSC_DECIDE);
  if(npes > 1) {
    VecSetType(fNall1, VECMPI);
  } else {
    VecSetType(fNall1, VECSEQ);
  }

  for(i = 0; i < 2; i++) {
    VecDuplicate(fNall1, &(fNall2[i]));
  }//end for i

  for(i = 0; i < 4; i++) {
    VecDuplicate(fNall1, &(fNall3[i]));
  }//end for i

  for(i = 0; i < 8; i++) {
    VecDuplicate(fNall1, &(fNall4[i]));
  }//end for i

  VecSet(fNall1, -1.0);

  for(i = 0; i < 2; i++) {
    VecSet(fNall2[i], -1.0);
  }//end for i

  for(i = 0; i < 4; i++) {
    VecSet(fNall3[i], -1.0);
  }//end for i

  for(i = 0; i < 8; i++) {
    VecSet(fNall4[i], -1.0);
  }//end for i

  if(dac) {
    Vec cidGlobal;
    Vec cidLocal;
    PetscScalar* cidArr;
    PetscScalar*** cidLocalArr;
    PetscInt cStId, cEndId;
    PetscInt Nc, Nf, xsc, ysc, zsc, nxc, nyc, nzc;
    PetscInt xci, yci, zci;

    DAGetInfo(dac, PETSC_NULL, &Nc, PETSC_NULL, PETSC_NULL,
        PETSC_NULL, PETSC_NULL, PETSC_NULL, 
        PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL);

    Nf = (2*(Nc - 1)) + 1;

    DAGetCorners(dac, &xsc, &ysc, &zsc, &nxc, &nyc, &nzc);

    DACreateGlobalVector(dac, &cidGlobal);
    DACreateLocalVector(dac, &cidLocal);

    VecGetOwnershipRange(cidGlobal, &cStId, &cEndId);

    VecGetArray(cidGlobal, &cidArr);

    for(i = cStId; i < cEndId; i++) {
      cidArr[i - cStId] = i;
    }//end for i

    VecRestoreArray(cidGlobal, &cidArr);

    DAGlobalToLocalBegin(dac, cidGlobal, INSERT_VALUES, cidLocal);
    DAGlobalToLocalEnd(dac, cidGlobal, INSERT_VALUES, cidLocal);

    DAVecGetArray(dac, cidLocal, &cidLocalArr);

    //Z = 0 and Z = (N - 1) are Dirichlet Boundaries

    for(zci = zsc; zci < (zsc + nzc); zci++) {
      for(yci = ysc; yci < (ysc + nyc); yci++) {
        for(xci = xsc; xci < (xsc + nxc); xci++) {
          PetscInt row;
          PetscScalar value;

          if( (zci != 0) && (zci != (Nc - 1)) ) {
            //Vertices
            row = __DOF__((2*xci), (2*yci), (2*zci), Nf);

            value = cidLocalArr[zci][yci][xci];
            VecSetValue(fNall1, row, value, INSERT_VALUES);

            //Right Edge
            if(xci < (Nc - 1)) {
              row = __DOF__(((2*xci) + 1), (2*yci), (2*zci), Nf);

              value = cidLocalArr[zci][yci][xci]; 
              VecSetValue(fNall2[0], row, value, INSERT_VALUES);

              value = cidLocalArr[zci][yci][xci + 1]; 
              VecSetValue(fNall2[1], row, value, INSERT_VALUES);
            }

            //Back Edge
            if(yci < (Nc - 1)) {
              row = __DOF__((2*xci), ((2*yci) + 1), (2*zci), Nf);

              value = cidLocalArr[zci][yci][xci]; 
              VecSetValue(fNall2[0], row, value, INSERT_VALUES);

              value = cidLocalArr[zci][yci + 1][xci];
              VecSetValue(fNall2[1], row, value, INSERT_VALUES);
            }

            //Z-Face
            if( (xci < (Nc - 1)) && (yci < (Nc - 1)) ) {
              row = __DOF__(((2*xci) + 1), ((2*yci) + 1), (2*zci), Nf);

              value = cidLocalArr[zci][yci][xci]; 
              VecSetValue(fNall3[0], row, value, INSERT_VALUES);

              value = cidLocalArr[zci][yci][xci + 1]; 
              VecSetValue(fNall3[1], row, value, INSERT_VALUES);

              value = cidLocalArr[zci][yci + 1][xci]; 
              VecSetValue(fNall3[2], row, value, INSERT_VALUES);

              value = cidLocalArr[zci][yci + 1][xci + 1];
              VecSetValue(fNall3[3], row, value, INSERT_VALUES);
            }
          }

          //Top Edge
          if(zci < (Nc - 1)) {
            row = __DOF__((2*xci), (2*yci), ((2*zci) + 1), Nf);

            if(zci) {
              if( zci != (Nc - 2) ) {
                value = cidLocalArr[zci][yci][xci]; 
                VecSetValue(fNall2[0], row, value, INSERT_VALUES);

                value = cidLocalArr[zci + 1][yci][xci];
                VecSetValue(fNall2[1], row, value, INSERT_VALUES);
              } else {
                value = cidLocalArr[zci][yci][xci];
                VecSetValue(fNall2[0], row, value, INSERT_VALUES);
              }
            } else {
              if( zci != (Nc - 2) ) {
                value = cidLocalArr[zci + 1][yci][xci];
                VecSetValue(fNall2[0], row, value, INSERT_VALUES);
              }
            }
          }

          //Y-Face
          if( (xci < (Nc - 1)) && (zci < (Nc - 1)) ) {
            row = __DOF__(((2*xci) + 1), (2*yci), ((2*zci) + 1), Nf);

            if(zci) {
              if( zci != (Nc - 2) ) {
                value = cidLocalArr[zci][yci][xci]; 
                VecSetValue(fNall3[0], row, value, INSERT_VALUES);

                value = cidLocalArr[zci + 1][yci][xci];
                VecSetValue(fNall3[1], row, value, INSERT_VALUES);

                value = cidLocalArr[zci][yci][xci + 1];
                VecSetValue(fNall3[2], row, value, INSERT_VALUES);

                value = cidLocalArr[zci + 1][yci][xci + 1];
                VecSetValue(fNall3[3], row, value, INSERT_VALUES);
              } else {
                value = cidLocalArr[zci][yci][xci]; 
                VecSetValue(fNall3[0], row, value, INSERT_VALUES);

                value = cidLocalArr[zci][yci][xci + 1];
                VecSetValue(fNall3[1], row, value, INSERT_VALUES);
              }
            } else {
              if( zci != (Nc - 2) ) {
                value = cidLocalArr[zci + 1][yci][xci]; 
                VecSetValue(fNall3[0], row, value, INSERT_VALUES);

                value = cidLocalArr[zci + 1][yci][xci + 1];
                VecSetValue(fNall3[1], row, value, INSERT_VALUES);
              }
            }
          }

          //X-Face
          if( (yci < (Nc - 1)) && (zci < (Nc - 1)) ) {
            row = __DOF__((2*xci), ((2*yci) + 1), ((2*zci) + 1), Nf);

            if(zci) {
              if( zci != (Nc - 2) ) {
                value = cidLocalArr[zci][yci][xci]; 
                VecSetValue(fNall3[0], row, value, INSERT_VALUES);

                value = cidLocalArr[zci + 1][yci][xci];
                VecSetValue(fNall3[1], row, value, INSERT_VALUES);

                value = cidLocalArr[zci][yci + 1][xci]; 
                VecSetValue(fNall3[2], row, value, INSERT_VALUES);

                value = cidLocalArr[zci + 1][yci + 1][xci];
                VecSetValue(fNall3[3], row, value, INSERT_VALUES);
              } else {
                value = cidLocalArr[zci][yci][xci]; 
                VecSetValue(fNall3[0], row, value, INSERT_VALUES);

                value = cidLocalArr[zci][yci + 1][xci];
                VecSetValue(fNall3[1], row, value, INSERT_VALUES);
              }
            } else {
              if( zci != (Nc - 2) ) {
                value = cidLocalArr[zci + 1][yci][xci]; 
                VecSetValue(fNall3[0], row, value, INSERT_VALUES);

                value = cidLocalArr[zci + 1][yci + 1][xci];
                VecSetValue(fNall3[1], row, value, INSERT_VALUES);
              }
            }
          }

          //Cell Center
          if( (xci < (Nc - 1)) && (yci < (Nc - 1)) && (zci < (Nc - 1)) ) {
            row = __DOF__(((2*xci) + 1), ((2*yci) + 1), ((2*zci) + 1), Nf);

            if(zci) {
              if( zci != (Nc - 2) ) {
                value = cidLocalArr[zci][yci][xci]; 
                VecSetValue(fNall4[0], row, value, INSERT_VALUES);

                value = cidLocalArr[zci][yci][xci + 1]; 
                VecSetValue(fNall4[1], row, value, INSERT_VALUES);

                value = cidLocalArr[zci][yci + 1][xci]; 
                VecSetValue(fNall4[2], row, value, INSERT_VALUES);

                value = cidLocalArr[zci][yci + 1][xci + 1];
                VecSetValue(fNall4[3], row, value, INSERT_VALUES);

                value = cidLocalArr[zci + 1][yci][xci]; 
                VecSetValue(fNall4[4], row, value, INSERT_VALUES);

                value = cidLocalArr[zci + 1][yci][xci + 1]; 
                VecSetValue(fNall4[5], row, value, INSERT_VALUES);

                value = cidLocalArr[zci + 1][yci + 1][xci]; 
                VecSetValue(fNall4[6], row, value, INSERT_VALUES);

                value = cidLocalArr[zci + 1][yci + 1][xci + 1];
                VecSetValue(fNall4[7], row, value, INSERT_VALUES);
              } else {
                value = cidLocalArr[zci][yci][xci]; 
                VecSetValue(fNall4[0], row, value, INSERT_VALUES);

                value = cidLocalArr[zci][yci][xci + 1]; 
                VecSetValue(fNall4[1], row, value, INSERT_VALUES);

                value = cidLocalArr[zci][yci + 1][xci]; 
                VecSetValue(fNall4[2], row, value, INSERT_VALUES);

                value = cidLocalArr[zci][yci + 1][xci + 1];
                VecSetValue(fNall4[3], row, value, INSERT_VALUES);
              }
            } else {
              if( zci != (Nc - 2) ) {
                value = cidLocalArr[zci + 1][yci][xci]; 
                VecSetValue(fNall4[0], row, value, INSERT_VALUES);

                value = cidLocalArr[zci + 1][yci][xci + 1]; 
                VecSetValue(fNall4[1], row, value, INSERT_VALUES);

                value = cidLocalArr[zci + 1][yci + 1][xci]; 
                VecSetValue(fNall4[2], row, value, INSERT_VALUES);

                value = cidLocalArr[zci + 1][yci + 1][xci + 1];
                VecSetValue(fNall4[3], row, value, INSERT_VALUES);
              }
            }
          }

        }//end for xci
      }//end for yci
    }//end for zci

    DAVecRestoreArray(dac, cidLocal, &cidLocalArr);

    VecDestroy(cidGlobal);
    VecDestroy(cidLocal);
  }//end if coarse active

  VecAssemblyBegin(fNall1);
  VecAssemblyEnd(fNall1);

  for(i = 0; i < 2; i++) {
    VecAssemblyBegin(fNall2[i]);
    VecAssemblyEnd(fNall2[i]);
  }//end for i

  for(i = 0; i < 4; i++) {
    VecAssemblyBegin(fNall3[i]);
    VecAssemblyEnd(fNall3[i]);
  }//end for i

  for(i = 0; i < 8; i++) {
    VecAssemblyBegin(fNall4[i]);
    VecAssemblyEnd(fNall4[i]);
  }//end for i

  MatZeroEntries(Pmat);

  if(daf) {
    PetscScalar* farr1;
    PetscScalar* farr2[2];
    PetscScalar* farr3[4];
    PetscScalar* farr4[8];
    PetscScalar* fidArr;

    VecGetArray(fNall1, &farr1);

    for(i = 0; i < 2; i++) {
      VecGetArray(fNall2[i], &(farr2[i]));
    }//end for i

    for(i = 0; i < 4; i++) {
      VecGetArray(fNall3[i], &(farr3[i]));
    }//end for i

    for(i = 0; i < 8; i++) {
      VecGetArray(fNall4[i], &(farr4[i]));
    }//end for i

    VecGetArray(fidNatural, &fidArr);

    for(i = 0; i < fLocalSize; i++) {
      int j;
      PetscInt row = (PetscInt)(fidArr[i]);

      if(farr1[i] >= 0) {
        PetscInt col = (PetscInt)(farr1[i]);
        PetscScalar value = 1.0;
        MatSetValues(Pmat, 1, &row, 1, &col, &value, INSERT_VALUES);
      }

      for(j = 0; j < 2; j++) {
        if(farr2[j][i] >= 0) {
          PetscInt col = (PetscInt)(farr2[j][i]);
          PetscScalar value = 0.5;
          MatSetValues(Pmat, 1, &row, 1, &col, &value, INSERT_VALUES);
        }
      }//end for j

      for(j = 0; j < 4; j++) {
        if(farr3[j][i] >= 0) {
          PetscInt col = (PetscInt)(farr3[j][i]);
          PetscScalar value = 0.25;
          MatSetValues(Pmat, 1, &row, 1, &col, &value, INSERT_VALUES);
        }
      }//end for j

      for(j = 0; j < 8; j++) {
        if(farr4[j][i] >= 0) {
          PetscInt col = (PetscInt)(farr4[j][i]);
          PetscScalar value = 0.125;
          MatSetValues(Pmat, 1, &row, 1, &col, &value, INSERT_VALUES);
        }
      }//end for j
    }//end for i

    VecRestoreArray(fidNatural, &fidArr);

    VecRestoreArray(fNall1, &farr1);

    for(i = 0; i < 2; i++) {
      VecRestoreArray(fNall2[i], &(farr2[i]));
    }//end for i

    for(i = 0; i < 4; i++) {
      VecRestoreArray(fNall3[i], &(farr3[i]));
    }//end for i

    for(i = 0; i < 8; i++) {
      VecRestoreArray(fNall4[i], &(farr4[i]));
    }//end for i

  }//end if fine active

  MatAssemblyBegin(Pmat, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(Pmat, MAT_FINAL_ASSEMBLY);

  if(daf) {
    VecDestroy(fidNatural);
  }//end if fine active

  VecDestroy(fNall1);

  for(i = 0; i < 2; i++) {
    VecDestroy(fNall2[i]);
  }//end for i

  for(i = 0; i < 4; i++) {
    VecDestroy(fNall3[i]);
  }//end for i

  for(i = 0; i < 8; i++) {
    VecDestroy(fNall4[i]);
  }//end for i

  PetscLogEventEnd(buildPmatEvent, 0, 0, 0, 0);

  PetscFunctionReturn(0);

}



