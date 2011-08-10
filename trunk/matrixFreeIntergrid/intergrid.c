
#include "fuseMg3d.h"
#include <stdlib.h>

extern int rMatvecEvent;
extern int pMatvecEvent;

void createRmat(Mat** _Rmat, TransferOpData* rpData, DA* da, PetscInt nlevels) {
  Mat* Rmat;
  int lev;
  PetscInt xs, ys, zs, nxc, nyc, nzc, nxf, nyf, nzf;

  Rmat = (Mat*)(malloc( sizeof(Mat)*(nlevels - 1) ));

  for(lev = 0; lev < (nlevels - 1); lev++) {
    nxc = nyc = nzc = 0;
    nxf = nyf = nzf = 0;

    if(da[lev]) {
      DAGetCorners(da[lev], &xs, &ys, &zs, &nxc, &nyc, &nzc);
    } 

    if(da[lev + 1]) {
      DAGetCorners(da[lev + 1], &xs, &ys, &zs, &nxf, &nyf, &nzf);
    }

    MatCreateShell(PETSC_COMM_WORLD, (nxc*nyc*nzc), (nxf*nyf*nzf), 
        PETSC_DETERMINE, PETSC_DETERMINE, (rpData + lev), (Rmat + lev));
    MatShellSetOperation(Rmat[lev], MATOP_MULT_TRANSPOSE, (void(*)(void)) prolongMatvec);
    MatShellSetOperation(Rmat[lev], MATOP_MULT, (void(*)(void)) restrictMatvec);
    MatShellSetOperation(Rmat[lev], MATOP_MULT_ADD, (void(*)(void)) addRestrictMatvec);
    MatShellSetOperation(Rmat[lev], MATOP_MULT_TRANSPOSE_ADD, (void(*)(void)) addProlongMatvec);
    MatShellSetOperation(Rmat[lev], MATOP_DESTROY, (void(*)(void)) rpDestroy);
  }//end for lev

  *_Rmat = Rmat;
}

void createRmatData(TransferOpData** _rpData, DA* da, PetscInt nlevels) {
  TransferOpData* rpData;
  int npes, lev;
  PetscInt cLocalSize, fLocalSize;

  MPI_Comm_size(PETSC_COMM_WORLD, &npes);

  rpData =  (TransferOpData*)(malloc( sizeof(TransferOpData)*(nlevels - 1) ));

  for(lev = 0; lev < (nlevels - 1); lev++) {
    rpData[lev].dac = NULL;
    rpData[lev].daf = NULL;
    rpData[lev].addRtmp = NULL;
    rpData[lev].addPtmp = NULL;
    rpData[lev].coarseGlobalActive = NULL;
    rpData[lev].coarseNaturalActive = NULL;
    rpData[lev].coarseNaturalAll = NULL;
    rpData[lev].fineGlobalActive = NULL;
    rpData[lev].fineNaturalActive = NULL;
    rpData[lev].fineNaturalAll = NULL;

    VecCreate(PETSC_COMM_WORLD, &(rpData[lev].coarseNaturalAll));
    if(da[lev]) {
      rpData[lev].dac = da[lev];
      DACreateGlobalVector(da[lev], &(rpData[lev].coarseGlobalActive));
      DACreateNaturalVector(da[lev], &(rpData[lev].coarseNaturalActive));
      VecGetLocalSize(rpData[lev].coarseNaturalActive, &cLocalSize);
      VecSetSizes(rpData[lev].coarseNaturalAll, cLocalSize, PETSC_DECIDE);
    } else {
      VecSetSizes(rpData[lev].coarseNaturalAll, 0, PETSC_DECIDE);
    }
    if(npes > 1) {
      VecSetType(rpData[lev].coarseNaturalAll, VECMPI);
    } else {
      VecSetType(rpData[lev].coarseNaturalAll, VECSEQ);
    }

    VecCreate(PETSC_COMM_WORLD, &(rpData[lev].fineNaturalAll));
    if(da[lev + 1]) {
      rpData[lev].daf = da[lev + 1];
      DACreateGlobalVector(da[lev + 1], &(rpData[lev].fineGlobalActive));
      DACreateNaturalVector(da[lev + 1], &(rpData[lev].fineNaturalActive));
      VecGetLocalSize(rpData[lev].fineNaturalActive, &fLocalSize);
      VecSetSizes(rpData[lev].fineNaturalAll, fLocalSize, PETSC_DECIDE);
    } else {
      VecSetSizes(rpData[lev].fineNaturalAll, 0, PETSC_DECIDE);
    }
    if(npes > 1) {
      VecSetType(rpData[lev].fineNaturalAll, VECMPI);
    } else {
      VecSetType(rpData[lev].fineNaturalAll, VECSEQ);
    }

  }//end for lev

  *_rpData = rpData;
}

PetscErrorCode prolongMatvec(Mat R, Vec coarse, Vec fine) {

  TransferOpData *data; 
  DA dac;
  DA daf;
  PetscInt Nc, Nf, xsc, ysc, zsc, nxc, nyc, nzc;
  PetscInt xci, yci, zci;
  Vec coarseLocal;
  Vec coarseActive;
  Vec fineGlobalActive;
  Vec fineNaturalActive;
  Vec fineNaturalAll;
  PetscScalar*** coarseArr;
  PetscScalar* fNaturalActiveArr;
  PetscScalar* fGlobalAllArr;
  PetscScalar* coarseAllArr;
  PetscScalar vals[8];
  PetscInt rows[8];

  PetscFunctionBegin;

  PetscLogEventBegin(pMatvecEvent, 0, 0, 0, 0);

  MatShellGetContext(R, (void **)&data);

  dac = data->dac;
  daf = data->daf;

  fineNaturalAll = data->fineNaturalAll;

  if(daf) {
    fineGlobalActive = data->fineGlobalActive;
    fineNaturalActive = data->fineNaturalActive;
    VecGetArray(fineNaturalActive, &fNaturalActiveArr);

    VecPlaceArray(fineNaturalAll, fNaturalActiveArr);
  }

  VecZeroEntries(fineNaturalAll);

  if(dac) {
    coarseActive = data->coarseGlobalActive;

    VecGetArray(coarse, &coarseAllArr);

    VecPlaceArray(coarseActive, coarseAllArr);

    DAGetInfo(dac, PETSC_NULL, &Nc, PETSC_NULL, PETSC_NULL,
        PETSC_NULL, PETSC_NULL, PETSC_NULL, 
        PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL);

    Nf = (2*(Nc - 1)) + 1;

    DAGetCorners(dac, &xsc, &ysc, &zsc, &nxc, &nyc, &nzc);

    DAGetLocalVector(dac, &coarseLocal);

    DAGlobalToLocalBegin(dac, coarseActive, INSERT_VALUES, coarseLocal);
    DAGlobalToLocalEnd(dac, coarseActive, INSERT_VALUES, coarseLocal);

    DAVecGetArray(dac, coarseLocal, &coarseArr);

    //Z = 0 and Z = (N - 1) are Dirichlet Boundaries

    for(zci = zsc; zci < (zsc + nzc); zci++) {
      for(yci = ysc; yci < (ysc + nyc); yci++) {
        for(xci = xsc; xci < (xsc + nxc); xci++) {
          PetscInt valCount = 0;

          if( (zci != 0) && (zci != (Nc - 1)) ) {
            //Vertices
            vals[valCount] = coarseArr[zci][yci][xci];
            rows[valCount] = __DOF__((2*xci), (2*yci), (2*zci), Nf);
            valCount++;

            //Right Edge
            if(xci < (Nc - 1)) {
              vals[valCount] = 0.5*(coarseArr[zci][yci][xci] + coarseArr[zci][yci][xci + 1]);
              rows[valCount] = __DOF__(((2*xci) + 1), (2*yci), (2*zci), Nf);
              valCount++;
            }

            //Back Edge
            if(yci < (Nc - 1)) {
              vals[valCount] = 0.5*(coarseArr[zci][yci][xci] + coarseArr[zci][yci + 1][xci]);
              rows[valCount] = __DOF__((2*xci), ((2*yci) + 1), (2*zci), Nf);
              valCount++;
            }

            //Z-Face
            if( (xci < (Nc - 1)) && (yci < (Nc - 1)) ) {
              vals[valCount] = 0.25*(coarseArr[zci][yci][xci] + coarseArr[zci][yci][xci + 1] 
                  + coarseArr[zci][yci + 1][xci] + coarseArr[zci][yci + 1][xci + 1]);
              rows[valCount] = __DOF__(((2*xci) + 1), ((2*yci) + 1), (2*zci), Nf);
              valCount++;
            }
          }

          //Top Edge
          if(zci < (Nc - 1)) {
            rows[valCount] = __DOF__((2*xci), (2*yci), ((2*zci) + 1), Nf);
            if(zci) {
              if( zci != (Nc - 2) ) {
                vals[valCount] = 0.5*(coarseArr[zci][yci][xci] + coarseArr[zci + 1][yci][xci]);
                valCount++;
              } else {
                vals[valCount] = 0.5*coarseArr[zci][yci][xci];
                valCount++;
              }
            } else {
              if( zci != (Nc - 2) ) {
                vals[valCount] = 0.5*coarseArr[zci + 1][yci][xci];
                valCount++;
              }
            }
          }

          //Y-Face
          if( (xci < (Nc - 1)) && (zci < (Nc - 1)) ) {
            rows[valCount] = __DOF__(((2*xci) + 1), (2*yci), ((2*zci) + 1), Nf);
            if(zci) {
              if( zci != (Nc - 2) ) {
                vals[valCount] = 0.25*(coarseArr[zci][yci][xci] + coarseArr[zci + 1][yci][xci]
                    + coarseArr[zci][yci][xci + 1] + coarseArr[zci + 1][yci][xci + 1]);
                valCount++;
              } else {
                vals[valCount] = 0.25*(coarseArr[zci][yci][xci] + coarseArr[zci][yci][xci + 1]);
                valCount++;
              }
            } else {
              if( zci != (Nc - 2) ) {
                vals[valCount] = 0.25*(coarseArr[zci + 1][yci][xci] + coarseArr[zci + 1][yci][xci + 1]);
                valCount++;
              }
            }
          }

          //X-Face
          if( (yci < (Nc - 1)) && (zci < (Nc - 1)) ) {
            rows[valCount] = __DOF__((2*xci), ((2*yci) + 1), ((2*zci) + 1), Nf);
            if(zci) {
              if( zci != (Nc - 2) ) {
                vals[valCount] = 0.25*(coarseArr[zci][yci][xci] + coarseArr[zci + 1][yci][xci]
                    + coarseArr[zci][yci + 1][xci] + coarseArr[zci + 1][yci + 1][xci]);
                valCount++;
              } else {
                vals[valCount] = 0.25*(coarseArr[zci][yci][xci] + coarseArr[zci][yci + 1][xci]);
                valCount++;
              }
            } else {
              if( zci != (Nc - 2) ) {
                vals[valCount] = 0.25*(coarseArr[zci + 1][yci][xci] + coarseArr[zci + 1][yci + 1][xci]);
                valCount++;
              }
            }
          }

          //Cell Center
          if( (xci < (Nc - 1)) && (yci < (Nc - 1)) && (zci < (Nc - 1)) ) {
            rows[valCount] = __DOF__(((2*xci) + 1), ((2*yci) + 1), ((2*zci) + 1), Nf);
            if(zci) {
              if( zci != (Nc - 2) ) {
                vals[valCount] = 0.125*(coarseArr[zci][yci][xci] + coarseArr[zci][yci][xci + 1] 
                    + coarseArr[zci][yci + 1][xci] + coarseArr[zci][yci + 1][xci + 1]
                    + coarseArr[zci + 1][yci][xci] + coarseArr[zci + 1][yci][xci + 1] 
                    + coarseArr[zci + 1][yci + 1][xci] + coarseArr[zci + 1][yci + 1][xci + 1]);
                valCount++;
              } else {
                vals[valCount] = 0.125*(coarseArr[zci][yci][xci] + coarseArr[zci][yci][xci + 1] 
                    + coarseArr[zci][yci + 1][xci] + coarseArr[zci][yci + 1][xci + 1]);
                valCount++;
              }
            } else {
              if( zci != (Nc - 2) ) {
                vals[valCount] = 0.125*(coarseArr[zci + 1][yci][xci] + coarseArr[zci + 1][yci][xci + 1] 
                    + coarseArr[zci + 1][yci + 1][xci] + coarseArr[zci + 1][yci + 1][xci + 1]);
                valCount++;
              }
            }
          }

          VecSetValues(fineNaturalAll, valCount, rows, vals, INSERT_VALUES);
        }//end for xci
      }//end for yci
    }//end for zci

    DAVecRestoreArray(dac, coarseLocal, &coarseArr);

    DARestoreLocalVector(dac, &coarseLocal);

    VecResetArray(coarseActive);

    VecRestoreArray(coarse, &coarseAllArr);
  }//end if coarse active

  VecAssemblyBegin(fineNaturalAll);
  VecAssemblyEnd(fineNaturalAll);

  if(daf) {
    VecResetArray(fineNaturalAll);

    VecRestoreArray(fineNaturalActive, &fNaturalActiveArr);

    VecGetArray(fine, &fGlobalAllArr);

    VecPlaceArray(fineGlobalActive, fGlobalAllArr);

    DANaturalToGlobalBegin(daf, fineNaturalActive, INSERT_VALUES, fineGlobalActive);
    DANaturalToGlobalEnd(daf, fineNaturalActive, INSERT_VALUES, fineGlobalActive);

    VecResetArray(fineGlobalActive);

    VecRestoreArray(fine, &fGlobalAllArr);
  }

  PetscLogEventEnd(pMatvecEvent, 0, 0, 0, 0);

  PetscFunctionReturn(0);
}

PetscErrorCode restrictMatvec(Mat R, Vec fine, Vec coarse) {

  TransferOpData *data; 
  DA dac;
  DA daf;
  PetscInt Nc, Nf, xsf, ysf, zsf, nxf, nyf, nzf;
  PetscInt xsfeven, ysfeven, zsfeven;
  PetscInt xsfodd, ysfodd, zsfodd;
  PetscInt xfi, yfi, zfi;
  PetscInt xci, yci, zci;
  Vec coarseNaturalAll;
  Vec coarseNaturalActive;
  Vec coarseGlobalActive;
  Vec fineActive;
  Vec fineLocal;
  PetscScalar*** fineArr;
  PetscScalar* fineAllArr;
  PetscScalar* cNaturalActiveArr;
  PetscScalar* cGlobalAllArr;
  PetscScalar vals[8];
  PetscInt rows[8];

  PetscFunctionBegin;

  PetscLogEventBegin(rMatvecEvent, 0, 0, 0, 0);

  MatShellGetContext(R, (void **)&data);

  dac = data->dac;
  daf = data->daf;

  coarseNaturalAll = data->coarseNaturalAll;

  if(dac) {
    coarseGlobalActive = data->coarseGlobalActive;
    coarseNaturalActive = data->coarseNaturalActive;

    VecGetArray(coarseNaturalActive, &cNaturalActiveArr);

    VecPlaceArray(coarseNaturalAll, cNaturalActiveArr);
  }

  VecZeroEntries(coarseNaturalAll);

  if(daf) {
    fineActive = data->fineGlobalActive;

    VecGetArray(fine, &fineAllArr);

    VecPlaceArray(fineActive, fineAllArr);

    DAGetInfo(daf, PETSC_NULL, &Nf, PETSC_NULL, PETSC_NULL,
        PETSC_NULL, PETSC_NULL, PETSC_NULL, 
        PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL);

    Nc = ((Nf - 1)/2) + 1;

    DAGetCorners(daf, &xsf, &ysf, &zsf, &nxf, &nyf, &nzf);

    if( (xsf%2) == 0 ) {
      xsfeven = xsf;
      xsfodd = xsf + 1;
    } else {
      xsfodd = xsf;
      xsfeven = xsf + 1;
    }

    if( (ysf%2) == 0 ) {
      ysfeven = ysf;
      ysfodd = ysf + 1;
    } else {
      ysfodd = ysf;
      ysfeven = ysf + 1;
    }

    if( (zsf%2) == 0 ) {
      zsfeven = zsf;
      zsfodd = zsf + 1;
    } else {
      zsfodd = zsf;
      zsfeven = zsf + 1;
    }

    DAGetLocalVector(daf, &fineLocal);

    DAGlobalToLocalBegin(daf, fineActive, INSERT_VALUES, fineLocal);
    DAGlobalToLocalEnd(daf, fineActive, INSERT_VALUES, fineLocal);

    DAVecGetArray(daf, fineLocal, &fineArr);

    //Z = 0 and Z = (N - 1) are Dirichlet Boundaries

    for(zfi = zsfeven; zfi < (zsf + nzf); zfi += 2) {
      if( (zfi != 0) && (zfi != (Nf - 1)) ) {
        zci = (zfi/2);

        for(yfi = ysfeven; yfi < (ysf + nyf); yfi += 2) {
          yci = (yfi/2);

          //Vertices
          for(xfi = xsfeven; xfi < (xsf + nxf); xfi += 2) {
            vals[0] = fineArr[zfi][yfi][xfi];
            xci = (xfi/2);
            rows[0] = __DOF__(xci, yci, zci, Nc);
            VecSetValues(coarseNaturalAll, 1, rows, vals, ADD_VALUES);
          }//end for xfi

          //Right Edge
          for(xfi = xsfodd; xfi < (xsf + nxf); xfi += 2) {
            vals[1] = vals[0] = 0.5*fineArr[zfi][yfi][xfi];
            xci = ((xfi - 1)/2);
            rows[0] = __DOF__(xci, yci, zci, Nc);
            rows[1] = __DOF__(xci + 1, yci, zci, Nc);
            VecSetValues(coarseNaturalAll, 2, rows, vals, ADD_VALUES);
          }//end for xfi
        }//end for yfi

        for(yfi = ysfodd; yfi < (ysf + nyf); yfi += 2) {
          yci = (yfi - 1)/2;

          //Back Edge
          for(xfi = xsfeven; xfi < (xsf + nxf); xfi += 2) {
            vals[1] = vals[0] = 0.5*fineArr[zfi][yfi][xfi];
            xci = xfi/2;
            rows[0] = __DOF__(xci, yci, zci, Nc);
            rows[1] = __DOF__(xci, yci + 1, zci, Nc);
            VecSetValues(coarseNaturalAll, 2, rows, vals, ADD_VALUES);
          }//end for xfi

          //Z Face
          for(xfi = xsfodd; xfi < (xsf + nxf); xfi += 2) {
            vals[3] = vals[2] = vals[1] = vals[0] = 0.25*fineArr[zfi][yfi][xfi];
            xci = (xfi - 1)/2;
            rows[0] = __DOF__(xci, yci, zci, Nc);
            rows[1] = __DOF__(xci + 1, yci, zci, Nc);
            rows[2] = __DOF__(xci, yci + 1, zci, Nc);
            rows[3] = __DOF__(xci + 1, yci + 1, zci, Nc);
            VecSetValues(coarseNaturalAll, 4, rows, vals, ADD_VALUES);
          }//end for xfi
        }//end for yfi
      }
    }//end for zfi

    for(zfi = zsfodd; zfi < (zsf + nzf); zfi += 2) {
      zci = (zfi - 1)/2;

      for(yfi = ysfeven; yfi < (ysf + nyf); yfi += 2) {
        yci = yfi/2;

        //Top Edge
        for(xfi = xsfeven; xfi < (xsf + nxf); xfi += 2) {
          PetscInt off = 0;
          vals[1] = vals[0] = 0.5*fineArr[zfi][yfi][xfi];
          xci = xfi/2;
          if(zci) {
            rows[0] = __DOF__(xci, yci, zci, Nc);
            off = 1;
          }
          if(zci != (Nc - 2)) {
            rows[off] = __DOF__(xci, yci, zci + 1, Nc);
            off += 1;
          }
          VecSetValues(coarseNaturalAll, off, rows, vals, ADD_VALUES);
        }//end for xfi

        //Y Face
        for(xfi = xsfodd; xfi < (xsf + nxf); xfi += 2) {
          PetscInt off = 0;
          vals[3] = vals[2] = vals[1] = vals[0] = 0.25*fineArr[zfi][yfi][xfi];
          xci = (xfi - 1)/2;
          if(zci) {
            rows[0] = __DOF__(xci, yci, zci, Nc);
            rows[1] = __DOF__(xci + 1, yci, zci, Nc);
            off = 2;
          }
          if(zci != (Nc - 2)) {
            rows[off] = __DOF__(xci, yci, zci + 1, Nc);
            rows[off + 1] = __DOF__(xci + 1, yci, zci + 1, Nc);
            off += 2;
          }
          VecSetValues(coarseNaturalAll, off, rows, vals, ADD_VALUES);
        }//end for xfi
      }//end for yfi

      for(yfi = ysfodd; yfi < (ysf + nyf); yfi += 2) {
        yci = (yfi - 1)/2;

        //X Face
        for(xfi = xsfeven; xfi < (xsf + nxf); xfi += 2) {
          PetscInt off = 0;
          vals[3] = vals[2] = vals[1] = vals[0] = 0.25*fineArr[zfi][yfi][xfi];
          xci = xfi/2;
          if(zci) {
            rows[0] = __DOF__(xci, yci, zci, Nc);
            rows[1] = __DOF__(xci, yci + 1, zci, Nc);
            off = 2;
          }
          if(zci != (Nc - 2)) {
            rows[off] = __DOF__(xci, yci, zci + 1, Nc);
            rows[off + 1] = __DOF__(xci, yci + 1, zci + 1, Nc);
            off += 2;
          }
          VecSetValues(coarseNaturalAll, off, rows, vals, ADD_VALUES);
        }//end for xfi

        //Cell Center
        for(xfi = xsfodd; xfi < (xsf + nxf); xfi += 2) {
          PetscInt off = 0;
          vals[7] = vals[6] = vals[5] = vals[4] = vals[3] = vals[2] = vals[1] = vals[0] = 0.125*fineArr[zfi][yfi][xfi];
          xci = (xfi - 1)/2;
          if(zci) {
            rows[0] = __DOF__(xci, yci, zci, Nc);
            rows[1] = __DOF__(xci + 1, yci, zci, Nc);
            rows[2] = __DOF__(xci, yci + 1, zci, Nc);
            rows[3] = __DOF__(xci + 1, yci + 1, zci, Nc);
            off = 4;
          }
          if(zci != (Nc - 2)) {
            rows[off] = __DOF__(xci, yci, zci + 1, Nc);
            rows[off + 1] = __DOF__(xci + 1, yci, zci + 1, Nc);
            rows[off + 2] = __DOF__(xci, yci + 1, zci + 1, Nc);
            rows[off + 3] = __DOF__(xci + 1, yci + 1, zci + 1, Nc);
            off += 4;
          }
          VecSetValues(coarseNaturalAll, off, rows, vals, ADD_VALUES);
        }//end for xfi
      }//end for yfi
    }//end for zfi

    DAVecRestoreArray(daf, fineLocal, &fineArr);

    DARestoreLocalVector(daf, &fineLocal);

    VecResetArray(fineActive);

    VecRestoreArray(fine, &fineAllArr);
  }//end if fine active

  VecAssemblyBegin(coarseNaturalAll);
  VecAssemblyEnd(coarseNaturalAll);

  VecScale(coarseNaturalAll, 0.125);

  if(dac) {
    VecResetArray(coarseNaturalAll);

    VecRestoreArray(coarseNaturalActive, &cNaturalActiveArr);

    VecGetArray(coarse, &cGlobalAllArr);

    VecPlaceArray(coarseGlobalActive, cGlobalAllArr);

    DANaturalToGlobalBegin(dac, coarseNaturalActive, INSERT_VALUES, coarseGlobalActive);
    DANaturalToGlobalEnd(dac, coarseNaturalActive, INSERT_VALUES, coarseGlobalActive);

    VecResetArray(coarseGlobalActive);

    VecRestoreArray(coarse, &cGlobalAllArr);
  }

  PetscLogEventEnd(rMatvecEvent, 0, 0, 0, 0);

  PetscFunctionReturn(0);
}

PetscErrorCode addProlongMatvec(Mat R, Vec v1, Vec v2, Vec v3) {
  PetscScalar one;
  TransferOpData *data;			
  Vec tmp;

  PetscFunctionBegin;

  one = 1.0;

  if((v2 != v3) && (v1 != v3)) {
    //Note This will fail only if v2==v3 or v1 ==v3!
    //(i.e they are identical copies pointing to the same memory location)
    MatMultTranspose(R, v1, v3);//v3 = R*v1
    VecAXPY(v3,one,v2);//v3 = v3+ v2=v2 + R*v1
  }else {
    //This is less efficient but failproof.
    MatShellGetContext( R, (void **)&data);
    tmp = data->addPtmp;
    if(tmp == NULL) {
      VecDuplicate(v3,&tmp);
      data->addPtmp = tmp;
    }
    MatMultTranspose(R, v1, tmp);//tmp=R'*v1;
    VecWAXPY(v3,one,v2,tmp);//v3 = (1*v2)+tmp=v2 + R'*v1
  }

  PetscFunctionReturn(0);
}

PetscErrorCode  addRestrictMatvec(Mat R, Vec v1, Vec v2, Vec v3) {

  PetscScalar one;
  TransferOpData *data;
  Vec tmp;

  PetscFunctionBegin;

  one = 1.0;

  if((v2 != v3) && (v1 != v3)) {
    //Note This will fail only if v2==v3 or v1 ==v3!(i.e they are identical copies pointing to the same memory location)
    MatMult(R, v1, v3);//v3 = R*v1
    VecAXPY(v3, one, v2);//v3 = v3+ v2=v2 + R*v1
  }else {
    //This is less efficient but failproof.
    MatShellGetContext( R, (void **)&data);
    tmp = data->addRtmp;
    if(tmp == NULL) {
      VecDuplicate(v3,&tmp);
      data->addRtmp = tmp;
    }
    MatMult(R, v1, tmp);//tmp=R*v1;
    VecWAXPY(v3, one, v2, tmp);//v3 = (1*v2)+tmp=v2 + R*v1
  }

  PetscFunctionReturn(0);
}

PetscErrorCode  rpDestroy(Mat R) {
  TransferOpData *data;

  PetscFunctionBegin;

  MatShellGetContext( R, (void **)&data);

  if(data->addRtmp) {
    VecDestroy(data->addRtmp);
    data->addRtmp = NULL;
  }
  if(data->addPtmp) {
    VecDestroy(data->addPtmp);
    data->addPtmp = NULL;
  }
  if(data->coarseGlobalActive) {
    VecDestroy(data->coarseGlobalActive);
    data->coarseGlobalActive = NULL;
  }
  if(data->coarseNaturalActive) {
    VecDestroy(data->coarseNaturalActive);
    data->coarseNaturalActive = NULL;
  }
  if(data->coarseNaturalAll) {
    VecDestroy(data->coarseNaturalAll);
    data->coarseNaturalAll = NULL;
  }
  if(data->fineGlobalActive) {
    VecDestroy(data->fineGlobalActive);
    data->fineGlobalActive = NULL;
  }
  if(data->fineNaturalActive) {
    VecDestroy(data->fineNaturalActive);
    data->fineNaturalActive = NULL;
  }
  if(data->fineNaturalAll) {
    VecDestroy(data->fineNaturalAll);
    data->fineNaturalAll = NULL;
  }

  PetscFunctionReturn(0);
}


