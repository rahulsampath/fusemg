
#include "fuseMg3d.h"
#include <stdlib.h>

extern PetscLogEvent kMatvecEvent;
extern PetscLogEvent nMatvecEvent;
extern PetscLogEvent coarsenKdataEvent;
extern PetscLogEvent resetKdataEvent;

PetscErrorCode coarseMatvec(Mat mat, Vec in, Vec out) {

  CoarseMatData *data;  
  PetscScalar* inarr;
  PetscScalar* outarr;

  PetscFunctionBegin;

  PetscLogEventBegin(kMatvecEvent, 0, 0, 0, 0);

  MatShellGetContext( mat, (void **)&data);

  VecGetArray(in, &inarr);
  VecGetArray(out, &outarr);

  if(data->mat) {
    VecPlaceArray(data->inTmp, inarr);
    VecPlaceArray(data->outTmp, outarr);

    MatMult(data->mat, data->inTmp, data->outTmp);

    VecResetArray(data->inTmp);
    VecResetArray(data->outTmp);
  }

  VecRestoreArray(in, &inarr);
  VecRestoreArray(out, &outarr);

  PetscLogEventEnd(kMatvecEvent, 0, 0, 0, 0);

  PetscFunctionReturn(0);
}

void buildStiffnessMat(Mat mat, StiffnessData* data) {

  PetscScalar h;
  int xi, yi, zi;
  PetscInt N, xs, ys, zs, nx, ny, nz;
  DA da;
  PetscScalar val;
  PetscScalar diagVal;
  MatStencil row;
  MatStencil col;
  Vec rightLocal;
  Vec backLocal;
  Vec topLocal;
  PetscScalar*** rightArr;
  PetscScalar*** backArr;
  PetscScalar*** topArr;

  da = data->da;

  rightLocal = data->rightPropLocal;
  backLocal = data->backPropLocal;
  topLocal = data->topPropLocal;

  if(da) {
    MatZeroEntries(mat);

    DAGetInfo(da, PETSC_NULL, &N, PETSC_NULL, PETSC_NULL,
        PETSC_NULL, PETSC_NULL, PETSC_NULL, 
        PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL);

    DAGetCorners(da, &xs, &ys, &zs, &nx, &ny, &nz);

    h = 1.0/((double)(N - 1));

    DAVecGetArray(da, rightLocal, &rightArr);
    DAVecGetArray(da, backLocal, &backArr);
    DAVecGetArray(da, topLocal, &topArr);

    //Z = 0 and Z = (N-1) are Dirichlet Boundaries
    for(zi = zs; zi < (zs + nz); zi++) {
      for(yi = ys; yi < (ys + ny); yi++) {
        for(xi = xs; xi < (xs + nx); xi++) {
          row.i = xi; 
          row.j = yi;
          row.k = zi;

          if( (zi == 0) || (zi == (N - 1)) ) {
            diagVal = h*h;
            MatSetValuesStencil(mat, 1, &row, 1, &row, &diagVal, INSERT_VALUES);
          } else {
            diagVal = 0.0;

            //Right
            if( xi < (N - 1) ) {
              diagVal += rightArr[zi][yi][xi];
              val = -rightArr[zi][yi][xi];
              col.i = xi + 1;
              col.j = yi;
              col.k = zi;
              MatSetValuesStencil(mat, 1, &row, 1, &col, &val, INSERT_VALUES);
            }

            //Left
            if(xi) {
              diagVal += rightArr[zi][yi][xi - 1];
              val = -rightArr[zi][yi][xi - 1];
              col.i = xi - 1;
              col.j = yi;
              col.k = zi;
              MatSetValuesStencil(mat, 1, &row, 1, &col, &val, INSERT_VALUES);
            }

            //Back
            if( yi < (N - 1) ) {
              diagVal += backArr[zi][yi][xi];
              val = -backArr[zi][yi][xi];
              col.i = xi;
              col.j = yi + 1;
              col.k = zi;
              MatSetValuesStencil(mat, 1, &row, 1, &col, &val, INSERT_VALUES);
            }

            //Front
            if(yi) {
              diagVal += backArr[zi][yi - 1][xi];
              val = -backArr[zi][yi - 1][xi];
              col.i = xi;
              col.j = yi - 1;
              col.k = zi;
              MatSetValuesStencil(mat, 1, &row, 1, &col, &val, INSERT_VALUES);
            }

            //Top
            if( zi < (N - 2) ) {
              diagVal += topArr[zi][yi][xi];
              val = -topArr[zi][yi][xi];
              col.i = xi;
              col.j = yi;
              col.k = zi + 1;
              MatSetValuesStencil(mat, 1, &row, 1, &col, &val, INSERT_VALUES);
            } else {
              diagVal += topArr[zi][yi][xi];
            }

            //Bottom
            if(zi > 1) {
              diagVal += topArr[zi - 1][yi][xi];
              val = -topArr[zi - 1][yi][xi];
              col.i = xi;
              col.j = yi;
              col.k = zi - 1;
              MatSetValuesStencil(mat, 1, &row, 1, &col, &val, INSERT_VALUES);
            } else {
              diagVal += topArr[zi - 1][yi][xi];
            }

            MatSetValuesStencil(mat, 1, &row, 1, &row, &diagVal, INSERT_VALUES);
          }
        }//end for xi
      }//end for yi
    }//end for zi

    DAVecRestoreArray(da, rightLocal, &rightArr);
    DAVecRestoreArray(da, backLocal, &backArr);
    DAVecRestoreArray(da, topLocal, &topArr);

    MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY);

    MatScale(mat, (1.0/(h*h)));

  }//end if active
}

PetscErrorCode stiffnessGetDiagonal(Mat mat, Vec diag) {

  StiffnessData *data; 
  DA da;
  int xi, yi, zi;
  PetscInt N, xs, ys, zs, nx, ny, nz;
  PetscScalar h;
  PetscScalar*** diagArr;
  PetscScalar* diagAllArr;
  Vec diagActive;
  Vec rightLocal;
  Vec backLocal;
  Vec topLocal;
  PetscScalar*** rightArr;
  PetscScalar*** backArr;
  PetscScalar*** topArr;

  PetscFunctionBegin;

  MatShellGetContext( mat, (void **)&data);

  da = data->da;

  rightLocal = data->rightPropLocal;
  backLocal = data->backPropLocal;
  topLocal = data->topPropLocal;

  if(da) {
    diagActive = data->diagActive;

    VecGetArray(diag, &diagAllArr);

    VecPlaceArray(diagActive, diagAllArr);

    VecZeroEntries(diagActive);

    DAGetInfo(da, PETSC_NULL, &N, PETSC_NULL, PETSC_NULL,
        PETSC_NULL, PETSC_NULL, PETSC_NULL, 
        PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL);

    DAGetCorners(da, &xs, &ys, &zs, &nx, &ny, &nz);

    h = 1.0/((double)(N - 1));

    DAVecGetArray(da, diagActive, &diagArr);

    DAVecGetArray(da, rightLocal, &rightArr);
    DAVecGetArray(da, backLocal, &backArr);
    DAVecGetArray(da, topLocal, &topArr);

    //Z = 0 and Z = (N - 1) are Dirichlet Boundaries
    for(zi = zs; zi < (zs + nz); zi++) {
      for(yi = ys; yi < (ys + ny); yi++) {
        for(xi = xs; xi < (xs + nx); xi++) {
          double outVal = 0.0;

          if( (zi == 0) || (zi == (N - 1)) ) {
            outVal = h*h;
          } else {
            //Right
            if( xi < (N - 1) ) {
              outVal += rightArr[zi][yi][xi];
            }

            //Left
            if(xi) {
              outVal += rightArr[zi][yi][xi - 1];
            }

            //Back
            if( yi < (N - 1) ) {
              outVal += backArr[zi][yi][xi];
            }

            //Front
            if(yi) {
              outVal += backArr[zi][yi - 1][xi];
            }

            //Top
            outVal += topArr[zi][yi][xi];

            //Bottom
            outVal += topArr[zi - 1][yi][xi];
          }

          diagArr[zi][yi][xi] = outVal;
        }//end for xi
      }//end for yi
    }//end for zi

    DAVecRestoreArray(da, rightLocal, &rightArr);
    DAVecRestoreArray(da, backLocal, &backArr);
    DAVecRestoreArray(da, topLocal, &topArr);

    DAVecRestoreArray(da, diagActive, &diagArr);

    VecScale(diagActive, (1.0/(h*h)));

    VecResetArray(diagActive);

    VecRestoreArray(diag, &diagAllArr);
  }//end if active

  PetscFunctionReturn(0);
}

PetscErrorCode stiffnessMatvec(Mat mat, Vec in, Vec out) {

  StiffnessData *data; 
  Vec inLocal;
  Vec outActive;
  Vec inActive;
  DA da;
  int xi, yi, zi;
  PetscInt N, xs, ys, zs, nx, ny, nz;
  PetscScalar h;
  PetscScalar*** inArr;
  PetscScalar*** outArr;
  PetscScalar* inAllArr;
  PetscScalar* outAllArr;
  Vec rightLocal;
  Vec backLocal;
  Vec topLocal;
  PetscScalar*** rightArr;
  PetscScalar*** backArr;
  PetscScalar*** topArr;

  PetscFunctionBegin;

  PetscLogEventBegin(kMatvecEvent, 0, 0, 0, 0);

  MatShellGetContext( mat, (void **)&data);

  da = data->da;

  rightLocal = data->rightPropLocal;
  backLocal = data->backPropLocal;
  topLocal = data->topPropLocal;

  if(da) {
    inActive = data->inActive;
    outActive = data->outActive;

    VecGetArray(in, &inAllArr);
    VecGetArray(out, &outAllArr);

    VecPlaceArray(inActive, inAllArr);
    VecPlaceArray(outActive, outAllArr);

    VecZeroEntries(outActive);

    DAGetInfo(da, PETSC_NULL, &N, PETSC_NULL, PETSC_NULL,
        PETSC_NULL, PETSC_NULL, PETSC_NULL, 
        PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL);

    DAGetCorners(da, &xs, &ys, &zs, &nx, &ny, &nz);

    h = 1.0/((double)(N - 1));

    DAGetLocalVector(da, &inLocal);

    DAGlobalToLocalBegin(da, inActive, INSERT_VALUES, inLocal);
    DAGlobalToLocalEnd(da, inActive, INSERT_VALUES, inLocal);

    DAVecGetArray(da, inLocal, &inArr);

    DAVecGetArray(da, outActive, &outArr);

    DAVecGetArray(da, rightLocal, &rightArr);
    DAVecGetArray(da, backLocal, &backArr);
    DAVecGetArray(da, topLocal, &topArr);

    //Z = 0 and Z = (N - 1) are Dirichlet Boundaries
    for(zi = zs; zi < (zs + nz); zi++) {
      for(yi = ys; yi < (ys + ny); yi++) {
        for(xi = xs; xi < (xs + nx); xi++) {
          double outVal = 0.0;

          if( (zi == 0) || (zi == (N - 1)) ) {
            outVal = h*h*inArr[zi][yi][xi];
          } else {
            //Right
            if( xi < (N - 1) ) {
              outVal += (rightArr[zi][yi][xi]*(inArr[zi][yi][xi] - inArr[zi][yi][xi + 1]));
            }

            //Left
            if(xi) {
              outVal += (rightArr[zi][yi][xi - 1]*(inArr[zi][yi][xi] - inArr[zi][yi][xi - 1]));
            }

            //Back
            if( yi < (N - 1) ) {
              outVal += (backArr[zi][yi][xi]*(inArr[zi][yi][xi] - inArr[zi][yi + 1][xi]));
            }

            //Front
            if(yi) {
              outVal += (backArr[zi][yi - 1][xi]*(inArr[zi][yi][xi] - inArr[zi][yi - 1][xi]));
            }

            //Top
            if( zi < (N - 2) ) {
              outVal += (topArr[zi][yi][xi]*(inArr[zi][yi][xi] - inArr[zi + 1][yi][xi]));
            } else {
              outVal += (topArr[zi][yi][xi]*inArr[zi][yi][xi]);
            }

            //Bottom
            if(zi > 1) {
              outVal += (topArr[zi - 1][yi][xi]*(inArr[zi][yi][xi] - inArr[zi - 1][yi][xi]));
            } else {
              outVal += (topArr[zi - 1][yi][xi]*inArr[zi][yi][xi]);
            }
          }

          outArr[zi][yi][xi] = outVal;
        }//end for xi
      }//end for yi
    }//end for zi

    DAVecRestoreArray(da, rightLocal, &rightArr);
    DAVecRestoreArray(da, backLocal, &backArr);
    DAVecRestoreArray(da, topLocal, &topArr);

    DAVecRestoreArray(da, outActive, &outArr);

    DAVecRestoreArray(da, inLocal, &inArr);

    DARestoreLocalVector(da, &inLocal);

    VecScale(outActive, (1.0/(h*h)));

    VecResetArray(inActive);
    VecResetArray(outActive);

    VecRestoreArray(in, &inAllArr);
    VecRestoreArray(out, &outAllArr);
  }//end if active

  PetscLogEventEnd(kMatvecEvent, 0, 0, 0, 0);

  PetscFunctionReturn(0);
}

PetscErrorCode neumannMatvec(Mat mat, Vec in, Vec out) {

  StiffnessData *data; 
  Vec inLocal;
  Vec outActive;
  Vec inActive;
  DA da;
  int xi, yi, zi;
  PetscInt N, xs, ys, zs, nx, ny, nz;
  PetscScalar h;
  PetscScalar*** inArr;
  PetscScalar*** outArr;
  PetscScalar* inAllArr;
  PetscScalar* outAllArr;
  Vec rightLocal;
  Vec backLocal;
  Vec topLocal;
  PetscScalar*** rightArr;
  PetscScalar*** backArr;
  PetscScalar*** topArr;

  PetscFunctionBegin;

  PetscLogEventBegin(nMatvecEvent, 0, 0, 0, 0);

  MatShellGetContext( mat, (void **)&data);

  da = data->da;

  rightLocal = data->rightPropLocal;
  backLocal = data->backPropLocal;
  topLocal = data->topPropLocal;

  if(da) {
    inActive = data->inActive;
    outActive = data->outActive;

    VecGetArray(in, &inAllArr);
    VecGetArray(out, &outAllArr);

    VecPlaceArray(inActive, inAllArr);
    VecPlaceArray(outActive, outAllArr);

    VecZeroEntries(outActive);

    DAGetInfo(da, PETSC_NULL, &N, PETSC_NULL, PETSC_NULL,
        PETSC_NULL, PETSC_NULL, PETSC_NULL, 
        PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL);

    DAGetCorners(da, &xs, &ys, &zs, &nx, &ny, &nz);

    h = 1.0/((double)(N - 1));

    DAGetLocalVector(da, &inLocal);

    DAGlobalToLocalBegin(da, inActive, INSERT_VALUES, inLocal);
    DAGlobalToLocalEnd(da, inActive, INSERT_VALUES, inLocal);

    DAVecGetArray(da, inLocal, &inArr);

    DAVecGetArray(da, outActive, &outArr);

    DAVecGetArray(da, rightLocal, &rightArr);
    DAVecGetArray(da, backLocal, &backArr);
    DAVecGetArray(da, topLocal, &topArr);

    for(zi = zs; zi < (zs + nz); zi++) {
      for(yi = ys; yi < (ys + ny); yi++) {
        for(xi = xs; xi < (xs + nx); xi++) {
          double outVal = 0.0;

          //Right
          if( xi < (N - 1) ) {
            outVal += (rightArr[zi][yi][xi]*(inArr[zi][yi][xi] - inArr[zi][yi][xi + 1]));
          }

          //Left
          if(xi) {
            outVal += (rightArr[zi][yi][xi - 1]*(inArr[zi][yi][xi] - inArr[zi][yi][xi - 1]));
          }

          //Back
          if( yi < (N - 1) ) {
            outVal += (backArr[zi][yi][xi]*(inArr[zi][yi][xi] - inArr[zi][yi + 1][xi]));
          }

          //Front
          if(yi) {
            outVal += (backArr[zi][yi - 1][xi]*(inArr[zi][yi][xi] - inArr[zi][yi - 1][xi]));
          }

          //Top
          if( zi < (N - 1) ) {
            outVal += (topArr[zi][yi][xi]*(inArr[zi][yi][xi] - inArr[zi + 1][yi][xi]));
          }

          //Bottom
          if(zi) {
            outVal += (topArr[zi - 1][yi][xi]*(inArr[zi][yi][xi] - inArr[zi - 1][yi][xi]));
          }

          outArr[zi][yi][xi] = outVal;
        }//end for xi
      }//end for yi
    }//end for zi

    DAVecRestoreArray(da, rightLocal, &rightArr);
    DAVecRestoreArray(da, backLocal, &backArr);
    DAVecRestoreArray(da, topLocal, &topArr);

    DAVecRestoreArray(da, outActive, &outArr);

    DAVecRestoreArray(da, inLocal, &inArr);

    DARestoreLocalVector(da, &inLocal);

    VecScale(outActive, (1.0/(h*h)));

    VecResetArray(inActive);
    VecResetArray(outActive);

    VecRestoreArray(in, &inAllArr);
    VecRestoreArray(out, &outAllArr);
  }//end if active

  PetscLogEventEnd(nMatvecEvent, 0, 0, 0, 0);

  PetscFunctionReturn(0);
}

PetscErrorCode dummyMatDestroy(Mat mat) {
  PetscFunctionBegin;

  PetscFunctionReturn(0);
}

void destroyCoarseMatData(CoarseMatData *data) {

  if(data->mat) {
    MatDestroy(data->mat);
    data->mat = NULL;
  }
  if(data->inTmp) {
    VecDestroy(data->inTmp);
    data->inTmp = NULL;
  }
  if(data->outTmp) {
    VecDestroy(data->outTmp);
    data->outTmp = NULL;
  }

}

void destroyStiffnessData(StiffnessData* data, PetscInt nlevels) {
  int lev;

  for(lev = 0; lev < nlevels; lev++) {

    if(data[lev].inActive) {
      VecDestroy(data[lev].inActive);
      data[lev].inActive = NULL;
    }

    if(data[lev].outActive) {
      VecDestroy(data[lev].outActive);
      data[lev].outActive = NULL;
    }

    if(data[lev].diagActive) {
      VecDestroy(data[lev].diagActive);
      data[lev].diagActive = NULL;
    }

    if(data[lev].rightPropLocal) {
      VecDestroy(data[lev].rightPropLocal);
      data[lev].rightPropLocal = NULL;
    }

    if(data[lev].backPropLocal) {
      VecDestroy(data[lev].backPropLocal);
      data[lev].backPropLocal = NULL;
    }

    if(data[lev].topPropLocal) {
      VecDestroy(data[lev].topPropLocal);
      data[lev].topPropLocal = NULL;
    }

  }//end for lev

}

void createNeumannMat(Mat* _NeumannMat, StiffnessData* kData) {

  Mat NeumannMat;
  PetscInt xs, ys, zs, nx, ny, nz;

  if(kData->da) {
    DAGetCorners(kData->da, &xs, &ys, &zs, &nx, &ny, &nz);
    MatCreateShell(PETSC_COMM_WORLD, (nx*ny*nz), (nx*ny*nz), 
        PETSC_DETERMINE, PETSC_DETERMINE, kData, &NeumannMat);
  } else {
    MatCreateShell(PETSC_COMM_WORLD, 0, 0, 
        PETSC_DETERMINE, PETSC_DETERMINE, kData, &NeumannMat);
  }
  MatShellSetOperation(NeumannMat, MATOP_MULT, (void(*)(void)) neumannMatvec);
  MatShellSetOperation(NeumannMat, MATOP_DESTROY, (void(*)(void)) dummyMatDestroy);

  *_NeumannMat = NeumannMat;
}


void createStiffnessMat(Mat** _Kmat, StiffnessData* kData, CoarseMatData* cData,
    PetscInt nlevels) {

  Mat* Kmat;
  int lev;
  PetscInt xs, ys, zs, nx, ny, nz;

  Kmat = (Mat*)(malloc( sizeof(Mat)*nlevels ));

  if(kData[0].da) {
    DAGetCorners(kData[0].da, &xs, &ys, &zs, &nx, &ny, &nz);
    MatCreateShell(PETSC_COMM_WORLD, (nx*ny*nz), (nx*ny*nz), 
        PETSC_DETERMINE, PETSC_DETERMINE, cData, Kmat);
  } else {
    MatCreateShell(PETSC_COMM_WORLD, 0, 0, 
        PETSC_DETERMINE, PETSC_DETERMINE, cData, Kmat);
  }
  MatShellSetOperation(Kmat[0], MATOP_MULT, (void(*)(void)) coarseMatvec);
  MatShellSetOperation(Kmat[0], MATOP_DESTROY, (void(*)(void)) dummyMatDestroy);

  for(lev = 1; lev < nlevels; lev++) {
    if(kData[lev].da) {
      DAGetCorners(kData[lev].da, &xs, &ys, &zs, &nx, &ny, &nz);
      MatCreateShell(PETSC_COMM_WORLD, (nx*ny*nz), (nx*ny*nz), 
          PETSC_DETERMINE, PETSC_DETERMINE, (kData + lev), (Kmat + lev));
    } else {
      MatCreateShell(PETSC_COMM_WORLD, 0, 0, 
          PETSC_DETERMINE, PETSC_DETERMINE, (kData + lev), (Kmat + lev));
    }
    MatShellSetOperation(Kmat[lev], MATOP_MULT, (void(*)(void)) stiffnessMatvec);
    MatShellSetOperation(Kmat[lev], MATOP_GET_DIAGONAL, (void(*)(void)) stiffnessGetDiagonal);
    MatShellSetOperation(Kmat[lev], MATOP_DESTROY, (void(*)(void)) dummyMatDestroy);
  }//end for lev

  *_Kmat = Kmat;
}

void createCoarseMatData(CoarseMatData** _cData, DA da) {
  CoarseMatData* cData;

  cData = (CoarseMatData*)(malloc( sizeof(CoarseMatData) ));
  cData->mat = NULL;
  cData->inTmp = NULL;
  cData->outTmp = NULL;

  if(da) {
    DAGetMatrix(da, MATAIJ, &(cData->mat));
    MatGetVecs( cData->mat, &(cData->inTmp), &(cData->outTmp) );
  }

  *_cData = cData;
}

void createStiffnessData(StiffnessData** _kData, DA* da, PetscInt nlevels) {
  StiffnessData* kData;
  int lev;

  kData = (StiffnessData*)(malloc( sizeof(StiffnessData)*nlevels ));
  for(lev = 0; lev < nlevels; lev++) {
    (kData[lev]).da = da[lev];
    (kData[lev]).inActive = NULL;
    (kData[lev]).outActive = NULL;
    (kData[lev]).diagActive = NULL;
    (kData[lev]).rightPropLocal = NULL;
    (kData[lev]).backPropLocal = NULL;
    (kData[lev]).topPropLocal = NULL;
    if(da[lev]) {
      DACreateGlobalVector(da[lev], (&((kData[lev]).inActive)));
      DACreateGlobalVector(da[lev], (&((kData[lev]).outActive)));
      DACreateGlobalVector(da[lev], (&((kData[lev]).diagActive)));
      DACreateLocalVector(da[lev], (&((kData[lev]).rightPropLocal)));
      DACreateLocalVector(da[lev], (&((kData[lev]).backPropLocal)));
      DACreateLocalVector(da[lev], (&((kData[lev]).topPropLocal)));
    }
  }//end for lev

  *_kData = kData;
}

PetscErrorCode coarsenStiffnessData(StiffnessData* kData, PetscInt nlevels) {
  int lev;
  int npes;

  PetscFunctionBegin;

  PetscLogEventBegin(coarsenKdataEvent, 0, 0, 0, 0);

  MPI_Comm_size(PETSC_COMM_WORLD, &npes);

  for(lev = (nlevels - 2); lev >= 0; lev--) {
    Vec cRightNaturalAll;
    Vec cRightNaturalActive;
    Vec cRightGlobal;
    Vec cBackNaturalAll;
    Vec cBackNaturalActive;
    Vec cBackGlobal;
    Vec cTopNaturalAll;
    Vec cTopNaturalActive;
    Vec cTopGlobal;
    PetscInt xfi, yfi, zfi;
    PetscInt xsf, ysf, zsf, nxf, nyf, nzf;
    PetscInt xsfeven, ysfeven, zsfeven;
    PetscInt Nf, Nc;
    PetscScalar*** fRightArr;
    PetscScalar*** fBackArr;
    PetscScalar*** fTopArr;
    PetscScalar* cRnaturalActiveArr;
    PetscScalar* cBnaturalActiveArr;
    PetscScalar* cTnaturalActiveArr;
    DA dac = kData[lev].da;
    DA daf = kData[lev + 1].da;
    Vec fRightLocal = kData[lev + 1].rightPropLocal;
    Vec fBackLocal = kData[lev + 1].backPropLocal;
    Vec fTopLocal = kData[lev + 1].topPropLocal;
    Vec cRightLocal = kData[lev].rightPropLocal;
    Vec cBackLocal = kData[lev].backPropLocal;
    Vec cTopLocal = kData[lev].topPropLocal;

    VecCreate(PETSC_COMM_WORLD, &cRightNaturalAll);
    if(dac) {
      PetscInt localSize;
      DACreateNaturalVector(dac, &cRightNaturalActive);
      VecDuplicate(cRightNaturalActive, &cBackNaturalActive);
      VecDuplicate(cRightNaturalActive, &cTopNaturalActive);

      VecGetLocalSize(cRightNaturalActive, &localSize);
      VecSetSizes(cRightNaturalAll, localSize, PETSC_DECIDE);
    } else {
      VecSetSizes(cRightNaturalAll, 0, PETSC_DECIDE);
    }
    if(npes > 1) {
      VecSetType(cRightNaturalAll, VECMPI);
    } else {
      VecSetType(cRightNaturalAll, VECSEQ);
    }
    VecDuplicate(cRightNaturalAll, &cBackNaturalAll);
    VecDuplicate(cRightNaturalAll, &cTopNaturalAll);

    if(dac) {
      VecGetArray(cRightNaturalActive, &cRnaturalActiveArr);
      VecGetArray(cBackNaturalActive, &cBnaturalActiveArr);
      VecGetArray(cTopNaturalActive, &cTnaturalActiveArr);

      VecPlaceArray(cRightNaturalAll, cRnaturalActiveArr);
      VecPlaceArray(cBackNaturalAll, cBnaturalActiveArr);
      VecPlaceArray(cTopNaturalAll, cTnaturalActiveArr);
    }

    VecZeroEntries(cRightNaturalAll);
    VecZeroEntries(cBackNaturalAll);
    VecZeroEntries(cTopNaturalAll);

    if(daf) {

      DAGetInfo(daf, PETSC_NULL, &Nf, PETSC_NULL, PETSC_NULL,
          PETSC_NULL, PETSC_NULL, PETSC_NULL, 
          PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL);

      Nc = ((Nf - 1)/2) + 1;

      DAGetCorners(daf, &xsf, &ysf, &zsf, &nxf, &nyf, &nzf);

      if( (xsf%2) == 0 ) {
        xsfeven = xsf;
      } else {
        xsfeven = xsf + 1;
      }

      if( (ysf%2) == 0 ) {
        ysfeven = ysf;
      } else {
        ysfeven = ysf + 1;
      }

      if( (zsf%2) == 0 ) {
        zsfeven = zsf;
      } else {
        zsfeven = zsf + 1;
      }

      DAVecGetArray(daf, fRightLocal, &fRightArr);
      DAVecGetArray(daf, fBackLocal, &fBackArr);
      DAVecGetArray(daf, fTopLocal, &fTopArr);

      for(zfi = zsfeven; zfi < (zsf + nzf); zfi += 2) {
        for(yfi = ysfeven; yfi < (ysf + nyf); yfi += 2) {
          for(xfi = xsfeven; xfi < (xsf + nxf); xfi += 2) {
            PetscInt xci = xfi/2;
            PetscInt yci = yfi/2;
            PetscInt zci = zfi/2;
            PetscInt row = __DOF__(xci, yci, zci, Nc);
            if(xfi < (Nf - 1)) {
              PetscScalar val = 0.5*(fRightArr[zfi][yfi][xfi] + fRightArr[zfi][yfi][xfi + 1]);
              VecSetValue(cRightNaturalAll, row, val, INSERT_VALUES);
            }
            if(yfi < (Nf - 1)) {
              PetscScalar val = 0.5*(fBackArr[zfi][yfi][xfi] + fBackArr[zfi][yfi + 1][xfi]);
              VecSetValue(cBackNaturalAll, row, val, INSERT_VALUES);
            }
            if(zfi < (Nf - 1)) {
              PetscScalar val = 0.5*(fTopArr[zfi][yfi][xfi] + fTopArr[zfi + 1][yfi][xfi]);
              VecSetValue(cTopNaturalAll, row, val, INSERT_VALUES);
            }
          }//end for xfi
        }//end for yfi
      }//end for zfi

      DAVecRestoreArray(daf, fRightLocal, &fRightArr);
      DAVecRestoreArray(daf, fBackLocal, &fBackArr);
      DAVecRestoreArray(daf, fTopLocal, &fTopArr);
    }//end if fine active

    VecAssemblyBegin(cRightNaturalAll);
    VecAssemblyBegin(cBackNaturalAll);
    VecAssemblyBegin(cTopNaturalAll);

    VecAssemblyEnd(cRightNaturalAll);
    VecAssemblyEnd(cBackNaturalAll);
    VecAssemblyEnd(cTopNaturalAll);

    if(dac) {

      VecResetArray(cRightNaturalAll);
      VecResetArray(cBackNaturalAll);
      VecResetArray(cTopNaturalAll);

      VecRestoreArray(cRightNaturalActive, &cRnaturalActiveArr);
      VecRestoreArray(cBackNaturalActive, &cBnaturalActiveArr);
      VecRestoreArray(cTopNaturalActive, &cTnaturalActiveArr);

      DAGetGlobalVector(dac, &cRightGlobal);
      DAGetGlobalVector(dac, &cBackGlobal);
      DAGetGlobalVector(dac, &cTopGlobal);

      DANaturalToGlobalBegin(dac, cRightNaturalActive, INSERT_VALUES, cRightGlobal);
      DANaturalToGlobalEnd(dac, cRightNaturalActive, INSERT_VALUES, cRightGlobal);

      DANaturalToGlobalBegin(dac, cBackNaturalActive, INSERT_VALUES, cBackGlobal);
      DANaturalToGlobalEnd(dac, cBackNaturalActive, INSERT_VALUES, cBackGlobal);

      DANaturalToGlobalBegin(dac, cTopNaturalActive, INSERT_VALUES, cTopGlobal);
      DANaturalToGlobalEnd(dac, cTopNaturalActive, INSERT_VALUES, cTopGlobal);

      DAGlobalToLocalBegin(dac, cRightGlobal, INSERT_VALUES, cRightLocal);
      DAGlobalToLocalEnd(dac, cRightGlobal, INSERT_VALUES, cRightLocal);

      DAGlobalToLocalBegin(dac, cBackGlobal, INSERT_VALUES, cBackLocal);
      DAGlobalToLocalEnd(dac, cBackGlobal, INSERT_VALUES, cBackLocal);

      DAGlobalToLocalBegin(dac, cTopGlobal, INSERT_VALUES, cTopLocal);
      DAGlobalToLocalEnd(dac, cTopGlobal, INSERT_VALUES, cTopLocal);

      DARestoreGlobalVector(dac, &cRightGlobal);
      DARestoreGlobalVector(dac, &cBackGlobal);
      DARestoreGlobalVector(dac, &cTopGlobal);

      VecDestroy(cRightNaturalActive);
      VecDestroy(cBackNaturalActive);
      VecDestroy(cTopNaturalActive);
    }

    VecDestroy(cRightNaturalAll);
    VecDestroy(cBackNaturalAll);
    VecDestroy(cTopNaturalAll);
  }//end for lev

  PetscLogEventEnd(coarsenKdataEvent, 0, 0, 0, 0);

  PetscFunctionReturn(0);
}

void initFinestStiffnessData(StiffnessData* data) {
  if(data->da) {
    VecSet(data->rightPropLocal, 1.0); 
    VecSet(data->backPropLocal, 1.0); 
    VecSet(data->topPropLocal, 1.0); 
  }
}

PetscErrorCode resetFinestStiffnessDataFromFile(char* fname, PetscInt initNumBrokenBonds, StiffnessData* data) {
  
  PetscFunctionBegin;
  
  PetscLogEventBegin(resetKdataEvent, 0, 0, 0, 0);
  
  if(data->da) {
    PetscInt gxs, gys, gzs, gnx, gny, gnz;
    PetscScalar*** rightArr;
    PetscScalar*** backArr;
    PetscScalar*** topArr;
    FILE* ifp = fopen(fname,"r");
    int bCnt;
    int xb, yb, zb, rbt;

    DAGetGhostCorners(data->da, &gxs, &gys, &gzs, &gnx, &gny, &gnz);

    DAVecGetArray(data->da, data->rightPropLocal, &rightArr);
    DAVecGetArray(data->da, data->backPropLocal, &backArr);
    DAVecGetArray(data->da, data->topPropLocal, &topArr);

    for(bCnt = 0; bCnt < initNumBrokenBonds; bCnt++) {
      fscanf(ifp, "%d", &xb);
      fscanf(ifp, "%d", &yb);
      fscanf(ifp, "%d", &zb);
      fscanf(ifp, "%d", &rbt);
      if( (xb >= gxs) && (xb < (gxs + gnx)) ) {
        if( (yb >= gys) && (yb < (gys + gny)) ) {
          if( (zb >= gzs) && (zb < (gzs + gnz)) ) {
            if(rbt == __RIGHT__) {
              rightArr[zb][yb][xb] = 0;
            } else if(rbt == __BACK__) {
              backArr[zb][yb][xb] = 0;
            } else {
              topArr[zb][yb][xb] = 0;
            }
          }
        }
      }
    }//end for bCnt

    DAVecRestoreArray(data->da, data->rightPropLocal, &rightArr);
    DAVecRestoreArray(data->da, data->backPropLocal, &backArr);
    DAVecRestoreArray(data->da, data->topPropLocal, &topArr);

    fclose(ifp);
  }//end if active
  
  PetscLogEventEnd(resetKdataEvent, 0, 0, 0, 0);
  
  PetscFunctionReturn(0);
}

PetscErrorCode resetFinestStiffnessData(StiffnessData* data, PetscInt xb, PetscInt yb, PetscInt zb, PetscInt rbt) {
 
  PetscFunctionBegin;

  PetscLogEventBegin(resetKdataEvent, 0, 0, 0, 0);
  
  if(data->da) {
    PetscInt gxs, gys, gzs, gnx, gny, gnz;
    PetscScalar*** rightArr;
    PetscScalar*** backArr;
    PetscScalar*** topArr;

    DAGetGhostCorners(data->da, &gxs, &gys, &gzs, &gnx, &gny, &gnz);

    DAVecGetArray(data->da, data->rightPropLocal, &rightArr);
    DAVecGetArray(data->da, data->backPropLocal, &backArr);
    DAVecGetArray(data->da, data->topPropLocal, &topArr);

    if( (xb >= gxs) && (xb < (gxs + gnx)) ) {
      if( (yb >= gys) && (yb < (gys + gny)) ) {
        if( (zb >= gzs) && (zb < (gzs + gnz)) ) {
          if(rbt == __RIGHT__) {
            rightArr[zb][yb][xb] = 0;
          } else if(rbt == __BACK__) {
            backArr[zb][yb][xb] = 0;
          } else {
            topArr[zb][yb][xb] = 0;
          }
        }
      }
    }

    DAVecRestoreArray(data->da, data->rightPropLocal, &rightArr);
    DAVecRestoreArray(data->da, data->backPropLocal, &backArr);
    DAVecRestoreArray(data->da, data->topPropLocal, &topArr);
  }//end if active
  
  PetscLogEventEnd(resetKdataEvent, 0, 0, 0, 0);
 
  PetscFunctionReturn(0);
  
}



