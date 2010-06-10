
#include "fuseMg3d.h"
#include "petscmg.h"
#include <assert.h>
#include <stdlib.h>

extern PetscLogEvent createThresholdEvent;

PetscErrorCode createThresholds(char* fname, DA da, Vec* _rightThreshold, Vec* _backThreshold, Vec* _topThreshold) {
  Vec rightThreshold = NULL;
  Vec backThreshold = NULL;
  Vec topThreshold = NULL;

  PetscFunctionBegin;

  PetscLogEventBegin(createThresholdEvent, 0, 0, 0, 0);
  
  if(da) {
    PetscScalar*** rightArr;
    PetscScalar*** backArr;
    PetscScalar*** topArr;
    PetscInt N, xs, ys, zs, nx, ny, nz;
    int cnt, xi, yi, zi, rbt, numBonds;
    double val;
    FILE* fp = fopen(fname, "r");

    DAGetInfo(da, PETSC_NULL, &N, PETSC_NULL, PETSC_NULL,
        PETSC_NULL, PETSC_NULL, PETSC_NULL, 
        PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL);

    numBonds = 3*N*N*(N - 1);

    DAGetCorners(da, &xs, &ys, &zs, &nx, &ny, &nz);

    DACreateGlobalVector(da, &rightThreshold);
    VecDuplicate(rightThreshold, &backThreshold);
    VecDuplicate(rightThreshold, &topThreshold);

    VecZeroEntries(rightThreshold);
    VecZeroEntries(backThreshold);
    VecZeroEntries(topThreshold);

    DAVecGetArray(da, rightThreshold, &rightArr);
    DAVecGetArray(da, backThreshold, &backArr);
    DAVecGetArray(da, topThreshold, &topArr);

    for(cnt = 0; cnt < numBonds; cnt++) {
      fscanf(fp, "%d", &xi);
      fscanf(fp, "%d", &yi);
      fscanf(fp, "%d", &zi);
      fscanf(fp, "%d", &rbt);
      fscanf(fp, "%lf", &val);
      if(val <= 0.0) {
        printf("At xi = %d, yi = %d, zi = %d, rbt = %d, found val = %lf\n", xi, yi, zi, rbt, val);
      }
      assert(val > 0.0);

      if( (zi >= zs) && (zi < (zs + nz)) ) {
        if( (yi >= ys) && (yi < (ys + ny)) ) {
          if( (xi >= xs) && (xi < (xs + nx)) ) {
            if(rbt == __RIGHT__) {
              rightArr[zi][yi][xi] = val;
            } else if(rbt == __BACK__) {
              backArr[zi][yi][xi] = val;
            } else {
              topArr[zi][yi][xi] = val;
            }
          }
        }
      }
    }//end for cnt

    DAVecRestoreArray(da, rightThreshold, &rightArr);
    DAVecRestoreArray(da, backThreshold, &backArr);
    DAVecRestoreArray(da, topThreshold, &topArr);

    fclose(fp);
  }//end if active

  *_rightThreshold = rightThreshold;
  *_backThreshold = backThreshold;
  *_topThreshold = topThreshold;
  
  PetscLogEventEnd(createThresholdEvent, 0, 0, 0, 0);
 
  PetscFunctionReturn(0);
  
}

double computeTotalReaction(Vec sol, StiffnessData* data, PetscScalar stiffness) {

  PetscInt N, xi, yi, zi, xs, ys, zs, nx, ny, nz;
  DA da = data->da;
  Vec solActive;
  Vec solLocal;
  Vec topProp = data->topPropLocal;
  PetscScalar*** topPropArr;
  PetscScalar* solAllArr; 
  PetscScalar*** solArr;
  double totalReaction = 0;
  double localReaction = 0;

  if(da) {

    DAGetInfo(da, PETSC_NULL, &N, PETSC_NULL, PETSC_NULL,
        PETSC_NULL, PETSC_NULL, PETSC_NULL, 
        PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL);

    DAGetCorners(da, &xs, &ys, &zs, &nx, &ny, &nz);

    DAGetGlobalVector(da, &solActive);

    DAGetLocalVector(da, &solLocal);

    VecGetArray(sol, &solAllArr);

    VecPlaceArray(solActive, solAllArr);

    DAGlobalToLocalBegin(da, solActive, INSERT_VALUES, solLocal);
    DAGlobalToLocalEnd(da, solActive, INSERT_VALUES, solLocal);

    DAVecGetArray(da, solLocal, &solArr);
    DAVecGetArray(da, topProp, &topPropArr);

    zi = (N - 2);
    if( (zi >= zs) && (zi < (zs + nz)) ) {
      for(yi = ys; yi < (ys + ny); yi++) {
        for(xi = xs; xi < (xs + nx); xi++) {
          //Top
          localReaction += (topPropArr[zi][yi][xi]*stiffness*(solArr[zi + 1][yi][xi] - solArr[zi][yi][xi]));
        }//end for xi
      }//end for yi
    }

    DAVecRestoreArray(da, solLocal, &solArr);
    DAVecRestoreArray(da, topProp, &topPropArr);

    VecResetArray(solActive);

    VecRestoreArray(sol, &solAllArr);

    DARestoreLocalVector(da, &solLocal);

    DARestoreGlobalVector(da, &solActive);

  }//end if active

  MPI_Allreduce(&localReaction, &totalReaction, 1, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);

  return totalReaction;
}

void breakBond(Vec sol, Vec rightThreshold, Vec backThreshold, Vec topThreshold,
    StiffnessData* data, PetscScalar stiffness, 
    PetscInt* _xb, PetscInt* _yb, PetscInt* _zb, PetscInt* _rbt, double* _maxGlobalLambda) {

  PetscInt xb, yb, zb, rbt;
  PetscInt N, xi, yi, zi, xs, ys, zs, nx, ny, nz;
  PetscScalar* solAllArr; 
  Vec solActive;
  Vec solLocal;
  PetscScalar*** solArr;
  PetscScalar*** rightPropArr;
  PetscScalar*** backPropArr;
  PetscScalar*** topPropArr;
  PetscScalar*** rightThresArr;
  PetscScalar*** backThresArr;
  PetscScalar*** topThresArr;
  DA da = data->da;
  Vec rightProp = data->rightPropLocal;
  Vec backProp = data->backPropLocal;
  Vec topProp = data->topPropLocal;
  double maxLocalLambda = -1.0;
  double maxGlobalLambda;
  int rank;
  int bestRank;
  int buf[4];

  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  if(da) {

    DAGetInfo(da, PETSC_NULL, &N, PETSC_NULL, PETSC_NULL,
        PETSC_NULL, PETSC_NULL, PETSC_NULL, 
        PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL);

    DAGetCorners(da, &xs, &ys, &zs, &nx, &ny, &nz);

    DAGetGlobalVector(da, &solActive);

    DAGetLocalVector(da, &solLocal);

    VecGetArray(sol, &solAllArr);

    VecPlaceArray(solActive, solAllArr);

    DAGlobalToLocalBegin(da, solActive, INSERT_VALUES, solLocal);
    DAGlobalToLocalEnd(da, solActive, INSERT_VALUES, solLocal);

    DAVecGetArray(da, solLocal, &solArr);
    DAVecGetArray(da, rightProp, &rightPropArr);
    DAVecGetArray(da, backProp, &backPropArr);
    DAVecGetArray(da, topProp, &topPropArr);
    DAVecGetArray(da, rightThreshold, &rightThresArr);
    DAVecGetArray(da, backThreshold, &backThresArr);
    DAVecGetArray(da, topThreshold, &topThresArr);

    for(zi = zs; zi < (zs + nz); zi++) {
      for(yi = ys; yi < (ys + ny); yi++) {
        for(xi = xs; xi < (xs + nx); xi++) {
          //Right
          if(xi < (N - 1)) {
            double current = rightPropArr[zi][yi][xi]*stiffness*(solArr[zi][yi][xi + 1] - solArr[zi][yi][xi]);
            double lambda = (fabs(current))/(rightThresArr[zi][yi][xi]);
            if(lambda > maxLocalLambda) {
              maxLocalLambda = lambda;
              xb = xi;
              yb = yi;
              zb = zi;
              rbt = __RIGHT__;
            }
          }

          //Back
          if(yi < (N - 1)) {
            double current = backPropArr[zi][yi][xi]*stiffness*(solArr[zi][yi + 1][xi] - solArr[zi][yi][xi]);
            double lambda = (fabs(current))/(backThresArr[zi][yi][xi]);
            if(lambda > maxLocalLambda) {
              maxLocalLambda = lambda;
              xb = xi;
              yb = yi;
              zb = zi;
              rbt = __BACK__;
            }
          }

          //Top
          if(zi < (N - 1)) {
            double current = topPropArr[zi][yi][xi]*stiffness*(solArr[zi + 1][yi][xi] - solArr[zi][yi][xi]);
            double lambda = (fabs(current))/(topThresArr[zi][yi][xi]);
            if(lambda > maxLocalLambda) {
              maxLocalLambda = lambda;
              xb = xi;
              yb = yi;
              zb = zi;
              rbt = __TOP__;
            }
          }
        }//end for xi
      }//end for yi
    }//end for zi

    DAVecRestoreArray(da, solLocal, &solArr);
    DAVecRestoreArray(da, rightProp, &rightPropArr);
    DAVecRestoreArray(da, backProp, &backPropArr);
    DAVecRestoreArray(da, topProp, &topPropArr);
    DAVecRestoreArray(da, rightThreshold, &rightThresArr);
    DAVecRestoreArray(da, backThreshold, &backThresArr);
    DAVecRestoreArray(da, topThreshold, &topThresArr);

    VecResetArray(solActive);

    VecRestoreArray(sol, &solAllArr);

    DARestoreLocalVector(da, &solLocal);

    DARestoreGlobalVector(da, &solActive);

  }//end if active

  MPI_Allreduce(&maxLocalLambda, &maxGlobalLambda, 1, MPI_DOUBLE, MPI_MAX, PETSC_COMM_WORLD);

  if(maxLocalLambda != maxGlobalLambda) {
    rank = -1;
  }

  MPI_Allreduce(&rank, &bestRank, 1, MPI_INT, MPI_MAX, PETSC_COMM_WORLD);

  buf[0] = xb;
  buf[1] = yb;
  buf[2] = zb;
  buf[3] = rbt;

  MPI_Bcast(buf, 4, MPI_INT, bestRank, PETSC_COMM_WORLD);

  xb  = buf[0];
  yb  = buf[1];
  zb  = buf[2];
  rbt = buf[3];

  *_xb = xb;
  *_yb = yb;
  *_zb = zb;
  *_rbt = rbt;
  *_maxGlobalLambda = maxGlobalLambda;
}

void createSolver(KSP* _ksp, PC_KSP_Shell* kspShellData, Mat* Rmat, Mat* Pmat, MPI_Comm* comms, PetscInt nlevels) {
  KSP ksp;
  KSP lksp;
  PC pc;
  PC lpc;
  int lev;
  PetscTruth ismg;
  PetscInt singleLevelFullMatrix;
  const char* clearOptionPrefix;
  char optionName[256];

  KSPCreate(PETSC_COMM_WORLD, &ksp);
  KSPSetType(ksp, KSPCG);

  KSPGetPC(ksp, &pc);
  PCSetType(pc, PCMG);
  PCMGSetLevels(pc, nlevels, comms);
  PCMGSetType(pc, PC_MG_MULTIPLICATIVE);

  for( lev = 1; lev < nlevels; lev++) {
    PCMGSetInterpolation(pc, lev, Pmat[lev - 1]);
    PCMGSetRestriction(pc, lev, Rmat[lev - 1]);
  }//end for lev
  KSPSetFromOptions(ksp);

  KSPSetInitialGuessNonzero(ksp, PETSC_TRUE);

  PetscTypeCompare((PetscObject)pc, PCMG, &ismg);

  singleLevelFullMatrix = 0;
  PetscOptionsGetInt(PETSC_NULL, "-singleLevelFullMatrix", &singleLevelFullMatrix, PETSC_NULL);

  if(ismg) {
    PCMGGetSmoother(pc, 0, &lksp);

    KSPGetPC(lksp, &lpc);
    PCGetOptionsPrefix(lpc, &clearOptionPrefix);

    sprintf(optionName, "-%spc_type", clearOptionPrefix);
    PetscOptionsClearValue(optionName); 
    PCSetType(lpc, PCSHELL); 
    PCShellSetName(lpc, "PC_KSP_Shell"); 

    kspShellData->pc = lpc;
    PCShellSetContext(lpc, kspShellData); 
    PCShellSetSetUp(lpc, PC_KSP_Shell_SetUp); 
    PCShellSetApply(lpc, PC_KSP_Shell_Apply); 
    PCShellSetDestroy(lpc, PC_KSP_Shell_Destroy); 
  } else if(singleLevelFullMatrix) {
    PCGetOptionsPrefix(pc, &clearOptionPrefix);

    sprintf(optionName, "-%spc_type", clearOptionPrefix);
    PetscOptionsClearValue(optionName); 
    PCSetType(pc, PCSHELL); 
    PCShellSetName(pc, "PC_KSP_Shell"); 

    kspShellData->pc = pc;
    PCShellSetContext(pc, kspShellData); 
    PCShellSetSetUp(pc, PC_KSP_Shell_SetUp); 
    PCShellSetApply(pc, PC_KSP_Shell_Apply); 
    PCShellSetDestroy(pc, PC_KSP_Shell_Destroy); 
  }

  *_ksp = ksp;
}

void resetBC(Vec rhs, DA da) {

  PetscInt N, xs, ys, zs, nx, ny, nz, xi, yi, zi;
  PetscScalar* rhsAllArr;
  PetscScalar*** rhsActiveArr;
  Vec rhsActive;

  if(da) {

    DAGetInfo(da, PETSC_NULL, &N, PETSC_NULL, PETSC_NULL,
        PETSC_NULL, PETSC_NULL, PETSC_NULL, 
        PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL);

    DAGetCorners(da, &xs, &ys, &zs, &nx, &ny, &nz);

    DAGetGlobalVector(da, &rhsActive);

    VecGetArray(rhs, &rhsAllArr);

    VecPlaceArray(rhsActive, rhsAllArr);

    DAVecGetArray(da, rhsActive, &rhsActiveArr);

    zi = 0;
    if( (zi >= zs) && (zi < (zs + nz)) ) {
      for(yi = ys; yi < (ys + ny); yi++) {
        for(xi = xs; xi < (xs + nx); xi++) {
          rhsActiveArr[zi][yi][xi] = 0.0;
        }//end for xi
      }//end for yi
    }//end if bottom surface

    zi = (N - 1);
    if( (zi >= zs) && (zi < (zs + nz)) ) {
      for(yi = ys; yi < (ys + ny); yi++) {
        for(xi = xs; xi < (xs + nx); xi++) {
          rhsActiveArr[zi][yi][xi] = 1.0;
        }//end for xi
      }//end for yi
    }//end if top surface

    DAVecRestoreArray(da, rhsActive, &rhsActiveArr);

    VecResetArray(rhsActive);

    VecRestoreArray(rhs, &rhsAllArr);

    DARestoreGlobalVector(da, &rhsActive);

  }//end if active

}

void setRHSinp(Vec rhsInp, DA da) {

  PetscInt N, xs, ys, zs, nx, ny, nz, xi, yi, zi;
  PetscScalar* rhsInpAllArr;
  PetscScalar*** rhsInpActiveArr;
  Vec rhsInpActive;

  if(da) {

    DAGetInfo(da, PETSC_NULL, &N, PETSC_NULL, PETSC_NULL,
        PETSC_NULL, PETSC_NULL, PETSC_NULL, 
        PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL);

    DAGetCorners(da, &xs, &ys, &zs, &nx, &ny, &nz);

    DAGetGlobalVector(da, &rhsInpActive);

    VecGetArray(rhsInp, &rhsInpAllArr);

    VecPlaceArray(rhsInpActive, rhsInpAllArr);

    VecZeroEntries(rhsInpActive);

    DAVecGetArray(da, rhsInpActive, &rhsInpActiveArr);

    zi = (N - 1);
    if( (zi >= zs) && (zi < (zs + nz)) ) {
      for(yi = ys; yi < (ys + ny); yi++) {
        for(xi = xs; xi < (xs + nx); xi++) {
          rhsInpActiveArr[zi][yi][xi] = -1.0;
        }//end for xi
      }//end for yi
    }//end if top surface

    DAVecRestoreArray(da, rhsInpActive, &rhsInpActiveArr);

    VecResetArray(rhsInpActive);

    VecRestoreArray(rhsInp, &rhsInpAllArr);

    DARestoreGlobalVector(da, &rhsInpActive);

  }//end if active

}

void createRHSandSol(Vec* _rhs, Vec* _sol, DA da) {
  Vec rhs;
  Vec sol;
  int npes;
  PetscInt xs, ys, zs, nx, ny, nz;

  MPI_Comm_size(PETSC_COMM_WORLD, &npes);

  VecCreate(PETSC_COMM_WORLD, &rhs);
  if(da) {
    DAGetCorners(da, &xs, &ys, &zs, &nx, &ny, &nz);
    VecSetSizes(rhs, (nx*ny*nz), PETSC_DECIDE);
  } else {
    VecSetSizes(rhs, 0, PETSC_DECIDE);
  }
  if(npes > 1) {
    VecSetType(rhs, VECMPI);
  } else {
    VecSetType(rhs, VECSEQ);
  }

  VecDuplicate(rhs, &sol);

  *_rhs = rhs;
  *_sol = sol;
}

void createDA(DA** _da, int* iAmActive, MPI_Comm* activeComms, PetscInt* N, PetscInt nlevels) {
  DA* da;
  int lev;

  da = (DA*)(malloc( sizeof(DA)*nlevels ));
  for(lev = 0; lev < nlevels; lev++) {
    da[lev] = NULL;
    if(iAmActive[lev]) {
      DACreate3d(activeComms[lev], DA_NONPERIODIC, DA_STENCIL_BOX,
          N[lev], N[lev], N[lev], PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE,
          1, 1, PETSC_NULL, PETSC_NULL, PETSC_NULL, (da + lev) );
    }
  }//end for lev

  *_da = da;
}

void createComms(int** _activeNpes, int** _iAmActive, MPI_Comm** _activeComms,
    MPI_Comm** _comms, PetscInt* N, PetscInt nlevels) {
  int* activeNpes;
  int* iAmActive;
  MPI_Comm* activeComms;
  MPI_Comm* comms;
  PetscInt maxCoarseNpes;
  int rank, npes, lev;

  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  MPI_Comm_size(PETSC_COMM_WORLD, &npes);

  activeNpes = (int*)(malloc( sizeof(int)*nlevels ));
  iAmActive = (int*)(malloc( sizeof(int)*nlevels ));

  maxCoarseNpes = npes;
  PetscOptionsGetInt(PETSC_NULL, "-maxCoarseNpes", &maxCoarseNpes, PETSC_NULL);
  activeNpes[0] = (__CUBE__(N[0] - 1))/(__PROC_FACTOR__);
  if(activeNpes[0] == 0) {
    activeNpes[0] = 1;
  }
  if(activeNpes[0] > maxCoarseNpes) {
    activeNpes[0] = maxCoarseNpes;
  }
  while( !foundValidDApart(N[0], activeNpes[0]) ) {
    activeNpes[0]--;
  }
  for(lev = 1; lev < nlevels; lev++) {
    activeNpes[lev] = (__CUBE__(N[lev] - 1))/(__PROC_FACTOR__);
    if(activeNpes[lev] == 0) {
      activeNpes[lev] = 1;
    }
    if(activeNpes[lev] > npes) {
      activeNpes[lev] = npes;
    }
    while( !foundValidDApart(N[lev], activeNpes[lev]) ) {
      activeNpes[lev]--;
    }
  }//end for lev

  if(!rank) {
    for(lev = 0; lev < nlevels; lev++) {
      printf("Lev: %d, N: %d, P: %d \n", lev, N[lev], activeNpes[lev]);
    }//end for lev
  }

  for(lev = 0; lev < nlevels; lev++) {
    if( rank < activeNpes[lev] ) {
      iAmActive[lev] = 1;
    } else {
      iAmActive[lev] = 0;
    }
  }//end for lev

  activeComms = (MPI_Comm*)(malloc( sizeof(MPI_Comm)*nlevels ));
  for(lev = 0; lev < nlevels; lev++) {
    MPI_Comm_split(PETSC_COMM_WORLD, iAmActive[lev], rank, (activeComms + lev) );
  }//end for lev

  comms = (MPI_Comm*)(malloc( sizeof(MPI_Comm)*nlevels ));
  for(lev = 0; lev < nlevels; lev++) {
    comms[lev] = PETSC_COMM_WORLD;
  }//end for lev

  *_activeNpes = activeNpes;
  *_iAmActive = iAmActive;
  *_activeComms = activeComms;
  *_comms = comms;
}

void createGridSizes(PetscInt** _N, PetscInt nlevels) {
  int lev;
  PetscInt* N;

  //0 is the coarsest
  N = (PetscInt*)(malloc( sizeof(PetscInt)*nlevels ));
  PetscOptionsGetInt(PETSC_NULL, "-N", N + nlevels - 1, PETSC_NULL);

  for(lev = (nlevels - 2); lev >= 0; lev--) {
    N[lev] =  ((N[lev + 1] - 1)/2) + 1;
  }//end for lev

  *_N = N;
}

int foundValidDApart(int N, int npes) {
  int m, n, p;
  int foundValidPart;

  n = (int)(0.5 + pow( ((double)npes), (1.0/3.0) ));

  if(!n) { 
    n = 1;
  }

  while (n > 0) {
    int pm = npes/n;

    if (n*pm == npes) {
      break;
    }

    n--;
  }

  if (!n) {
    n = 1; 
  }

  m = (int)(0.5 + sqrt( ((double)npes)/((double)n) ));

  if(!m) { 
    m = 1; 
  }

  while (m > 0) {
    p = npes/(m*n);
    if ((m*n*p) == npes) {
      break;
    }
    m--;
  }

  foundValidPart = 1;
  if((m*n*p) != npes) {
    foundValidPart = 0;
  }
  if(N < m) {
    foundValidPart = 0;
  }
  if(N < n) {
    foundValidPart = 0;
  }
  if(N < p) {
    foundValidPart = 0;
  }

  return foundValidPart;
}




