
#include "mpi.h"
#include "petsc.h"
#include "petscvec.h"
#include "petscmat.h"
#include "petscda.h"
#include "petscksp.h"
#include "petscpc.h"
#include "private/pcimpl.h"
#include "petscmg.h"
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <math.h>
#include "fuseMg3d.h"

PetscLogEvent createThresholdEvent;

PetscLogEvent nMatvecEvent;
PetscLogEvent kMatvecEvent;
PetscLogEvent coarsenKdataEvent;
PetscLogEvent resetKdataEvent;

PetscLogEvent createRPmatEvent;
PetscLogEvent buildPmatEvent;

PetscCookie auxillaryCookie;
PetscCookie stiffnessCookie;
PetscCookie intergridCookie;

PetscLogStage solveStage; 
PetscLogStage finalStage; 
PetscLogStage initialStage; 

int main(int argc, char** argv) {

  int lev;
  PetscInt nlevels;
  PetscInt* N;
  DA* da;
  Mat NeumannMat;
  Mat* Kmat;
  Mat* Rmat;
  Mat* Pmat;
  CoarseMatData* cData;
  StiffnessData* kData;
  PC_KSP_Shell* kspShellData;
  int* activeNpes;
  int* iAmActive;
  MPI_Comm* comms;
  MPI_Comm* activeComms;
  KSP ksp;
  KSP lksp;
  PC pc;
  Vec rhsInp;
  Vec rhs;
  Vec sol;
  Vec rightThreshold;
  Vec backThreshold;
  Vec topThreshold;
  PetscTruth ismg;
  PetscInt xb, yb, zb, rbt;
  PetscInt useZeroGuess; 
  PetscInt maxIts;
  PetscScalar stiffness;
  PetscScalar stopTol;
  int repeatLoop;
  int iter;
  double totalReaction;
  double breakFactor;
  int rank;
  FILE* ofpB = NULL;
  FILE* ofpR = NULL;
  PetscReal solNorm2, solNormInf;
  PetscInt initFromFile;
  PetscInt initNumBrokenBonds;

  PetscInitialize(&argc, &argv, argv[3], "Parallel Multigrid for the 3-D Fuse Problem (Displacement Loading)");

  PetscCookieRegister("AUXILLARY", &auxillaryCookie);
  PetscCookieRegister("STIFFNESS", &stiffnessCookie);
  PetscCookieRegister("INTERGRID", &intergridCookie);

  PetscLogEventRegister("CreateThreshold", auxillaryCookie, &createThresholdEvent);
  
  PetscLogEventRegister("Nmatvec", stiffnessCookie, &nMatvecEvent);
  PetscLogEventRegister("Kmatvec", stiffnessCookie, &kMatvecEvent);
  PetscLogEventRegister("CoarsenKdata", stiffnessCookie, &coarsenKdataEvent);
  PetscLogEventRegister("ResetKdata", stiffnessCookie, &resetKdataEvent);

  PetscLogEventRegister("CreateRP", intergridCookie, &createRPmatEvent);
  PetscLogEventRegister("BuildPmat", intergridCookie, &buildPmatEvent);

  PetscLogStageRegister("Initial Stage", &initialStage);
  PetscLogStageRegister("Solve Stage", &solveStage);
  PetscLogStageRegister("Final Stage", &finalStage);

  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  initFromFile = 0;
  PetscOptionsGetInt(PETSC_NULL, "-initFromFile", &initFromFile, PETSC_NULL);

  initNumBrokenBonds = 0;
  PetscOptionsGetInt(PETSC_NULL, "-initNumBrokenBonds", &initNumBrokenBonds, PETSC_NULL);

  nlevels = 2;
  PetscOptionsGetInt(PETSC_NULL, "-nlevels", &nlevels, PETSC_NULL);

  maxIts = 1; 
  PetscOptionsGetInt(PETSC_NULL, "-maxIts", &maxIts, PETSC_NULL);

  useZeroGuess = 0;
  PetscOptionsGetInt(PETSC_NULL, "-useZeroGuess", &useZeroGuess, PETSC_NULL);

  stiffness = 1.0;
  PetscOptionsGetScalar(PETSC_NULL, "-stiffness", &stiffness, PETSC_NULL);

  stopTol = 1.0e-10;
  PetscOptionsGetScalar(PETSC_NULL, "-stopTol", &stopTol, PETSC_NULL);

  PetscLogStagePush(initialStage);
  
  createGridSizes(&N, nlevels);

  createComms(&activeNpes, &iAmActive, &activeComms, &comms, N, nlevels);

  createDA(&da, iAmActive, activeComms, N, nlevels);

  createThresholds(argv[1], da[nlevels - 1], &rightThreshold, &backThreshold, &topThreshold);

  createStiffnessData(&kData, da, nlevels);

  createCoarseMatData(&cData, da[0]);

  createKSPShellData(&kspShellData);

  createNeumannMat(&NeumannMat, (kData + (nlevels - 1)));

  createStiffnessMat(&Kmat, kData, cData, nlevels);

  createRPmats(&Rmat, &Pmat, da, nlevels);

  createSolver(&ksp, kspShellData, Rmat, Pmat, comms, nlevels);

  createRHSandSol(&rhs, &sol, da[nlevels - 1]);

  VecDuplicate(rhs, &rhsInp);

  setRHSinp(rhsInp, da[nlevels - 1]);

  if(!useZeroGuess) {
    VecZeroEntries(sol);
  }

  initFinestStiffnessData(kData + (nlevels - 1));

  if(initFromFile) {
    char fname[100];
    sprintf(fname, "%s_Bonds.txt", argv[2]);
    resetFinestStiffnessDataFromFile(fname, initNumBrokenBonds, (kData + (nlevels - 1)));
  }

  PetscLogStagePop();

  PetscLogStagePush(solveStage);

  if(!rank) {
    char fnameB[100];
    char fnameR[100];
    sprintf(fnameB, "%s_Bonds.txt", argv[2]);
    sprintf(fnameR, "%s_Reactions.txt", argv[2]);
    ofpB = fopen(fnameB, "a"); 
    ofpR = fopen(fnameR, "a"); 
  }

  repeatLoop = 1;
  iter = 0;
  while(repeatLoop) {

    coarsenStiffnessData(kData, nlevels);

    if(da[0]) {
      buildStiffnessMat(cData->mat, kData);
    }

    MatMult(NeumannMat, rhsInp, rhs);
    resetBC(rhs, da[nlevels - 1]);

    KSPGetPC(ksp, &pc);

    PetscTypeCompare((PetscObject)pc, PCMG, &ismg);

    KSPSetOperators(ksp, Kmat[nlevels - 1], Kmat[nlevels - 1], SAME_NONZERO_PATTERN);

    if(ismg) {
      //0 is the coarsest
      for( lev = 0; lev < nlevels; lev++) {
        PCMGGetSmoother(pc, lev, &lksp);
        KSPSetOperators(lksp, Kmat[lev], Kmat[lev], SAME_NONZERO_PATTERN);
      }//end for lev
    }

    if(useZeroGuess) {
      VecZeroEntries(sol);
    }

    KSPSolve(ksp, rhs, sol);

    VecNorm(sol, NORM_2, &solNorm2);
    VecNorm(sol, NORM_INFINITY, &solNormInf);

    totalReaction = computeTotalReaction(sol, (kData + (nlevels - 1)), stiffness);

    breakBond(sol, rightThreshold, backThreshold, topThreshold,
        (kData + (nlevels - 1)), stiffness, &xb, &yb, &zb, &rbt, &breakFactor);

    if(!rank) {
      fprintf(ofpB, "%d %d %d %d\n", xb, yb, zb, rbt);
      fprintf(ofpR, "%lE %lE %lf %lf\n", (1.0/breakFactor), (totalReaction/breakFactor), solNorm2, solNormInf);
    }

    resetFinestStiffnessData((kData + (nlevels - 1)), xb, yb, zb, rbt);

    if(iter == (maxIts - 1)) {
      repeatLoop = 0;
    }

    if( (fabs(totalReaction)) < stopTol ) {
      repeatLoop = 0;
    }

    if( (fabs(breakFactor)) < stopTol ) {
      repeatLoop = 0;
    }

    iter++;
  }//end while

  if(!rank) {
    fclose(ofpB);
    fclose(ofpR);
  }

  PetscLogStagePop();

  PetscLogStagePush(finalStage);
  
  KSPDestroy(ksp);

  VecDestroy(sol);
  VecDestroy(rhs);
  VecDestroy(rhsInp);

  for(lev = 0; lev < nlevels; lev++) {
    MatDestroy(Kmat[lev]);
  }//end for lev
  free(Kmat);
  MatDestroy(NeumannMat);

  for(lev = 0; lev < (nlevels - 1); lev++) {
    MatDestroy(Rmat[lev]);
    MatDestroy(Pmat[lev]);
  }//end for lev
  free(Rmat);
  free(Pmat);

  destroyStiffnessData(kData, nlevels);
  free(kData);

  destroyCoarseMatData(cData);
  free(cData);

  free(kspShellData);

  for(lev = 0; lev < nlevels; lev++) {
    if(da[lev]) {
      DADestroy(da[lev]);
      da[lev] = NULL;
    }
  }//end for lev
  free(da);

  free(comms);
  free(activeComms);
  free(activeNpes);
  free(iAmActive);

  free(N);

  if(rightThreshold) {
    VecDestroy(rightThreshold);
  }
  if(backThreshold) {
    VecDestroy(backThreshold);
  }
  if(topThreshold) {
    VecDestroy(topThreshold);
  }

  PetscLogStagePop();

  PetscFinalize();

  return 1;

}


