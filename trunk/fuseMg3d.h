
#ifndef __FUSE_MG_3D__
#define __FUSE_MG_3D__

#include "petsc.h"
#include "petscvec.h"
#include "petscmat.h"
#include "petscda.h"
#include "petscksp.h"
#include "petscpc.h"

#define __DOF__(x, y, z, N) ( ((N)*( ((N)*(z)) + (y) )) + (x) )

#define __CUBE__(x) ((x)*(x)*(x))

#define __PROC_FACTOR__ 4000

#define __RIGHT__ 0
#define __BACK__ 1
#define __TOP__ 2

extern PetscLogEvent createThresholdEvent;

extern PetscLogEvent nMatvecEvent;
extern PetscLogEvent kMatvecEvent;
extern PetscLogEvent coarsenKdataEvent;
extern PetscLogEvent resetKdataEvent;

extern PetscLogEvent createRPmatEvent;
extern PetscLogEvent buildPmatEvent;

//N is the number of nodes (vertices) in each direction

typedef struct {
  Mat mat;
  Vec inTmp;
  Vec outTmp;
} CoarseMatData;

typedef struct {
  PC pc; 
  KSP ksp_private; 
  Vec rhs_private;  
  Vec sol_private; 
} PC_KSP_Shell;

typedef struct {
  DA da;
  Vec inActive;
  Vec outActive;
  Vec diagActive;
  Vec rightPropLocal;
  Vec backPropLocal;
  Vec topPropLocal;
} StiffnessData;

//Aux Functions

void createSolver(KSP* _ksp, PC_KSP_Shell* kspShellData, Mat* Rmat, Mat* Pmat, MPI_Comm* comms, PetscInt nlevels);

double computeTotalReaction(Vec sol, StiffnessData* data, PetscScalar stiffness);

void breakBond(Vec sol, Vec rightThreshold, Vec backThreshold, Vec topThreshold,
    StiffnessData* data, PetscScalar stiffness, 
    PetscInt* _xb, PetscInt* _yb, PetscInt* _zb, PetscInt* _rbt, double* _maxGlobalLambda);

PetscErrorCode createThresholds(char* fname, DA da, Vec* _rightThreshold, Vec* _backThreshold, Vec* _topThreshold);

void setRHSinp(Vec rhsInp, DA da);

void resetBC(Vec rhs, DA da);

void createRHSandSol(Vec* _rhs, Vec* _sol, DA da);

void createDA(DA** _da, int* iAmActive, MPI_Comm* activeComms, PetscInt* N, PetscInt nlevels);

void createComms(int** _activeNpes, int** _iAmActive, MPI_Comm** _activeComms,
    MPI_Comm** _comms, PetscInt* N, PetscInt nlevels);

void createGridSizes(PetscInt** _N, PetscInt nlevels);

int foundValidDApart(int N, int npes);

//Stiffness Functions

void createCoarseMatData(CoarseMatData** _cData, DA da);

void destroyCoarseMatData(CoarseMatData* _cData);

void createStiffnessData(StiffnessData** _kData, DA* da, PetscInt nlevels);

void destroyStiffnessData(StiffnessData* _kData, PetscInt nlevels);

void initFinestStiffnessData(StiffnessData* kData);

PetscErrorCode resetFinestStiffnessData(StiffnessData* kData, PetscInt xb, PetscInt yb, PetscInt zb, PetscInt rbt);

PetscErrorCode resetFinestStiffnessDataFromFile(char* fname, PetscInt initNumBrokenBonds, StiffnessData* kData);

PetscErrorCode coarsenStiffnessData(StiffnessData* kData, PetscInt nlevels);

PetscErrorCode dummyMatDestroy(Mat mat);

PetscErrorCode stiffnessGetDiagonal(Mat mat, Vec diag);

PetscErrorCode stiffnessMatvec(Mat mat, Vec in, Vec out);

PetscErrorCode neumannMatvec(Mat mat, Vec in, Vec out);

PetscErrorCode coarseMatvec(Mat mat, Vec in, Vec out);

void buildStiffnessMat(Mat mat, StiffnessData* data);

void createStiffnessMat(Mat** _Kmat, StiffnessData* kData, CoarseMatData* cData, PetscInt nlevels);

void createNeumannMat(Mat* _NeumannMat, StiffnessData* kData);

//Intergrid Functions

PetscErrorCode createRPmats(Mat** _Rmat, Mat** _Pmat, DA* da, PetscInt nlevels);

PetscErrorCode buildPmat(Mat Pmat, DA dac, DA daf);

//KSP_Shell Functions

PetscErrorCode PC_KSP_Shell_SetUp(void* ctx);

PetscErrorCode PC_KSP_Shell_Destroy(void* ctx);

PetscErrorCode PC_KSP_Shell_Apply(void* ctx, Vec rhs, Vec sol);

void createKSPShellData(PC_KSP_Shell** _kspShellData);

#endif



