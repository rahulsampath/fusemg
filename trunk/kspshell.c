
#include "fuseMg3d.h"
#include "private/pcimpl.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

void createKSPShellData(PC_KSP_Shell** _kspShellData) {
  PC_KSP_Shell* kspShellData;

  kspShellData = (PC_KSP_Shell*)(malloc( sizeof(PC_KSP_Shell) ));
  kspShellData->sol_private = NULL;
  kspShellData->rhs_private = NULL;
  kspShellData->ksp_private = NULL;

  *_kspShellData = kspShellData;
}

PetscErrorCode PC_KSP_Shell_SetUp(void* ctx) {

  PC_KSP_Shell* data = (PC_KSP_Shell*)(ctx); 

  CoarseMatData* cData;

  //This points to the shell itself
  PC pc = data->pc;

  Mat Amat;

  PetscFunctionBegin;

 // printf("Inside KSP_Shell_Setup\n");
 // printf("PCSetupCalled = %d\n", (pc->setupcalled));

  PCGetOperators(pc, &Amat, NULL, NULL);      

  MatShellGetContext(Amat, (void **)&cData);

  //Create ksp_private, rhs_private, sol_private,
  if(cData->mat) {
    Mat Amat_private = cData->mat;
    Mat Pmat_private = cData->mat;
    MatStructure pFlag = SAME_NONZERO_PATTERN;

    if(pc->setupcalled == 0) {
      assert(data->ksp_private == NULL);
      assert(data->rhs_private == NULL);
      assert(data->sol_private == NULL);
    } else {
      assert(data->ksp_private != NULL);
      assert(data->rhs_private != NULL);
      assert(data->sol_private != NULL);
    }

    if(pc->setupcalled == 0) {
      const char *prefix;
      PC privatePC;
      MPI_Comm commActive;

      PetscObjectGetComm((PetscObject)Amat_private, &commActive);

      KSPCreate(commActive, &(data->ksp_private));

      PCGetOptionsPrefix(pc, &prefix);

      //These functions also set the correct prefix for the inner pc 
      KSPSetOptionsPrefix(data->ksp_private, prefix);
      KSPAppendOptionsPrefix(data->ksp_private, "private_");

      //Default Types for KSP and PC
      KSPSetType(data->ksp_private, KSPPREONLY);

      KSPGetPC(data->ksp_private, &privatePC);
      PCSetType(privatePC, PCLU);

      //The command line options get higher precedence.
      //This also calls PCSetFromOptions for the private pc internally
      KSPSetFromOptions(data->ksp_private);  

      MatGetVecs(Amat_private, &(data->sol_private), &(data->rhs_private));
    }

    KSPSetOperators(data->ksp_private, Amat_private, Pmat_private, pFlag);

  } else {
    data->sol_private = NULL;
    data->rhs_private = NULL;
    data->ksp_private = NULL;
  }//end if active

  PetscFunctionReturn(0);
}

PetscErrorCode PC_KSP_Shell_Destroy(void* ctx) {

  PC_KSP_Shell* data = (PC_KSP_Shell*)(ctx); 

  PetscFunctionBegin;

  //printf("Inside KSP_Shell_Destroy\n");

  if(data->ksp_private) {
    KSPDestroy(data->ksp_private);
    data->ksp_private = NULL;
  }

  if(data->rhs_private) {
    VecDestroy(data->rhs_private);
    data->rhs_private = NULL;
  }

  if(data->sol_private) {
    VecDestroy(data->sol_private);
    data->sol_private = NULL;
  }

  PetscFunctionReturn(0);
}

PetscErrorCode PC_KSP_Shell_Apply(void* ctx, Vec rhs, Vec sol) {

  PC_KSP_Shell* data = (PC_KSP_Shell*)(ctx); 

  PetscFunctionBegin;

  //printf("Inside KSP_Shell_Apply\n");

  if(data->ksp_private) {      
    PetscScalar* rhsArray;
    PetscScalar* solArray;

    //There are no copies and no mallocs involved.

    VecGetArray(rhs, &rhsArray);
    VecGetArray(sol, &solArray);

    VecPlaceArray(data->rhs_private, rhsArray);
    VecPlaceArray(data->sol_private, solArray);

    KSPSolve(data->ksp_private, data->rhs_private, data->sol_private);

    VecResetArray(data->rhs_private);
    VecResetArray(data->sol_private);

    VecRestoreArray(rhs, &rhsArray);
    VecRestoreArray(sol, &solArray);
  }

  PetscFunctionReturn(0);
}


