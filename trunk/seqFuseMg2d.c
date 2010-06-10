
#include "petsc.h"
#include "petscsys.h"
#include "petscmat.h"
#include "petscvec.h"
#include "petscksp.h"
#include "petscpc.h"
#include "petscmg.h"
#include "stdio.h"
#include "assert.h"
#include "stdlib.h"
#include "math.h"

int kMatvecEvent;
int rMatvecEvent;
int pMatvecEvent;

//N is the number of nodes (vertices) in each direction

#define __DOF__(xi, yi, N) (((N)*(yi)) + (xi)) 

//My Bond Numbering Scheme

#define __TOP__ 0
#define __RIGHT__ 1

#define __LB_OFF__(N) ( (2*((N) - 1)*((N) - 1)) )
#define __BB_OFF__(N) ( (__LB_OFF__(N)) + ((N) - 1) )

#define __I_BOND__(xi, yi, T_R, N) ( (2*((((N) - 1)*(yi)) + (xi))) + (T_R) )
#define __L_BOND__(yi, N) ( (__LB_OFF__(N)) + (yi) )
#define __B_BOND__(xi, N) ( (__BB_OFF__(N)) + (xi) )

//Pallab's Bond Numbering Scheme

#define __VOFF__(N) ( ((N - 1)*(N - 1)) + (N) )

#define __VBONDNUM__(BondNum, N) ( (BondNum) - (__VOFF__(N)) )

#define __XH__(BondNum, N) ( ((BondNum) - 1) % ((N) - 1) )

#define __YH__(BondNum, N) ( ((BondNum) - 1) / ((N) - 1)  )

#define __I_XV__(VBondNum, N) ( (VBondNum) / ((N) - 1)  )

#define __I_YV__(VBondNum, N) ( (VBondNum) % ((N) - 1) )

#define __XV__(BondNum, N) ( __I_XV__((__VBONDNUM__(BondNum, N)), N) )

#define __YV__(BondNum, N) ( __I_YV__((__VBONDNUM__(BondNum, N)), N) )

#define __IS_V__(BondNum, N) ( (BondNum) >= (__VOFF__((N))) )

#define __XB__(BondNum, N)  ( ( __IS_V__(BondNum, N) ) ? ( __XV__((BondNum), (N)) ) : ( __XH__((BondNum), (N)) ) ) 

#define __YB__(BondNum, N)  ( ( __IS_V__(BondNum, N) ) ? ( __YV__((BondNum), (N)) ) : ( __YH__((BondNum), (N)) ) ) 

//Mapping between Pallab's Numbering and My Numbering

#define __IS_L__(X) ( (X) == 0)

#define __IS_B__(Y) ( (Y) == 0)

#define __L_ID__(BondNum, N) ( __L_BOND__( (__YB__(BondNum, N)), (N) ) )

#define __B_ID__(BondNum, N) ( __B_BOND__( (__XB__(BondNum, N)), (N) ) )

#define __T_ID__(BondNum, N) ( __I_BOND__( (__XB__(BondNum, N)), ((__YB__(BondNum, N)) - 1), __TOP__,  (N) ) )

#define __R_ID__(BondNum, N) ( __I_BOND__( ((__XB__(BondNum, N)) - 1), (__YB__(BondNum, N)), __RIGHT__, (N) ) )

#define __L_R_ID__(BondNum, N) ( (__IS_L__(__XB__(BondNum, N))) ? (__L_ID__(BondNum, N)) : (__R_ID__(BondNum, N)) )

#define __T_B_ID__(BondNum, N) ( (__IS_B__(__YB__(BondNum, N))) ? (__B_ID__(BondNum, N)) : (__T_ID__(BondNum, N)) )

#define __MY_BOND_ID__(BondNum, N) ( (__IS_V__(BondNum, N)) ? (__L_R_ID__(BondNum, N)) : (__T_B_ID__(BondNum, N)) )

typedef struct {
  Vec addRtmp;
  Vec addPtmp;
} TransferOpData;

typedef struct {
  double* matProp;
} StiffnessData;

void setCoarseMat(Mat mat, int N, StiffnessData* data) {

  PetscScalar h;
  PetscInt row, col;
  PetscScalar val;
  int xi, yi;
  double topProp, rightProp, leftProp;

  MatZeroEntries(mat);

  h = (1.0/((double)(N - 1)));

  //Element based assembly
  //Every element adds the contribution for the top and right edges

  //Bottom Boundary is Dirichlet boundary

  for(yi = 0; yi < (N - 1); yi++) {
    for(xi = 0; xi < (N - 1); xi++) {
      topProp = (data->matProp)[__I_BOND__(xi, yi, __TOP__, N)];
      rightProp = (data->matProp)[__I_BOND__(xi, yi, __RIGHT__, N)];

      //Top Edge
      row = __DOF__(xi, yi + 1, N);
      col = __DOF__(xi, yi + 1, N); 
      val = topProp;
      MatSetValues(mat, 1, &row, 1, &col, &val, ADD_VALUES);

      row = __DOF__(xi, yi + 1, N);
      col = __DOF__(xi + 1, yi + 1, N); 
      val = -topProp;
      MatSetValues(mat, 1, &row, 1, &col, &val, ADD_VALUES);

      row = __DOF__(xi + 1, yi + 1, N);
      col = __DOF__(xi, yi + 1, N); 
      val = -topProp;
      MatSetValues(mat, 1, &row, 1, &col, &val, ADD_VALUES);

      row = __DOF__(xi + 1, yi + 1, N);
      col = __DOF__(xi + 1, yi + 1, N); 
      val = topProp;
      MatSetValues(mat, 1, &row, 1, &col, &val, ADD_VALUES);

      //Right Edge
      if(yi) {
        row = __DOF__(xi + 1, yi, N);
        col = __DOF__(xi + 1, yi, N); 
        val = rightProp;
        MatSetValues(mat, 1, &row, 1, &col, &val, ADD_VALUES);

        row = __DOF__(xi + 1, yi, N);
        col = __DOF__(xi + 1, yi + 1, N); 
        val = -rightProp;
        MatSetValues(mat, 1, &row, 1, &col, &val, ADD_VALUES);

        row = __DOF__(xi + 1, yi + 1, N);
        col = __DOF__(xi + 1, yi, N); 
        val = -rightProp;
        MatSetValues(mat, 1, &row, 1, &col, &val, ADD_VALUES);

        row = __DOF__(xi + 1, yi + 1, N);
        col = __DOF__(xi + 1, yi + 1, N); 
        val = rightProp;
        MatSetValues(mat, 1, &row, 1, &col, &val, ADD_VALUES);
      } else {
        row = __DOF__(xi + 1, yi + 1, N);
        col = __DOF__(xi + 1, yi + 1, N); 
        val = rightProp;
        MatSetValues(mat, 1, &row, 1, &col, &val, ADD_VALUES);
      }

    }//end for xi
  }//end for yi

  //Left Boundary
  {
    xi = 0;
    for( yi = 0; yi < (N - 1); yi++) {
      leftProp = (data->matProp)[__L_BOND__(yi, N)];

      if(yi) {
        row = __DOF__(xi, yi, N);
        col = __DOF__(xi , yi, N); 
        val = leftProp;
        MatSetValues(mat, 1, &row, 1, &col, &val, ADD_VALUES);

        row = __DOF__(xi, yi, N);
        col = __DOF__(xi , yi + 1, N); 
        val = -leftProp;
        MatSetValues(mat, 1, &row, 1, &col, &val, ADD_VALUES);

        row = __DOF__(xi, yi + 1, N);
        col = __DOF__(xi , yi, N); 
        val = -leftProp;
        MatSetValues(mat, 1, &row, 1, &col, &val, ADD_VALUES);

        row = __DOF__(xi, yi + 1, N);
        col = __DOF__(xi , yi + 1, N); 
        val = leftProp;
        MatSetValues(mat, 1, &row, 1, &col, &val, ADD_VALUES);
      } else {
        row = __DOF__(xi, yi + 1, N);
        col = __DOF__(xi , yi + 1, N); 
        val = leftProp;
        MatSetValues(mat, 1, &row, 1, &col, &val, ADD_VALUES);
      }
    }//end for yi
  }

  MatAssemblyBegin(mat, MAT_FLUSH_ASSEMBLY);
  MatAssemblyEnd(mat, MAT_FLUSH_ASSEMBLY);

  //Bottom Boundary 
  {
    yi = 0;
    for( xi = 0; xi < N; xi++) {
      row = __DOF__(xi, yi, N);
      col = __DOF__(xi, yi, N); 
      val = h*h;
      MatSetValues(mat, 1, &row, 1, &col, &val, INSERT_VALUES);
    }//end for xi
  }

  MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY);

  MatScale(mat, (1.0/(h*h)));

}

PetscErrorCode stiffnessMatvec(Mat mat, Vec in, Vec out) {

  StiffnessData *data;  
  PetscScalar* inarr;
  PetscScalar* outarr;
  PetscScalar h;
  PetscInt Nsq, N;
  int xi, yi;
  double topProp, leftProp, rightProp;

  PetscFunctionBegin;

  PetscLogEventBegin(kMatvecEvent, 0, 0, 0, 0);

  MatShellGetContext( mat, (void **)&data);

  VecGetSize(in, &Nsq);

  //Number of nodes
  N = sqrt(Nsq);
  h = (1.0/((double)(N - 1)));

  VecZeroEntries(out);

  VecGetArray(in, &inarr);
  VecGetArray(out, &outarr);

  //Element based assembly
  //Every element adds the contribution for the top and right edges

  //Bottom Boundary is Dirichlet boundary

  for( yi = 0; yi < (N - 1); yi++) {
    for( xi = 0; xi < (N - 1); xi++) {
      topProp = (data->matProp)[__I_BOND__(xi, yi, __TOP__, N)];
      rightProp = (data->matProp)[__I_BOND__(xi, yi, __RIGHT__, N)];

      //Top Edge
      outarr[__DOF__(xi, yi + 1, N)] += (topProp*(inarr[__DOF__(xi, yi + 1, N)] 
            - inarr[__DOF__(xi + 1, yi + 1, N)]));
      outarr[__DOF__(xi + 1, yi + 1, N)] += (topProp*(-inarr[__DOF__(xi, yi + 1, N)] 
            + inarr[__DOF__(xi + 1, yi + 1, N)]));

      //Right Edge
      if(yi) {
        outarr[__DOF__(xi + 1, yi, N)] += (rightProp*(inarr[__DOF__(xi + 1, yi, N)] 
              - inarr[__DOF__(xi + 1, yi + 1, N)]));
        outarr[__DOF__(xi + 1, yi + 1, N)] += (rightProp*(-inarr[__DOF__(xi + 1, yi, N)] 
              + inarr[__DOF__(xi + 1, yi + 1, N)]));
      } else {
        outarr[__DOF__(xi + 1, yi + 1, N)] += (rightProp*inarr[__DOF__(xi + 1, yi + 1, N)]);
      }

    }//end for xi
  }//end for yi

  //Left Boundary
  {
    xi = 0;
    for( yi = 0; yi < (N - 1); yi++) {
      leftProp = (data->matProp)[__L_BOND__(yi, N)];

      if(yi) {
        outarr[__DOF__(xi , yi, N)] += (leftProp*(inarr[__DOF__(xi , yi, N)] 
              - inarr[__DOF__(xi , yi + 1, N)]));
        outarr[__DOF__(xi , yi + 1, N)] += (leftProp*(-inarr[__DOF__(xi , yi, N)] 
              + inarr[__DOF__(xi , yi + 1, N)]));
      } else {
        outarr[__DOF__(xi , yi + 1, N)] += (leftProp*inarr[__DOF__(xi , yi + 1, N)]);
      }
    }//end for yi
  }

  //Bottom Boundary 
  {
    yi = 0;
    for( xi = 0; xi < N; xi++) {
      outarr[__DOF__(xi, yi , N)] = h*h*inarr[__DOF__(xi, yi , N)];
    }//end for xi
  }

  VecRestoreArray(in, &inarr);
  VecRestoreArray(out, &outarr);

  VecScale(out, (1.0/(h*h)));

  PetscLogEventEnd(kMatvecEvent, 0, 0, 0, 0);

  PetscFunctionReturn(0);
}

PetscErrorCode stiffnessGetDiagonal(Mat mat, Vec diag) {

  StiffnessData *data;  
  PetscScalar* diagarr;
  double topProp, rightProp, leftProp;
  PetscInt Nsq, N;
  PetscScalar h;
  int xi, yi;

  PetscFunctionBegin;

  MatShellGetContext( mat, (void **)&data);

  VecGetSize(diag, &Nsq);

  //Number of nodes
  N = sqrt(Nsq);
  h = (1.0/((double)(N - 1)));

  VecZeroEntries(diag);

  VecGetArray(diag, &diagarr);

  //Element based assembly
  //Every element adds the contribution for the top and right edges

  //Bottom Boundary is Dirichlet boundary

  for( yi = 0; yi < (N - 1); yi++) {
    for( xi = 0; xi < (N - 1); xi++) {
      topProp = (data->matProp)[__I_BOND__(xi, yi, __TOP__, N)];
      rightProp = (data->matProp)[__I_BOND__(xi, yi, __RIGHT__, N)];

      //Top Edge
      diagarr[__DOF__(xi, yi + 1, N)] += topProp;
      diagarr[__DOF__(xi + 1, yi + 1, N)] += topProp;

      //Right Edge
      if(yi) {
        diagarr[__DOF__(xi + 1, yi, N)] += rightProp;
        diagarr[__DOF__(xi + 1, yi + 1, N)] += rightProp;
      } else {
        diagarr[__DOF__(xi + 1, yi + 1, N)] += rightProp;
      }

    }//end for xi
  }//end for yi

  //Left Boundary
  {
    xi = 0;
    for( yi = 0; yi < (N - 1); yi++) {
      leftProp = (data->matProp)[__L_BOND__(yi, N)];

      if(yi) {
        diagarr[__DOF__(xi , yi, N)] += leftProp;
        diagarr[__DOF__(xi , yi + 1, N)] += leftProp;
      } else {
        diagarr[__DOF__(xi , yi + 1, N)] += leftProp;
      }
    }//end for yi
  }

  //Bottom Boundary 
  {
    yi = 0;
    for( xi = 0; xi < N; xi++) {
      diagarr[__DOF__(xi, yi , N)] = h*h;
    }//end for xi
  }

  VecRestoreArray(diag, &diagarr);

  VecScale(diag, (1.0/(h*h)));

  PetscFunctionReturn(0);
}

PetscErrorCode  addRestrictMatvec(Mat R, Vec v1, Vec v2, Vec v3)	
{

  PetscScalar one;
  TransferOpData *data;
  Vec tmp;

  PetscFunctionBegin;

  one = 1.0;

  if((v2!=v3) && (v1!=v3)) {
    //Note This will fail only if v2==v3 or v1 ==v3!(i.e they are identical copies pointing to the same memory location)
    MatMult(R, v1, v3);//v3 = R*v1
    VecAXPY(v3,one,v2);//v3 = v3+ v2=v2 + R*v1
  }else {
    //This is less efficient but failproof.
    MatShellGetContext( R, (void **)&data);
    tmp = data->addRtmp;
    if(tmp == NULL) {
      VecDuplicate(v3,&tmp);
      data->addRtmp = tmp;
    }
    MatMult(R, v1, tmp);//tmp=R*v1;
    VecWAXPY(v3,one,v2,tmp);//v3 = (1*v2)+tmp=v2 + R*v1
  }

  PetscFunctionReturn(0);
}

PetscErrorCode addProlongMatvec(Mat R, Vec v1, Vec v2, Vec v3)
{
  PetscScalar one;
  TransferOpData *data;			
  Vec tmp;

  PetscFunctionBegin;

  one = 1.0;

  if((v2!=v3) && (v1!=v3)) {
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

PetscErrorCode prolongMatvecType(Mat R, Vec coarse, Vec fine) {

  PetscInt Nfsq, Ncsq, Nc, Nf;
  PetscScalar* farr;
  PetscScalar* carr;
  int xci, yci;

  PetscFunctionBegin;

  PetscLogEventBegin(pMatvecEvent, 0, 0, 0, 0);

  VecGetSize(fine, &Nfsq);
  VecGetSize(coarse, &Ncsq);

  Nf = sqrt(Nfsq);
  Nc = sqrt(Ncsq);

  assert( (Nf - 1) == (2*(Nc - 1)) );

  VecZeroEntries(fine);

  VecGetArray(fine, &farr);
  VecGetArray(coarse, &carr);

  //Bottom is dirichlet boundary

  //Loop over coarse nodes 
  for( yci = 1; yci < Nc; yci++) {
    for( xci = 0; xci < Nc; xci++) {
      farr[__DOF__((2*xci), (2*yci), Nf)] = carr[__DOF__(xci, yci, Nc)];
    }//end for xci
  }//end for yci

  //Loop over coarse elements
  for( yci = 0; yci < (Nc - 1); yci++) {
    for( xci = 0; xci < (Nc - 1); xci++) {
      //Center
      if(yci) {
        farr[__DOF__(((2*xci) + 1), ((2*yci) + 1), Nf)] = (0.25*(carr[__DOF__(xci, yci, Nc)] +
              carr[__DOF__((xci + 1), yci, Nc)] + carr[__DOF__(xci, (yci + 1), Nc)] 
              + carr[__DOF__((xci + 1), (yci + 1), Nc)]));
      } else {
        farr[__DOF__(((2*xci) + 1), ((2*yci) + 1), Nf)] = (0.25*(carr[__DOF__(xci, (yci + 1), Nc)] 
              + carr[__DOF__((xci + 1), (yci + 1), Nc)]));
      }

      //Top edge
      farr[__DOF__(((2*xci) + 1), (2*(yci + 1)), Nf)] = (0.5*(carr[__DOF__(xci, (yci + 1), Nc)] 
            + carr[__DOF__((xci + 1), (yci + 1), Nc)]));

      //Right edge
      if(yci) {
        farr[__DOF__((2*(xci + 1)), ((2*yci) + 1), Nf)] = (0.5*(carr[__DOF__((xci + 1), yci, Nc)] 
              + carr[__DOF__((xci + 1), (yci + 1), Nc)]));
      } else {
        farr[__DOF__((2*(xci + 1)), ((2*yci) + 1), Nf)] = (0.5*carr[__DOF__((xci + 1), (yci + 1), Nc)]);
      }
    }//end for xci
  }//end for yci

  //Left Boundary
  {
    xci = 0;
    for( yci = 0; yci < (Nc - 1); yci++) {
      if(yci) {
        farr[__DOF__((2*xci), ((2*yci) + 1), Nf)] = (0.5*(carr[__DOF__(xci, yci, Nc)] + carr[__DOF__(xci, (yci + 1), Nc)]));
      } else {
        farr[__DOF__((2*xci), ((2*yci) + 1), Nf)] = (0.5*carr[__DOF__(xci, (yci + 1), Nc)]);
      }
    }//end for yci
  }

  VecRestoreArray(fine, &farr);
  VecRestoreArray(coarse, &carr);

  PetscLogEventEnd(pMatvecEvent, 0, 0, 0, 0);

  PetscFunctionReturn(0);
}

PetscErrorCode restrictMatvecType(Mat R, Vec fine, Vec coarse) {

  PetscScalar* farr;
  PetscScalar* carr;
  PetscInt Nfsq, Ncsq, Nc, Nf;
  int xci, yci;

  PetscFunctionBegin;

  PetscLogEventBegin(rMatvecEvent, 0, 0, 0, 0);

  VecGetSize(fine, &Nfsq);
  VecGetSize(coarse, &Ncsq);

  Nf = sqrt(Nfsq);
  Nc = sqrt(Ncsq);

  assert( (Nf - 1) == (2*(Nc - 1)) );

  VecZeroEntries(coarse);

  VecGetArray(fine, &farr);
  VecGetArray(coarse, &carr);

  //Bottom is dirichlet boundary

  //Loop over coarse nodes 
  for( yci = 1; yci < Nc; yci++) {
    for( xci = 0; xci < Nc; xci++) {
      carr[__DOF__(xci, yci, Nc)] = farr[__DOF__((2*xci), (2*yci), Nf)];
    }//end for xci
  }//end for yci

  //Loop over coarse elements
  for( yci = 0; yci < (Nc - 1); yci++) {
    for( xci = 0; xci < (Nc - 1); xci++) {
      //Center
      if(yci) {
        carr[__DOF__(xci, yci, Nc)] += (0.25*farr[__DOF__(((2*xci) + 1), ((2*yci) + 1), Nf)]);
        carr[__DOF__((xci + 1), yci, Nc)] += (0.25*farr[__DOF__(((2*xci) + 1), ((2*yci) + 1), Nf)]);
        carr[__DOF__(xci, (yci + 1), Nc)] += (0.25*farr[__DOF__(((2*xci) + 1), ((2*yci) + 1), Nf)]);
        carr[__DOF__((xci + 1), (yci + 1), Nc)] += (0.25*farr[__DOF__(((2*xci) + 1), ((2*yci) + 1), Nf)]);
      } else {
        carr[__DOF__(xci, (yci + 1), Nc)] += (0.25*farr[__DOF__(((2*xci) + 1), ((2*yci) + 1), Nf)]);
        carr[__DOF__((xci + 1), (yci + 1), Nc)] += (0.25*farr[__DOF__(((2*xci) + 1), ((2*yci) + 1), Nf)]);
      }

      //Top edge
      carr[__DOF__(xci, (yci + 1), Nc)] += (0.5*farr[__DOF__(((2*xci) + 1), (2*(yci + 1)), Nf)]);
      carr[__DOF__((xci + 1), (yci + 1), Nc)] += (0.5*farr[__DOF__(((2*xci) + 1), (2*(yci + 1)), Nf)]);

      //Right edge
      if(yci) {
        carr[__DOF__((xci + 1), yci, Nc)] += (0.5*farr[__DOF__((2*(xci + 1)), ((2*yci) + 1), Nf)]);
        carr[__DOF__((xci + 1), (yci + 1), Nc)] += (0.5*farr[__DOF__((2*(xci + 1)), ((2*yci) + 1), Nf)]);
      } else {
        carr[__DOF__((xci + 1), (yci + 1), Nc)] += (0.5*farr[__DOF__((2*(xci + 1)), ((2*yci) + 1), Nf)]);
      }
    }//end for xci
  }//end for yci

  //Left Boundary
  {
    xci = 0;
    for( yci = 0; yci < (Nc - 1); yci++) {
      if(yci) {
        carr[__DOF__(xci, yci, Nc)] += (0.5*farr[__DOF__((2*xci), ((2*yci) + 1), Nf)]);
        carr[__DOF__(xci, (yci + 1), Nc)] += (0.5*farr[__DOF__((2*xci), ((2*yci) + 1), Nf)]);
      } else {
        carr[__DOF__(xci, (yci + 1), Nc)] += (0.5*farr[__DOF__((2*xci), ((2*yci) + 1), Nf)]);
      }
    }//end for yci
  }

  VecRestoreArray(fine, &farr);
  VecRestoreArray(coarse, &carr);

  VecScale(coarse, 0.25);

  PetscLogEventEnd(rMatvecEvent, 0, 0, 0, 0);

  PetscFunctionReturn(0);
}

PetscErrorCode stiffnessDestroy(Mat mat) {
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

PetscErrorCode  rpDestroy(Mat R) {

  TransferOpData *data;

  PetscFunctionBegin;

  MatShellGetContext( R, (void **)&data);
  if(data) {
    if(data->addRtmp) {
      VecDestroy(data->addRtmp);
      data->addRtmp = NULL;
    }
    if(data->addPtmp) {
      VecDestroy(data->addPtmp);
      data->addPtmp = NULL;
    }
    free(data);
    data = NULL;
  }

  PetscFunctionReturn(0);
}

int main(int argc, char** argv) {

  PetscInt nlevels, numBrokenBonds;
  int numBonds, i, lev, totalBrokenBonds, bondId;
  int Nc, Nf, xci, yci;
  PetscInt* N;
  Mat* Kmat;
  Mat* Rmat;
  StiffnessData* kData;
  TransferOpData* rpData;
  MPI_Comm* comms;
  FILE* fin;
  KSP ksp;
  KSP lksp;
  PC pc;
  Vec rhs, sol;

  PetscInitialize(&argc, &argv, "options", "Testing Multigrid for the 2-D Fuse Problem");

  PetscLogEventRegister("Kmatvec", PETSC_VIEWER_COOKIE, &kMatvecEvent);
  PetscLogEventRegister("Rmatvec", PETSC_VIEWER_COOKIE, &rMatvecEvent);
  PetscLogEventRegister("Pmatvec", PETSC_VIEWER_COOKIE, &pMatvecEvent);

  nlevels = 2;
  PetscOptionsGetInt(PETSC_NULL, "-nlevels", &nlevels, PETSC_NULL);

  numBrokenBonds = 0;
  PetscOptionsGetInt(PETSC_NULL, "-numBrokenBonds", &numBrokenBonds, PETSC_NULL);

  //0 is the coarsest
  N = (PetscInt*)(malloc( sizeof(PetscInt)*nlevels ));
  PetscOptionsGetInt(PETSC_NULL, "-N", N + nlevels - 1, PETSC_NULL);

  for(lev = (nlevels - 2); lev >= 0; lev--) {
    N[lev] =  ((N[lev + 1] - 1)/2) + 1;
  }//end for lev

  Kmat = (Mat*)(malloc( sizeof(Mat)*nlevels ));
  Rmat = (Mat*)(malloc( sizeof(Mat)*(nlevels - 1) ));

  kData = (StiffnessData*)(malloc( sizeof(StiffnessData)*nlevels ));

  for(lev = 0; lev < nlevels; lev++) {
    numBonds = (2*(N[lev])*((N[lev]) - 1));
    (kData[lev]).matProp = (double*)(malloc( sizeof(double)*numBonds ));
  }//end for lev

  //Finest Grid 
  {
    lev = (nlevels - 1);
    numBonds = (2*(N[lev])*((N[lev]) - 1));
    for(i = 0; i < numBonds; i++) {
      ((kData[lev]).matProp)[i] = 1.0;
    }//end for i

    fin = fopen("wrst_0","r");
    fscanf(fin,"%d",&totalBrokenBonds);
    assert(numBrokenBonds <= totalBrokenBonds);

    for(i = 0; i < numBrokenBonds; i++) {
      fscanf(fin,"%d",&bondId);
      ((kData[lev]).matProp)[__MY_BOND_ID__( bondId, (N[lev]))] = 0;
    }//end for i

    fclose(fin);
  }

  //Coarser Grids
  for(lev = (nlevels - 2); lev >= 0; lev--) {

    Nc = N[lev];
    Nf = N[lev + 1];

    for( yci = 0; yci < (Nc - 1); yci++) {
      for( xci = 0; xci < (Nc - 1); xci++) {
        ((kData[lev]).matProp)[__I_BOND__(xci, yci, __TOP__, Nc)] = (0.5*(
            ((kData[lev + 1]).matProp)[__I_BOND__( (2*xci), ((2*yci) + 1), __TOP__, Nf)] +
            ((kData[lev + 1]).matProp)[__I_BOND__( ((2*xci) + 1), ((2*yci) + 1), __TOP__, Nf)] ));

        ((kData[lev]).matProp)[__I_BOND__(xci, yci, __RIGHT__, Nc)] = (0.5*(
            ((kData[lev + 1]).matProp)[__I_BOND__( ((2*xci) + 1), (2*yci), __RIGHT__, Nf)] +
            ((kData[lev + 1]).matProp)[__I_BOND__( ((2*xci) + 1), ((2*yci) + 1), __RIGHT__, Nf)] ));         
      }//end for xci
    }//end for yci

    {
      for( yci = 0; yci < (Nc - 1); yci++) {
        ((kData[lev]).matProp)[__L_BOND__(yci, Nc)] =  (0.5*(
            ((kData[lev + 1]).matProp)[__L_BOND__( (2*yci), Nf)] +
            ((kData[lev + 1]).matProp)[__L_BOND__( ((2*yci) + 1), Nf)] )); 
      }//end for yci
    }

    {
      for( xci = 0; xci < (Nc - 1); xci++) {
        ((kData[lev]).matProp)[__B_BOND__(xci, Nc)] = (0.5*(
            ((kData[lev + 1]).matProp)[__B_BOND__( (2*xci), Nf)] +
            ((kData[lev + 1]).matProp)[__B_BOND__( ((2*xci) + 1), Nf)] ));
      }//end for xci
    }

  }//end for lev

  for(lev = 1; lev < nlevels; lev++) {
    MatCreateShell(PETSC_COMM_WORLD, (N[lev]*N[lev]), (N[lev]*N[lev]), 
        PETSC_DETERMINE, PETSC_DETERMINE, (kData + lev), (Kmat + lev));
    MatShellSetOperation(Kmat[lev], MATOP_MULT, (void(*)(void)) stiffnessMatvec);
    MatShellSetOperation(Kmat[lev], MATOP_GET_DIAGONAL, (void(*)(void)) stiffnessGetDiagonal);
    MatShellSetOperation(Kmat[lev], MATOP_DESTROY, (void(*)(void)) stiffnessDestroy);

    rpData = (TransferOpData*)(malloc( sizeof(TransferOpData) ));
    rpData->addRtmp = NULL;
    rpData->addPtmp = NULL;
    MatCreateShell(PETSC_COMM_WORLD, (N[lev - 1]*N[lev - 1]), (N[lev]*N[lev]), 
        PETSC_DETERMINE, PETSC_DETERMINE, rpData, (Rmat + lev - 1));
    MatShellSetOperation(Rmat[lev - 1], MATOP_MULT_TRANSPOSE, (void(*)(void)) prolongMatvecType);
    MatShellSetOperation(Rmat[lev - 1], MATOP_MULT, (void(*)(void)) restrictMatvecType);
    MatShellSetOperation(Rmat[lev - 1], MATOP_MULT_ADD, (void(*)(void)) addRestrictMatvec);
    MatShellSetOperation(Rmat[lev - 1], MATOP_MULT_TRANSPOSE_ADD, (void(*)(void)) addProlongMatvec);
    MatShellSetOperation(Rmat[lev - 1], MATOP_DESTROY, (void(*)(void)) rpDestroy);
  }//end for lev

  MatCreateSeqAIJ(PETSC_COMM_WORLD, (N[0]*N[0]), (N[0]*N[0]), 5, PETSC_NULL, Kmat);
  setCoarseMat(Kmat[0], N[0], kData);

  //MatView(Kmat[0], 0);

  VecCreate(PETSC_COMM_WORLD, &rhs);
  VecSetSizes(rhs, (N[nlevels - 1]*N[nlevels - 1]), PETSC_DECIDE);
  VecSetType(rhs, VECSEQ);

  VecDuplicate(rhs, &sol);
  VecSetRandom(sol, PETSC_NULL);
  MatMult(Kmat[nlevels - 1], sol, rhs);
  VecZeroEntries(sol);

  KSPCreate(PETSC_COMM_WORLD, &ksp);
  KSPSetOperators(ksp, Kmat[nlevels - 1], Kmat[nlevels - 1], SAME_NONZERO_PATTERN);
  KSPSetType(ksp, KSPCG);

  comms = (MPI_Comm*)(malloc( sizeof(MPI_Comm)*nlevels ));
  for(lev = 0; lev < nlevels; lev++) {
    comms[lev] = PETSC_COMM_WORLD;
  }//end for lev

  KSPGetPC(ksp, &pc);
  PCSetType(pc, PCMG);
  PCMGSetLevels(pc, nlevels, comms);
  PCMGSetType(pc, PC_MG_MULTIPLICATIVE);

  //0 is the coarsest
  for( lev = 0; lev < nlevels; lev++) {
    PCMGGetSmoother(pc, lev, &lksp);
    KSPSetOperators(lksp, Kmat[lev], Kmat[lev], SAME_NONZERO_PATTERN);
  }//end for lev

  for( lev = 1; lev < nlevels; lev++) {
    PCMGSetInterpolation(pc, lev, Rmat[lev - 1]);
    PCMGSetRestriction(pc, lev, Rmat[lev - 1]);
  }//end for lev

  KSPSetFromOptions(ksp);
  KSPSetUp(ksp);

  KSPSolve(ksp, rhs, sol);

  KSPDestroy(ksp);

  VecDestroy(sol);
  VecDestroy(rhs);

  for( lev = 0; lev < nlevels; lev++) {
    MatDestroy(Kmat[lev]);
  }//end for lev

  for( lev = 0; lev < (nlevels - 1); lev++) {
    MatDestroy(Rmat[lev]);
  }//end for lev 

  for( lev = 0; lev < nlevels; lev++) {
    free((kData[lev]).matProp);
  }//end for lev
  free(kData);

  free(N);
  free(Kmat);
  free(Rmat);
  free(comms);

  PetscFinalize();

  return 1;

}




