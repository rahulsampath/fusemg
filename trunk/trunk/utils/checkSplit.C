
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cassert>

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

#define __H_BOND__(x, y, N) ( 1 + ((y)*((N) - 1)) + (x) )

#define __V_BOND__(x, y, N) ( (__VOFF__(N)) + ((x)*((N) - 1)) + (y) )

bool isBondOnBoundary(int bondNum, int N) {

  int keyL = __V_BOND__(0, 0, N);
  int keyR = __V_BOND__( (N - 1), 0, N);
  int keyT = __H_BOND__( 0, (N - 1), N);
  int keyB = __H_BOND__( 0, 0, N);

  if ( ( bondNum >= keyL ) &&
      ( bondNum < (keyL + (N - 1)) ) ) {
    return true;
  }

  if ( ( bondNum >= keyR ) &&
      ( bondNum < (keyR + (N - 1)) ) ) {
    return true;
  }

  if ( ( bondNum >= keyT ) &&
      ( bondNum < (keyT + (N - 1)) ) ) {
    return true;
  }

  if ( ( bondNum >= keyB ) &&
      ( bondNum < (keyB + (N - 1)) ) ) {
    return true;
  }

  return false;

}

bool anyBondsOnBoundary(const std::vector<int> & bonds, int N) {

  int keyL = __V_BOND__(0, 0, N);
  int keyR = __V_BOND__( (N - 1), 0, N);
  int keyT = __H_BOND__( 0, (N - 1), N);
  int keyB = __H_BOND__( 0, 0, N);

  std::vector<int>::iterator posL = std::lower_bound(bonds.begin(), bonds.end(), keyL); 
  std::vector<int>::iterator posR = std::lower_bound(bonds.begin(), bonds.end(), keyR); 
  std::vector<int>::iterator posT = std::lower_bound(bonds.begin(), bonds.end(), keyT); 
  std::vector<int>::iterator posB = std::lower_bound(bonds.begin(), bonds.end(), keyB); 

  if (posL != bonds.end()) {
    if( (*posL) == keyL ) {
      return true;
    } else {
      assert( (*posL) > keyL );
      if( (*posL) < (keyL + (N - 1)) ) {
        return true;
      }
    }
  }

  if (posR != bonds.end()) {
    if( (*posR) == keyR ) {
      return true;
    } else {
      assert( (*posR) > keyR );
      if( (*posR) < (keyR + (N - 1)) ) {
        return true;
      }
    }
  }

  if (posT != bonds.end()) {
    if( (*posT) == keyT ) {
      return true;
    } else {
      assert( (*posT) > keyT );
      if( (*posT) < (keyT + (N - 1)) ) {
        return true;
      }
    }
  }

  if (posB != bonds.end()) {
    if( (*posB) == keyB ) {
      return true;
    } else {
      assert( (*posB) > keyB );
      if( (*posB) < (keyB + (N - 1)) ) {
        return true;
      }
    }
  }

  return false;
}



int main(int argc, char** argv) {

  std::vector<int> brokenBonds;

  assert(argc == 3);

  //N is the number of nodes in each direction
  int N = atoi(argv[1]);

  FILE* fp = fopen(argv[2], "r");

  int numBrokenBonds;

  fscanf(fp,"%d",&numBrokenBonds);

  brokenBonds.resize(numBrokenBonds);

  for(int i = 0; i < numBrokenBonds; i++) {
    fscanf(fp, "%d", (&(brokenBonds[i])) );
  }//end for i

  std::sort(brokenBonds.begin(), brokenBonds.end());

  bool bdyFlag = anyBondsOnBoundary(brokenBonds, N);

  fclose(fp);

}




