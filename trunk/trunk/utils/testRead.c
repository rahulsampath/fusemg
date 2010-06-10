
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

int main(int argc, char** argv) {

  if(argc < 2) {
    printf("Usage: exe inpFilename N\n");
    return 1;
  }

  FILE* fp = fopen(argv[1],"r");
  int N = atoi(argv[2]);
  int numBonds = 3*N*N*(N - 1);
  int cnt, xi, yi, zi, rbt;
  double val;

  for(cnt = 0; cnt < numBonds; cnt++) {
    fscanf(fp, "%d", &xi);
    fscanf(fp, "%d", &yi);
    fscanf(fp, "%d", &zi);
    fscanf(fp, "%d", &rbt);
    fscanf(fp, "%lf", &val);
    assert(val > 0.0);
    printf("xi = %d, yi = %d, zi = %d, rbt = %d, val = %lf\n", xi, yi, zi, rbt, val);
  }

  fclose(fp);
  return 1;
}

