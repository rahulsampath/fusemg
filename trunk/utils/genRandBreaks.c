
#include "stdio.h"
#include "stdlib.h"

int main(int argc, char** argv) {

  int xi, yi, zi;
  int N = atoi(argv[1]);
  int numBroken = atoi(argv[2]);
  int seed = atoi(argv[3]);
  FILE* fpTmp = NULL; 
  FILE* fp = NULL;
  int count = 0;
  int i;

  srand(seed);

  fpTmp = fopen("tmpBrokenBonds.txt","w");
  for(zi = 1; (count < numBroken) && (zi < (N - 1)); zi++) {
    for(yi = 1; (count < numBroken) && (yi < (N - 1)); yi++) {
      for(xi = 1; (count < numBroken) && (xi < (N - 1)); xi++) {
        double prob = ((double)(rand()))/((double)(RAND_MAX));
        if( prob > 0.5 ) {
          double probX = ((double)(rand()))/((double)(RAND_MAX));
          if( probX > 0.5 ) {
            int tmp = 0;
            fprintf(fpTmp, "%d %d %d %d\n", xi, yi, zi, tmp);
          } else {
            double probY = ((double)(rand()))/((double)(RAND_MAX));
            if( probY > 0.5 ) {
              int tmp = 1;
              fprintf(fpTmp, "%d %d %d %d\n", xi, yi, zi, tmp);
            } else {
              int tmp = 2;
              fprintf(fpTmp, "%d %d %d %d\n", xi, yi, zi, tmp);
            }
          }
          count++;
        }
      }//end for xi
    }//end for yi
  }//end for zi
  fclose(fpTmp);

  printf("count = %d\n",count);

  fpTmp = fopen("tmpBrokenBonds.txt","r");
  fp = fopen("brokenBonds.txt","w");

  fprintf(fp," %d \n", count);
  for(i = 0; i < count; i++) {
    int txi, tyi, tzi, rbt;
    fscanf(fpTmp, "%d", &txi);
    fscanf(fpTmp, "%d", &tyi);
    fscanf(fpTmp, "%d", &tzi);
    fscanf(fpTmp, "%d", &rbt);
    fprintf(fp, "%d %d %d %d\n", txi, tyi, tzi, rbt);
  }//end for i

  fclose(fpTmp);
  fclose(fp);

  return 1;
}

