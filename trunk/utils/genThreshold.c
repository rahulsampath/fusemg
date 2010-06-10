#include <stdio.h>
#include <math.h>
#include <stdlib.h>

FILE  *fout,*fnode,*felem,*fprob,*fout_map;
int main(int argc, char **argv)
{
    long iseed;
    double bcc, bcr;
    int node,inode,jnode,elem,refnode,knode;
    double x,y,z;
    int elnum,mtype,iflag;
    int ibcc, ibcr,ndbcmin,numdc;
    int topfirstnode;
    double fload,thk;
    double refdisp,refload;
    int i,j,k,num_nodes = 0,lat_size,latx_size, laty_size, latz_size;
    int numx_rows,numy_rows,numz_rows;
    double xmax,ymax,zmax,xref,yref,zref;
    int R,B,T,Pointer,xi,yi,zi;
    int xperiod = 1,npxcols = 1;     /** periodicity in X **/
    int yperiod = 1,npycols = 1;     /** periodicity in Y **/

    if(argc < 2)
    {
        printf("Usage: latx_size laty_size latz_size xref yref zref iseed filename mapped_filename periodicity\n(Currently put 2 2 for periodicity which means non-periodic)\n");
        return 1;
    }

    lat_size = latx_size = atoi(argv[1]);
    laty_size = atoi(argv[2]);
    latz_size = atoi(argv[3]);
    xmax = lat_size;
    ymax = lat_size;
    zmax = lat_size;
    xref = atof(argv[4]);
    yref = atof(argv[5]);
    zref = atof(argv[6]);
    iseed = atol(argv[7]);
    srand48(iseed);
    fout = fopen(argv[8],"w");
    fout_map = fopen(argv[9],"w");
    if(argc > 10) 
    {
        xperiod = atoi(argv[10]);   /** period != 2 then periodicity **/
        yperiod = atoi(argv[11]);   /** period != 2 then periodicity **/
    }

    if(xperiod == 2) npxcols = 0;    /** No periodicity **/
    if(yperiod == 2) npycols = 0;    /** No periodicity **/

    numx_rows = latx_size + 1 + npxcols;
    numy_rows = laty_size + 1 + npycols;
    numx_rows = latx_size + 1;
    numy_rows = laty_size + 1;
    numz_rows = latz_size + 1;
    bcc = 0.0;
    bcr = 7.0;

/*  Generate Nodes */

    for(k=0;k<numz_rows;k++)
    {
        for(j=0;j<numy_rows;j++)
        {
            for(i=0;i<numx_rows;i++)
            {
                x = i;
                y = j;
                z = k;
                bcc = 4.0;
                if(k==0) bcc = 7.0;
                bcr = 7.0;

                num_nodes++;
                if(k==numz_rows-1 && j==0 && i==0) topfirstnode = num_nodes;
                fprintf(fout,"%10d%5.1lf%20lf%20lf%20lf%5.1lf\n",num_nodes,bcc, x,y,z,bcr);
            }
        }
    }
    refnode = num_nodes + 1;
    fprintf(fout,"%10d%5.1lf%20lf%20lf%20lf%5.1lf\n",refnode,bcc, xref,yref,zref,bcr);

/* Generate Element Connectivity */

    elnum = 0;
    mtype = 1;
    thk = 0.0;
    iflag = 1;
    for(k=0; k<numz_rows; k++)
    {
        knode = k*numx_rows*numy_rows;
        for(j=0;j<laty_size+1;j++)
        {
            for(i=0;i<numx_rows-1;i++)
            {
                elnum++;
                inode = knode + j*numx_rows + i + 1;
                jnode = knode + j*numx_rows + i + 2;
                fload = drand48();
                fprintf(fout,"%10d%5d%10d%10d%10d%8.4lf%8.4lf%8.4lf%8.4lf%2d%15.8lE\n", \
                              elnum,mtype,inode,jnode,refnode,thk,thk,thk,thk,iflag,fload);
                xi = i;
                yi = j;
                zi = k;
                R = 1;
                B = 0;
                T = 0;
                Pointer = 0;
                fprintf(fout_map,"%d %d %d %d %lE\n",xi,yi,zi,Pointer,fload);
                
            }
        }

        for(i=0;i<latx_size+1;i++)
        {
            for(j=0;j<numy_rows-1;j++)
            {
                elnum++;
                inode = knode + j*numx_rows + i + 1;
                jnode = knode + (j+1)*numx_rows + i + 1;
                fload = drand48();
                fprintf(fout,"%10d%5d%10d%10d%10d%8.4lf%8.4lf%8.4lf%8.4lf%2d%15.8lE\n", \
                              elnum,mtype,inode,jnode,refnode,thk,thk,thk,thk,iflag,fload);
                xi = i;
                yi = j;
                zi = k;
                R = 0;
                B = 1;
                T = 0;
                Pointer = 1;
                fprintf(fout_map,"%d %d %d %d %lE\n",xi,yi,zi,Pointer,fload);
            }
        }
    }

    for(j=0;j<laty_size+1;j++)
    {
        for(i=0;i<latx_size+1;i++)
        {
            for(k=0;k<latz_size;k++)
            {
                elnum++;
                inode = k*numx_rows*numy_rows + j*numx_rows + i + 1;
                jnode = (k+1)*numx_rows*numy_rows + j*numx_rows + i + 1;
                fload = drand48();
                fprintf(fout,"%10d%5d%10d%10d%10d%8.4lf%8.4lf%8.4lf%8.4lf%2d%15.8lE\n", \
                              elnum,mtype,inode,jnode,refnode,thk,thk,thk,thk,iflag,fload);
                xi = i;
                yi = j;
                zi = k;
                R = 0;
                B = 0;
                T = 1;
                Pointer = 2;
                fprintf(fout_map,"%d %d %d %d %lE\n",xi,yi,zi,Pointer,fload);
            }
        }
    }
    fclose(fout_map);
    printf("elnum = %d\n",elnum);

/*  Generate Nodal Constraint Cards  */

    if ((xperiod == 1) || (yperiod == 1))
    {
    for(k=1; k<numz_rows; k++)
    {
        knode = k*numx_rows*numy_rows;
        ibcc = 3;
        ibcr = 0;
        if (xperiod == 1)
        {
        for(j=0; j<laty_size+1; j++)
        {
            inode = knode + j*numx_rows + 1;
            jnode = knode + (j+1)*numx_rows;
            fprintf(fout,"%10d%10d%5d%5d\n", \
                              inode,jnode,ibcc,ibcr);
        }
        }

        ibcr = 0;
        if (yperiod == 1)
        {
        for(i=0; i<latx_size+1; i++)
        {
            inode = knode + i + 1;
            jnode = knode + (laty_size+1)*numx_rows + i + 1;
            fprintf(fout,"%10d%10d%5d%5d\n", \
                              inode,jnode,ibcc,ibcr);
        }
        }
    }
    }

/*  Generate Loading Curves  */

    ibcc = 1;
    ibcr = 2;
    fprintf(fout,"%5d%5d\n",ibcc,ibcr);
    refdisp = 0.0;
    refload = 0.0;
    fprintf(fout,"%10.5lf%10.5lf\n",refdisp,refload);
    refdisp = 1.0;
    refload = 1.0;
    fprintf(fout,"%10.5lf%10.5lf\n",refdisp,refload);

/*  Generate Displacement Loading Cards  */

    ndbcmin=0;
    numdc=0;
    ibcc = 3;
    ibcr = 1;
    refload = 1.0;
    knode = latz_size*numx_rows*numy_rows;
    for(j=0;j<laty_size+1;j++)
    {
        for(i=0;i<latx_size+1;i++)
        {
            numdc++;
	    inode = knode + j*numx_rows + i + 1;
            fprintf(fout,"%10d%5d%5d%10.5lf\n",inode,ibcc,ibcr,refload);
            if(ndbcmin==0) ndbcmin=inode;
        }
    }
    fprob=fopen("prob_0","w");
    fprintf(fprob,"%10d%10d%10d%10d%10d\n%10d%10d%10d%10d%10d%10d\n",refnode,elnum,ndbcmin,50,10,1,0,1,2,numdc,1);
    fclose(fprob);
}



