#include <stdio.h>

// Previous version using BLAS

void K4S(double *p, double **X, double *u, int M, int N, double sampleTime); 

void K4S(double *p, double **X, double *u, int M, int N, double sampleTime) {
    double xi[M];
    double Q[] = { 0, 0, 0, 0,
                   0, 0, 0, 0,
                   0, 0, 0, 0,
                   0, 0, 0, 0 };
                /* Index
                 *  0  1  2  3
                 *  4  5  6  7
                 *  8  9 10 11
                 * 12 13 14 15
                 */

    double xo[] = {0, 0, 0, 0};
    int i;

    Q[0]  = -(p[0]*u[0]+p[6]);     Q[1]  =  0;        Q[2]  = p[2];               Q[3]  = 0; 
    Q[4]  =  (p[0]*u[0]+p[6]);     Q[5]  = -p[1];     Q[6]  = 0;                  Q[7]  = 0; 
    Q[8]  =     0;                 Q[9]  =  p[1];     Q[10] = -(p[2]+p[3]);       Q[11] = p[4]*u[0]+p[5]; 
    Q[12] =     0;                 Q[13] =  0;        Q[14] = p[3];               Q[15] = -(p[4]*u[0]+p[5]); 


    for (i=0; i<(N-1); i++) {


        Q[0]  = -(p[0]*u[i]+p[6]);
        Q[4]  =  (p[0]*u[i]+p[6]);
        Q[11] =  (p[4]*u[i]+p[5]); 
        Q[15] = -(p[4]*u[i]+p[5]); 
        xi[0] = X[0][i];
        xi[1] = X[1][i];
        xi[2] = X[2][i];
        xi[3] = X[3][i];


        xo[0] = (1 + sampleTime*Q[0]) * xi[0] + sampleTime*Q[2] * xi[2];
        xo[1] = sampleTime*Q[4] * xi[0] + (1 + sampleTime*Q[5]) * xi[1];
        xo[2] = sampleTime*Q[9] * xi[1] + (1 + sampleTime*Q[10]) * xi[2] + sampleTime*Q[11]*xi[3];
        xo[3] = sampleTime*Q[14] * xi[2] + (1 + sampleTime*Q[15]) *xi[3];

//        BLAS library
//        xo = matrixOperation(Q, xi, M);

        X[0][i+1] = xo[0];
        X[1][i+1] = xo[1];
        X[2][i+1] = xo[2];
        X[3][i+1] = xo[3];
    }
}

