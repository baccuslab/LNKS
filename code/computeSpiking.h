#include <stddef.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/*
 * SC1DF
 */
void SC1DF(double *p, double *v, double *dv, double *r, int N); 
void SC1DF_gain(double *p, double *v, double *dv, double *r, double *gain, int N); 
void SC1DF_get_h(double *p, double *v, double *dv, double *r, double *h, int N); 
void SC1DF_get_m(double *p, double *v, double *dv, double *r, double *h, int N); 

/*
 * SCIF
 */
void SCIF(double *p, double *v, double *dv, double *r, int N); 
void SCIF_gain(double *p, double *v, double *dv, double *r, double *gain, int N); 
void SCIF_get_h(double *p, double *v, double *dv, double *r, double *h, int N); 

/*
 * SCIF2
 */
void SCIF2(double *p, double *v, double *dv, double *r, int N); 
void SCIF2_gain(double *p, double *v, double *dv, double *r, double *gain, int N); 
void SCIF2_get_h(double *p, double *v, double *dv, double *r, double *h, int N); 
void SCIF2_constFB(double *p, double *v, double *dv, double *r, double *h, int N); 

/*
 * SCIEF
 */
void SCIEF(double *p, double *v, double *dv, double *r, int N); 
void SCIEF_gain(double *p, double *v, double *dv, double *r, double *gain, int N); 
void SCIEF_get_h(double *p, double *v, double *dv, double *r, double *h, int N); 


/*
 * SDF
 */
void SDF(double *p, double *v, double *r, int N);

/*
 * helper functions
 */
double sigmoid(double x);
double *createVector(int n); 
double *feedbackfilter(double *theta, int N); 


