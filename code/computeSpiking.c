#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define lenP 10000

/*
 * SC1DF
 */
void SC1DF(double *p, double *v, double *dv, double *r, int N); 
void SC1DF_gain(double *p, double *v, double *dv, double *r, double *gain, int N); 
void SC1DF_get_h(double *p, double *v, double *dv, double *r, double *h, int N); 

/*
 * SCIF
 */
void SCIF(double *p, double *v, double *dv, double *r, int N); 
void SCIF_gain(double *p, double *v, double *dv, double *r, double *gain, int N); 
void SCIF_get_h(double *p, double *v, double *dv, double *r, double *h, int N); 
void SCIF_constFB(double *p, double *v, double *dv, double *r, double *h, int N); 


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
double *expfeedbackfilter(double *theta, int N); 
double *getfeedbackfilter(double *theta);
double *get_initial_h(double *v, int N); 




void SCIF(double *p, double *v, double *dv, double *r, int N) {
/* Spiking block returns the firing rate output given the input membrane potential.
 *
 * Input
 * -----
 *  p (double *array):
 *      The parameter array
 *
 *  v (double *array):
 *      The input of the Spiking block(membrane potential). This is the output of LNK model.
 *
 *  dv (double *array):
 *      The derivative of the input v of the Spiking block(derivative of the membrane potential).
 *  h (double *array):
 *  r (double *array):
 *  fb (double *array):
 *
 *  N (int size_t):
 *      The column size of the Kinetics block state matrix X, which represents the number of samples, or size of input u(uSize).
 *
 * Output
 * ------
 *  r (double *array):
 *      The spiking block output, firing rate.
 */
    double ptr[2] = {0.0, 0.0};
    ptr[0] = p[4];
    ptr[1] = p[5];
    double *h, *fb, *rfb;
    int i, k, l;

    h = createVector(N);
    fb = expfeedbackfilter(ptr, lenP); 
    rfb = createVector(lenP);

    for (i=0; i<N; i++) {
        h[i] = v[i];
    }

    for (i=0; i<N; i++) {

        r[i] = sigmoid(p[0] + p[1]*h[i]) * sigmoid(p[2] + p[3]*dv[i]);

        if (i < (N-lenP) ){

            for(k=0; k<lenP; k++) {
                rfb[k] = r[i]*fb[k];
                h[i+1+k] += rfb[k];
            }
        }
        else{
            l=0;
            for(k=i; k<N; k++) {
                rfb[l] = r[i]*fb[l];
                h[k+1] += rfb[l];
                ++l;
            }
        }
    }

    free(fb);
    free(rfb);
    free(h);

}

void SCIF_gain(double *p, double *v, double *dv, double *r, double *gain, int N) {

    double ptr[2] = {0.0, 0.0};
    ptr[0] = p[4];
    ptr[1] = p[5];
    double *h, *fb, *rfb;
    int i, k, l;

    h = createVector(N);
    fb = expfeedbackfilter(ptr, lenP); 
    rfb = createVector(lenP);

    for (i=0; i<N; i++) {
        h[i] = v[i];
    }

    for (i=0; i<N; i++) {

        r[i] = sigmoid(p[0] + p[1]*h[i]) * sigmoid(p[2] + p[3]*dv[i]);

        if (i < (N-lenP) ){

            for(k=0; k<lenP; k++) {
                rfb[k] = r[i]*fb[k];
                h[i+1+k] += rfb[k];
            }
        }
        else{
            l=0;
            for(k=i; k<N; k++) {
                rfb[l] = r[i]*fb[l];
                h[k+1] += rfb[l];
                ++l;
            }
        }

        //gain[i] = r[i] * ( (1 - sigmoid(p[0] + p[1]*h[i])) * p[1] + (1 - sigmoid(p[2] + p[3]*dv[i])) * p[3] * cg );
        gain[i] = r[i] * (1 - sigmoid(p[0] + p[1]*h[i])) * p[1];

    }

    free(fb);
    free(rfb);
    free(h);

}

void SCIF_get_h(double *p, double *v, double *dv, double *r, double *h, int N) {

    double ptr[2] = {0.0, 0.0};
    ptr[0] = p[4];
    ptr[1] = p[5];
    double *fb, *rfb;
    int i, k, l;

    fb = expfeedbackfilter(ptr, lenP); 
    rfb = createVector(lenP);

    for (i=0; i<N; i++) {
        h[i] = v[i];
    }

    for (i=0; i<N; i++) {

        r[i] = sigmoid(p[0] + p[1]*h[i]) * sigmoid(p[2] + p[3]*dv[i]);

        if (i < (N-lenP) ){

            for(k=0; k<lenP; k++) {
                rfb[k] = r[i]*fb[k];
                h[i+1+k] += rfb[k];
            }
        }
        else{
            l=0;
            for(k=i; k<N; k++) {
                rfb[l] = r[i]*fb[l];
                h[k+1] += rfb[l];
                ++l;
            }
        }
    }

    free(fb);
    free(rfb);
}

/*
 * Spiking Continuous Independent Feedback 2 (SCIF2)
 */
void SCIF2(double *p, double *v, double *dv, double *r, int N) {
/* Spiking block returns the firing rate output given the input membrane potential.
 *
 * Input
 * -----
 *  p (double *array):
 *      The parameter array
 *
 *  v (double *array):
 *      The input of the Spiking block(membrane potential). This is the output of LNK model.
 *
 *  dv (double *array):
 *      The derivative of the input v of the Spiking block(derivative of the membrane potential).
 *  h (double *array):
 *  r (double *array):
 *  fb (double *array):
 *
 *  N (int size_t):
 *      The column size of the Kinetics block state matrix X, which represents the number of samples, or size of input u(uSize).
 *
 * Output
 * ------
 *  r (double *array):
 *      The spiking block output, firing rate.
 */
    double ptr1[2] = {0.0, 0.0};
    double ptr2[2] = {0.0, 0.0};
    ptr1[0] = p[4];
    ptr1[1] = p[5];
    ptr2[0] = p[6];
    ptr2[1] = p[7];
    double *h, *fb, *fb1, *fb2, *rfb;
    int i, k, l;

    h = createVector(N);
    fb = createVector(lenP);
    fb1 = expfeedbackfilter(ptr1, lenP); 
    fb2 = expfeedbackfilter(ptr2, lenP); 
    for (i=0; i<lenP; i++) {
        fb[i] = (fb1[i] + fb2[i])/2;
    }
    rfb = createVector(lenP);

    for (i=0; i<N; i++) {
        h[i] = v[i];
    }

    for (i=0; i<N; i++) {

        r[i] = sigmoid(p[0] + p[1]*h[i]) * sigmoid(p[2] + p[3]*dv[i]);

        if (i < (N-lenP) ){

            for(k=0; k<lenP; k++) {
                rfb[k] = r[i]*fb[k];
                h[i+1+k] += rfb[k];
            }
        }
        else{
            l=0;
            for(k=i; k<N; k++) {
                rfb[l] = r[i]*fb[l];
                h[k+1] += rfb[l];
                ++l;
            }
        }
    }

    free(fb);
    free(rfb);
    free(h);

}

void SCIF2_gain(double *p, double *v, double *dv, double *r, double *gain, int N) {

    double ptr1[2] = {0.0, 0.0};
    double ptr2[2] = {0.0, 0.0};
    ptr1[0] = p[4];
    ptr1[1] = p[5];
    ptr2[0] = p[6];
    ptr2[1] = p[7];
    double *h, *fb, *fb1, *fb2, *rfb;
    int i, k, l;

    h = createVector(N);
    fb = createVector(lenP);
    fb1 = expfeedbackfilter(ptr1, lenP); 
    fb2 = expfeedbackfilter(ptr2, lenP); 
    for (i=0; i<lenP; i++) {
        fb[i] = (fb1[i] + fb2[i])/2;
    }
    rfb = createVector(lenP);

    for (i=0; i<N; i++) {
        h[i] = v[i];
    }

    for (i=0; i<N; i++) {

        r[i] = sigmoid(p[0] + p[1]*h[i]) * sigmoid(p[2] + p[3]*dv[i]);

        if (i < (N-lenP) ){

            for(k=0; k<lenP; k++) {
                rfb[k] = r[i]*fb[k];
                h[i+1+k] += rfb[k];
            }
        }
        else{
            l=0;
            for(k=i; k<N; k++) {
                rfb[l] = r[i]*fb[l];
                h[k+1] += rfb[l];
                ++l;
            }
        }

        //gain[i] = r[i] * ( (1 - sigmoid(p[0] + p[1]*h[i])) * p[1] + (1 - sigmoid(p[2] + p[3]*dv[i])) * p[3] * cg );
        gain[i] = r[i] * (1 - sigmoid(p[0] + p[1]*h[i])) * p[1];

    }

    free(fb);
    free(rfb);
    free(h);

}

void SCIF2_get_h(double *p, double *v, double *dv, double *r, double *h, int N) {

    double ptr1[2] = {0.0, 0.0};
    double ptr2[2] = {0.0, 0.0};
    ptr1[0] = p[4];
    ptr1[1] = p[5];
    ptr2[0] = p[6];
    ptr2[1] = p[7];
    double *fb, *fb1, *fb2, *rfb;
    int i, k, l;

    fb = createVector(lenP);
    fb1 = expfeedbackfilter(ptr1, lenP); 
    fb2 = expfeedbackfilter(ptr2, lenP); 
    for (i=0; i<lenP; i++) {
        fb[i] = (fb1[i] + fb2[i])/2;
    }
    rfb = createVector(lenP);


    for (i=0; i<N; i++) {
        h[i] = v[i];
    }

    for (i=0; i<N; i++) {

        r[i] = sigmoid(p[0] + p[1]*h[i]) * sigmoid(p[2] + p[3]*dv[i]);

        if (i < (N-lenP) ){

            for(k=0; k<lenP; k++) {
                rfb[k] = r[i]*fb[k];
                h[i+1+k] += rfb[k];
            }
        }
        else{
            l=0;
            for(k=i; k<N; k++) {
                rfb[l] = r[i]*fb[l];
                h[k+1] += rfb[l];
                ++l;
            }
        }
    }

    free(fb);
    free(rfb);
}



/*
 * Spiking Continuous Independent Ellipsoid Feedback(SCIEF)
 */
void SCIEF(double *p, double *v, double *dv, double *r, int N) {
/* Spiking block returns the firing rate output given the input membrane potential.
 *
 * Input
 * -----
 *  p (double *array):
 *      The parameter array
 *
 *  v (double *array):
 *      The input of the Spiking block(membrane potential). This is the output of LNK model.
 *
 *  dv (double *array):
 *      The derivative of the input v of the Spiking block(derivative of the membrane potential).
 *  h (double *array):
 *  r (double *array):
 *  fb (double *array):
 *
 *  N (int size_t):
 *      The column size of the Kinetics block state matrix X, which represents the number of samples, or size of input u(uSize).
 *
 * Output
 * ------
 *  r (double *array):
 *      The spiking block output, firing rate.
 */
    double ptr1[2] = {0.0, 0.0};
    double ptr2[2] = {0.0, 0.0};
    ptr1[0] = p[10];
    ptr1[1] = p[11];
    ptr2[0] = p[12];
    ptr2[1] = p[13];
    double *h, *fb, *fb1, *fb2, *rfb, r1, r2, r3;
    int i, k, l;

    h = createVector(N);
    fb = createVector(lenP);
    fb1 = expfeedbackfilter(ptr1, lenP); 
    fb2 = expfeedbackfilter(ptr2, lenP); 
    for (i=0; i<lenP; i++) {
        fb[i] = (fb1[i] + fb2[i])/2;
    }
    rfb = createVector(lenP);

    for (i=0; i<N; i++) {
        h[i] = v[i];
    }

    for (i=0; i<N; i++) {

        r1 = sigmoid(p[0] + p[1]*h[i]);
        r2 = sigmoid(p[2] + p[3]*dv[i]);
        r3 = sigmoid(p[4] + p[5]*h[i] + p[6]*dv[i] + p[7]*h[i]*dv[i] + p[8]*pow(h[i], 2) + p[9]*pow(dv[i], 2));
        r[i] =  r1 * r2 * r3;

        if (i < (N-lenP) ){

            for(k=0; k<lenP; k++) {
                rfb[k] = r[i]*fb[k];
                h[i+1+k] += rfb[k];
            }
        }
        else{
            l=0;
            for(k=i; k<N; k++) {
                rfb[l] = r[i]*fb[l];
                h[k+1] += rfb[l];
                ++l;
            }
        }
    }

    free(fb);
    free(rfb);
    free(h);

}


void SCIEF_gain(double *p, double *v, double *dv, double *r, double *gain, int N) {

    double ptr1[2] = {0.0, 0.0};
    double ptr2[2] = {0.0, 0.0};
    ptr1[0] = p[4];
    ptr1[1] = p[5];
    ptr2[0] = p[6];
    ptr2[1] = p[7];
    double *h, *fb, *fb1, *fb2, *rfb, r1, r2, r3;
    double wx, wz;
    int i, k, l;

    h = createVector(N);
    fb = createVector(lenP);
    fb1 = expfeedbackfilter(ptr1, lenP); 
    fb2 = expfeedbackfilter(ptr2, lenP); 
    for (i=0; i<lenP; i++) {
        fb[i] = (fb1[i] + fb2[i])/2;
    }
    rfb = createVector(lenP);

    for (i=0; i<N; i++) {
        h[i] = v[i];
    }

    for (i=0; i<N; i++) {

        r1 = sigmoid(p[0] + p[1]*h[i]);
        r2 = sigmoid(p[2] + p[3]*dv[i]);
        r3 = sigmoid(p[4] + p[5]*h[i] + p[6]*dv[i] + p[7]*h[i]*dv[i] + p[8]*pow(h[i], 2) + p[9]*pow(dv[i], 2));
        r[i] =  r1 * r2 * r3;

        if (i < (N-lenP) ){

            for(k=0; k<lenP; k++) {
                rfb[k] = r[i]*fb[k];
                h[i+1+k] += rfb[k];
            }
        }
        else{
            l=0;
            for(k=i; k<N; k++) {
                rfb[l] = r[i]*fb[l];
                h[k+1] += rfb[l];
                ++l;
            }
        }

        wx = 1 - r1;
        wz = 1 - r3;

        gain[i] = r[i] * (p[1] * wx + (p[5] + p[7]*dv[i] + 2*p[8]*h[i]) * wz);

    }

    free(fb);
    free(rfb);
    free(h);

}

void SCIEF_get_h(double *p, double *v, double *dv, double *r, double *h, int N) {

    double ptr1[2] = {0.0, 0.0};
    double ptr2[2] = {0.0, 0.0};
    ptr1[0] = p[4];
    ptr1[1] = p[5];
    ptr2[0] = p[6];
    ptr2[1] = p[7];
    double *fb, *fb1, *fb2, *rfb, r1, r2, r3;
    int i, k, l;

    fb = createVector(lenP);
    fb1 = expfeedbackfilter(ptr1, lenP); 
    fb2 = expfeedbackfilter(ptr2, lenP); 
    for (i=0; i<lenP; i++) {
        fb[i] = (fb1[i] + fb2[i])/2;
    }
    rfb = createVector(lenP);


    for (i=0; i<N; i++) {
        h[i] = v[i];
    }

    for (i=0; i<N; i++) {

        r1 = sigmoid(p[0] + p[1]*h[i]);
        r2 = sigmoid(p[2] + p[3]*dv[i]);
        r3 = sigmoid(p[4] + p[5]*h[i] + p[6]*dv[i] + p[7]*h[i]*dv[i] + p[8]*pow(h[i], 2) + p[9]*pow(dv[i], 2));
        r[i] =  r1 * r2 * r3;

        if (i < (N-lenP) ){

            for(k=0; k<lenP; k++) {
                rfb[k] = r[i]*fb[k];
                h[i+1+k] += rfb[k];
            }
        }
        else{
            l=0;
            for(k=i; k<N; k++) {
                rfb[l] = r[i]*fb[l];
                h[k+1] += rfb[l];
                ++l;
            }
        }
    }

    free(fb);
    free(rfb);
}



void SDF(double *p, double *v, double *r, int N){
/* Spiking Discrete Feedback
 * spiking block returns the spike response output given the input membrane potential.
 *
 * Input
 * -----
 *  p (double *array):
 *      The parameter array
 *
 *  v (double *array):
 *      The input of the Spiking block(membrane potential). This is the output of LNK model.
 *
 *  N (int size_t):
 *      The column size of the Kinetics block state matrix X, which represents the number of samples, or size of input u(uSize).
 *
 * Output
 * ------
 *  r (double *array):
 *      The spiking block output, firing rate.
 */
    double *g, *fb, *fb1, *fb2, thr;
    double ptr1[2] = {0.0, 0.0};
    double ptr2[2] = {0.0, 0.0};

    int i, k, l;
    int len_fb = 5000;

    thr = p[0];
    ptr1[0] = p[1];
    ptr1[1] = p[2];
    ptr2[0] = p[3];
    ptr2[1] = p[4];

    // feedback filter
    fb = createVector(len_fb);
    fb1 = expfeedbackfilter(ptr1, len_fb); 
    fb2 = expfeedbackfilter(ptr2, len_fb); 
    for (i=0; i<len_fb; i++) {
        fb[i] = (fb1[i] + fb2[i])/2;
    }


    g = createVector(N);
    //for (i=0; i<N; i++) {
    //    g[i] = v[i];
    //}

    g[0] = v[0];
    for (i=1; i<N; i++) {

        g[i] += v[i];

        if ((g[i] > thr) && (g[i] > g[i-1])) {
            r[i] = 1;

            if (i < (N-len_fb) ){

                for(k=0; k<len_fb; k++) {
                    g[i+1+k] += fb[k];
                }
            }
            else{
                if (~(i == N-1)) {
                    l=0;
                    for(k=i; k<N; k++) {
                        g[k+1] += fb[l];
                        ++l;
                    }
                }
            }
        }
    }

    free(fb);
    free(fb1);
    free(fb2);
    free(g);


}


/*
 * SC1DF
 */
void SC1DF(double *p, double *v, double *dv, double *r, int N) {
/* Spiking block returns the firing rate output given the input membrane potential.
 *
 * Input
 * -----
 *  p (double *array):
 *      The parameter array
 *
 *  v (double *array):
 *      The input of the Spiking block(membrane potential). This is the output of LNK model.
 *
 *  dv (double *array):
 *      The derivative of the input v of the Spiking block(derivative of the membrane potential).
 *  h (double *array):
 *  r (double *array):
 *  fb (double *array):
 *
 *  N (int size_t):
 *      The column size of the Kinetics block state matrix X, which represents the number of samples, or size of input u(uSize).
 *
 * Output
 * ------
 *  r (double *array):
 *      The spiking block output, firing rate.
 */

    double *h, *fb, *rfb;
    int i, k, l;

    fb = getfeedbackfilter(p);
    rfb = createVector(lenP);
    h = get_initial_h(v, N);

    for (i=0; i<N; i++) {

        r[i] = sigmoid(p[0] + p[1]*h[i] + p[2]*dv[i]);

        if (i < (N-lenP) ){

            for(k=0; k<lenP; k++) {
                rfb[k] = r[i]*fb[k];
                h[i+1+k] += rfb[k];
            }
        }
        else{
            l=0;
            for(k=i; k<N; k++) {
                rfb[l] = r[i]*fb[l];
                h[k+1] += rfb[l];
                ++l;
            }
        }
    }

    free(fb);
    free(rfb);
    free(h);

}

void SC1DF_gain(double *p, double *v, double *dv, double *r, double *gain, int N) {

    double *h, *fb, *rfb;
    int i, k, l;

    fb = getfeedbackfilter(p);
    rfb = createVector(lenP);
    h = get_initial_h(v, N);

    for (i=0; i<N; i++) {

        r[i] = sigmoid(p[0] + p[1]*h[i] + p[2]*dv[i]);

        if (i < (N-lenP) ){

            for(k=0; k<lenP; k++) {
                rfb[k] = r[i]*fb[k];
                h[i+1+k] += rfb[k];
            }
        }
        else{
            l=0;
            for(k=i; k<N; k++) {
                rfb[l] = r[i]*fb[l];
                h[k+1] += rfb[l];
                ++l;
            }
        }

        gain[i] = r[i] * (1 - r[i]) * p[1];

    }

    free(fb);
    free(rfb);
    free(h);

}

void SC1DF_get_h(double *p, double *v, double *dv, double *r, double *h, int N) {

    double *fb, *rfb;
    int i, k, l;

    fb = getfeedbackfilter(p);
    rfb = createVector(lenP);
    h = get_initial_h(v, N);

    for (i=0; i<N; i++) {

        r[i] = sigmoid(p[0] + p[1]*h[i] + p[2]*dv[i]);

        if (i < (N-lenP) ){

            for(k=0; k<lenP; k++) {
                rfb[k] = r[i]*fb[k];
                h[i+1+k] += rfb[k];
            }
        }
        else{
            l=0;
            for(k=i; k<N; k++) {
                rfb[l] = r[i]*fb[l];
                h[k+1] += rfb[l];
                ++l;
            }
        }
    }

    free(fb);
    free(rfb);
}



void SC1DF_get_m(double *p, double *v, double *dv, double *r, double *m, int N) {

    double *h, *fb, *rfb;
    int i, k, l;

    fb = getfeedbackfilter(p);
    rfb = createVector(lenP);
    h = get_initial_h(v, N);

    for (i=0; i<N; i++) {

        m[i] = p[1]*h[i] + p[2]*dv[i];
        r[i] = sigmoid(p[0] + p[1]*h[i] + p[2]*dv[i]);

        if (i < (N-lenP) ){

            for(k=0; k<lenP; k++) {
                rfb[k] = r[i]*fb[k];
                h[i+1+k] += rfb[k];
            }
        }
        else{
            l=0;
            for(k=i; k<N; k++) {
                rfb[l] = r[i]*fb[l];
                h[k+1] += rfb[l];
                ++l;
            }
        }
    }

    free(h);
    free(fb);
    free(rfb);
}


double sigmoid(double x){

    double y;

    y = 1 / (1 + exp(-x));

    return y;
}

double *createVector(int n) {

    double *array;
    int i;
    array = malloc(n * sizeof(double));
    for (i = 0; i < n; i++) {
        array[i] = 0.0;
    }

    return array;

}

double *expfeedbackfilter(double *theta, int N) {

    double *fb;
    int i;
    fb = createVector(N);

    for (i=0; i<N; i++) {

        fb[i] = - theta[0] * exp(- i / theta[1]);

    }

    return fb;
}

double *getfeedbackfilter(double *theta) {

    double ptr1[2] = {0.0, 0.0};
    double ptr2[2] = {0.0, 0.0};
    double ptr3[2] = {0.0, 0.0};
    ptr1[0] = theta[3];
    ptr1[1] = theta[4];
    ptr2[0] = theta[5];
    ptr2[1] = theta[6];
    ptr3[0] = theta[7];
    ptr3[1] = theta[8];
    double *fb, *fb1, *fb2, *fb3;

    fb = createVector(lenP);
    fb1 = expfeedbackfilter(ptr1, lenP); 
    fb2 = expfeedbackfilter(ptr2, lenP); 
    fb3 = expfeedbackfilter(ptr3, lenP); 

    for (int i=0; i<lenP; i++) {
        fb[i] = (fb1[i] + fb2[i] + fb3[i])/3;
    }

    return fb;

}

double *get_initial_h(double *v, int N) {

    double *h;
    h = createVector(N);

    for (int i=0; i<N; i++) {
        h[i] = v[i];
    }

    return h;

}
