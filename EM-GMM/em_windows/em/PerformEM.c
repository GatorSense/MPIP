// Expectation Maximization algorithm for image segmentation into 2 clusters.

#include "PerformEM.h"
#include "explookup.h"

// Defaulting to Intel AVX 2 and FMA
#include <immintrin.h>
#include <pthread.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#define TSTART      (*inf).nstart
#define TEND        (*inf).nend
#define TID         (*inf).threadid

#ifdef __GNUC__
 #define max(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })
#endif

#define     RANDOM     ((0.0)+((double)((double)rand()/(double)RAND_MAX))*(1.0))

#ifdef __GNUC__
#define VEC(x)    (x)
#else
#define VEC(x)    (x).m256d_f64
#endif

// Computation of exponential
// --------------------------
// Compare if the values exceed limits
// If so, then set the exponential to zero.
// Else preform lookups.
#define EXPCOMPUTE(ymm) \
x = _mm256_set1_pd(-745.0); \
u = _mm256_cmp_pd(ymm, x, _CMP_GE_OQ); \
ymm = _mm256_and_pd(u, ymm); \
x = _mm256_set1_pd(10.0); \
x = _mm256_mul_pd(ymm, x); \
x = _mm256_floor_pd(x); \
y = _mm256_set1_pd(10.0); \
y = _mm256_sub_pd(ymm, _mm256_div_pd(x, y)); \
z = _mm256_set1_pd(-7450.0); \
z = _mm256_sub_pd(x, z); \
x = _mm256_set_pd(explookuphi[(int)VEC(z)[3]], explookuphi[(int)VEC(z)[2]], explookuphi[(int)VEC(z)[1]], explookuphi[(int)VEC(z)[0]]); \
z = _mm256_set1_pd(100000); \
s = _mm256_mul_pd(y, z); \
s = _mm256_floor_pd(s); \
y = _mm256_sub_pd(y, _mm256_div_pd(s, z)); \
s = _mm256_set_pd(explookupmid[(int)VEC(s)[3]], explookupmid[(int)VEC(s)[2]], explookupmid[(int)VEC(s)[1]], explookupmid[(int)VEC(s)[0]]); \
z = _mm256_set1_pd(1000000000); \
q = _mm256_mul_pd(y, z); \
q = _mm256_floor_pd(q); \
q = _mm256_set_pd(explookuplow[(int)VEC(q)[3]], explookuplow[(int)VEC(q)[2]], explookuplow[(int)VEC(q)[1]], explookuplow[(int)VEC(q)[0]]); \
ymm = _mm256_mul_pd(x, s); \
ymm = _mm256_mul_pd(ymm, q); \
ymm = _mm256_and_pd(u, ymm);

#define UPDATE_PROBABILITIES_AVX    \
s = _mm256_mul_pd(s, g); \
s = _mm256_add_pd(s, t); \
z = _mm256_set1_pd(lpis); \
r = _mm256_add_pd(s, z); \
    \
if (k == 1) \
{ \
    p = _mm256_load_pd(&dist[0][i]); \
    q = _mm256_max_pd(p, r); \
    p = _mm256_sub_pd(p, q); \
    r = _mm256_sub_pd(r, q); \
    \
    EXPCOMPUTE(p); \
    EXPCOMPUTE(r); \
    \
    q = _mm256_add_pd(p, r); \
    p = _mm256_div_pd(p, q); \
    r = _mm256_div_pd(r, q); \
    \
    _mm256_store_pd(&dist[0][i], p); \
    _mm256_store_pd(&dist[1][i], r); \
    \
    p = _mm256_load_pd(&N0[0]); \
    p = _mm256_add_pd(p, r); \
    \
    _mm256_store_pd(&N0[0], p); \
} \
else \
_mm256_store_pd(&dist[k][i], r);

#define UPDATE_PROBABILITIES    \
af *= df; \
af += logdet[k]; \
dist[k][i + m] = af + log(pis[k]); \
    \
if (k == 1) \
{ \
    af = max(dist[k][i + m], dist[0][i + m]); \
    dist[k][i + m] -= af; \
    dist[0][i + m] -= af; \
    \
    dist[k][i + m] = exp(dist[k][i + m]); \
    dist[0][i + m] = exp(dist[0][i + m]); \
    \
    af = dist[k][i + m] + dist[0][i + m]; \
    dist[k][i + m] /= af; \
    dist[0][i + m] /= af; \
    \
    N0[0] += dist[k][i + m]; \
}

unsigned char *Y;
static double *Xarr[3], *Xdiff[2][3], *dist[2], Means[3][2], MeansOld[3][2], *Sigmas[2], *SigmasOld[2], pis[2] = { 0.1, 0.9 }, pisOld[2];
int cols, ncores, ndim = 3, ncomp = 2, sigl = 9, *N;
double logdet[2], SigmasInv[2][9], *N2, Occ[2], Msum[2];
static double diffth;
int stepsize = 4, rollback = 0;
static int imgreduce = 0;

// Pthread data
pthread_t *initthreads;
pthread_t *mainthreads;
pthread_mutex_t ptm;
pthread_barrier_t ptb;
double *ptsigmas[2];

typedef struct
{
    int threadid;
    int nstart;
    int nend;
} EMThreadInfo;

EMThreadInfo *Info;

void *InitEMThread(void *info)
{
    EMThreadInfo *inf = (EMThreadInfo *)info;

    double af, bf;
    const double df = -2.0;
    int N0 = 0;
    int i, j, k, m;

    __m256d a, b, c, d, e, f;

    d = _mm256_set1_pd(df);
    /****** To compute euclidean distance *******/
    for (j = 0; j < ndim; j++)
    {
        for (k = 0; k < ncomp; k++)
        {
            b = _mm256_set1_pd(Means[j][k]);

            for (i = TSTART; i < TEND; i += stepsize)
            {
                if ((i + stepsize) <= TEND)
                {
                    a = _mm256_load_pd(&Xarr[j][i]);
                    c = _mm256_sub_pd(a, b);
                    _mm256_store_pd(&Xdiff[k][j][i], c);
                }
                else
                {
                    for (m = 0; m < (TEND - i + 1); m++)
                        Xdiff[k][j][i + m] = (Xarr[j][i + m] - Means[j][k]);
                }
            }
        }
    }
    
    if (imgreduce == 1 || rollback == 1)
        pthread_exit(NULL);

    pthread_barrier_wait(&ptb);
    
    for (k = 0; k < ncomp; k++)
    {
        for (i = TSTART; i < TEND; i += stepsize)
        {
            if ((i + stepsize) <= TEND)
            {
                c = _mm256_set1_pd(0.0);

                for (j = 0; j < ndim; j++)
                {
                    a = _mm256_load_pd(&Xdiff[k][j][i]);
                    c = _mm256_fmadd_pd(a, a, c);
                }
                
                //a = _mm256_sqrt_pd(a);
                _mm256_store_pd(&dist[k][i], c);
            }
            else
            {
                for (m = 0; m < (TEND - i + 1); m++)
                {
                    bf = 0.0;

                    for (j = 0; j < ndim; j++)
                        bf += Xdiff[k][j][i + m] * Xdiff[k][j][i + m];
                    
                    //for (int j = 0; j < ncomp; j++)
                    dist[k][i + m] = sqrt(bf);
                }
            }
        }
    }
    
    pthread_barrier_wait(&ptb);
    /**********************************************/

    /********* To compute the sigmas **************/
    if (TID == 0)
    {
        for (k = 0; k < ncomp; k++)
            for (i = 0; i < (ncores * 9); i++)
                ptsigmas[k][i] = 0.0;
    }

    pthread_barrier_wait(&ptb);
    
    for (j = 0; j < ndim; j++)
    {
        for (k = j; k < ndim; k++)
        {
            e = _mm256_set1_pd(0.0);
            f = _mm256_set1_pd(0.0);
            af = 0.0;
            bf = 0.0;

            for (i = TSTART; i < TEND; i += stepsize)
            {
                if ((i + stepsize) <= TEND)
                {
                    a = _mm256_load_pd(&dist[0][i]);
                    b = _mm256_load_pd(&dist[1][i]);

                    c = _mm256_cmp_pd(a, b, _CMP_LT_OQ);

                    // For component 1
                    a = _mm256_load_pd(&Xdiff[0][j][i]);
                    b = _mm256_load_pd(&Xdiff[0][k][i]);
                    d = _mm256_and_pd(c, a);
                    e = _mm256_fmadd_pd(b, d, e);

                    // For component 2
                    a = _mm256_load_pd(&Xdiff[1][j][i]);
                    b = _mm256_load_pd(&Xdiff[1][k][i]);
                    d = _mm256_andnot_pd(c, a);
                    f = _mm256_fmadd_pd(b, d, f);
                    
                    if (j == 0 && k == 0)
                        for (m = 0; m < 4; m++)
                            if (VEC(c)[m] == 0)
                                N0++;
                }
                else
                {
                    for (m = 0; m < (TEND - i + 1); m++)
                    {
                        if (dist[0][i + m] < dist[1][i + m])
                            af += (Xdiff[0][j][i] * Xdiff[0][k][i]);
                        else
                        {
                            bf += (Xdiff[1][j][i] * Xdiff[1][k][i]);

                            if (j == 0 && k == 0)
                                N0++;
                        }
                    }
                }
            }

            pthread_barrier_wait(&ptb);

            af += (VEC(e)[0] + VEC(e)[1] + VEC(e)[2] + VEC(e)[3]);
            bf += (VEC(f)[0] + VEC(f)[1] + VEC(f)[2] + VEC(f)[3]);
            
            ptsigmas[0][TID * sigl + j * ndim + k] = af;
            ptsigmas[0][TID * sigl + k * ndim + j] = af;
            ptsigmas[1][TID * sigl + j * ndim + k] = bf;
            ptsigmas[1][TID * sigl + k * ndim + j] = bf;
        }
    }

    N[TID] = N0;
    /**********************************************/
    
    pthread_exit(NULL);
}

double getdet(int k)
{
    /*for (int i = 0; i < 9; i++)
    {
        printf("%lf - ", Sigmas[k][i]);
    }
    printf("\n");*/

    return Sigmas[k][0] * (Sigmas[k][4] * Sigmas[k][8] - Sigmas[k][5] * Sigmas[k][7]) -
        Sigmas[k][1] * (Sigmas[k][3] * Sigmas[k][8] - Sigmas[k][5] * Sigmas[k][6]) +
        Sigmas[k][2] * (Sigmas[k][3] * Sigmas[k][7] - Sigmas[k][6] * Sigmas[k][4]);
}

void ComputeSigmaDetInv()
{
    int k ;

    if (ndim == 3)
    {
        for (k = 0; k < ncomp; k++)
        {
            logdet[k] = getdet(k);

            SigmasInv[k][0] = (Sigmas[k][4] * Sigmas[k][8] - Sigmas[k][5] * Sigmas[k][7]) / logdet[k];
            SigmasInv[k][1] = (-1.0) * (Sigmas[k][3] * Sigmas[k][8] - Sigmas[k][5] * Sigmas[k][6]) / logdet[k];
            SigmasInv[k][2] = (Sigmas[k][3] * Sigmas[k][7] - Sigmas[k][6] * Sigmas[k][4]) / logdet[k];
            SigmasInv[k][3] = (-1.0) * (Sigmas[k][1] * Sigmas[k][8] - Sigmas[k][2] * Sigmas[k][7]) / logdet[k];
            SigmasInv[k][4] = (Sigmas[k][0] * Sigmas[k][8] - Sigmas[k][6] * Sigmas[k][2]) / logdet[k];
            SigmasInv[k][5] = (-1.0) * (Sigmas[k][0] * Sigmas[k][7] - Sigmas[k][6] * Sigmas[k][1]) / logdet[k];
            SigmasInv[k][6] = (Sigmas[k][1] * Sigmas[k][5] - Sigmas[k][4] * Sigmas[k][2]) / logdet[k];
            SigmasInv[k][7] = (-1.0) * (Sigmas[k][0] * Sigmas[k][5] - Sigmas[k][3] * Sigmas[k][2]) / logdet[k];
            SigmasInv[k][8] = (Sigmas[k][0] * Sigmas[k][4] - Sigmas[k][3] * Sigmas[k][1]) / logdet[k];

            logdet[k] = log(logdet[k]) / (-2.0) - 1.5 * log(2 * 3.1415926535897932384626);
        }
    }
    else if (ndim == 2)
    {
        for (k = 0; k < ncomp; k++)
        {
            logdet[k] = (Sigmas[k][0] * Sigmas[k][3] - Sigmas[k][1] * Sigmas[k][2]);

            SigmasInv[k][0] = (Sigmas[k][3]) / logdet[k];
            SigmasInv[k][1] = ((-1.0) * Sigmas[k][1]) / logdet[k];
            SigmasInv[k][2] = ((-1.0) * Sigmas[k][2]) / logdet[k];
            SigmasInv[k][3] = (Sigmas[k][0]) / logdet[k];
            
            logdet[k] = log(logdet[k]) / (-2.0) - log(2 * 3.1415926535897932384626);
        }
    }
    else
    {
        for (k = 0; k < ncomp; k++)
        {
            logdet[k] = (Sigmas[k][0]);
            SigmasInv[k][0] = 1 / logdet[k];
            logdet[k] = log(logdet[k]) / (-2.0) - 0.5 * log(2 * 3.1415926535897932384626);
        }
    }
}

void InitEM()
{
    int i = 0, N0 = 0, rc, j, k, m;
    void *status;
    int datastep = cols / ncores;
    datastep = datastep - (datastep % 32);

    Info = (EMThreadInfo *)malloc(sizeof(EMThreadInfo) * ncores);
    for (i = 0; i < ncores; i++)
    {
        Info[i].threadid = i;
        Info[i].nstart = i * datastep;
        Info[i].nend = (i == (ncores - 1)) ? (cols - 1) : ((i + 1) * datastep - 1);
        pthread_create(&initthreads[i], NULL, InitEMThread, &Info[i]);
    }
    for (i = 0; i<ncores; i++)
        rc = pthread_join(initthreads[i], &status);
    free(Info);
    if (imgreduce == 0 && rollback == 0)
    {
        // Combine results from all threads.
        for (i = 0; i < ncores; i++)
            N0 += N[i];

        for (k = 0; k < ncomp; k++)
        {
            for (j = 0; j < ndim; j++)
            {
                for (m = j; m < ndim; m++)
                {
                    Sigmas[k][j * ndim + m] = 0.0;

                    for (i = 0; i < ncores; i++)
                        Sigmas[k][j * ndim + m] += ptsigmas[k][i * 9 + j * ndim + m];

                    Sigmas[k][j * ndim + m] = (k == 0) ? (Sigmas[k][j * ndim + m] / (cols - N0 - 1)) : (Sigmas[k][j * ndim + m] / (N0 - 1));
                    Sigmas[k][m * ndim + j] = Sigmas[k][j * ndim + m];
                }
            }
        }

        /*Sigmas[1][0] = 0.002871146439417; Sigmas[1][1] = 0.001712753760565; Sigmas[1][2] = 0.001188333998772;
        Sigmas[1][3] = 0.001712753760565; Sigmas[1][4] = 0.001304209888813; Sigmas[1][5] = 0.000918000785036;
        Sigmas[1][6] = 0.001188333998772; Sigmas[1][7] = 0.000918000785036; Sigmas[1][8] = 0.000748057549503;

        Sigmas[0][0] = 0.005809259846787; Sigmas[0][1] = 0.003273877221238; Sigmas[0][2] = 0.003746825966364;
        Sigmas[0][3] = 0.003273877221238; Sigmas[0][4] = 0.003918767270040; Sigmas[0][5] = 0.000328252743649;
        Sigmas[0][6] = 0.003746825966364; Sigmas[0][7] = 0.000328252743649; Sigmas[0][8] = 0.031898116598367;*/
    }
    /* InitEM Output */
    /*for (int k = 0; k < ncomp; k++)
    {
        printf("\nComponent %d: \n", k);

        for (int j = 0; j < ndim; j++)
        {
            for (int m = 0; m < ndim; m++)
            {
                printf("%lf  ", Sigmas[k][j * ndim + m]);
            }
            printf("\n");
        }
    }

    printf("\nMeans:\n");
    for (int j = 0; j < ncomp; j++)
    {
        printf("\nComponent %d: \n", j);

        for (int k = 0; k < ndim; k++)
        {
            printf("%lf    ", Means[k][j]);
        }

        printf("\n");
    }*/

    ComputeSigmaDetInv();
}

void *DoEStepThread(void *info)
{
    EMThreadInfo *inf = (EMThreadInfo *)info;

    int i = TSTART, k, m;
    double af, maxv = 0, lpis = 0;
    const double df = -0.5;

#ifdef __GNUC__
    __attribute__ ((aligned (32))) double N0[4] = { 0, 0, 0, 0 };
#else
    __declspec(align(32)) double N0[4] = { 0, 0, 0, 0 };
#endif

    __m256d a, b, c, d, e, f, g, x, y, z, p, q, r, s, t, u;
    g = _mm256_set1_pd(df);

    for (k = 0; k < ncomp; k++)
    {
        if (ndim == 3)
        {
            a = _mm256_set1_pd(SigmasInv[k][0]);
            b = _mm256_set1_pd(SigmasInv[k][1] * 2);
            c = _mm256_set1_pd(SigmasInv[k][2] * 2);
            d = _mm256_set1_pd(SigmasInv[k][4]);
            e = _mm256_set1_pd(SigmasInv[k][5] * 2);
            f = _mm256_set1_pd(SigmasInv[k][8]);
        }
        else if (ndim == 2)
        {
            a = _mm256_set1_pd(SigmasInv[k][0]);
            b = _mm256_set1_pd(SigmasInv[k][1] * 2);
            d = _mm256_set1_pd(SigmasInv[k][3]);
        }
        else
            a = _mm256_set1_pd(SigmasInv[k][0]);

        t = _mm256_set1_pd(logdet[k]);
        lpis = log(pis[k]);

        if (ndim == 3)
        {
            for (i = TSTART; i < TEND; i += stepsize)
            {
                if ((i + stepsize) <= TEND)
                {
                    x = _mm256_load_pd(&Xdiff[k][0][i]);
                    y = _mm256_load_pd(&Xdiff[k][1][i]);
                    z = _mm256_load_pd(&Xdiff[k][2][i]);

                    p = _mm256_mul_pd(x, a);
                    p = _mm256_fmadd_pd(b, y, p);

                    q = _mm256_mul_pd(d, y);
                    q = _mm256_fmadd_pd(e, z, q);

                    r = _mm256_mul_pd(f, z);
                    r = _mm256_fmadd_pd(c, x, r);

                    s = _mm256_mul_pd(p, x);
                    s = _mm256_fmadd_pd(q, y, s);
                    s = _mm256_fmadd_pd(r, z, s);
                    
                    UPDATE_PROBABILITIES_AVX;
                }
                else
                {
                    for (m = 0; m < (TEND - i + 1); m++)
                    {
                        af = ((SigmasInv[k][0] * Xdiff[k][0][m + i] + 2 * SigmasInv[k][1] * Xdiff[k][1][m + i])  * Xdiff[k][0][m + i]);
                        af += ((SigmasInv[k][4] * Xdiff[k][1][m + i] + 2 * SigmasInv[k][5] * Xdiff[k][2][m + i]) * Xdiff[k][1][m + i]);
                        af += ((SigmasInv[k][8] * Xdiff[k][2][m + i] + 2 * SigmasInv[k][2] * Xdiff[k][0][m + i])   * Xdiff[k][2][m + i]);
                        UPDATE_PROBABILITIES;
                    }
                }
            }
        }
        else if (ndim == 2)
        {
            for (i = TSTART; i < TEND; i += stepsize)
            {
                if ((i + stepsize) <= TEND)
                {
                    x = _mm256_load_pd(&Xdiff[k][0][i]);
                    y = _mm256_load_pd(&Xdiff[k][1][i]);
                    
                    p = _mm256_mul_pd(x, a);
                    p = _mm256_fmadd_pd(b, y, p);

                    q = _mm256_mul_pd(d, y);
                    
                    s = _mm256_mul_pd(p, x);
                    s = _mm256_fmadd_pd(q, y, s);

                    UPDATE_PROBABILITIES_AVX;
                }
                else
                {
                    for (m = 0; m < (TEND - i + 1); m++)
                    {
                        af = ((SigmasInv[k][0] * Xdiff[k][0][m + i] + 2 * SigmasInv[k][1] * Xdiff[k][1][m + i])  * Xdiff[k][0][m + i]);
                        af += ((SigmasInv[k][3] * Xdiff[k][1][m + i]) * Xdiff[k][1][m + i]);
                        UPDATE_PROBABILITIES;
                    }
                }
            }
        }
        else
        {
            for (i = TSTART; i < TEND; i += stepsize)
            {
                if ((i + stepsize) <= TEND)
                {
                    x = _mm256_load_pd(&Xdiff[k][0][i]);
                    p = _mm256_mul_pd(x, a);
                    s = _mm256_mul_pd(p, x);

                    UPDATE_PROBABILITIES_AVX;
                }
                else
                {
                    for (m = 0; m < (TEND - i + 1); m++)
                    {
                        af = ((SigmasInv[k][0] * Xdiff[k][0][m + i])  * Xdiff[k][0][m + i]);
                        UPDATE_PROBABILITIES;
                    }
                }
            }
        }
    }
    
    N2[TID] = N0[0] + N0[1] + N0[2] + N0[3];
    pthread_exit(NULL);
}

void *DoMStepThread(void *info)
{
    EMThreadInfo *inf = (EMThreadInfo *)info;

    double af, bf;

#ifdef __GNUC__
    __attribute__ ((aligned (32))) double me[2][3][4] = { { { 0, 0, 0, 0 }, { 0, 0, 0, 0 }, { 0, 0, 0, 0 } }, { { 0, 0, 0, 0 }, { 0, 0, 0, 0 }, { 0, 0, 0, 0 } } };
#else
    __declspec(align(32)) double me[2][3][4] = { { { 0, 0, 0, 0 }, { 0, 0, 0, 0 }, { 0, 0, 0, 0 } }, { { 0, 0, 0, 0 }, { 0, 0, 0, 0 }, { 0, 0, 0, 0 } } };
#endif

    double me2[2][3] = { { 0, 0, 0 }, { 0, 0, 0 } };
    __m256d a, b, c, d, e, f, g, h;
    int i, j, k, m;
    
    // To compute Means
    for (k = 0; k < ncomp; k++)
    {
        for (j = 0; j < ndim; j++)
        {
            d = _mm256_set1_pd(0);

            for (i = TSTART; i < TEND; i += stepsize)
            {
                if ((i + stepsize) <= TEND)
                {
                    a = _mm256_load_pd(&Xarr[j][i]);
                    g = _mm256_load_pd(&dist[k][i]);
                    d = _mm256_fmadd_pd(a, g, d);
                }
                else
                {
                    for (m = 0; m < (TEND - i + 1); m++)
                        me2[k][j] += (dist[k][i + m] * Xarr[j][m + i]);
                }
            }

            _mm256_store_pd(&me[k][j][0], d);
        }
    }

    pthread_mutex_lock(&ptm);
    for (j = 0; j < ndim; j++)
        for (k = 0; k < ncomp; k++)
            Means[j][k] += me[k][j][0] + me[k][j][1] + me[k][j][2] + me[k][j][3] + me2[k][j];
    pthread_mutex_unlock(&ptm);

    pthread_barrier_wait(&ptb);

    if ((*inf).threadid == 1)
    {
        for (k = 0; k < ncomp; k++)
            for (j = 0; j < ndim; j++)
                Means[j][k] /= Occ[k];
    }

    pthread_barrier_wait(&ptb);

    // Rebuild Xdiff matrices
    for (j = 0; j < ndim; j++)
    {
        for (k = 0; k < ncomp; k++)
        {
            b = _mm256_set1_pd(Means[j][k]);

            for (i = TSTART; i < TEND; i += stepsize)
            {
                if ((i + stepsize) <= TEND)
                {
                    a = _mm256_load_pd(&Xarr[j][i]);
                    c = _mm256_sub_pd(a, b);
                    _mm256_store_pd(&Xdiff[k][j][i], c);
                }
                else
                {
                    for (m = 0; m < (TEND - i + 1); m++)
                        Xdiff[k][j][i + m] = (Xarr[j][i + m] - Means[j][k]);
                }
            }
        }
    }

    pthread_barrier_wait(&ptb);

    // To update Sigmas
    for (j = 0; j < ndim; j++)
    {
        for (k = j; k < ndim; k++)
        {
            e = _mm256_set1_pd(0.0);
            f = _mm256_set1_pd(0.0);
            af = 0.0;
            bf = 0.0;

            for (i = TSTART; i < TEND; i += stepsize)
            {
                if ((i + stepsize) <= TEND)
                {
                    a = _mm256_load_pd(&dist[0][i]);
                    b = _mm256_load_pd(&dist[1][i]);

                    // For component 1
                    c = _mm256_load_pd(&Xdiff[0][j][i]);
                    d = _mm256_load_pd(&Xdiff[0][k][i]);
                    c = _mm256_mul_pd(a, c);
                    e = _mm256_fmadd_pd(c, d, e);

                    // For component 2
                    g = _mm256_load_pd(&Xdiff[1][j][i]);
                    h = _mm256_load_pd(&Xdiff[1][k][i]);
                    g = _mm256_mul_pd(b, g);
                    f = _mm256_fmadd_pd(g, h, f);
                }
                else
                {
                    for (m = 0; m < (TEND - i + 1); m++)
                    {
                        af += (dist[0][i + m] * Xdiff[0][j][i + m] * Xdiff[0][k][i + m]);
                        bf += (dist[1][i + m] * Xdiff[1][j][i + m] * Xdiff[1][k][i + m]);
                    }
                }
            }

            af += (VEC(e)[0] + VEC(e)[1] + VEC(e)[2] + VEC(e)[3]);
            bf += (VEC(f)[0] + VEC(f)[1] + VEC(f)[2] + VEC(f)[3]);

            ptsigmas[0][TID * sigl + j * ndim + k] = af;
            ptsigmas[0][TID * sigl + k * ndim + j] = af;
            ptsigmas[1][TID * sigl + j * ndim + k] = bf;
            ptsigmas[1][TID * sigl + k * ndim + j] = bf;
        }
    }

    pthread_exit(NULL);
}

void DoEStep()
{
    int i = 0, rc;
    void *status;
    int datastep = cols / ncores;
    datastep = datastep - (datastep % 32);

    Info = (EMThreadInfo *)malloc(sizeof(EMThreadInfo) * ncores);

    for (i = 0; i < ncores; i++)
    {
        Info[i].threadid = i;
        Info[i].nstart = i * datastep;
        Info[i].nend = (i == (ncores - 1)) ? (cols - 1) : ((i + 1) * datastep - 1);

        pthread_create(&mainthreads[i], NULL, DoEStepThread, &Info[i]);
    }

    for (i = 0; i<ncores; i++)
        rc = pthread_join(mainthreads[i], &status);
    free(Info);
}

void DoMStep()
{
    int i = 0, rc;
    void *status;
    int datastep = cols / ncores;
    datastep = datastep - (datastep % 32);

    Info = (EMThreadInfo *)malloc(sizeof(EMThreadInfo) * ncores);

    for (i = 0; i < ncores; i++)
    {
        Info[i].threadid = i;
        Info[i].nstart = i * datastep;
        Info[i].nend = (i == (ncores - 1)) ? (cols - 1) : ((i + 1) * datastep - 1);

        pthread_create(&mainthreads[i], NULL, DoMStepThread, &Info[i]);
    }

    for (i = 0; i<ncores; i++)
        rc = pthread_join(mainthreads[i], &status);
    free(Info);
}

void EMMainLoop()
{
    double error = 10000000, permissibleError = 0.00001;
    int nitr = 1, i, j, k, m;
    double mindet = 1.0E-300;
    int maxitr = 500;

    while (error > permissibleError && nitr <= maxitr)
    {
        printf("Iteration = %d \t Error = %lf\n", nitr++, error);

        // Do the E-step
        DoEStep();

        // To compute the occupancy count
        // Combine results from all threads.
        Occ[0] = 0; Occ[1] = 0;
        for (i = 0; i < ncores; i++)
            Occ[1] += N2[i];
        Occ[0] = cols - Occ[1];

        // To store old values
        for (k = 0; k < ncomp; k++)
        {
            for (j = 0; j < ndim; j++)
            {
                MeansOld[j][k] = Means[j][k];
                Means[j][k] = 0;
            }

            for (j = 0; j < sigl; j++)
            {
                SigmasOld[k][j] = Sigmas[k][j];
                Sigmas[k][j] = 0;
            }
        }

        pisOld[0] = pis[0]; pisOld[1] = pis[1];

        // Do the M-step
        DoMStep();

        // To combine the results
        for (k = 0; k < ncomp; k++)
        {
            for (j = 0; j < ndim; j++)
            {
                for (m = j; m < ndim; m++)
                {
                    Sigmas[k][j * ndim + m] = 0.0;

                    for (i = 0; i < ncores; i++)
                        Sigmas[k][j * ndim + m] += ptsigmas[k][i * sigl + j * ndim + m];

                    Sigmas[k][j * ndim + m] = Sigmas[k][j * ndim + m] / Occ[k];
                    Sigmas[k][m * ndim + j] = Sigmas[k][j * ndim + m];
                }
            }
        }

        pis[0] = Occ[0] / cols; pis[1] = Occ[1] / cols;
        ComputeSigmaDetInv();

        /* InitEM Output */
        /*printf("Sigmas:\n");
        for (int k = 0; k < ncomp; k++)
        {
        printf("\nComponent %d: \n", k);

        for (int j = 0; j < ndim; j++)
        {
        for (int m = 0; m < ndim; m++)
        {
        printf("%lf  ", Sigmas[k][j * ndim + m]);
        }
        printf("\n");
        }

        printf("Determinant: %e\n", getdet(k));
        }

        printf("\nMeans:\n");
        for (int j = 0; j < ncomp; j++)
        {
        printf("\nComponent %d: \n", j);

        for (int k = 0; k < ndim; k++)
        {
        printf("%lf    ", Means[k][j]);
        }

        printf("\n");
        }*/

        // To compute the error
        error = 0;
        for (k = 0; k < ncomp; k++)
        {
            for (j = 0; j < ndim; j++)
                error += fabs(MeansOld[j][k] - Means[j][k]);

            for (j = 0; j < sigl; j++)
                error += fabs(SigmasOld[k][j] - Sigmas[k][j]);
        }

        error += (fabs(pisOld[0] - pis[0]) + fabs(pisOld[1] - pis[1]));
        error *= 10000;

        // We do not want the determinant values of covariance matrices to 
        // get closer to minimum machine supported precision. In that case, 
        // we rollback parameters estimated in previous iteration.
        if (getdet(0) < mindet || getdet(1) < mindet)
        {
            for (k = 0; k < ncomp; k++)
            {
                for (j = 0; j < ndim; j++)
                    Means[j][k] = MeansOld[j][k];

                for (j = 0; j < sigl; j++)
                    Sigmas[k][j] = SigmasOld[k][j];
            }

            rollback = 1;
            InitEM();
            DoEStep();
            rollback = 0;
            break;
        }

        /*double sum[3] = { 0.0, 0.0, 0.0 };
        printf("Component 1 : \n");
        for (int i = 0; i < 10; i++)
            printf("%lf - %lf - %lf\n", Xdiff[0][0][i], Xdiff[0][1][i], Xdiff[0][2][i]);
        printf("\n Component 2 : \n");
        for (int i = 0; i < 10; i++)
            printf("%lf - %lf - %lf\n", Xdiff[1][0][i], Xdiff[1][1][i], Xdiff[1][2][i]);

        for (int i = 0; i < cols; i++)
        {
            sum[0] += Xdiff[0][0][i];
            sum[1] += Xdiff[0][1][i];
            sum[2] += Xdiff[0][2][i];
        }

        printf("Sum 0 = %lf - %lf - %lf\n", sum[0], sum[1], sum[2]);

        sum[0] = 0.0; sum[1] = 0.0; sum[2] = 0.0;
        for (int i = 0; i < cols; i++)
        {
            sum[0] += Xdiff[1][0][i];
            sum[1] += Xdiff[1][1][i];
            sum[2] += Xdiff[1][2][i];
        }

        printf("Sum 1 = %lf - %lf - %lf\n", sum[0], sum[1], sum[2]);

        printf("\n*********\n");
        printf("Distance : \n");
        for (int i = 0; i < 10; i++)
            printf("%lf - %lf\n", dist[0][i], dist[1][i]);

        sum[0] = 0.0; sum[1] = 0.0;
        for (int i = 0; i < cols; i++)
        {
            sum[0] += dist[0][i];
            sum[1] += dist[1][i];
        }

        printf("Dist = %lf - %lf\n", sum[0], sum[1]);
        printf("\n*********\n");
        for (int k = 0; k < 2; k++)
        {
            for (int i = 0; i < 9; i++)
                printf("%lf - ", Sigmas[k][i]);
            printf("\n");
        }
        printf("\n*********\n");
        for (int k = 0; k < 2; k++)
        {
            for (int j = 0; j < ndim; j++)
                printf("%lf - ", Means[j][k]);
            
            printf("\n");
        }
        printf("\n*********\n");
        printf("\n");
        system("pause");*/
    }

    Msum[0] = 0;
    Msum[1] = 0;
    for (k = 0; k < ndim; k++)
    {
        Msum[0] += Means[k][0];
        Msum[1] += Means[k][1];
    }
}

//void *DiscretizeResultThread(void *info)
//{
//    EMThreadInfo *inf = (EMThreadInfo *)info;
//
//    __m256d a, b, c, d, e;
//
//    d = _mm256_set1_pd(1.0);
//    e = _mm256_set1_pd(255.0);
//    
//    for (int i = TSTART; i < TEND; i += stepsize)
//    {
//        if ((i + stepsize) <= TEND)
//        {
//            a = _mm256_load_pd(&dist[0][i]);
//            b = _mm256_load_pd(&dist[1][i]);
//
//            if (Msum[0] > Msum[1])
//                c = _mm256_cmp_pd(a, b, _CMP_LT_OQ);
//            else
//                c = _mm256_cmp_pd(a, b, _CMP_GT_OQ);
//
//            c = _mm256_and_pd(c, d);
//            c = _mm256_mul_pd(c, e);
//
//            _mm256_store_pd(&Y[i], c);
//        }
//        else
//        {
//            for (int m = 0; m < (TEND - i + 1); m++)
//            {
//                Y[i + m] = (Msum[0] > Msum[1]) ? ((dist[0][i + m] < dist[1][i + m]) ? 255.0 : 0) : ((dist[0][i + m] > dist[1][i + m]) ? 255.0 : 0);
//            }
//        }
//    }
//}
//
//void DiscretizeResult()
//{
//    int i = 0, rc;
//    void *status;
//    int datastep = cols / ncores;
//
//    Info = (EMThreadInfo *)malloc(sizeof(EMThreadInfo) * ncores);
//
//    for (i = 0; i < ncores; i++)
//    {
//        Info[i].threadid = i;
//        Info[i].nstart = i * datastep;
//        Info[i].nend = (i == (ncores - 1)) ? cols : (i + 1) * datastep - 1;
//
//        pthread_create(&mainthreads[i], NULL, DiscretizeResultThread, &Info[i]);
//    }
//
//    free(Info);
//    for (int i = 0; i<ncores; i++)
//        rc = pthread_join(mainthreads[i], &status);
//}

void DiscretizeResultST()
{
    double diff = (diffth == 100.0) ? 0.0 : diffth;
    int i;

    for (i = 0; i < cols; i++)
    {
        // dist[0] is the P(Z = 0|X)
        // dist[1] is the P(Z = 1|X)
        if (Msum[0] > Msum[1])    // If first component is background
        {
            if ((dist[0][i] + diff) <= (dist[1][i] - diff))
            {
                Y[i] = 255;
            }
            else
            {
                Y[i] = 0;
            }
        }
        else                      // If first component is foreground
        {
            if ((dist[0][i] - diff) >= (dist[1][i] + diff))
            {
                Y[i] = 255;
            }
            else
            {
                Y[i] = 0;
            }
        }
    }
}

void PerformEM(double *_X, double **_Means, unsigned char *result, int _rows, int _cols, int _ncores, int _imgreduce, double *_Xfull, int _colsfull, double _fuzzifier, double _diffth)
{
    int dswitch = 0;
    double sum[3] = {0.0, 0.0, 0.0};
    //Xarr = _X;
    Y = result;
    cols = _cols;
    ndim = _rows;
    ncores = _ncores;
    diffth = _diffth;
    sigl = ndim * ndim;

    initthreads = (pthread_t *)_mm_malloc(sizeof(pthread_t) * ncores, 32);
    mainthreads = (pthread_t *)_mm_malloc(sizeof(pthread_t) * ncores, 32);

    Sigmas[0] = (double *)_mm_malloc(sizeof(double) * sigl, 32);
    Sigmas[1] = (double *)_mm_malloc(sizeof(double) * sigl, 32);

    SigmasOld[0] = (double *)_mm_malloc(sizeof(double) * sigl, 32);
    SigmasOld[1] = (double *)_mm_malloc(sizeof(double) * sigl, 32);

    ptsigmas[0] = (double *)_mm_malloc(sizeof(double) * sigl * ncores, 32);
    ptsigmas[1] = (double *)_mm_malloc(sizeof(double) * sigl * ncores, 32);
    
    dist[0] = (double *)((_imgreduce == 0) ? _mm_malloc(sizeof(double) * cols, 32) : _mm_malloc(sizeof(double) * _colsfull, 32));
    dist[1] = (double *)((_imgreduce == 0) ? _mm_malloc(sizeof(double) * cols, 32) : _mm_malloc(sizeof(double) * _colsfull, 32));

    Xarr[0] = (double *)((_imgreduce == 0) ? _mm_malloc(sizeof(double) * cols, 32) : _mm_malloc(sizeof(double) * _colsfull, 32));
    memcpy(Xarr[0], &_X[0], cols * sizeof(double));
    Xdiff[0][0] = (double *)((_imgreduce == 0) ? _mm_malloc(sizeof(double) * cols, 32) : _mm_malloc(sizeof(double) * _colsfull, 32));
    Xdiff[1][0] = (double *)((_imgreduce == 0) ? _mm_malloc(sizeof(double) * cols, 32) : _mm_malloc(sizeof(double) * _colsfull, 32));
    if (ndim > 1)
    {
        Xarr[1] = (double *)((_imgreduce == 0) ? _mm_malloc(sizeof(double) * cols, 32) : _mm_malloc(sizeof(double) * _colsfull, 32));
        memcpy(Xarr[1], &_X[cols], cols * sizeof(double));
        Xdiff[0][1] = (double *)((_imgreduce == 0) ? _mm_malloc(sizeof(double) * cols, 32) : _mm_malloc(sizeof(double) * _colsfull, 32));
        Xdiff[1][1] = (double *)((_imgreduce == 0) ? _mm_malloc(sizeof(double) * cols, 32) : _mm_malloc(sizeof(double) * _colsfull, 32));
    }
    if (ndim > 2)
    {
        Xarr[2] = (double *)((_imgreduce == 0) ? _mm_malloc(sizeof(double) * cols, 32) : _mm_malloc(sizeof(double) * _colsfull, 32));
        memcpy(Xarr[2], &_X[2 * cols], cols * sizeof(double));
        Xdiff[0][2] = (double *)((_imgreduce == 0) ? _mm_malloc(sizeof(double) * cols, 32) : _mm_malloc(sizeof(double) * _colsfull, 32));
        Xdiff[1][2] = (double *)((_imgreduce == 0) ? _mm_malloc(sizeof(double) * cols, 32) : _mm_malloc(sizeof(double) * _colsfull, 32));
    }
    
    N = (int *)_mm_malloc(sizeof(int) * ncores, 32);
    N2 = (double *)_mm_malloc(sizeof(double) * ncores, 32);

    Means[0][0] = _Means[0][0]; Means[0][1] = _Means[0][1];
    if (ndim > 1) { Means[1][0] = _Means[1][0]; Means[1][1] = _Means[1][1]; }
    if (ndim > 2) { Means[2][0] = _Means[2][0]; Means[2][1] = _Means[2][1]; }
    
    /*Means[0][0] = 0.05882; Means[0][1] = 0.3137;
    Means[1][0] = 0.07059; Means[1][1] = 0.3098;
    Means[2][0] = 0.1255; Means[2][1] = 0.2941;*/
    
    /*Means[0][0] = 0.53236; Means[0][1] = 0.897565;
    Means[1][0] = 0.59904; Means[1][1] = 0.895;
    Means[2][0] = 0.751795; Means[2][1] = 0.88;*/
    
    /*Means[0][0] = 0.53236; Means[0][1] = 0.897565;
    Means[1][0] = 0.59904; Means[1][1] = 0.895;
    Means[2][0] = 0.751795; Means[2][1] = 0.88;*/

    /*Means[0][0] = 0.538558; Means[0][1] = 0.85491;
    Means[1][0] = 0.612171; Means[1][1] = 0.82;
    Means[2][0] = 0.717873; Means[2][1] = 0.755;*/
    
    /*Means[0][0] = 0.3137; Means[0][1] = 0.5843;
    Means[1][0] = 0.4471; Means[1][1] = 0.5608;
    Means[2][0] = 0.6157; Means[2][1] = 0.5176;*/
    
    /*Means[0][0] = RANDOM; Means[0][1] = RANDOM;
    Means[1][0] = RANDOM; Means[1][1] = RANDOM;
    Means[2][0] = RANDOM; Means[2][1] = RANDOM;*/
    
    pthread_mutex_init(&ptm, NULL);
    pthread_barrier_init(&ptb, NULL, ncores);
    
    // To select suitable initialization parameters for EM means
    InitEM();

    /*printf("Data : \n");
    for (int i = 0; i < 10; i++)
        printf("%lf - %lf - %lf\n", Xarr[i], Xarr[cols + i], Xarr[2 * cols + i]);
    printf("\n*********\n");
    printf("Component 1 : \n");
    for (int i = 0; i < 10; i++)
        printf("%lf - %lf - %lf\n", Xdiff[0][0][i], Xdiff[0][1][i], Xdiff[0][2][i]);
    printf("\n Component 2 : \n");
    for (int i = 0; i < 10; i++)
        printf("%lf - %lf - %lf\n", Xdiff[1][0][i], Xdiff[1][1][i], Xdiff[1][2][i]);

    for (int i = 0; i < cols; i++)
    {
        sum[0] += Xdiff[0][0][i];
        sum[1] += Xdiff[0][1][i];
        sum[2] += Xdiff[0][2][i];
    }

    printf("Sum 0 = %lf - %lf - %lf\n", sum[0], sum[1], sum[2]);

    sum[0] = 0.0; sum[1] = 0.0; sum[2] = 0.0;
    for (int i = 0; i < cols; i++)
    {
        sum[0] += Xdiff[1][0][i];
        sum[1] += Xdiff[1][1][i];
        sum[2] += Xdiff[1][2][i];
    }

    printf("Sum 1 = %lf - %lf - %lf\n", sum[0], sum[1], sum[2]);
    
    printf("\n*********\n");
    printf("Distance : \n");
    for (int i = 0; i < 10; i++)
        printf("%lf - %lf\n", dist[0][i], dist[1][i]);

    sum[0] = 0.0; sum[1] = 0.0;
    for (int i = 0; i < cols; i++)
    {
        sum[0] += dist[0][i];
        sum[1] += dist[1][i];
    }

    printf("Dist = %lf - %lf\n", sum[0], sum[1]);
    printf("\n*********\n");
    for (int k = 0; k < 2; k++)
    {
        for (int i = 0; i < 9; i++)
            printf("%lf - ", Sigmas[k][i]);
        printf("\n");
    }
    printf("\n*********\n");
    for (int k = 0; k < 2; k++)
    {
        for (int j = 0; j < ndim; j++)
            printf("%lf - ", Means[j][k]);

        printf("\n");
    }
    printf("\n*********\n");
    printf("\n");
    system("pause");*/

    // To perform EM
    EMMainLoop();
    // To discretize the result
    imgreduce = _imgreduce;
    if (imgreduce == 0)
        DiscretizeResultST();
    else
    {
        //Xarr = _Xfull;
        cols = _colsfull;
        
        memcpy(Xarr[0], &_Xfull[0], cols * sizeof(double));
        if (ndim > 1)
            memcpy(Xarr[1], &_Xfull[cols], cols * sizeof(double));
        if (ndim > 2)
            memcpy(Xarr[2], &_Xfull[2 * cols], cols * sizeof(double));
        
        InitEM();
        EMMainLoop();
        //DoEStep();
        DiscretizeResultST();
    }

    _Means[0][0] = Means[0][0]; _Means[0][1] = Means[0][1];
    if (ndim > 1) { _Means[1][0] = Means[1][0]; Means[1][1] = Means[1][1]; }
    if (ndim > 2) { _Means[2][0] = Means[2][0]; Means[2][1] = Means[2][1]; }
}

