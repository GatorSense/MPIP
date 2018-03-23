#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define NMAINTHREADS 1024
#define NBLOCKS      8
#define NTHREADS     128

//#pragma once
//typedef double tfloat;

#include "emkernel.h"

tfloat *devXptr;
unsigned char *devY;
static int cols, rows, ndim, sigl, ncomp;
static tfloat *Means, *Sigmas, *Herror, *Derror;
static tfloat diffth;
tfloat *hptrs;

__device__ tfloat *devX, devOcc[2] = {0.0, 0.0}, *devDist[2], *devXdiff[2][3], logdet[2], MeansOld[6], SigmasInv[18], SigmasOld[18], pis[2] = { 0.1, 0.9 }, pisOld[2] = { 0.1, 0.9 }, error = 10000000;
__device__ tfloat *devMeans, *devSigmas, *devnitr;
__device__ static int devcols, devndim, devncomp;
__device__ tfloat bdata[13][NBLOCKS];

#if (defined __GNUC__ && (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600)) || CUDA_VERSION >= 8000
#else
  static __inline__ __device__ double atomicAdd(double *address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    if (val==0.0)
      return __longlong_as_double(old);
    do {
      assumed = old;
      old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val +__longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
  }
#endif
//__device__ double atomicAdd(double* address, double val)
//{
//    unsigned long long int* address_as_ull =
//        (unsigned long long int*)address;
//    unsigned long long int old = *address_as_ull, assumed;
//    do {
//        assumed = old;
//        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
//    } while (assumed != old);
//    return __longlong_as_double(old);
//}

__device__ void ComputeSigmaDetInv()
{
    for (int k = 0; k < devncomp; k++)
    {
        logdet[k] = devSigmas[k * 9 + 0] * (devSigmas[k * 9 + 4] * devSigmas[k * 9 + 8] - devSigmas[k * 9 + 5] * devSigmas[k * 9 + 7]) -
            devSigmas[k * 9 + 1] * (devSigmas[k * 9 + 3] * devSigmas[k * 9 + 8] - devSigmas[k * 9 + 5] * devSigmas[k * 9 + 6]) +
            devSigmas[k * 9 + 2] * (devSigmas[k * 9 + 3] * devSigmas[k * 9 + 7] - devSigmas[k * 9 + 6] * devSigmas[k * 9 + 4]);

        SigmasInv[k * 9 + 0] = (devSigmas[k * 9 + 4] * devSigmas[k * 9 + 8] - devSigmas[k * 9 + 5] * devSigmas[k * 9 + 7]) / logdet[k];
        SigmasInv[k * 9 + 1] = (-1.0) * (devSigmas[k * 9 + 3] * devSigmas[k * 9 + 8] - devSigmas[k * 9 + 5] * devSigmas[k * 9 + 6]) / logdet[k];
        SigmasInv[k * 9 + 2] = (devSigmas[k * 9 + 3] * devSigmas[k * 9 + 7] - devSigmas[k * 9 + 6] * devSigmas[k * 9 + 4]) / logdet[k];
        SigmasInv[k * 9 + 3] = (-1.0) * (devSigmas[k * 9 + 1] * devSigmas[k * 9 + 8] - devSigmas[k * 9 + 2] * devSigmas[k * 9 + 7]) / logdet[k];
        SigmasInv[k * 9 + 4] = (devSigmas[k * 9 + 0] * devSigmas[k * 9 + 8] - devSigmas[k * 9 + 6] * devSigmas[k * 9 + 2]) / logdet[k];
        SigmasInv[k * 9 + 5] = (-1.0) * (devSigmas[k * 9 + 0] * devSigmas[k * 9 + 7] - devSigmas[k * 9 + 6] * devSigmas[k * 9 + 1]) / logdet[k];
        SigmasInv[k * 9 + 6] = (devSigmas[k * 9 + 1] * devSigmas[k * 9 + 5] - devSigmas[k * 9 + 4] * devSigmas[k * 9 + 2]) / logdet[k];
        SigmasInv[k * 9 + 7] = (-1.0) * (devSigmas[k * 9 + 0] * devSigmas[k * 9 + 5] - devSigmas[k * 9 + 3] * devSigmas[k * 9 + 2]) / logdet[k];
        SigmasInv[k * 9 + 8] = (devSigmas[k * 9 + 0] * devSigmas[k * 9 + 4] - devSigmas[k * 9 + 3] * devSigmas[k * 9 + 1]) / logdet[k];

        logdet[k] = logf(logdet[k]) / (-2.0) - 1.5 * logf(2 * 3.1415926535897932384626);
    }
}

__device__ void UpdateXdiff(tfloat *devX, tfloat *devXdiff[2][3], tfloat *devMeans, int cols, int ndim, int ncomp)
{
    int iinit = blockDim.x * blockIdx.x + threadIdx.x, i, k, d;
    float md;

    for (k = 0; k < ncomp; k++) // For each component
    {
        for (d = 0; d < ndim; d++) // For each dimension
        {
            md = devMeans[k * ndim + d];
            for (i = iinit; i < cols; i += NMAINTHREADS)
                devXdiff[k][d][i] = devX[d * cols + i] - md;
        }
    }

    __syncthreads();
}

__global__ void init(tfloat *devXptr, tfloat *devptrs, tfloat *Means, tfloat *Sigmas, int _cols, int _ndim, int _ncomp)
{
    devDist[0] = devptrs;
    devDist[1] = devptrs + _cols;

    devXdiff[0][0] = devptrs + _cols * 2;
    devXdiff[0][1] = devptrs + _cols * 3;
    devXdiff[0][2] = devptrs + _cols * 4;
    devXdiff[1][0] = devptrs + _cols * 5;
    devXdiff[1][1] = devptrs + _cols * 6;
    devXdiff[1][2] = devptrs + _cols * 7;

    devX = devXptr;

    devMeans = Means;
    devSigmas = Sigmas;

    devcols = _cols;
    devndim = _ndim;
    devncomp = _ncomp;

    for (int k = 0; k < devncomp; k++)
    {
        for (int i = 0; i < 9; i++)
            devSigmas[k * 9 + i] = 0.0f;
    }
}

__global__ void InitEM()
{
    int iinit = blockDim.x * blockIdx.x + threadIdx.x, i, k, cols = devcols, ndim = devndim, ncomp = devncomp;
    tfloat md[12] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f }, n = 0.0f;
    
    __shared__ tfloat sn, sd[12];

    if (threadIdx.x == 1)
    {
        sn = 0.0f;
        
        for (i = 0; i < 12; i++)
            sd[i] = 0.0f;
    }

    __syncthreads();
    UpdateXdiff(devX, devXdiff, devMeans, cols, ndim, ncomp);

    for (k = 0; k < ncomp; k++) // For each component
    {
        for (i = iinit; i < cols; i += NMAINTHREADS)
        {
            md[0] = devXdiff[k][0][i];
            md[1] = devXdiff[k][1][i];
            md[2] = devXdiff[k][2][i];

            devDist[k][i] = (md[0] * md[0] + md[1] * md[1] + md[2] * md[2]);
        }
    }

    __syncthreads();

    md[0] = 0.0f;
    md[1] = 0.0f;
    md[2] = 0.0f;

    for (i = iinit; i < cols; i += NMAINTHREADS)
    {
        if (devDist[0][i] < devDist[1][i])
        {
            md[0] += (devXdiff[0][0][i] * devXdiff[0][0][i]);
            md[1] += (devXdiff[0][0][i] * devXdiff[0][1][i]);
            md[2] += (devXdiff[0][0][i] * devXdiff[0][2][i]);
            md[3] += (devXdiff[0][1][i] * devXdiff[0][1][i]);
            md[4] += (devXdiff[0][1][i] * devXdiff[0][2][i]);
            md[5] += (devXdiff[0][2][i] * devXdiff[0][2][i]);

            n++;
        }
        else
        {
            md[6]  += (devXdiff[1][0][i] * devXdiff[1][0][i]);
            md[7]  += (devXdiff[1][0][i] * devXdiff[1][1][i]);
            md[8]  += (devXdiff[1][0][i] * devXdiff[1][2][i]);
            md[9]  += (devXdiff[1][1][i] * devXdiff[1][1][i]);
            md[10] += (devXdiff[1][1][i] * devXdiff[1][2][i]);
            md[11] += (devXdiff[1][2][i] * devXdiff[1][2][i]);
        }
    }

    __syncthreads();

    atomicAdd(&sn, n);
    
    for (i = 0; i < 12; i++)
        atomicAdd(&sd[i], md[i]);

    __syncthreads();

    if (threadIdx.x == 1)
    {
        for (i = 0; i < 12; i++)
            bdata[i][blockIdx.x] = sd[i];

        bdata[12][blockIdx.x] = sn;
    }
}

__global__ void EStep(double fuzzifier)
{
    int iinit = blockDim.x * blockIdx.x + threadIdx.x, i, k, cols = devcols, ncomp = devncomp;

    // Utility registers
    tfloat a, b, c, d, e, f, g, q, r, s, x, y, z, me, fuz;

    __shared__ tfloat se;

	fuz = ((float)fuzzifier);
    if (threadIdx.x == 1)
        se = 0.0;

    for (k = 0, me = 0; k < ncomp; k++)
    {
        a = SigmasInv[k * 9 + 0];
        b = SigmasInv[k * 9 + 1] * 2;
        c = SigmasInv[k * 9 + 2] * 2;
        d = SigmasInv[k * 9 + 4];
        e = SigmasInv[k * 9 + 5] * 2;
        f = SigmasInv[k * 9 + 8];

        g = logdet[k] + logf(pis[k]);

        for (i = iinit; i < cols; i += NMAINTHREADS)
        {
            x = devXdiff[k][0][i];
            y = devXdiff[k][1][i];
            z = devXdiff[k][2][i];

            if (k == 1)
            {
                s = (((x * a + y * b) * x + (y * d + z * e) * y + (z * f + x * c) * z) * (-0.5) + g) / (fuz - 1.0);
                r = devDist[0][i];
                q = (s > r) ? s : r;
                s = exp(s - q);
                r = exp(r - q);
                q = s + r;
                s /= q;
                r /= q;
				s = pow(s, fuz);
				r = pow(r, fuz);
                me += r;
                devDist[0][i] = r;
                devDist[1][i] = s;
            }
            else
				devDist[k][i] = (((x * a + y * b) * x + (y * d + z * e) * y + (z * f + x * c) * z) * (-0.5) + g) / (fuz - 1.0);
        }
    }

    atomicAdd(&se, me);

    __syncthreads();
    if (threadIdx.x == 1) // To be executed for every block.
    {
        bdata[0][blockIdx.x] = se;
    }
    __syncthreads();
}

__global__ void EStep()
{
	int iinit = blockDim.x * blockIdx.x + threadIdx.x, i, k, cols = devcols, ncomp = devncomp;

	// Utility registers
	tfloat a, b, c, d, e, f, g, q, r, s, x, y, z, me;

	__shared__ tfloat se;

	if (threadIdx.x == 1)
		se = 0.0;

	for (k = 0, me = 0; k < ncomp; k++)
	{
		a = SigmasInv[k * 9 + 0];
		b = SigmasInv[k * 9 + 1] * 2;
		c = SigmasInv[k * 9 + 2] * 2;
		d = SigmasInv[k * 9 + 4];
		e = SigmasInv[k * 9 + 5] * 2;
		f = SigmasInv[k * 9 + 8];

		g = logdet[k] + logf(pis[k]);

		for (i = iinit; i < cols; i += NMAINTHREADS)
		{
			x = devXdiff[k][0][i];
			y = devXdiff[k][1][i];
			z = devXdiff[k][2][i];

			if (k == 1)
			{
				s = ((x * a + y * b) * x + (y * d + z * e) * y + (z * f + x * c) * z) * (-0.5) + g;
				r = devDist[0][i];
				q = (s > r) ? s : r;
				s = exp(s - q);
				r = exp(r - q);
				q = s + r;
				s /= q;
				r /= q;
				me += r;
				devDist[0][i] = r;
				devDist[1][i] = s;
			}
			else
				devDist[k][i] = ((x * a + y * b) * x + (y * d + z * e) * y + (z * f + x * c) * z) * (-0.5) + g;
		}
	}

	atomicAdd(&se, me);

	__syncthreads();
	if (threadIdx.x == 1) // To be executed for every block.
	{
		bdata[0][blockIdx.x] = se;
	}
	__syncthreads();
}

__global__ void StoreOldValues()
{
    int i;

    devOcc[0] = 0.0;
    for (i = 0; i < NBLOCKS; i++)
        devOcc[0] += bdata[0][i];
    
    devOcc[1] = devcols - devOcc[0];
    pisOld[0] = pis[0];
    pisOld[1] = pis[1];

    pis[0] = devOcc[0] / devcols;
    pis[1] = devOcc[1] / devcols;

    for (i = 0; i < 18; i++)
    {
        SigmasOld[i] = devSigmas[i];
        devSigmas[i] = 0.0f;
    }

    for (i = 0; i < 6; i++)
    {
        MeansOld[i] = devMeans[i];
        devMeans[i] = 0.0f;
    }

    pisOld[0] = pis[0];
    pisOld[1] = pis[1];

    pis[0] = devOcc[0] / devcols;
    pis[1] = devOcc[1] / devcols;
}

__global__ void UpdateMeansMain()
{
    int iinit = blockDim.x * blockIdx.x + threadIdx.x, i, cols = devcols;

    // Utility registers
    tfloat a, b, c, x, y, mf[6];

    __shared__ tfloat se[6];

    if (threadIdx.x == 1)
        for (i = 0; i < 6; i++)
            se[i] = 0.0f;
    for (i = 0; i < 6; i++)
        mf[i] = 0.0f;

    for (i = iinit; i < cols; i += NMAINTHREADS)
    {
        a = devX[i];
        b = devX[cols + i];
        c = devX[2 * cols + i];

        x = devDist[0][i];
        y = devDist[1][i];

        mf[0] += (a * x);
        mf[1] += (b * x);
        mf[2] += (c * x);

        mf[3] += (a * y);
        mf[4] += (b * y);
        mf[5] += (c * y);
    }

    for (i = 0; i < 6; i++)
        atomicAdd(&se[i], mf[i]);
    __syncthreads();

    if (threadIdx.x == 1)
    {
        for (i = 0; i < 6; i++)
            bdata[i][blockIdx.x] = se[i];
    }
}

__global__ void UpdateMeans()
{
    int i;

    for (i = 0; i < NBLOCKS; i++)
    {
        devMeans[0] += bdata[0][i];
        devMeans[1] += bdata[1][i];
        devMeans[2] += bdata[2][i];
        devMeans[3] += bdata[3][i];
        devMeans[4] += bdata[4][i];
        devMeans[5] += bdata[5][i];
    }

    devMeans[0] /= (devOcc[0] - 1);
    devMeans[1] /= (devOcc[0] - 1);
    devMeans[2] /= (devOcc[0] - 1);
    devMeans[3] /= (devOcc[1] - 1);
    devMeans[4] /= (devOcc[1] - 1);
    devMeans[5] /= (devOcc[1] - 1);
}

__global__ void UpdateSigmasMain()
{
    int iinit = blockDim.x * blockIdx.x + threadIdx.x, i, cols = devcols, ndim = devndim, ncomp = devncomp;
    tfloat a, b, c, x, y, mf[12];

    __shared__ tfloat sf[12];
    UpdateXdiff(devX, devXdiff, devMeans, cols, ndim, ncomp);

    // To update covariances
    if (threadIdx.x == 1)
        for (i = 0; i < 12; i++)
            sf[i] = 0.0f;
    for (i = 0; i < 12; i++)
        mf[i] = 0.0f;

    for (i = iinit; i < cols; i += NMAINTHREADS)
    {
        a = devXdiff[0][0][i];
        b = devXdiff[0][1][i];
        c = devXdiff[0][2][i];

        x = devDist[0][i];
        y = devDist[1][i];

        mf[0] += a * a * x; // 0-0
        mf[1] += a * b * x; // 0-1
        mf[2] += a * c * x; // 0-2
        mf[3] += b * b * x; // 1-1
        mf[4] += b * c * x; // 1-2
        mf[5] += c * c * x; // 2-2

        a = devXdiff[1][0][i];
        b = devXdiff[1][1][i];
        c = devXdiff[1][2][i];

        mf[6] += a * a * y; // 0-0
        mf[7] += a * b * y; // 0-1
        mf[8] += a * c * y; // 0-2
        mf[9] += b * b * y; // 1-1
        mf[10] += b * c * y; // 1-2
        mf[11] += c * c * y; // 2-2
    }

    for (i = 0; i < 12; i++)
        atomicAdd(&sf[i], mf[i]);
    __syncthreads();

    if (threadIdx.x == 1)
    {
        for (i = 0; i < 12; i++)
            bdata[i][blockIdx.x] = sf[i];
    }
}

__global__ void UpdateSigmas(int updateOcc)
{
    if (updateOcc == 1)
        devOcc[0] = 0.0;

    for (int i = 0; i < 18; i++)
        devSigmas[i] = 0.0f;

    for (int i = 0; i < NBLOCKS; i++)
    {
        devSigmas[0] += bdata[0][i];
        devSigmas[1] += bdata[1][i];
        devSigmas[2] += bdata[2][i];
        devSigmas[3] += bdata[1][i];
        devSigmas[4] += bdata[3][i];
        devSigmas[5] += bdata[4][i];
        devSigmas[6] += bdata[2][i];
        devSigmas[7] += bdata[4][i];
        devSigmas[8] += bdata[5][i];

        devSigmas[9] += bdata[6][i];
        devSigmas[10] += bdata[7][i];
        devSigmas[11] += bdata[8][i];
        devSigmas[12] += bdata[7][i];
        devSigmas[13] += bdata[9][i];
        devSigmas[14] += bdata[10][i];
        devSigmas[15] += bdata[8][i];
        devSigmas[16] += bdata[10][i];
        devSigmas[17] += bdata[11][i];

        if (updateOcc == 1)
            devOcc[0] += bdata[12][i];
    }

    if (updateOcc == 1)
        devOcc[1] = devcols - devOcc[0];

    for (int i = 0; i < 18; i++)
    {
        if (i < 9)
            devSigmas[i] /= (devOcc[0] - 1);
        else
            devSigmas[i] /= (devOcc[1] - 1);
    }

    ComputeSigmaDetInv();
}

__global__ void ComputeError(tfloat *error)
{
    int i;
    error[0] = 0;
    for (i = 0; i < 6; i++)
        error[0] += abs(MeansOld[i] - devMeans[i]);

    for (int i = 0; i < 18; i++)
        error[0] += abs(SigmasOld[i] - devSigmas[i]);

    error[0] += (abs(pisOld[0] - pis[0]) + abs(pisOld[1] - pis[1]));
    error[0] *= 10000;
}

__global__ void discretizeOutput(unsigned char *devY, tfloat diffth)
{
    int iinit = blockDim.x * blockIdx.x + threadIdx.x, i;
    tfloat a, b;
	float diff = (float)diffth;
    a = pis[0];
    b = pis[1];

	diff = (diff == 100.0) ? 0.0 : diff;

    for (i = iinit; i < devcols; i += NMAINTHREADS)
    {
        if (a > b) // If first component is background
        {
            if ((devDist[0][i] + diff) < (devDist[1][i] - diff)) // Probability that the pixel is foreground is higher.
                devY[i] = 255;
            else
                devY[i] = 0;
        }
        else // If first component is foreground
        {
			if ((devDist[0][i] - diff) > (devDist[1][i] + diff)) // Probability that the pixel is foreground is higher.
                devY[i] = 255;
            else
                devY[i] = 0;
        }
    }
}

void PerformEMcuda(tfloat *_X, double **_Means, unsigned char *_Y, int _rows, int _cols, tfloat _fuzzifier, tfloat _diffth)
{
    int itr, maxitr = 100;
    tfloat Meansh[6], HMeans[6], HSigmas[18], *dist[2], *Xdiff[2][3], sum[3] = { 0.0, 0.0, 0.0 };
    tfloat *ptrdata, scale = 0.5;

    cudaSetDevice(0);

    rows = _rows;
    cols = _cols;
    ndim = rows;
    sigl = ndim * ndim;
    ncomp = 2;
	diffth = _diffth;

    ptrdata = (tfloat *)malloc(sizeof(tfloat) * cols * 8);
    
    // Memory allocation for input and output
    cudaMalloc((void**)&devXptr, rows * cols * sizeof(tfloat));
    cudaMalloc((void**)&devY, cols * sizeof(unsigned char));
    cudaMalloc((void**)&hptrs, sizeof(tfloat) * cols * 8);

    // Memory allocation for Means and Sigmas
    cudaMalloc((void**)&Means, sizeof(tfloat) * ndim * 2);
    
    cudaMalloc((void**)&Sigmas, sizeof(tfloat) * sigl * 2);
    
    cudaMalloc((void**)&Derror, sizeof(tfloat));
    Herror = (tfloat *)malloc(sizeof(tfloat));
    
    cudaMemcpy(devXptr, _X, rows * cols * sizeof(tfloat), cudaMemcpyHostToDevice);

    Meansh[0] = scale * ((tfloat)_Means[0][0]); Meansh[3] = scale * ((tfloat)_Means[0][1]);
    Meansh[1] = ((tfloat)_Means[1][0]); Meansh[4] = ((tfloat)_Means[1][1]);
    Meansh[2] = ((tfloat)_Means[2][0]); Meansh[5] = ((tfloat)_Means[2][1]);

    cudaMemcpy(Means, Meansh, 6 * sizeof(tfloat), cudaMemcpyHostToDevice);

    init <<< 1, 1 >>>(devXptr, hptrs, Means, Sigmas, cols, ndim, ncomp);
    cudaDeviceSynchronize();
    InitEM <<< NBLOCKS, NTHREADS >>>();
    cudaDeviceSynchronize();
    UpdateSigmas <<< 1, 1 >>>(1);
    cudaDeviceSynchronize();

    Herror[0] = 10000000;
    itr = 0;
    while (Herror[0] > 0.01 && itr < maxitr)
    {
        printf("Iteration = %d \t Error = %lf\n", itr, Herror[0]);

		//_sleep(100);

		if (_fuzzifier > 0.0)
			EStep <<< NBLOCKS, NTHREADS >>>(_fuzzifier);
		else
			EStep <<< NBLOCKS, NTHREADS >>>();
        cudaDeviceSynchronize();
        StoreOldValues <<< 1, 1 >>>();
        cudaDeviceSynchronize();
        UpdateMeansMain <<< NBLOCKS, NTHREADS >>>();
        cudaDeviceSynchronize();
        UpdateMeans <<< 1, 1 >>>();
        cudaDeviceSynchronize();
        UpdateSigmasMain <<< NBLOCKS, NTHREADS >>>();
        cudaDeviceSynchronize();
        UpdateSigmas <<< 1, 1 >>>(0);
        cudaDeviceSynchronize();
        ComputeError <<< 1, 1 >>>(Derror);
        cudaDeviceSynchronize();

        cudaMemcpy(Herror, Derror, sizeof(tfloat), cudaMemcpyDeviceToHost);

        itr++;
    }

    discretizeOutput<<< NBLOCKS, NTHREADS >>>(devY, diffth);
    cudaDeviceSynchronize();

    cudaMemcpy(_Y, devY, cols * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaFree(hptrs);
}





