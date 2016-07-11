#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <helper_functions.h> 
#include <helper_cuda.h>      
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/sort.h>

#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>

using namespace std;


#define checkCudaErrors(X) X
//const size_t M = 2560;
//const size_t M = 3200;
//const size_t M = 160;
//const size_t N = 600000;
//#define M 256
//#define N 100000

#define WARPS  32 
#define BLOCKDIMY  8
//#define BLOCKDIMY 16

#if 1
/*
测试256特征码1:N搜索
*/
const size_t M = 256;
//const size_t NBLOCKS = 128;
const size_t NBLOCKS = 24 * 6;
const size_t N = NBLOCKS*6*10000;

#else

/*
测试4096特征码1:N搜索
*/
const size_t M = 4096;
const size_t NBLOCKS = 24;
const size_t N = 50 * 10000;

#endif

//=========================================================
/*
 技术选型：
    用opencv Mat 实现矩阵运算！
*/

/*
	在N个特征码中进行相似性搜索
	相似度计算：
	     a*svm_score+(1-a)cos_score;
*/

class CFeaSim1vsN
{
public:
	struct IndexScore
	{
		float score;
		int index;
	};
private:
	static const size_t M = 90;
	static const size_t N = 2282211;
private:
	float *h_j1;
private:
	float *h_f2j1f2; //Vector(N)
	float *h_2j2f2;  //Matrix(M,N)
	float *h_normf2; //Matrix(M,N)
private:
	//GPU Device上的资源
	float *d_f2j1f2; //Vector(N)   
	float *d_2j2f2;  //Matrix(M,N)
	float *d_normf2; //Matrix(M,N)

	float *d_svm_score; //Vector(N)
	float *d_cos_score; //Vector(N)
private:
	float *d_f1;
	float *d_select_flags;
private:
	IndexScore *h_score;
	IndexScore *d_score;
private:
	cublasStatus_t cublas_status;
	cublasHandle_t cublas_handle;
public:
	CFeaSim1vsN(int max_num, float min_score);
private:
	void GPU_Init()
	{
		cublas_status = cublasCreate(&cublas_handle);
	}
	/*
		输入：
		   f1　　　　　 : 特征码, Vector(90)
		   min_score 　 : 大于此值得结果才会返回，[0,1], 默认0.5
		   maxN         : 最多返回多少个结果，maxN<1000， 默认100
		   select_flags : 计算结果过滤数组，select_flags=1.0 结果会被返回，0.0的会被抛弃， Vector(N)， 默认NULL，所有结果会被比较 
	   输出：
	      result: 保存返回结果
	   返回：
	      返回满足条件的结果个数

	*/
	int GPU_1vsN(float *f1, IndexScore *result, float min_score=0.5f, int maxN=100, float *select_flags=0);

	void GPU_end()
	{
		cublas_status = cublasDestroy(cublas_handle);
	}
};


int CFeaSim1vsN::GPU_1vsN(float *f1, IndexScore *result, float min_score, int maxN, float *select_flags)
{
	/*
	第零步:计算<f1,f1>和<f1,J1*f1>, 传入GPU Kernel
	第一步:计算SVM得分， 利用sgemv计算<f1, TwoJ2F2>, 然后根据已有公式进行计算进行计算
	第二步:计算COS得分， 利用sgemv计算<f1,normF2>
	第三步:计算得分： 0.8*SVM+0.2*COS, 如果有select_flags的话， 乘上select_flags
	第四步:thrust sort 全体排序
	第五步:CopyOut maxN个结果，用CPU判断需要返回的结果数目
	*/
}


namespace PCA
{
	const size_t M = 90;
	const size_t N = 10000;
	template<class T>
	T norm(int n, T *a,int inc=1)
	{
		T s = 0;
		T *p = a;
		for (int i = 0; i < n; i++, p += inc)
		{
			s += *p * *p;
		}

		return sqrt(s);
	}


	int test_gpu_score()
	{
		//前处理可以用fortran写，非常方便，也可以用matlab
		float *h_F1;
		float *h_J1; //J1
		float *h_J2;

		float *h_F2;
		float *h_J2xF2;  //J2*F2，  （N,M)矩阵， 预先算出来
		float *h_normF2; //F2/|F2|， (N,M)矩阵， 预先算出来
		float *h_F2t_dot_J1xF2; //F2'*J1*F2, （N)数组，预先算出来



		float h_1absF1; //1/|F1|, 需要传入GPU
		float *d_F1;
		float *d_normF2;
		float *d_F2t_dot_J1xF2;
		
		h_J1 = new float[M*M]; //J1(M,M), J1=J1'
		h_J2 = new float[M*M]; //J2(M,M), J2=J2'
		h_F1 = new float[M];
		h_F2 = new float[M];  //F2(M)

		h_J2xF2 = new float[N*M];
		h_normF2 = new float[N*M];
		h_F2t_dot_J1xF2 = new float[N];
		//假设从文件中读入的F2是M一组一组的，一共N组
		FILE *fp = fopen("f2.bin", "r+");
		for (int n = 0; n < N; ++n)
		{
			//读入一个F2进来

			size_t nread = fread(h_F2, sizeof(*h_F2), M, fp);
			if (nread != M)
			{
			}
			float nf2= 1/norm(M, h_F2);
			for (int m = 0; m < M; m++)
			{
				h_normF2[n + m*N] = h_F2[m] * nf2;
			}
			
		}
		fclose(fp);

		return(0);

	}

}

template<class T>
static inline __host__ __device__ T Op(T x, T y)
{
	return x*y;
	//T t = x - y;
	//return t*t;
}


//CXG, 20160527
/*
便于sort
*/
struct ScoreIndex
{
	float score; //[0,1]
	int index;
};

/*
   计算Y=A'*X
   A: real, dimension(M,N) ::A
   X: real, dimension(M)   ::X
   Y: real, dimension(N)   ::Y
   调用： sgemv<<<128,(32,8)>>>(A,X,Y)
*/
__global__ void 
sgemv4096(int M/*=4096*/, int N/*500000*/, float *d_A, float *d_X, float *d_Y)
{
	//int lda = 4096;
	//int loops = 16;// 4096 / 256;

	//---------------------------------------------
	//把d_Y清零
	int gid = threadIdx.x + threadIdx.y*blockDim.x+blockIdx.x*blockDim.x*blockDim.y;
	int nthreads = gridDim.x*blockDim.x*blockDim.y;
	for (int i = 0; i < N; i += nthreads)
	{
		int idx = gid + i;
		if (idx< N)
			d_Y[idx] = 0;
	}
	//---------------------------------------------
	int tidx = threadIdx.x;
	int tidy = threadIdx.y;

	int tid = tidx + tidy * 32;

	int dn = BLOCKDIMY * gridDim.x;
	int dA = dn*M;

	int index0 = BLOCKDIMY * blockIdx.x + tidy;
	int index1 = index0*M;
	
	//下标临时变量
	int tidx32 = tidx + 32;
	int tidx64 = tidx + 64;
	int tidx96 = tidx + 96;
	int tidx128 = tidx + 128;
	int tidx160 = tidx + 160;
	int tidx192 = tidx + 192;
	int tidx224 = tidx + 224;

	//每次处理256个数据进行点乘，将结果加到输出数据
	for (int iloop = 0; iloop < 16; iloop++, d_X+=256, d_A+=256)
	{
		//将d_X中的256个数据载入共享内存
		__shared__ float sharedX[256];
		
		sharedX[tid] = d_X[tid];
		
		__syncthreads();

		float *A = d_A;
		float *Y = d_Y;
		for (int n = BLOCKDIMY*blockIdx.x; n < N; n += dn, A += dA, Y += dn)
		{
			float *myA;
			myA = A + index1;

			float s = 0;
			s += Op(myA[tidx], sharedX[tidx]);
			s += Op(myA[tidx32], sharedX[tidx32]);
			s += Op(myA[tidx64], sharedX[tidx64]);
			s += Op(myA[tidx96], sharedX[tidx96]);
			s += Op(myA[tidx128], sharedX[tidx128]);
			s += Op(myA[tidx160], sharedX[tidx160]);
			s += Op(myA[tidx192], sharedX[tidx192]);
			s += Op(myA[tidx224], sharedX[tidx224]);

			s += __shfl_down(s, 16);
			s += __shfl_down(s, 8);
			s += __shfl_down(s, 4);
			s += __shfl_down(s, 2);
			s += __shfl_down(s, 1);
			if (tidx == 0)
				Y[index0] += s;
		}
	}
	return;
}

int test4096()
{
	//-------------------
	cublasStatus_t status;
	cublasHandle_t handle;
	status = cublasCreate(&handle);

	//-------------------
	float *d_A;
	float *d_X;
	float *d_Y;

	float *h_X;
	float *h_Y;
	h_X = (float *)malloc(M * sizeof(h_X[0]));
	h_Y = (float *)malloc(2 * N * sizeof(h_Y[0]));

	for (int i = 0; i < M; i++)
		h_X[i] = 0.1f;

	int m = M;
	int n = N;
	printf("m=%d, n=%d\n", m, n);

	cudaError_t err;
	err = cudaMalloc((void **)&d_A, sizeof(float)*N*M);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to malloc d_A (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	err = cudaMalloc((void **)&d_X, M* sizeof(float));
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to malloc d_X (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	err = cudaMalloc((void **)&d_Y, 2 * N* sizeof(float));
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to malloc d_Y (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	int *d_indices = (int*)(d_Y + N);
	//checkCudaErrors(cudaMalloc((void **)&d_indices, N* sizeof(int)));
	thrust::device_ptr<int> indices(d_indices);




	//printf("After cudaMalloc\n");


	//初始化矩阵和向量
	int seed = 777;
	curandGenerator_t prngGPU;
	curandCreateGenerator(&prngGPU, CURAND_RNG_PSEUDO_MTGP32);
	curandSetPseudoRandomGeneratorSeed(prngGPU, seed);

	curandGenerateUniform(prngGPU, (float *)d_A, M*N);
	curandGenerateUniform(prngGPU, (float *)d_X, M);

	//---------------------------------
	StopWatchInterface *hTimer;
	float alpha = 1.0f;
	float beta = 0.0f;

	dim3 grid(NBLOCKS);
	dim3 threads(WARPS, BLOCKDIMY);

	const int numIterations = 20;
	checkCudaErrors(cudaDeviceSynchronize());
	sdkCreateTimer(&hTimer);
	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);

	for (int iloop = 0; iloop < numIterations; iloop++)
	{

		//cudaMemcpy(d_X, h_X, M* sizeof(float), cudaMemcpyHostToDevice);

#if 0
		//status = cublasSgemv(handle, CUBLAS_OP_N, N, M, &alpha, d_A, N, d_X, 1, &beta, d_Y, 1);
		//status = cublasSgemv(handle, CUBLAS_OP_T, M, N, &alpha, d_A, M, d_X, 1, &beta, d_Y, 1);

		//验证结果用
		status = cublasSgemv(handle, CUBLAS_OP_T, M, 10000, &alpha, d_A, M, d_X, 1, &beta, d_Y, 1);

#else
		sgemv4096 <<<grid, threads >>>(M, N, d_A, d_X, d_Y);  //Y(N)=A(M,N)'*X(M)
#endif

		//--------------------------------------------------------------------
#if 1

#if 0
		thrust::counting_iterator<int> iter(0);
		thrust::copy(iter, iter + N, indices);
		thrust::device_ptr<float> keys(d_Y);
		thrust::sort_by_key(keys, keys + N, indices, thrust::greater<float>());

#endif

		cudaMemcpy(h_Y, d_Y, 2 * N* sizeof(float), cudaMemcpyDeviceToHost);
#else 
		cudaMemcpy(h_Y, d_Y, N* sizeof(float), cudaMemcpyDeviceToHost);


		int *index = (int*)(h_Y + N);
		for (int i = 0; i < N; i++)
			index[i] = i;
		thrust::sort_by_key(h_Y, h_Y + N, index, thrust::greater<float>());

#endif

	}

	checkCudaErrors(cudaDeviceSynchronize());
	sdkStopTimer(&hTimer);

	printf("After Sgemv4096\n");
	double gpuTime = 1.0e-3 * sdkGetTimerValue(&hTimer) / (double)numIterations;

	printf("Time = %.5f s\n", gpuTime);

	cudaMemcpy(h_Y, d_Y, N* sizeof(float), cudaMemcpyDeviceToHost);
	if (1)
	{
		for (int i = 0; i < 1000; i += 11)
			//for (int i = 0; i < 1000; i++)
			//for (int i = 0; i < 32; i++)
		{
			printf("h_Y[%d]=%g\n", i, h_Y[i]);
		}
	}
	checkCudaErrors(curandDestroyGenerator(prngGPU));
	checkCudaErrors(cudaFree(d_A));
	checkCudaErrors(cudaFree(d_X));
	checkCudaErrors(cudaFree(d_Y));

	free(h_X);
	free(h_Y);
	sdkDeleteTimer(&hTimer);

	/* Shutdown */
	status = cublasDestroy(handle);
	cudaDeviceReset();
	if (status != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "!!!! shutdown error (A)\n");
		return EXIT_FAILURE;
	}

	return(0);
}




__global__ void 
sgemv256(int M/*=256*/, int N, float *d_A, float *d_X, /*float*/ ScoreIndex *d_Y)
{

	/*------------------------------------*/
	//将X加载到Shared Memory
	__shared__ float sharedX[256];
	

	int tidx = threadIdx.x;
	int tidy = threadIdx.y;
	
	int tid = tidx + tidy*32;

	sharedX[tid] = d_X[tid];
	

	//20160527
	//---------------------------------------------
	//设置d_Y的Index
	int gid = tidx + tidy*blockDim.x + blockIdx.x*blockDim.x*blockDim.y;
	int nthreads = gridDim.x*blockDim.x*blockDim.y;
	//20160531
	for (int n = gid; n< N; n+= nthreads)
	{
		if (n< N)
			d_Y[n].index= n;
	}

	/*------------------------------------------*/
	__syncthreads();

	float *A = d_A;
	
	//20160527
	//float *Y = d_Y;
	ScoreIndex *Y = d_Y;

	int dn = BLOCKDIMY * gridDim.x;
	int dA = dn*M;

	int index0 = BLOCKDIMY * blockIdx.x + tidy;
	int index1 = index0*M;

	int tidx32= tidx + 32;
	int tidx64 = tidx + 64;
	int tidx96 = tidx + 96;
	int tidx128 = tidx + 128;
	int tidx160 = tidx + 160;
	int tidx192 = tidx + 192;
	int tidx224 = tidx + 224;

	/*
	//不用x0~..., 反而速度更快，寄存器有压力！
	float x0, x32, x64, x96, x128, x160, x192, x224;
	x0 =sharedX[tidx];
	x32=sharedX[tidx32];
	x64=sharedX[tidx64];
	x96=sharedX[tidx96];
	x128=sharedX[tidx128];
	x160=sharedX[tidx160];
	x192=sharedX[tidx192];
	x224=sharedX[tidx224];
	*/

	for (int n = BLOCKDIMY*blockIdx.x; n < N; n += dn, A+=dA, Y+=dn)
	//for (int n = 0; n < 32; n+=8, A+=dA)
	{
		float *myA;
		//myA = A + (blockIdx.x*8+tidy)*M;
		myA = A + index1;
		float s = 0;
		
#if 0
		s += myA[tidx] * sharedX[tidx];
		s += myA[tidx + 32] * sharedX[tidx + 32];
		s += myA[tidx + 64] * sharedX[tidx + 64];
		s += myA[tidx + 96] * sharedX[tidx + 96];
		s += myA[tidx + 128] * sharedX[tidx + 128];
		s += myA[tidx + 160] * sharedX[tidx + 160];
		s += myA[tidx + 192] * sharedX[tidx + 192];
		s += myA[tidx + 224] * sharedX[tidx + 224];
#else 
		
		/*
		s += myA[tidx] * sharedX[tidx];
		s += myA[tidx32] * sharedX[tidx32];
		s += myA[tidx64] * sharedX[tidx64];
		s += myA[tidx96] * sharedX[tidx96];
		s += myA[tidx128] * sharedX[tidx128];
		s += myA[tidx160] * sharedX[tidx160];
		s += myA[tidx192] * sharedX[tidx192];
		s += myA[tidx224] * sharedX[tidx224];
		*/

		s += Op(myA[tidx], sharedX[tidx]);
		s += Op(myA[tidx32],sharedX[tidx32]);
		s += Op(myA[tidx64],sharedX[tidx64]);
		s += Op(myA[tidx96],sharedX[tidx96]);
		s += Op(myA[tidx128],sharedX[tidx128]);
		s += Op(myA[tidx160],sharedX[tidx160]);
		s += Op(myA[tidx192],sharedX[tidx192]);
		s += Op(myA[tidx224],sharedX[tidx224]);

		/*
		s += Op(myA[tidx],   x0);
		s += Op(myA[tidx32], x32);
		s += Op(myA[tidx64], x64);
		s += Op(myA[tidx96], x96);
		s += Op(myA[tidx128], x128);
		s += Op(myA[tidx160], x160);
		s += Op(myA[tidx192], x192);
		s += Op(myA[tidx224], x224);
		*/
#endif
		

#if 0
		__shared__ float sharedAccum[BLOCKDIMY][WARPS];

		sharedAccum[tidy][tidx] = s;
		//__syncthreads();

		if (tidx < 16)
			sharedAccum[tidy][tidx] += sharedAccum[tidy][tidx + 16];
		
		//__syncthreads();

		if (tidx < 8)
			sharedAccum[tidy][tidx] += sharedAccum[tidy][tidx + 8];

		//__syncthreads();
		if (tidx < 4)
			sharedAccum[tidy][tidx] += sharedAccum[tidy][tidx + 4];
		
		//__syncthreads();

		if (tidx < 2)
			sharedAccum[tidy][tidx] += sharedAccum[tidy][tidx + 2];
#if 1
		//__syncthreads();
		if (tidx == 0)
		{
			//Y[8 * blockIdx.x + tidy] = sharedAccum[tidy][0] + sharedAccum[tidy][1];
			Y[index0] = sharedAccum[tidy][0] + sharedAccum[tidy][1];
			//d_Y[n + blockIdx.x*8+tidy] = sharedAccum[tidy][0] + sharedAccum[tidy][1];
			//d_Y[n + tidy] = 0.11111f;
		}
#endif		

#else
		//使用shuffle命令，进行求和
		s += __shfl_down(s, 16);
		s += __shfl_down(s, 8);
		s += __shfl_down(s, 4);
		s += __shfl_down(s, 2);
		s += __shfl_down(s, 1);
		if (tidx == 0)
			Y[index0].score = s; //Y[index0] = s;
#endif


		//__syncthreads();
	}

	

	return;
}

/* 
    Y(N)=A(N,M)*X(M)
*/
__global__ void
sgemv256_1(int M/*=256*/, int N, float *d_A, float *d_X, /*float*/ ScoreIndex *d_Y)
{
	__shared__ float X[256];
	
	int tid = threadIdx.x + threadIdx.y*blockDim.x;
	int gid = threadIdx.x + blockDim.x*(threadIdx.y* +blockIdx.x*blockDim.y);
	int nthread = blockDim.x*blockDim.y*gridDim.x;

	X[tid] = d_X[tid];

	for (int n = gid; n< N; n+= nthread)
	{
		if (n< N)
			d_Y[n].index = n;
	}
	__syncthreads();

	auto Y = d_Y;

	for (int n = gid; n < N; n +=nthread)
	{
		float s = 0;
		auto A = d_A;
		for (int m = 0; m < M; m++,A+=N)
		{
			s += X[m] * A[n];
		}

		Y[n].score= s;
	}

	return;
}


struct MyCompare{
    __host__ __device__
	bool operator()(const ScoreIndex & a, const ScoreIndex &b)
	{
		return  a.score < b.score;

	}
};
struct MyPred{
	__host__ __device__
	bool operator()(const ScoreIndex &a)
	{
		return a.score < 10.0f;
	}
};

#include <algorithm>

int test256()
{
	//-------------------
	cublasStatus_t status;
	cublasHandle_t handle;
	status = cublasCreate(&handle);

	//-------------------
	float *d_A;
	float *d_X;

	float *h_X;
	
	h_X = (float *)malloc(M * sizeof(h_X[0]));

	//float *d_Y;
	//float *h_Y;
	//h_Y = (float *)malloc(2*N * sizeof(h_Y[0]));
	
	//20160527
	ScoreIndex *d_Y;
	ScoreIndex *h_Y = (ScoreIndex*)malloc(N * sizeof(h_Y[0]));

	for (int i = 0; i < M; i++)
		h_X[i] = 0.1f;

	int m = M;
	int n = N;
	printf("m=%d, n=%d\n", m, n);

	cudaError_t err;
	err = cudaMalloc((void **)&d_A, sizeof(float)*N*M);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to malloc d_A (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	err=cudaMalloc((void **)&d_X, M* sizeof(float));
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to malloc d_X (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	
	//err=cudaMalloc((void **)&d_Y, 2*N* sizeof(float));
	err=cudaMalloc((void **)&d_Y, N* sizeof(*d_Y));
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to malloc d_Y (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	//checkCudaErrors(cudaMalloc((void **)&d_indices, N* sizeof(int)));
	
	//20160527
	//int *d_indices = (int*)(d_Y + N);
	//thrust::device_ptr<int> indices(d_indices);


	

	//printf("After cudaMalloc\n");


	//初始化矩阵和向量
	int seed = 777;
	curandGenerator_t prngGPU;
	curandCreateGenerator(&prngGPU, CURAND_RNG_PSEUDO_MTGP32);
	curandSetPseudoRandomGeneratorSeed(prngGPU, seed);

	curandGenerateUniform(prngGPU, (float *)d_A, M*N);
	curandGenerateUniform(prngGPU, (float *)d_X, M);

	//---------------------------------
	StopWatchInterface *hTimer;
	float alpha = 1.0f;
	float beta = 0.0f;

	dim3 grid(NBLOCKS);
	dim3 threads(WARPS, BLOCKDIMY);

	const int numIterations = 20;
	checkCudaErrors(cudaDeviceSynchronize());
	sdkCreateTimer(&hTimer);
	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);

	for (int iloop = 0; iloop < numIterations; iloop++)
	{

		cudaMemcpy(d_X, h_X, M* sizeof(float), cudaMemcpyHostToDevice);

#if 0
		//status = cublasSgemv(handle, CUBLAS_OP_N, N, M, &alpha, d_A, N, d_X, 1, &beta, d_Y, 1);
		//status = cublasSgemv(handle, CUBLAS_OP_T, M, N, &alpha, d_A, M, d_X, 1, &beta, d_Y, 1);

		//验证结果用
		//status = cublasSgemv(handle, CUBLAS_OP_T, M, 10000, &alpha, d_A, M, d_X, 1, &beta, d_Y, 1);
		status = cublasSgemv(handle, CUBLAS_OP_T, M, 10000, &alpha, d_A, M, d_X, 1, &beta, (float*)d_Y, 2);

#else
		//sgemv256 <<<grid, threads >>>(M,N,d_A,d_X,d_Y);  //Y(N)=A(M,N)'*X(M)
		sgemv256_1<<<grid, threads >>>(M,N,d_A,d_X,d_Y);  //Y(N)=A(N,M)*X(M)
#endif

		//--------------------------------------------------------------------
#if 1
#if 0
		thrust::counting_iterator<int> iter(0);
		//thrust::device_vector<int> indices(N);
		//thrust::copy(iter, iter + indices.size(), indices.begin());
		thrust::copy(iter, iter + N, indices);

		thrust::device_ptr<float> keys(d_Y);
		//thrust::sort(keys, keys + N);
		//thrust::sort_by_key(keys, keys + N, indices.begin(), thrust::greater<float>());
		thrust::sort_by_key(keys, keys + N, indices, thrust::greater<float>());

		//cudaMemcpy(h_Y, d_Y, N* sizeof(float), cudaMemcpyDeviceToHost);
		//cudaMemcpy(h_indices, d_indices, N*sizeof(int), cudaMemcpyDeviceToHost);
#endif

		//thrust::device_ptr<ScoreIndex> keys(d_Y);
		//thrust::sort(keys, keys + N, MyCompare());

		//partition和sort的速度差不多
		//thrust::partition(keys, keys + N,MyPred());
		
		cudaMemcpy(h_Y, d_Y,  N* sizeof(*h_Y), cudaMemcpyDeviceToHost);

		//速度更慢
		//std::nth_element(h_Y, h_Y + 100, h_Y + N, MyCompare());
#else 
		cudaMemcpy(h_Y, d_Y, N* sizeof(float), cudaMemcpyDeviceToHost);
		
		
		int *index = (int*)(h_Y + N);
		for (int i = 0; i < N; i++)
			index[i] = i;
		thrust::sort_by_key(h_Y, h_Y + N, index, thrust::greater<float>());

#endif

	}

	checkCudaErrors(cudaDeviceSynchronize());
	sdkStopTimer(&hTimer);

	//printf("After Sgemv\n");
	double gpuTime = 1.0e-3 * sdkGetTimerValue(&hTimer) / (double)numIterations;

	printf("Time = %.5f s\n",gpuTime);
	
	cudaMemcpy(h_Y, d_Y, N* sizeof(*h_Y), cudaMemcpyDeviceToHost);
	if (1)
	{
		for (int i = 0; i < 1000; i+=11)
		//for (int i = 0; i < 1000; i++)
		//for (int i = 0; i < 32; i++)
		{
			printf("h_Y[%d].score=%g, h_Y[%d].index=%d\n", i, h_Y[i].score, i, h_Y[i].index);
		}
	}
	checkCudaErrors(curandDestroyGenerator(prngGPU));
	checkCudaErrors(cudaFree(d_A));
	checkCudaErrors(cudaFree(d_X));
	checkCudaErrors(cudaFree(d_Y));

	free(h_X);
	free(h_Y);
	sdkDeleteTimer(&hTimer);

	/* Shutdown */
	status = cublasDestroy(handle);
	cudaDeviceReset();
	if (status != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "!!!! shutdown error (A)\n");
		return EXIT_FAILURE;
	}

	return(0);
}

int test_cublas()
{

	size_t rand_n = M*N;
	int seed = 777;

	cublasStatus_t status;
	cublasHandle_t handle;
	status = cublasCreate(&handle);

	if (status != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "!!!! CUBLAS initialization error\n");
		return EXIT_FAILURE;
	}
	float *d_A;
	float *d_X;
	float *d_Y;

	float *h_X;
	float *h_Y;
	h_X = (float *)malloc(M * sizeof(h_X[0]));
	h_Y = (float *)malloc(2*N * sizeof(h_Y[0]));
	//int *h_indices;
	//h_indices = (int*)malloc(N*sizeof(int));


	for (int i = 0; i < M; i++)
		h_X[i] = 0.1f;

	checkCudaErrors(cudaMalloc((void **)&d_X, M* sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_Y, 2*N* sizeof(float)));


	checkCudaErrors(cudaMalloc((void **)&d_A, rand_n * sizeof(float)));

	int *d_indices=(int*)(d_Y+N);
	//checkCudaErrors(cudaMalloc((void **)&d_indices, N* sizeof(int)));
	thrust::device_ptr<int> indices(d_indices);

	

	curandGenerator_t prngGPU;
	checkCudaErrors(curandCreateGenerator(&prngGPU, CURAND_RNG_PSEUDO_MTGP32));
	checkCudaErrors(curandSetPseudoRandomGeneratorSeed(prngGPU, seed));

	checkCudaErrors(curandGenerateUniform(prngGPU, (float *)d_A, rand_n));
	checkCudaErrors(curandGenerateUniform(prngGPU, (float *)d_X, M));

	const int numIterations = 20;

	StopWatchInterface *hTimer;
	float alpha = 1.0f;
	float beta = 0.0f;

	

	checkCudaErrors(cudaDeviceSynchronize());
	sdkCreateTimer(&hTimer);
	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);
	
	for (int i = 0; i < numIterations; i++)
	{
		cudaMemcpy(d_X, h_X, M* sizeof(float), cudaMemcpyHostToDevice);
		//------------------------------------------------------------------

		//checkCudaErrors(curandGenerateUniform(prngGPU, (float *)d_Rand, rand_n));
		status = cublasSgemv(handle, CUBLAS_OP_N, N, M, &alpha, d_A, N, d_X, 1, &beta, d_Y, 1);

		//这种更快！
		//status = cublasSgemv(handle, CUBLAS_OP_T, M, N, &alpha, d_A, M, d_X, 1, &beta, d_Y, 1);
		if (status != CUBLAS_STATUS_SUCCESS)
		{
			fprintf(stderr, "!!!! shutdown error (A)\n");
			return EXIT_FAILURE;
		}
		//----------------------------------------------------------------
		//cudaDeviceSynchronize();
		//---------------------------------------------
		thrust::counting_iterator<int> iter(0);
		//thrust::device_vector<int> indices(N);
		//thrust::copy(iter, iter + indices.size(), indices.begin());
		thrust::copy(iter, iter + N, indices);

		thrust::device_ptr<float> keys(d_Y);
		//thrust::sort(keys, keys + N);
		//thrust::sort_by_key(keys, keys + N, indices.begin(), thrust::greater<float>());
		thrust::sort_by_key(keys, keys + N, indices, thrust::greater<float>());

		//cudaMemcpy(h_Y, d_Y, N* sizeof(float), cudaMemcpyDeviceToHost);
		//cudaMemcpy(h_indices, d_indices, N*sizeof(int), cudaMemcpyDeviceToHost);

		cudaMemcpy(h_Y, d_Y, 2 * N* sizeof(float), cudaMemcpyDeviceToHost);
		if (0)
		{
			int *h_ind = (int*)(h_Y + N);
			for (int i = 0; i < 10; ++i)
			{
				printf("h_Y[%d]=%g, indice[%d]=%d\n", i,h_Y[i], i,h_ind[i]);
			}
		}
	}

	checkCudaErrors(cudaDeviceSynchronize());
	sdkStopTimer(&hTimer);

	double gpuTime = 1.0e-3 * sdkGetTimerValue(&hTimer) / (double)numIterations;

	printf("MersenneTwisterGP11213, Throughput = %.4f GNumbers/s, Time = %.5f s, Size = %u Numbers\n",
		1.0e-9 * rand_n / gpuTime, gpuTime, rand_n);


	printf("Shutting down...\n");

	checkCudaErrors(curandDestroyGenerator(prngGPU));
	checkCudaErrors(cudaFree(d_A));
	checkCudaErrors(cudaFree(d_X));
	checkCudaErrors(cudaFree(d_Y));
	//checkCudaErrors(cudaFree(d_indices));

	free(h_X);
	free(h_Y);
	//free(h_indices);
	sdkDeleteTimer(&hTimer);



	/* Shutdown */
	status = cublasDestroy(handle);
	cudaDeviceReset();
	if (status != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "!!!! shutdown error (A)\n");
		return EXIT_FAILURE;
	}
	return 0;
}
int
main(int argc, char **argv)
{

	//return test_cublas();
	return test256();
	//return test4096();
}


