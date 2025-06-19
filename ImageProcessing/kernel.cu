#include "utils.h"

unsigned char *d_red, *d_green, *d_blue; 
float *d_sum, *d_stddev, *d_interSum, *d_interStd, *d_fintensity, *d_tempCalc;

void cleanup();

__global__ void byteToFloat(float *out, unsigned char* in, int n)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	for (; i < n; i += gridDim.x * blockDim.x) {
		out[i] = in[i];
	}
}

__global__
void calcStdDevPart(const float* const d_in, float *d_out, float *d_avg, int numRows, int numCols) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (i >= numCols || j >= numRows) {
		return;
	}

	float avg = *d_avg / (numRows*numCols);

    d_out[j * numCols + i] = pow(d_in[j * numCols + i]-avg,2);
}

__global__
void findSum(const float* const d_in, float *d_out) {
	extern __shared__ float sdata[];

	int id = threadIdx.x + blockDim.x * blockIdx.x;
	int tid = threadIdx.x;

	sdata[tid] = d_in[id];

	__syncthreads();

	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}

		__syncthreads();
	}

	if (tid == 0) {
		d_out[blockIdx.x] = sdata[0];
	}
}

__global__
void calculateIntensity(const unsigned char* const rCh,
	const unsigned char* const gCh,
	const unsigned char* const bCh,
    unsigned char* outputInt,
	int numRows,
	int numCols) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (i >= numCols || j >= numRows)
	{
		return;
	}

	outputInt[j * numCols + i] = (0.2126 * rCh[j * numCols + i]) + (0.7152 * gCh[j * numCols + i]) + (0.0722 * bCh[j * numCols + i]);
}

__global__
void separateChannels(const uchar4* const inputImageRGBA,
	int numRows,
	int numCols,
	unsigned char* const redChannel,
	unsigned char* const greenChannel,
	unsigned char* const blueChannel)
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (i >= numCols || j >= numRows) {
		return;
	}

	uchar4 rgba = inputImageRGBA[j * numCols + i];
	redChannel[j * numCols + i] = rgba.x;
	greenChannel[j * numCols + i] = rgba.y;
	blueChannel[j * numCols + i] = rgba.z;
}

void allocateMemory(const size_t numRowsImage, const size_t numColsImage) {
	checkCudaErrors(cudaMalloc(&d_red, sizeof(unsigned char) * numRowsImage * numColsImage));
	checkCudaErrors(cudaMalloc(&d_green, sizeof(unsigned char) * numRowsImage * numColsImage));
	checkCudaErrors(cudaMalloc(&d_blue, sizeof(unsigned char) * numRowsImage * numColsImage));
	checkCudaErrors(cudaMalloc((void **) &d_fintensity, sizeof(float) * numRowsImage * numColsImage));
	checkCudaErrors(cudaMalloc(&d_stddev, sizeof(float)));
	checkCudaErrors(cudaMalloc(&d_sum, sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_tempCalc, sizeof(float)*numRowsImage*numColsImage));
	checkCudaErrors(cudaMalloc((void **)&d_interStd, sizeof(float)*numRowsImage*numColsImage));
	checkCudaErrors(cudaMalloc((void **) &d_interSum, sizeof(float)*numRowsImage*numColsImage));
}

void processImage(uchar4 * const d_inputImageRGBA,
	unsigned char *d_intensity,
	const size_t numRows, 
	const size_t numCols) {

	const dim3 blockSize(5, 5, 1);

	int bx = (numCols + blockSize.x - 1) / blockSize.x;
	int by = (numRows + blockSize.y - 1) / blockSize.y;

	const dim3 gridSize(bx,by,1);

	separateChannels<<<gridSize, blockSize>>>(d_inputImageRGBA, numRows, numCols, d_red, d_green, d_blue);

	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	calculateIntensity<<<gridSize, blockSize>>>(d_red, d_green, d_blue, d_intensity, numRows, numCols);
    
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	byteToFloat<<<gridSize, blockSize>>>(d_fintensity, d_intensity, numRows*numCols);

	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	const int size = numRows*numCols;
	const int maxThreadsPerBlock = 1024;
	int threads = maxThreadsPerBlock;
	int blocks = size / maxThreadsPerBlock;
	
	findSum<<<blocks, threads, threads * sizeof(float)>>>(d_fintensity, d_interSum);

	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	threads = blocks;
	blocks = 1;

	findSum<<<blocks, threads, threads * sizeof(float)>>>(d_interSum, d_sum);

	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	float h_out;
	checkCudaErrors(cudaMemcpy(&h_out, d_sum, sizeof(float), cudaMemcpyDeviceToHost));

	printf("Average: %0.2f\n", h_out/(numRows*numCols));

	calcStdDevPart<<<gridSize, blockSize>>>(d_fintensity, d_tempCalc, d_sum, numRows, numCols);

	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	threads = maxThreadsPerBlock;
	blocks = size / maxThreadsPerBlock;

	findSum<<<blocks, threads, threads * sizeof(float)>>>(d_tempCalc, d_interStd);

	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	threads = blocks;
	blocks = 1;

	findSum <<<blocks, threads, threads * sizeof(float)>>>(d_interStd, d_stddev);

	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	float h_out2;
	checkCudaErrors(cudaMemcpy(&h_out2, d_stddev, sizeof(float), cudaMemcpyDeviceToHost));

	printf("StdDev: %0.2f\n", sqrt(h_out2/(numRows*numCols)));

	cleanup();
}

void cleanup() {
	checkCudaErrors(cudaFree(d_red));
	checkCudaErrors(cudaFree(d_green));
	checkCudaErrors(cudaFree(d_blue));
	checkCudaErrors(cudaFree(d_sum));
	checkCudaErrors(cudaFree(d_stddev));
}