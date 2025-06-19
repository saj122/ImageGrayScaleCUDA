#include <stdio.h>

#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\opencv.hpp>

#include "utils.h"

cv::Mat imageInputRGBA;

uchar4 *d_inputImageRGBA_;

void processImage(uchar4 * const d_inputImageRGBA,
	unsigned char *d_intensity,
	const size_t numRows,
	const size_t numCols);

void allocateMemory(const size_t numRowsImage, const size_t numColsImage);

void preProcess(uchar4 **h_inputImageRGBA,
	uchar4 **d_inputImageRGBA,
	unsigned char **d_intensity,
	const std::string &filename) {

	checkCudaErrors(cudaFree(0));

	cv::Mat image = cv::imread(filename.c_str(), CV_LOAD_IMAGE_COLOR);
	if (image.empty()) {
		std::cerr << "Couldn't open file: " << filename << std::endl;
		exit(1);
	}

	cv::imshow("RGB", image);

	cv::cvtColor(image, imageInputRGBA, CV_BGR2RGBA);

	if (!imageInputRGBA.isContinuous()) {
		std::cerr << "Images aren't continuous!! Exiting." << std::endl;
		exit(1);
	}

	*h_inputImageRGBA = (uchar4 *)imageInputRGBA.ptr<unsigned char>(0);

	const size_t numPixels = imageInputRGBA.rows*imageInputRGBA.cols;

	checkCudaErrors(cudaMalloc(d_inputImageRGBA, sizeof(uchar4) * numPixels));

	checkCudaErrors(cudaMemcpy(*d_inputImageRGBA, *h_inputImageRGBA, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice));

	d_inputImageRGBA_ = *d_inputImageRGBA;

	checkCudaErrors(cudaMalloc(d_intensity, sizeof(unsigned char) * numPixels));
	checkCudaErrors(cudaMemset(*d_intensity, 0, sizeof(unsigned char) * numPixels));
}

int main(int argc, char** argv) {
	int deviceCount, device;
	int gpuDeviceCount = 0;

	if (argc < 2) {
		printf("Input path of image.\n");
		return 1;
	}

	struct cudaDeviceProp properties;

	cudaError_t cudaResultCode = cudaGetDeviceCount(&deviceCount);

	if (cudaResultCode != cudaSuccess) {
		deviceCount = 0;
	}

	for (device = 0; device < deviceCount; ++device) {
		cudaGetDeviceProperties(&properties, device);
		if (properties.major != 9999) {
			++gpuDeviceCount;
		}
	}

	printf("%d GPU CUDA device(s) found.\n", gpuDeviceCount);

	if (gpuDeviceCount == 0) {
		return 1;
	}

	uchar4 *h_inRGBA, *d_inRGBA;

	unsigned char *d_intensity;

	std::string input_file = std::string(argv[1]);

	preProcess(&h_inRGBA, &d_inRGBA, &d_intensity, input_file);

	cv::namedWindow("RGB");

	allocateMemory(imageInputRGBA.rows, imageInputRGBA.cols);

	processImage(d_inRGBA, d_intensity, imageInputRGBA.rows, imageInputRGBA.cols);

	unsigned char* img;
	checkCudaErrors(cudaMemcpy(img, d_intensity, sizeof(unsigned char)*imageInputRGBA.rows*imageInputRGBA.cols, cudaMemcpyDeviceToHost));

	cv::Mat output(imageInputRGBA.rows, imageInputRGBA.cols, CV_8UC1, (void*)img);

	cv::namedWindow("CUDA");

	cv::imshow("CUDA", output);

	cv::waitKey(0);

	cudaFree(d_inputImageRGBA_);
	cudaFree(d_intensity);

	return 0;
}