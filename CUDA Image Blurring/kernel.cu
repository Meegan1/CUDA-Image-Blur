
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cstdlib>
#include <vector>

#define CHECK(e) { int res = (e); if (res) printf("CUDA ERROR %d\n", res); }

#define CHANNEL 3
#define MAX_THREADS 1024

struct Image {
	int width;
	int height;
	unsigned int bytes;
	unsigned char* img;
	unsigned char* dev_img;
};

typedef int Color;
struct RGB
{
	Color rgb[3]{0};

	__device__
	RGB() = default;

	__device__ RGB(int r, int g, int b)
	{
		rgb[0] = r;
		rgb[1] = g;
		rgb[2] = b;
	}

	__device__ Color & operator[](int i)
	{
		return rgb[i];
	}

	__device__ RGB operator / (int num) { return { rgb[0] / num, rgb[1] / num, rgb[2] / num }; }
	__device__ RGB operator + (const RGB& color) { return { rgb[0] + color.rgb[0], rgb[1] + color.rgb[1], rgb[2] + color.rgb[2] }; }
};

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
cudaError_t addBlur(Image& source);
__device__ RGB calculateBlur(unsigned char* source, int u, int v, int width, int height, int gridSize);
__device__ RGB getRGBAtPosition(unsigned char* source, int u, int v, int width);
__device__ int getPosition(int u, int v, int width);
int readInpImg(const char* fname, Image& source, int& max_col_val);
int writeOutImg(const char* fname, const Image& roted, const int max_col_val);

__global__ void rgbKernel(unsigned char* dev_source, unsigned char* dev_image, int width, int height, int gridSize)
{

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int v = tid / width;
	int u = tid % width;
	
	int p = getPosition(u, v, width);
	RGB rgb = calculateBlur(dev_source, u, v, width, height, gridSize);
	dev_image[p] = rgb[0];
	dev_image[p + 1] = rgb[1];
	dev_image[p + 2] = rgb[2];
}

int main(int argc, char** argv)
{
	if (argc != 2)
	{
		printf("Usage: exec filename\n");
		exit(1);
	}
	char* fname = argv[1];

	//Read the input file
	Image source;
	int max_col_val;
	if (readInpImg(fname, source, max_col_val) != 0)  exit(1);


	// Complete the code
	addBlur(source);
	cudaError_t cudaStatus = addBlur(source);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addBlur failed!");
		return 1;
	}


	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	// Write the output file
	if (writeOutImg("roted.ppm", source, max_col_val) != 0) // For demonstration, the input file is written to a new file named "roted.ppm" 
		exit(1);

	free(source.img);
    return 0;
}

cudaError_t addBlur(Image &source)
{
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}

	int width = source.width;
	int height = source.height;
	int size = width * height * 3 * sizeof(unsigned char);
	
	unsigned char* dev_source;
	cudaMalloc((void**)&dev_source, size);
	
	unsigned char* dev_image;
	cudaMalloc((void**)&dev_image, size);

	cudaMemcpy(dev_source, source.img, size, cudaMemcpyHostToDevice);

	rgbKernel <<< (width*height)/MAX_THREADS, MAX_THREADS >>> (dev_source, dev_image, width, height, 3);

	unsigned char* test = (unsigned char*) malloc(size);
	cudaMemcpy(source.dev_img, dev_image, size, cudaMemcpyDeviceToHost);
	

	cudaDeviceSynchronize();


	cudaFree(dev_source);
	cudaFree(dev_image);
	return cudaStatus;
}

__device__
RGB calculateBlur(unsigned char* source, int u, int v, int width, int height, int gridSize)
{
	int p = getPosition(u, v, width);
	RGB rgb;

	int count = 0;
	int i_bound = gridSize / 2;
	int j_bound = gridSize / 2;
	for (int j = -j_bound; j <= j_bound; j++) {
		for (int i = -i_bound; i <= i_bound; i++)
		{
			if (u + i < 0 || u + i >= width || v + j < 0 || v + j >= height) // skip if past edge of image
				continue;
			
			rgb = rgb + getRGBAtPosition(source, u + i, v + j, width);
			count++;
		}
	}

	rgb = rgb / count;
	return rgb;
}

__device__
RGB getRGBAtPosition(unsigned char* source, int u, int v, int width)
{
	int p = getPosition(u, v, width);
	RGB rgb = RGB(source[p], source[p + 1], source[p + 2]);
	return rgb;
}

__device__
int getPosition(int u, int v, int width)
{
	return (u + (v * width)) * CHANNEL;
}

// Reads a color PPM image file (name provided), and

// saves data in the provided Image structure. 
// The max_col_val is set to the value read from the 
// input file. This is used later for writing output image. 
int readInpImg(const char* fname, Image& source, int& max_col_val) {

	FILE* src;

	if (!(src = fopen(fname, "rb")))
	{
		printf("Couldn't open file %s for reading.\n", fname);
		return 1;
	}

	char p, s;
	fscanf(src, "%c%c\n", &p, &s);
	if (p != 'P' || s != '6')   // Is it a valid format?
	{
		printf("Not a valid PPM file (%c %c)\n", p, s);
		exit(1);
	}

	fscanf(src, "%d %d\n", &source.width, &source.height);
	fscanf(src, "%d\n", &max_col_val);

	int pixels = source.width * source.height;
	source.bytes = pixels * 3;  // 3 => colored image with r, g, and b channels 
	source.img = (unsigned char*)malloc(source.bytes);
	source.dev_img = (unsigned char*)malloc(source.bytes);


	if (fread(source.img, sizeof(unsigned char), source.bytes, src) != source.bytes)
	{
		printf("Error reading file.\n");
		exit(1);
	}
	fclose(src);
	return 0;
}

// Write a color image into a file (name provided) using PPM file format.  
// Image structure represents the image in the memory. 
int writeOutImg(const char* fname, const Image& roted, const int max_col_val) {

	FILE* out;
	if (!(out = fopen(fname, "wb")))
	{
		printf("Couldn't open file for output.\n");
		return 1;
	}
	fprintf(out, "P6\n%d %d\n%d\n", roted.width, roted.height, max_col_val);
	if (fwrite(roted.dev_img, sizeof(unsigned char), roted.bytes, out) != roted.bytes)
	{
		printf("Error writing file.\n");
		return 1;
	}
	fclose(out);
	return 0;
}
