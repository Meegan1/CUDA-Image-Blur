
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cstdlib>
#include <vector>
#include <iostream>

#define CHECK(e) { int res = (e); if (res) printf("CUDA ERROR %d\n", res); }

#define CHANNEL 3
#define BLOCK_SIZE 16
#define GRID_RADIUS 3

struct Image {
	int width;
	int height;
	unsigned int bytes;
	unsigned char* img;
	unsigned char* dev_img;
};

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
cudaError_t addBlur(Image& source);
int readInpImg(const char* fname, Image& source, int& max_col_val);
int writeOutImg(const char* fname, const Image& roted, const int max_col_val);

__global__ void rgbKernel(unsigned char* dev_source, unsigned char* dev_image, int width, int height)
{
	int bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x, ty = threadIdx.y, bdx = blockDim.x, bdy = blockDim.y;

	// variables for accessing shared image (block co-ords)
	int row = by * (bdy) + ty;
	int col = bx * (bdx) + tx;
	int index = (ty * (bdx) + tx) * CHANNEL;

	if (row >= height + GRID_RADIUS || col >= width + GRID_RADIUS)
		return;

	// variables for accessing input image (screen co-ords)
	int sRow = row - GRID_RADIUS;
	int sCol = col - GRID_RADIUS;
	int src = (sRow * width + sCol) * CHANNEL;

	__shared__ unsigned char shared_source[(BLOCK_SIZE + (GRID_RADIUS * 2)) * (BLOCK_SIZE + (GRID_RADIUS * 2)) * 3]; // size with apron

	if (sCol >= 0 && sCol < width && sRow >= 0 && sRow < height)
	{
		shared_source[index] = dev_source[src];
		shared_source[index + 1] = dev_source[src + 1];
		shared_source[index + 2] = dev_source[src + 2];
	}
	else
	{
		shared_source[index] = 0;
		shared_source[index + 1] = 0;
		shared_source[index + 2] = 0;
	}

	__syncthreads();

	int r = 0, g = 0, b = 0;
	int count = 0;


	int cornerRow = ty - GRID_RADIUS/2;
	int cornerCol = tx - GRID_RADIUS/2;
	if(cornerRow >= 0 && cornerCol >= 0) {
		for (int i = -GRID_RADIUS; i < GRID_RADIUS; i++) {
			for (int j = -GRID_RADIUS; j < GRID_RADIUS; j++) {
				int filterRow = cornerRow + j;
				int filterCol = cornerCol + i;

				if (filterRow >= 0 && filterRow < height && filterCol >= 0 && filterCol < width)
				{
					r += shared_source[(filterRow * bdx + filterCol) * CHANNEL];
					g += shared_source[((filterRow * bdx + filterCol) * CHANNEL) + 1];
					b += shared_source[((filterRow * bdx + filterCol) * CHANNEL) + 2];
					count++;
				}
			}
		}
		dev_image[src] = r / count;
		dev_image[src + 1] = g / count;
		dev_image[src + 2] = b / count;
	}
	return;

	// u = tx + bx * (BLOCK_SIZE - (GRID_RADIUS * 2));
	// v = ty + by * (BLOCK_SIZE - (GRID_RADIUS * 2));


	for (int j = -GRID_RADIUS; j <= GRID_RADIUS; j++) {
		for (int i = -GRID_RADIUS; i <= GRID_RADIUS; i++)
		{
			// index = ((tx+i + GRID_RADIUS) + ((ty+j + GRID_RADIUS) * (BLOCK_SIZE * (GRID_RADIUS * 2)))) * 3;
			// index = ((tx+i) + ((ty+j) * (BLOCK_SIZE + (GRID_RADIUS * 2)))) * 3;

			// r += shared_source[index];
			// g += shared_source[index + 1];
			// b += shared_source[index + 2];
			
			// int index = ((u+i) + ((v+j) * width)) * CHANNEL;
			// r += dev_source[index];
			// g += dev_source[index + 1];
			// b += dev_source[index + 2];

			//
			// r += dev_source[src];
			// g += dev_source[src + 1];
			// b += dev_source[src + 2];
			count++;
		}
	}
	//
	// dev_image[src] = r / count;
	// dev_image[src + 1] = g / count;
	// dev_image[src + 2] = b / count;
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

	dim3 thread_size(BLOCK_SIZE + (GRID_RADIUS*2), BLOCK_SIZE + (GRID_RADIUS*2));
	dim3 block_size(ceil(width/BLOCK_SIZE), ceil(height/BLOCK_SIZE));

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	rgbKernel <<< block_size, thread_size >>> (dev_source, dev_image, width, height);

	cudaMemcpy(source.dev_img, dev_image, size, cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0);	
	cudaEventSynchronize(stop);
	float t = 0;
	cudaEventElapsedTime(&t, start, stop);
	std::cout << "Elapsed Time: " << t << std::endl;

	cudaEventDestroy(start);
	cudaEventDestroy(stop);


	cudaFree(dev_source);
	cudaFree(dev_image);
	return cudaStatus;
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
