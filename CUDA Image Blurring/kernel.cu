
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <string>

#define CHECK(e) { int res = (e); if (res) printf("CUDA ERROR %d\n", res); }

// RGB channels
#define CHANNEL 3

// struct for image
struct Image {
	int width;
	int height;
	unsigned int bytes;
	unsigned char* img;
	unsigned char* dev_img;
};

// headers
void addBlur(Image& source, int block_size, int grid_size);
int readInpImg(const char* fname, Image& source, int& max_col_val);
int writeOutImg(const char* fname, const Image& roted, const int max_col_val);

/*
 * Kernel for generating blur from an image
 */
__global__ void rgbKernel(unsigned char* dev_source, unsigned char* dev_image, int width, int height, int grid_radius)
{
	// get block info
	int bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x, ty = threadIdx.y, bdx = blockDim.x, bdy = blockDim.y;

	// variables for accessing shared image (block co-ords)
	int row = by * bdy + ty;
	int col = bx * bdx + tx;
	// variable for accessing input image (1d image co-ords)
	int src = (row * width + col) * CHANNEL;

	extern __shared__ unsigned char shared_source[]; // create shared variable for input image

	if (col >= 0 && col < width && row >= 0 && row < height)
	{
		int index = (ty * (bdx)+tx) * CHANNEL; // get index for 1d shared memory image
		shared_source[index] = dev_source[src];
		shared_source[index + 1] = dev_source[src + 1];
		shared_source[index + 2] = dev_source[src + 2];
	}

	__syncthreads();

	int r = 0, g = 0, b = 0; // total rgb
	int count = 0; // count number of additions
	for (int i = -grid_radius; i <= grid_radius; i++) {
		for (int j = -grid_radius; j <= grid_radius; j++) {
			int filter_row = ty + j;
			int filter_col = tx + i;

			// if outside bounds of shared image, fetch from global memory
			if (filter_col >= bdx || filter_row >= bdy || filter_col < 0 || filter_row < 0)
			{
				int y = by * bdy + filter_row;
				int x = bx * bdx + filter_col;

				if (x < 0 || x >= width || y < 0 || y >= height) // make sure bounds inside image
					continue;

				int index = (y * width + x) * CHANNEL;
				r += dev_source[index];
				g += dev_source[index + 1];
				b += dev_source[index + 2];
			}
			else { // otherwise retrieve rgb from shared memory
				int index = (filter_row * bdx + filter_col) * CHANNEL;
				r += shared_source[index];
				g += shared_source[index + 1];
				b += shared_source[index + 2];
			}
			count++;
		}
	}

	// calculate and set average rgb
	dev_image[src] = r / count;
	dev_image[src + 1] = g / count;
	dev_image[src + 2] = b / count;
}

int main(int argc, char** argv)
{
	if (argc != 4)
	{
		printf("Usage: exec filename block_size grid_size\n");
		exit(1);
	}
	char* fname = argv[1];
	int blocksize = std::stoi(argv[2]); // set number of threads in block (n x n)
	int gridsize = std::stoi(argv[3]); // set total width of grid for filter

	//Read the input file
	Image source;
	int max_col_val;
	if (readInpImg(fname, source, max_col_val) != 0)  exit(1);

	// Complete the code
	addBlur(source, blocksize, gridsize);

	cudaDeviceReset();

	// Write the output file
	if (writeOutImg("roted.ppm", source, max_col_val) != 0) // For demonstration, the input file is written to a new file named "roted.ppm" 
		exit(1);

	free(source.img);
    return 0;
}

void addBlur(Image &source, int block_size, int grid_size)
{
	// get/set image variables
	int width = source.width;
	int height = source.height;
	int size = width * height * 3 * sizeof(unsigned char);

	// allocate device memory
	unsigned char* dev_source;
	cudaMalloc((void**)&dev_source, size);
	
	unsigned char* dev_image;
	cudaMalloc((void**)&dev_image, size);

	// copy image to device
	cudaMemcpy(dev_source, source.img, size, cudaMemcpyHostToDevice);

	// define threads/blocks/block size
	dim3 n_threads(block_size, block_size);
	dim3 n_blocks(ceil(width/ block_size), ceil(height/ block_size));
	int shared_size = (block_size * block_size * CHANNEL);

	// record cuda event
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	// execute kernel
	rgbKernel <<< n_blocks, n_threads, shared_size >>> (dev_source, dev_image, width, height, grid_size/2);

	// copy output image from device to host
	cudaMemcpy(source.dev_img, dev_image, size, cudaMemcpyDeviceToHost);

	// stop recording cuda event
	cudaEventRecord(stop, 0);	
	cudaEventSynchronize(stop);
	float t = 0;
	cudaEventElapsedTime(&t, start, stop); // get elapsed time
	std::cout << "Elapsed Time: " << t << std::endl; // output time

	// clear up memory
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(dev_source);
	cudaFree(dev_image);
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
