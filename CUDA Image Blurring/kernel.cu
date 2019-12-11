
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cstdlib>
#include <vector>

#define CHECK(e) { int res = (e); if (res) printf("CUDA ERROR %d\n", res); }

#define CHANNEL 3

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

	RGB() = default;

	RGB(int r, int g, int b)
	{
		rgb[0] = r;
		rgb[1] = g;
		rgb[2] = b;
	}

	Color & operator[](int i)
	{
		return rgb[i];
	}

	RGB operator / (int num) { return { rgb[0] / num, rgb[1] / num, rgb[2] / num }; }
	RGB operator + (const RGB& color) { return { rgb[0] + color.rgb[0], rgb[1] + color.rgb[1], rgb[2] + color.rgb[2] }; }
};

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
void addBlur(Image& source);
RGB calculateBlur(Image& source, int u, int v);
RGB getRGBAtPosition(Image& source, int u, int v);
int getPosition(Image& source, int u, int v);
int readInpImg(const char* fname, Image& source, int& max_col_val);
int writeOutImg(const char* fname, const Image& roted, const int max_col_val);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
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


	// Write the output file
	if (writeOutImg("roted.ppm", source, max_col_val) != 0) // For demonstration, the input file is written to a new file named "roted.ppm" 
		exit(1);

	free(source.img);

	exit(0);
	
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

void addBlur(Image &source)
{
	for (int v = 0; v < source.height; v++)
	{
		for(int u = 0; u < source.width; u++)
		{
			unsigned char r, g, b;
			int p = getPosition(source, u, v);
			r = source.img[p];
			g = source.img[p + 1];
			b = source.img[p + 2];

			RGB rgb = calculateBlur(source, u, v);
			source.dev_img[p] = rgb[0];
			source.dev_img[p+1] = rgb[1];
			source.dev_img[p+2] = rgb[2];
		}
	}
}

RGB calculateBlur(Image& source, int u, int v)
{
	int p = getPosition(source, u, v);
	RGB rgb;

	int r = 0, g = 0, b = 0;
	int count = 0;
	for (int j = -1; j <= 1; j++) {
		for (int i = -1; i <= 1; i++)
		{
			if (u + i < 0 || u + i >= source.width || v + j < 0 || v + j >= source.height)
				continue;
			
			rgb = rgb + getRGBAtPosition(source, u + i, v + j);
			count++;
		}
	}

	rgb = rgb / count;
	return rgb;
}

RGB getRGBAtPosition(Image &source, int u, int v)
{
	int p = getPosition(source, u, v);
	RGB rgb = RGB(source.img[p], source.img[p + 1], source.img[p + 2]);
	return rgb;
}

int getPosition(Image& source, int u, int v)
{
	return (u + (v * source.width)) * CHANNEL;
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

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
