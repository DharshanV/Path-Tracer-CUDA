#pragma once
#include <cuda_runtime.h>
#include <vector>
#include "Color.h"

class Image {
public:
	Image(int width, int height) : width(width), height(height) {
		cudaMallocManaged(&imageData, width * height * sizeof(Color));
	}

	virtual ~Image() { cudaFree(imageData); }

	virtual void writeToFile(const char* fileName) = 0;

	int index(int i, int j) { return i + j * width; }
public:
	int width;
	int height;
	Color* imageData;
};