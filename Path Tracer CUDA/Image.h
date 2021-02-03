#pragma once
#include <cuda_runtime.h>
#include <glm/glm.hpp>
class Image {
public:
	Image(int width, int height) : width(width), height(height) {
		cudaMallocManaged(&imageData, (size_t)width * height * sizeof(glm::vec3));
	}

	virtual ~Image() { cudaFree(imageData); }

	virtual void writeToFile(const char* fileName) = 0;

	int width;
	int height;
	glm::vec3* imageData;
};