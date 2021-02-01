#pragma once
#include <iostream>
#include <chrono>

#include "PPM.h"
#include "CUDAKernels.h"
#include "Scene.h"

class Renderer {
public:
	Renderer(Image* image,Camera* hCamera, int numOfThreads = 8) : image(image) {
		this->width = image->width;
		this->height = image->height;

		threadsPerBlock = dim3(numOfThreads, numOfThreads);
		numOfBlocks = dim3(width / numOfThreads, height / numOfThreads);

		//create camera
		cudaMalloc(&dCamera, sizeof(Camera));
		cudaMemcpy(dCamera, hCamera, sizeof(Camera), cudaMemcpyHostToDevice);

		//create scene
		cudaMalloc(&scene, sizeof(Scene*));
		createScene KERNEL_ARG2(1, 1)(scene);
	}

	~Renderer() {
		cudaFree(dCamera);
		cudaFree(scene);
	}


	void run(const char* fileName) {
		auto t1 = std::chrono::high_resolution_clock::now();
		render KERNEL_ARG2(numOfBlocks, threadsPerBlock)(image->imageData, width, height, dCamera, scene);
		cudaDeviceSynchronize();
		image->writeToFile(fileName);
		auto t2 = std::chrono::high_resolution_clock::now();
		elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
	}

	uint64_t getElapsed() { return elapsed; }

private:

private:
	Image* image;

	int width;
	int height;
	uint64_t elapsed;

	dim3 numOfBlocks;
	dim3 threadsPerBlock;

	Camera* dCamera;
	Scene** scene;
};