#pragma once
#include <vector>
#include <chrono>
#include <sstream>
#include <fstream>
#include "RenderKernel.h"
#include "CUDAHeaders.h"

class Renderer {
public:
	Renderer(int width,int height, int numOfThreads = 8) {
		this->width = width;
		this->height = height;
		commited = false;
		threadsPerBlock = dim3(numOfThreads, numOfThreads);
		numOfBlocks = dim3(width / numOfThreads, height / numOfThreads);

		//create random states
		cudaMalloc(&dRandState, width * height * sizeof(curandState));
		renderInit KERNEL_ARG2(numOfBlocks, threadsPerBlock)(width, height, dRandState);
		cudaDeviceSynchronize();
	}

	~Renderer() {
		cudaFree(dImageData);
		cudaFree(&scene.lights);
		cudaFree(&scene.spheres);
		cudaFree(&scene.materials);
		cudaFree(dRandState);
	}
	
	void addSphere(Sphere s,Material mat) {
		s.matIndex = (int)materials.size();
		spheres.push_back(s);
		materials.push_back(mat);
	}

	void addPlane(Plane p, Material mat) {
		p.matIndex = (int)materials.size();
		planes.push_back(p);
		materials.push_back(mat);
	}

	void addLight(Light l) {
		lights.push_back(l);
	}

	void render(cudaSurfaceObject_t surfaceObj,const Camera& camera, int samples, bool globalLight) {
		commit();

		auto start = std::chrono::steady_clock::now();
		renderKernel KERNEL_ARG2(numOfBlocks, threadsPerBlock)(surfaceObj, camera, samples, width, height, scene, dRandState, globalLight);
		cudaDeviceSynchronize();
		auto end = std::chrono::steady_clock::now();
		elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	}

	size_t getElapsed() { return elapsed; }
	
private:
	void commit() {
		if (commited) return;
		cudaMalloc(&dImageData, width * height * sizeof(glm::vec3));
		cudaMalloc(&scene.lights, lights.size() * sizeof(Light));
		cudaMalloc(&scene.spheres, spheres.size() * sizeof(Sphere));
		cudaMalloc(&scene.planes, planes.size() * sizeof(Plane));
		cudaMalloc(&scene.materials, materials.size() * sizeof(Material));

		scene.numSpheres = (int)spheres.size();
		scene.numLights = (int)lights.size();
		scene.numPlanes = (int)planes.size();

		cudaMemcpy(scene.lights, &lights[0], lights.size() * sizeof(Light), cudaMemcpyHostToDevice);
		cudaMemcpy(scene.spheres, &spheres[0], spheres.size() * sizeof(Sphere), cudaMemcpyHostToDevice);
		cudaMemcpy(scene.planes, &planes[0], planes.size() * sizeof(Plane), cudaMemcpyHostToDevice);
		cudaMemcpy(scene.materials, &materials[0], materials.size() * sizeof(Material), cudaMemcpyHostToDevice);
		commited = true;
	}

private:
	int width;
	int height;
	glm::vec3* dImageData;
	curandState* dRandState;
	bool commited;

	Scene scene;
	size_t elapsed;

	dim3 numOfBlocks;
	dim3 threadsPerBlock;

	std::vector<Light> lights;
	std::vector<Sphere> spheres;
	std::vector<Plane> planes;
	std::vector<Material> materials;
};