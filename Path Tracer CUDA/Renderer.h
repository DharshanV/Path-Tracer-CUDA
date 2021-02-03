#pragma once
#include <vector>
#include <chrono>
#include <sstream>
#include <fstream>

#include "RenderKernel.h"
#include "CUDAHeaders.h"

class Renderer {
public:
	Renderer(int width,int height, const Camera& cam, int numOfThreads = 8) : camera(cam) {
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
		cudaFree(&scene.camera);
		cudaFree(&scene.lights);
		cudaFree(&scene.spheres);
		cudaFree(&scene.materials);
		cudaFree(dRandState);
	}

	void updateCamera(const Camera& cam) {
		this->camera = cam;
		cudaMemcpy(scene.camera, &camera, sizeof(Camera), cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();
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

	void render(const char* fileName,int samples) {
		commit();

		auto start = std::chrono::steady_clock::now();
		renderKernel KERNEL_ARG2(numOfBlocks, threadsPerBlock)(dImageData, samples, width, height, scene, dRandState);
		cudaDeviceSynchronize();
		auto end = std::chrono::steady_clock::now();
		elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		writeToFile(fileName);
	}

	void render(float* imageTexture,int samples) {
		commit();

		auto start = std::chrono::steady_clock::now();
		renderKernel KERNEL_ARG2(numOfBlocks, threadsPerBlock)(dImageData, samples, width, height, scene, dRandState);
		cudaDeviceSynchronize();
		auto end = std::chrono::steady_clock::now();
		elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

		writeToImage(imageTexture);
	}

	size_t getElapsed() { return elapsed; }
	
private:
	void commit() {
		if (commited) return;
		cudaMalloc(&dImageData, width * height * sizeof(glm::vec3));
		cudaMalloc(&scene.camera, sizeof(Camera));
		cudaMalloc(&scene.lights, lights.size() * sizeof(Light));
		cudaMalloc(&scene.spheres, spheres.size() * sizeof(Sphere));
		cudaMalloc(&scene.planes, planes.size() * sizeof(Plane));
		cudaMalloc(&scene.materials, materials.size() * sizeof(Material));

		scene.numSpheres = (int)spheres.size();
		scene.numLights = (int)lights.size();
		scene.numPlanes = (int)planes.size();

		cudaMemcpy(scene.camera, &camera, sizeof(Camera), cudaMemcpyHostToDevice);
		cudaMemcpy(scene.lights, &lights[0], lights.size() * sizeof(Light), cudaMemcpyHostToDevice);
		cudaMemcpy(scene.spheres, &spheres[0], spheres.size() * sizeof(Sphere), cudaMemcpyHostToDevice);
		cudaMemcpy(scene.planes, &planes[0], planes.size() * sizeof(Plane), cudaMemcpyHostToDevice);
		cudaMemcpy(scene.materials, &materials[0], materials.size() * sizeof(Material), cudaMemcpyHostToDevice);
		commited = true;
	}

	void writeToFile(const char* fileName) {
		glm::vec3* imageData = new glm::vec3[width * height];
		cudaMemcpy(imageData, dImageData, width * height * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

		if (imageData == nullptr)return;
		std::stringstream ss;
		ss << "P3\n" << width << ' ' << height << " 255\n";
		for (int j = 0; j < height; ++j) {
			for (int i = 0; i < width; ++i) {
				glm::vec3 pixel = imageData[i + j * width] * 255.0f;
				int r = (int)clamp(pixel[0], 0, 255);
				int g = (int)clamp(pixel[1], 0, 255);
				int b = (int)clamp(pixel[2], 0, 255);
				ss << r << ' ' << g << ' ' << b << '\n';
			}
		}
		std::ofstream out(fileName);
		out << ss.rdbuf();
		out.close();
		delete[] imageData;
	}

	void writeToImage(float* imageTexture) {
		cudaMemcpy(imageTexture, dImageData, width * height * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
	}

	float clamp(float value, float low, float high) {
		return std::max(low, std::min(value, high));
	}
private:
	int width;
	int height;
	glm::vec3* dImageData;
	curandState* dRandState;
	bool commited;

	Scene scene;
	Camera camera;
	size_t elapsed;

	dim3 numOfBlocks;
	dim3 threadsPerBlock;

	std::vector<Light> lights;
	std::vector<Sphere> spheres;
	std::vector<Plane> planes;
	std::vector<Material> materials;
};