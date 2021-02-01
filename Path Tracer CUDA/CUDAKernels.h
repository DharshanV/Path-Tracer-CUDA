#pragma once
#include <cuda_runtime.h>
#include "CUDAHeader.h"
#include "Color.h"
#include "Sphere.h"
#include "Camera.h"
#include "Scene.h"
#include "Ray.h"
#include <vector>

__device__
Color castRay(const Ray& ray, Scene* scene, int depth = 0) {
	float t;
	Object* hitObj;
	if (scene->hit(ray, hitObj, t)) {
		return Color(0, 255, 0);
	}
	return Color(255, 127 , 51);
}

__global__
void render(Color* imageData, int width, int height,Camera* camera,Scene** scene) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x >= width || y >= height) return;

	uint32_t index = x + y * width;
	imageData[index] = castRay(camera->getRay((float)x/width, (float)y/height), *scene);
}

__global__
void createScene(Scene** scene) {
	if (threadIdx.x != 0 || blockIdx.x != 0) return;

	const int OBJECT_SIZE = 2;
	Object** objects = new Object * [OBJECT_SIZE];
	objects[0] = new Sphere(vec3(2, 0, -5));
	objects[1] = new Sphere(vec3(-2, 0, -5));
	*scene = new Scene(objects, OBJECT_SIZE);
}