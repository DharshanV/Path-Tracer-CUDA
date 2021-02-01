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
	vec3 hitPoint, N, dir(ray.dir);
	Material hitMaterial;
	Object* hitObj;

	if (depth > 5 || !scene->hit(ray, hitPoint, N, hitMaterial)) {
		return Color(1.0f, 0.5f, 0.2f);
	}
	return hitMaterial.diffuseColor;
}

__global__
void render(Color* imageData, int width, int height,Camera* camera,Scene** scene) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x >= width || y >= height) return;

	uint32_t index = x + y * width;
	float jitterValues[] = { 0.25f, 0.75f, -0.75f, 0.33f, 0.45f };
	vec3 pixelColor;
	for (int i = 0; i < 4; i++) {
		float pixelX = ((float)x + jitterValues[i]) / width;
		float pixelY = ((float)y + jitterValues[i+1]) / height;
		pixelColor += castRay(camera->getRay(pixelX, pixelY), *scene).RGB;
	}
	imageData[index] = pixelColor / 4;
}

__global__
void createScene(Scene** scene) {
	if (threadIdx.x != 0 || blockIdx.x != 0) return;

	Material* red = new Material();
	red->diffuseColor = vec3(1, 0, 0);

	Material* green = new Material();
	green->diffuseColor = vec3(0, 1, 0);


	const int OBJECT_SIZE = 2;
	Object** objects = new Object * [OBJECT_SIZE];
	objects[0] = new Sphere(vec3(2, 0, -5),red);
	objects[1] = new Sphere(vec3(-2, 0, -5),green);
	*scene = new Scene(objects, OBJECT_SIZE);
}