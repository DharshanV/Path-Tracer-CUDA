#pragma once
#include <glm/glm.hpp>
#include <cuda_runtime.h>

struct Material {
	__host__ __device__
	Material(glm::vec3 color = glm::vec3(1.0f),
			float isReflective = 0, float reflection = 0,
			float isRefractive = 0, float indexOfRefraction = 0) : color(color),
		isReflective(isReflective), reflection(reflection),
		isRefractive(isRefractive), indexOfRefraction(indexOfRefraction) {
	}

	glm::vec3 color;
	float isReflective;
	float reflection;
	float isRefractive;
	float indexOfRefraction;
	//float emittance;
	//float translucence;
};