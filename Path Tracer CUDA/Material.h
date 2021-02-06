#pragma once
#include <glm/glm.hpp>
#include <cuda_runtime.h>

struct Material {
	__host__ __device__
	Material(glm::vec3 color = glm::vec3(1.0f),
			bool isReflective = 0, float reflection = 0,
			bool isRefractive = 0, float indexOfRefraction = 0,bool isEmittance = 0) : color(color),
		isReflective(isReflective), reflection(reflection),
		isRefractive(isRefractive), indexOfRefraction(indexOfRefraction),
		isEmittance(isEmittance) {
	}

	glm::vec3 color;
	bool isReflective;
	bool isRefractive;
	bool isEmittance;

	float reflection;
	float indexOfRefraction;
	//float translucence;
};