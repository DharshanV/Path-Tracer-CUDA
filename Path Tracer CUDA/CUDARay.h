#pragma once
#include <glm/glm.hpp>
#include <cuda_runtime.h>

struct Ray {
	__host__ __device__
	Ray(const glm::vec3& origin, const glm::vec3& dir) : origin(origin), dir(dir) {}

	__host__ __device__
		glm::vec3 operator ()(float t) {
		return origin + dir * t;
	}
	glm::vec3 origin;
	glm::vec3 dir;
};