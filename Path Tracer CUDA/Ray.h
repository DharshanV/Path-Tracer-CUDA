#pragma once
#include "vec3.h"

class Ray {
public:
	__host__ __device__
	Ray(const vec3& origin, const vec3& dir) : origin(origin), dir(dir){ }

	__host__ __device__
	vec3 operator ()(float t) {
		return origin + dir * t;
	}
	vec3 origin;
	vec3 dir;
};