#pragma once
#include "vec3.h"
class Color {
public:
	__host__ __device__
	Color(float r = 0, float g = 0, float b = 0) : RGB(r,g,b) {}

	__host__ __device__
	Color(const vec3& RGB) : RGB(RGB) {}

	vec3 RGB;
};