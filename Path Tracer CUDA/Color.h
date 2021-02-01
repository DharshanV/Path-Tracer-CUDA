#pragma once
#include <cuda_runtime.h>
class Color {
public:
	__host__ __device__
	Color(float r = 0, float g = 0, float b = 0) : r(r), g(g), b(b) {}
	float r, g, b;
};