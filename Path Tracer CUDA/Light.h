#pragma once
#include <glm/glm.hpp>
#include <cuda_runtime.h>

typedef struct Light {
    __host__ __device__
    Light() : position(glm::vec3(0)), intensity(1){}

    __host__ __device__
    Light(glm::vec3 position = glm::vec3(0), float intensity = 0) :
        position(position), intensity(intensity) { }

    glm::vec3 position;
    float intensity;
} Light;