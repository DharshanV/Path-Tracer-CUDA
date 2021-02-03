#pragma once
#include <glm/glm.hpp>
#include <cuda_runtime.h>

typedef struct Sphere {
    __host__ __device__
    Sphere(glm::vec3 p, float r = 1.0f) :
            pos(p), radius(r), matIndex(0) { }

    __device__
    inline glm::vec3 normal(const glm::vec3& hitPoint) {
        return glm::normalize(hitPoint - pos);
    }

    glm::vec3 pos;
    float radius;
    uint16_t matIndex;
} Sphere;

typedef struct Plane {
    __host__ __device__
    Plane(glm::vec3 pos = glm::vec3(0), glm::vec3 N = glm::vec3(0),float w = 0,float h = 0) :
        pos(pos), N(N), width(w), height(h), matIndex(0) { }

    __device__
    inline glm::vec3 normal() {
        return N;
    }

    glm::vec3 pos, N;
    float width, height;
    uint16_t matIndex;
} Plane;