#pragma once
#include <glm/glm.hpp>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include "CUDAIntersections.h"
#include "Settings.h"

__device__ float schlick(float cosine, float ref_idx) {
    float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * pow((1.0f - cosine), 5.0f);
}

__device__ bool refract(const glm::vec3& v, const glm::vec3& n, float ni_over_nt, glm::vec3& refracted) {
    glm::vec3 uv = glm::normalize(v);
    float dt = glm::dot(uv, n);
    float discriminant = 1.0f - ni_over_nt * ni_over_nt * (1 - dt * dt);
    if (discriminant > 0) {
        refracted = ni_over_nt * (uv - n * dt) - n * sqrt(discriminant);
        return true;
    } else
        return false;
}

__device__
inline glm::vec3 randomInHemisphere(curandState* localRandState) {
    glm::vec3 p;
    do {
        glm::vec3 randVec(curand_uniform(localRandState), curand_uniform(localRandState), curand_uniform(localRandState));
        p = (2.0f * randVec) - glm::vec3(1.0f);
    } while (glm::length(p) >= 1.0f);
    return p;
}

__device__
bool scatter(const Ray& ray, const IntersectInfo& rec, glm::vec3& attenuation, Ray& scattered, curandState* local_rand_state) {
    const Material& mat = rec.hitMaterial;

    if (mat.isReflective) {
        glm::vec3 reflected = glm::reflect(glm::normalize(ray.dir), rec.N);
        scattered = Ray(rec.hitPoint, reflected + mat.reflection * randomInHemisphere(local_rand_state));
        attenuation = mat.color;
        return (glm::dot(scattered.dir, rec.N) > 0.0f);
    }
    else if (mat.isRefractive) {
        glm::vec3 outward_normal;
        glm::vec3 reflected = glm::reflect(ray.dir, rec.N);
        float ni_over_nt;
        attenuation = glm::vec3(1.0, 1.0, 1.0);
        glm::vec3 refracted;
        float reflect_prob;
        float cosine;
        if (glm::dot(ray.dir, rec.N) > 0.0f) {
            outward_normal = -rec.N;
            ni_over_nt = mat.indexOfRefraction;
            cosine = glm::dot(ray.dir, rec.N) / glm::length(ray.dir);
            cosine = sqrt(1.0f - mat.indexOfRefraction * mat.indexOfRefraction * (1 - cosine * cosine));
        } else {
            outward_normal = rec.N;
            ni_over_nt = 1.0f / mat.indexOfRefraction;
            cosine = -glm::dot(ray.dir, rec.N) / glm::length(ray.dir);
        }
        if (refract(ray.dir, outward_normal, ni_over_nt, refracted))
            reflect_prob = schlick(cosine, mat.indexOfRefraction);
        else
            reflect_prob = 1.0f;
        if (curand_uniform(local_rand_state) < reflect_prob)
            scattered = Ray(rec.hitPoint, reflected);
        else
            scattered = Ray(rec.hitPoint, refracted);
        return true;
    }
    else if (mat.emittance > 0.0f) {
        glm::vec3 target = rec.hitPoint + rec.N + randomInHemisphere(local_rand_state);
        attenuation *= mat.color * mat.emittance;
        scattered = Ray(rec.hitPoint, target - rec.hitPoint);
        return true;
    }
    else {
        glm::vec3 target = rec.hitPoint + rec.N + randomInHemisphere(local_rand_state);
        scattered = Ray(rec.hitPoint, target - rec.hitPoint);
        attenuation = mat.color;
        return true;
    }
    return false;
}

__device__
glm::vec3 castRay(const Ray& ray, Scene* scene, curandState* localRandState) {
    Ray curRay = ray;
    glm::vec3 cur_attenuation(1.0f);
    for (int i = 0; i < MAX_DEPTH; i++) {
        IntersectInfo hit;
        if (sceneIntersect(scene, curRay, hit)) {
            Ray scattered(glm::vec3(0), glm::vec3(0));
            glm::vec3 attenuation;
            if (scatter(curRay, hit, attenuation, scattered, localRandState)) {
                cur_attenuation *= attenuation;
                curRay = scattered;
            } else {
                return glm::vec3(0.0f);
            }
        } else {
            glm::vec3 unit_direction = glm::normalize(curRay.dir);
            float t = 0.5f * (unit_direction.y + 1.0f);
            glm::vec3 c = (1.0f - t) * glm::vec3(1.0f) + t * glm::vec3(0.3f);
            return cur_attenuation * c;
        }
    }
    return glm::vec3(0.0f);
}

__global__
void renderKernel(glm::vec3* imageData, int samples, int width, int height, Scene scene, curandState* randState) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x >= width || y >= height) return;

	uint32_t index = x + y * width;
	Camera* camera = scene.camera;
    curandState localRandState = randState[index];

    glm::vec3 pixelColor(0.0f);
    for (int i = 0; i < samples; i++) {
        float u = ((float)x + curand_uniform(&localRandState)) / width;
        float v = ((float)y + curand_uniform(&localRandState)) / height;
        pixelColor += castRay(camera->getRay(u, v), &scene, &localRandState);
    }
    randState[index] = localRandState;
    imageData[index] = glm::sqrt(pixelColor/ (float)samples);
}

__global__
void renderInit(int width, int height, curandState* randState) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= width || y >= height) return;

    uint32_t index = x + y * width;
    curand_init(1984, index, 0, &randState[index]);
}