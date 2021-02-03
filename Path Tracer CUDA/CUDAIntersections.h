#pragma once
#include "CUDARay.h"
#include "Scene.h"

typedef struct IntersectInfo {
    __host__ __device__
    IntersectInfo() : hitPoint(0), N(0) { }
    glm::vec3 hitPoint, N;
    Material hitMaterial;
} IntersectInfo;

__device__ __host__
bool sphereIntersection(const Ray& ray,const Sphere& sphere, float& t) {
    glm::vec3 oc = ray.origin - sphere.pos;
    float a = glm::dot(ray.dir, ray.dir);
    float b = glm::dot(oc, ray.dir);
    float c = glm::dot(oc, oc) - sphere.radius * sphere.radius;
    float discriminant = b * b - a * c;
    if (discriminant > 0) {
        float temp = (-b - glm::sqrt(discriminant)) / a;
        if (temp < FLT_MAX && temp > 0.001f) {
            t = temp;
            return true;
        }
        temp = (-b + glm::sqrt(discriminant)) / a;
        if (temp < FLT_MAX && temp > 0.001f) {
            t = temp;
            return true;
        }
    }
    return false;
}

__device__
bool planeIntersetion(const Ray& ray, const Plane& plane, float& t) {
    float denom = glm::dot(ray.dir, plane.N);
    if (fabs(denom) > 0.0001f) {
        t = (glm::dot(plane.pos - ray.origin, plane.N)) / denom;
        return t >= 0.0001f;
    }
    return false;
}

__device__
bool sceneIntersect(Scene* scene,const Ray& ray, IntersectInfo& info) {
    float maxDistance = 50.0f;
    float t;
    for (int i = 0; i < scene->numSpheres; i++) {
        if (sphereIntersection(ray, scene->spheres[i], t) && t < maxDistance) {
            info.hitPoint = ray.origin + ray.dir * t;
            info.N = scene->spheres[i].normal(info.hitPoint);
            info.hitMaterial = scene->materials[scene->spheres[i].matIndex];
            maxDistance = t;
        }
    }
    for (int i = 0; i < scene->numPlanes; i++) {
        if (planeIntersetion(ray, scene->planes[i], t) && t < maxDistance) {
            info.hitPoint = ray.origin + ray.dir * t;
            info.N = scene->planes[i].normal();
            info.hitMaterial = scene->materials[scene->planes[i].matIndex];
            maxDistance = t;
        }
    }
    return maxDistance < 50.0f;
}