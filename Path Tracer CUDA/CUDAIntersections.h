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
        if (temp < FLT_MAX && temp > EPSILON) {
            t = temp;
            return true;
        }
        //temp = (-b + glm::sqrt(discriminant)) / a;
        //if (temp < FLT_MAX && temp > EPSILON) {
        //    t = temp;
        //    return false;
        //}
    }
    return false;
}

__device__
bool planeIntersetion(const Ray& ray, const Plane& plane, float& t) {
    float denom = glm::dot(ray.dir, plane.N);
    if (-denom > EPSILON) {
        t = (glm::dot(plane.pos - ray.origin, plane.N)) / denom;
        return t >= EPSILON;
    }
    return false;
}

__device__
bool triangleIntersetion(const Ray& ray, const Triangle& triangle, float& t) {
    glm::vec3 v0 = triangle.vertices[0].position;
    glm::vec3 v1 = triangle.vertices[1].position;
    glm::vec3 v2 = triangle.vertices[2].position;
    glm::vec3 edge1, edge2, h, s, q;
    float a, f, u, v;
    edge1 = v1 - v0;
    edge2 = v2 - v0;
    h = glm::cross(ray.dir, edge2);
    a = glm::dot(edge1, h);
    if (a < EPSILON)
        return false;
    f = 1.0f / a;
    s = ray.origin - v0;
    u = f * glm::dot(s, h);
    if (u < 0.0f || u > 1.0f)
        return false;
    q = glm::cross(s, edge1);
    v = f * glm::dot(ray.dir, q);
    if (v < 0.0f || u + v > 1.0f)
        return false;

    t = f * glm::dot(edge2, q);
    return t > EPSILON;
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
            info.N = scene->planes[i].N;
            info.hitMaterial = scene->materials[scene->planes[i].matIndex];
            maxDistance = t;
        }
    }
    for (int i = 0; i < scene->numTriangles; i++) {
        if (triangleIntersetion(ray, scene->triangles[i], t) && t < maxDistance) {
            info.hitPoint = ray.origin + ray.dir * t;
            info.N = scene->triangles[i].N;
            info.hitMaterial = scene->materials[scene->triangles[i].matIndex];
            maxDistance = t;
        }
    }
    return maxDistance < 50.0f;
}