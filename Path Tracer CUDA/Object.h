#pragma once
#include "vec3.h"
#include "Material.h"

class Object {
public:
    __host__ __device__
    virtual bool rayIntersect(const vec3& origin, const vec3& dir, float& t) const = 0;
    __host__ __device__
    virtual const Material* getMaterial() const = 0;
    __host__ __device__
    virtual vec3 getNormal(const vec3& hit) const = 0;
};