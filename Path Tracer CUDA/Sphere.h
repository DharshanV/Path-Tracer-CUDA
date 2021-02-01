#pragma once
#include "Object.h"
#include "vec3.h"

class Sphere : public Object {
public:
    __host__ __device__
    Sphere() : position(0), radius(1) {}
    __host__ __device__
    Sphere(const vec3& position) : position(position), radius(1) {}
    __host__ __device__
    Sphere(const vec3& position, float r) : position(position), radius(r) {}
    __host__ __device__
    Sphere(const vec3& position, float r, const Material& mat)
        : position(position), radius(r), material(mat) {
    }
    ~Sphere() {}

public:
    __device__ __host__
    bool rayIntersect(const vec3& origin, const vec3& dir, float& t) const {
        vec3 oToC = vec3(position) - origin;
        float t1 = vec3(dir) * oToC;  // t1 when ray is closest to sphere
        float rayToC = (oToC * oToC) - (t1 * t1);
        // Don't need to check for sqrt or radius squared
        // since the inequality stays the same
        if (rayToC > radius) return false;
        float deltaT = sqrt(radius * radius - rayToC);
        t = t1 - deltaT;  // below point
        float abovePoint = t1 + deltaT;
        // check if the origin is inside left of center
        // if it is then t will be zero since will have to travel back to meet below
        // point
        if (t < 0) t = abovePoint;
        // if that the above point is still negative then sphere is behind origin
        if (t < 0) return false;
        return true;
    }
    __device__ __host__
    const Material* getMaterial() const { return &material; }

    __device__ __host__
    vec3 getNormal(const vec3& hit) const { return vec3(hit) - position; }

private:
    Material material;
    vec3 position;
    float radius;
};