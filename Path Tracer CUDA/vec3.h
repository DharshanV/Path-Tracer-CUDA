#pragma once
#include <math.h>
#include <iostream>

class vec3 {
public:

    union {
        float data[3];
        struct {
            float x;
            float y;
            float z;
        };
    };

    // Constructors

    // Vectors default to 0, 0, 0.
    __host__ __device__
    vec3() {
        x = 0;
        y = 0;
        z = 0;
    }

    __host__ __device__
    vec3(float value) {
        x = value;
        y = value;
        z = value;
    }

    // Construct with values, 3D
    __host__ __device__
    vec3(float ax, float ay, float az) {
        x = ax;
        y = ay;
        z = az;
    }

    // Construct with values, 2D
    __host__ __device__
    vec3(float ax, float ay) {
        x = ax;
        y = ay;
        z = 0;
    }

    // Copy constructor
    __host__ __device__
    vec3(const vec3& o) {
        x = o.x;
        y = o.y;
        z = o.z;
    }

    // Addition
    __host__ __device__
    vec3 operator+(const vec3& o) {
        return vec3(x + o.x, y + o.y, z + o.z);
    }
    __host__ __device__
    vec3 operator+(const vec3& o) const {
        return vec3(x + o.x, y + o.y, z + o.z);
    }

    __host__ __device__
    vec3& operator+=(const vec3& o) {
        x += o.x;
        y += o.y;
        z += o.z;
        return *this;
    }

    // Subtraction
    __host__ __device__
    vec3 operator-() {
        return vec3(-x, -y, -z);
    }

    __host__ __device__
    vec3 operator-(const vec3 o) {
        return vec3(x - o.x, y - o.y, z - o.z);
    }

    __host__ __device__
    vec3& operator-=(const vec3 o) {
        x -= o.x;
        y -= o.y;
        z -= o.z;
        return *this;
    }

    // Multiplication by scalars
    __host__ __device__
    vec3 operator*(const float s) {
        return vec3(x * s, y * s, z * s);
    }

    __host__ __device__
    vec3 operator*(const float s) const {
        return vec3(x * s, y * s, z * s);
    }

    __host__ __device__
    vec3& operator*=(const float s) {
        x *= s;
        y *= s;
        z *= s;
        return *this;
    }

    // Division by scalars
    __host__ __device__
    vec3 operator/(const float s) {
        return vec3(x / s, y / s, z / s);
    }

    __host__ __device__
    vec3& operator/=(const float s) {
        x /= s;
        y /= s;
        z /= s;
        return *this;
    }

    // Dot product
    __host__ __device__
    float operator*(const vec3 o) {
        return (x * o.x) + (y * o.y) + (z * o.z);
    }

    // An in-place dot product does not exist because
    // the result is not a vector.

    // Cross product
    __host__ __device__
    vec3 operator^(const vec3 o) {
        float nx = y * o.z - o.y * z;
        float ny = z * o.x - o.z * x;
        float nz = x * o.y - o.x * y;
        return vec3(nx, ny, nz);
    }
    __host__ __device__
    vec3& operator^=(const vec3 o) {
        float nx = y * o.z - o.y * z;
        float ny = z * o.x - o.z * x;
        float nz = x * o.y - o.x * y;
        x = nx;
        y = ny;
        z = nz;
        return *this;
    }

    float operator[](int i) {
        if (i == 0) return x;
        if (i == 1) return y;
        if (i == 2) return z;
        return -1;
    }

    // Other functions

    // Length of vector
    __host__ __device__
    float magnitude() {
        return sqrt(magnitude_sqr());
    }

    // Length of vector squared
    __host__ __device__
    float magnitude_sqr() {
        return (x * x) + (y * y) + (z * z);
    }

    // Returns a normalised copy of the vector
    // Will break if it's length is 0
    __host__ __device__
    vec3 normalised() {
        return vec3(*this) / magnitude();
    }

    // Modified the vector so it becomes normalised
    __host__ __device__
    vec3& normalise() {
        (*this) /= magnitude();
        return *this;
    }

};

std::ostream& operator<<(std::ostream& os,const vec3& vec) {
    os << "<" << vec.x << "," << vec.y << "," << vec.z << ">";
    return os;
}