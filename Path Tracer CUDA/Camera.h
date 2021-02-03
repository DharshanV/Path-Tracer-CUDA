#pragma once
#include <glm/glm.hpp>
#include <cuda_runtime.h>

#define PI 3.14159265f

struct Camera {
    __host__ __device__
    Camera(const glm::vec3& pos,const glm::vec3& lookAt,const float aspectRatio) : cp(pos), aspectRatio(aspectRatio) {
        this->LookAt(lookAt);
    }

    __host__ __device__
    Camera& operator=(const Camera& o) {
        cp = o.cp;
        cd = o.cd;
        cr = o.cr;
        cu = o.cu;
        aspectRatio = o.aspectRatio;
        return *this;
    }

    __host__ __device__
    inline void LookAt(const glm::vec3& p) {
        glm::vec3 up(0, 1, 0);
        cd = glm::normalize(p - cp);
        cr = glm::normalize(glm::cross(up, cd));
        cu = glm::normalize(glm::cross(cd, cr));
    }

    __host__ __device__
    inline Ray getRay(float x, float y) {
        float deltaX = (-2.0f * x + 1.0f) * tan(FOV / 2 * PI / 180) * aspectRatio;
        float deltaY = (-2.0f * y + 1.0f) * tan(FOV / 2 * PI / 180);
        glm::vec3 rd = glm::normalize(cr * deltaX + cu * deltaY + cd);
        return Ray(cp, rd);
    }

    glm::vec3 cp;
    glm::vec3 cd;
    glm::vec3 cr;
    glm::vec3 cu;
    float aspectRatio;
    const float FOV = 90.0f;
};