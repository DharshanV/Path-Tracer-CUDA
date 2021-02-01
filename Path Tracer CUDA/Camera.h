#pragma once
#include "vec3.h"
#include "Ray.h"
#define PI 3.14159265f

//class Camera {
//public:
//    __host__ __device__
//    Camera(int width, int height) : width(width), height(height) {
//        aspectRatio = (float)width / height;
//        origin = vec3(0);
//        FOV = 90;
//    }
//
//    __host__ __device__
//    Ray getRay(int x, int y) {
//        float Px = (2 * ((x + 0.5) / width) - 1) * tan(FOV / 2 * PI / 180) * aspectRatio;
//        float Py = (1 - 2 * ((y + 0.5) / height) * tan(FOV / 2 * PI / 180));
//        vec3 rayDirection = (vec3(Px, Py, -1) - origin).normalised();
//        return Ray(origin, rayDirection);
//    }
//private:
//    float aspectRatio;
//    float FOV;
//    vec3 origin;
//    int width;
//    int height;
//};

class Camera {
public:
    __host__ __device__
    Camera(vec3& pos, vec3& lookAt, float aspectRatio) : cp(pos), aspectRatio(aspectRatio){
        this->LookAt(lookAt);
    }
    __host__ __device__
    void LookAt(vec3& p) {
        vec3 up(0, 1, 0);
        cd = (p - cp).normalised();
        cr = (up ^ cd).normalised();
        cu = (cd ^ cr).normalised();
    }

    __host__ __device__
    Ray getRay(float x, float y) {
        float deltaX = (-2.0f * x + 1.0f) * tan(90.0f / 2 * PI / 180) * aspectRatio;
        float deltaY = (-2.0f * y + 1.0f) * tan(90.0f / 2 * PI / 180);
        vec3 rd = (cr * deltaX + cu * deltaY + cd).normalised();
        return Ray(cp, rd);
    }

private:
    vec3 cp;
    vec3 cd;
    vec3 cr;
    vec3 cu;
    float aspectRatio;
};