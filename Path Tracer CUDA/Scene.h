#pragma once
#include "CUDAObjects.h"
#include "Light.h"
#include "Material.h"
#include "Camera.h"

typedef struct Scene {
    Light* lights;
    Sphere* spheres;
    Plane* planes;
    Material* materials;
    Camera* camera;
    int numLights;
    int numSpheres;
    int numPlanes;
} Scene;