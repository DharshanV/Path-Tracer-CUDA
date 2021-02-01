#pragma once
#include "Object.h"

class Scene {
public:
	__device__
	Scene(Object** objects, int objectSize) : objects(objects), objectSize(objectSize) { }

	__device__
	bool hit(const Ray& ray,vec3& hitPoint,vec3& N,Material& mat) {
		float maxDistance = FLT_MAX;
		for (int i = 0; i < objectSize; i++) {
			float t;
			if (objects[i]->rayIntersect(ray.origin,ray.dir,t) && t < maxDistance) {
				hitPoint = ray.origin + ray.dir * t;
				N = objects[i]->getNormal(hitPoint).normalised();
				mat = *objects[i]->getMaterial();
				maxDistance = t;
			}
		}
		return maxDistance < 10000.0f;
	}

	__device__
	~Scene() {
		for (int i = 0; i < objectSize; i++) {
			delete objects[i];
		}
	}
private:
	Object** objects;
	int objectSize;
};