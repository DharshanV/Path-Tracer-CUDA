#pragma once
#include "Object.h"

class Scene {
public:
	__device__
	Scene(Object** objects, int objectSize) : objects(objects), objectSize(objectSize) { }

	__device__
	bool hit(const Ray& ray,vec3& hitPoint,vec3& N,Material& mat) {
		float t;
		for (int i = 0; i < objectSize; i++) {
			if (objects[i]->rayIntersect(ray.origin,ray.dir,t)) {
				hitPoint = ray.origin + ray.dir * t;
				N = objects[i]->getNormal(hitPoint);
				mat = *objects[i]->getMaterial();
				return true;
			}
		}
		return false;
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