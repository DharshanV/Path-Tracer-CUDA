#pragma once
#include "Object.h"

class Scene {
public:
	__device__
	Scene(Object** objects, int objectSize) : objects(objects), objectSize(objectSize) { }

	__device__
	bool hit(const Ray& ray,Object*& hitObject, float& t) {
		for (int i = 0; i < objectSize; i++) {
			if (objects[i]->rayIntersect(ray.origin,ray.dir,t)) {
				hitObject = objects[i];
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