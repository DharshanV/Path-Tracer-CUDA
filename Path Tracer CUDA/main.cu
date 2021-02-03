#include <iostream>
#include "Renderer.h"

using namespace std;
using namespace glm;

void createScene(Renderer& renderer) {
	vec3 red = vec3(0.8f, 0, 0);
	vec3 gray = vec3(0.9f);
	vec3 purple = vec3(0.8f, 0.1f, 1.0f);
	vec3 green = vec3(0.3f, 1.0f, 0.7f);
	vec3 blue = vec3(0.5f, 0.7f, 1.0f);
	vec3 lightGreen = vec3(0.7f, 0.7f, 0.5f);

	Material redMarble(red);
	Material greyMetal(gray, 1, 0.1f);
	Material purpleMarble(purple, 0, 0, 1, 1.0f);
	Material greenMarble(green);
	Material blueMarble(blue);
	Material planeMat(lightGreen);

	renderer.addSphere(Sphere(vec3(-0.5f, 0.0f, -5.0f), 1.0f), greyMetal);
	renderer.addSphere(Sphere(vec3(-0.3f, -0.6f, -3.0f), 0.4f), purpleMarble);
	renderer.addSphere(Sphere(vec3(1.5f, -0.6f, -3.0f), 0.4f), greenMarble);
	renderer.addSphere(Sphere(vec3(-2.1f, -0.3f, -3.0f), 0.7f), blueMarble);
	renderer.addSphere(Sphere(vec3(-2.0f, -0.6f, -5.0f), 0.4f), redMarble);
	renderer.addPlane(Plane(vec3(0.0f, -1.0f, -5.0f), vec3(0, 1, 0), 20, 20), planeMat);

	renderer.addLight(Light(glm::vec3(-7, 7, 3), 1.0f));
	renderer.addLight(Light(glm::vec3(7, 7, 3), 0.2f));
}

int main() {
	int width = 500;
	int height = 500;
	int numOfFrames = 5;
	Camera camera(vec3(0.05f, 0, 0.5f), vec3(0, 0, -1.0f), (float)width / (float)height);
	Renderer renderer(width, height, camera);
	createScene(renderer);

	renderer.updateCamera(camera);
	renderer.render("test.ppm");
	system("start test.ppm");
	system("PAUSE");
	return 0;
}