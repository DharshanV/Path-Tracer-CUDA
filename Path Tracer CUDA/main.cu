#include <iostream>
#include "Renderer.h"

using namespace std;

int main() {
	const int WIDTH = 1600;
	const int HEIGHT = 900;

	PPM image(WIDTH, HEIGHT);
	Camera camera(vec3(0), vec3(0, 0, -1), (float)WIDTH / HEIGHT);
	Renderer renderer(&image,&camera);

	renderer.run("test.ppm");

	cout << "Elapsed: " << renderer.getElapsed() << endl;
	system("start test.ppm");
	system("PAUSE");
	return 0;
}

/*
* float clamp(float value, float low, float high) {
	return std::max(low, std::min(value, high));
}
	Camera camera(WIDTH,HEIGHT);
	Color* imageData = new Color[WIDTH * HEIGHT];
	Object* sphere = new Sphere(vec3(0,0,-3));

	for (int j = 0; j < HEIGHT; ++j) {
		for (int i = 0; i < WIDTH; ++i) {
			Ray ray = camera.getRay(i, j);
			float t;
			if (sphere->rayIntersect(ray.origin, ray.dir, t)) {
				imageData[i + j * WIDTH] = Color(255, 0, 0);
			}
			else {
				imageData[i + j * WIDTH] = Color(0, 0, 0);
			}
		}
	}


	std::stringstream ss;
	ss << "P3\n" << WIDTH << ' ' << HEIGHT << " 255\n";
	for (int j = 0; j < HEIGHT; ++j) {
		for (int i = 0; i < WIDTH; ++i) {
			Color pixel = imageData[i + j * WIDTH];
			int r = (int)clamp(pixel.r, 0, 255);
			int g = (int)clamp(pixel.g, 0, 255);
			int b = (int)clamp(pixel.b, 0, 255);
			ss << r << ' ' << g << ' ' << b << '\n';
		}
	}
	std::ofstream out("test.ppm");
	out << ss.rdbuf();
	out.close();
	delete[] imageData;

*/