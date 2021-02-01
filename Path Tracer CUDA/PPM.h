#pragma once
#include <fstream>
#include <vector>
#include <sstream>

#include "Image.h"

class PPM : public Image {
public:
	PPM(int width, int height) : Image(width,height) { }

	void writeToFile(const char* fileName) override {
		std::stringstream ss;
		ss << "P3\n" << width << ' ' << height << " 255\n";
		for (int j = 0; j < height; ++j) {
			for (int i = 0; i < width; ++i) {
				vec3 pixel = imageData[index(i, j)].RGB * 255;
				int r = (int)clamp(pixel[0], 0, 255);
				int g = (int)clamp(pixel[1], 0, 255);
				int b = (int)clamp(pixel[2], 0, 255);
				ss << r << ' ' << g << ' ' << b << '\n';
			}
		}
		std::ofstream out(fileName);
		out << ss.rdbuf();
        out.close();
	}

	float clamp(float value, float low, float high) {
		return std::max(low, std::min(value, high));
	}
};

