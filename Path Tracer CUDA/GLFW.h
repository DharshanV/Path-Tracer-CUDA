#ifndef GLFW_H
#define GLFW_H
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include <iostream>
using namespace std;
class GLFW
{

public:
	GLFW(unsigned int width, unsigned int height, const char* name);
	bool close();
	bool isGood();

	void clear(GLbitfield clearBit = GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	void swapBuffers();
	void getEvents();

	~GLFW();
public:
	void setCursor(int value);
	void setClearColor(glm::vec4 clearColor);

	float getAspectRatio();
	int getInputMode(int mode);

	int getWidth() { return width; };
	int getHeight() { return height; };
	GLFWwindow* getWindow() { return window; };
private:
	GLFWwindow* window;
	bool good;
	int width;
	int height;
};
#endif