#include "GLFW.h"

GLFW::GLFW(unsigned int width, unsigned int height, const char* name) {
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

	window = glfwCreateWindow(width, height, name, NULL, NULL);
	good = true;

	if (window == NULL)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		good = false;
	}
	glfwMakeContextCurrent(window);
	glfwSwapInterval(1);

	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
		good = false;
	}

	this->width = width;
	this->height = height;
}

bool GLFW::close()
{
	return glfwWindowShouldClose(window);
}

bool GLFW::isGood()
{
	return good;
}

void GLFW::clear(GLbitfield clearBit)
{
	glClear(clearBit);
}

void GLFW::swapBuffers()
{
	glfwSwapBuffers(window);
}

void GLFW::getEvents()
{
	glfwPollEvents();
}

float GLFW::getAspectRatio()
{
	glfwGetWindowSize(window, &width, &height);
	return (float)width / (float)height;
}

int GLFW::getInputMode(int mode)
{
	return glfwGetInputMode(window, mode);
}

GLFW::~GLFW() {
	glfwTerminate();
}

void GLFW::setCursor(int value)
{
	glfwSetInputMode(window, GLFW_CURSOR, value);
}

void GLFW::setClearColor(glm::vec4 clearColor)
{
	glClearColor(clearColor.r, clearColor.g, clearColor.b, clearColor.a);
}
