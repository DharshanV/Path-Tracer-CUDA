#include <iostream>
#include "GLFW.h"
#include "Renderer.h"
#include "VertexArray.h"
#include "ElementBuffer.h"
#include "Shader.h"
#include "Texture.h"

using namespace std;
using namespace glm;

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void updateTexture(Renderer& renderer, cudaArray_t* writeTo);
void processInput(GLFWwindow* window);
void createScene(Renderer& renderer);
bool lockFPS(uint32_t FPS);

float lastX = WIDTH / 2.0f;
float lastY = HEIGHT / 2.0f;
bool firstMouse = true;

Camera camera;
int samples = SAMPLE_MIN;

float deltaTime = 0.0f;
float lastFrame = 0.0f;
bool globalLight = true;

int main() {
	srand(time(0));
	//==================
	//Setup GLFW and OpenGL
	GLFW window(WIDTH, HEIGHT, "Path Tracer");

	if (!window.isGood()) { glfwTerminate(); return EXIT_FAILURE; }
	window.setClearColor({ 0,0,0,1.0f });

	glfwSetInputMode(window.getWindow(), GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	glfwSetCursorPosCallback(window.getWindow(), mouse_callback);
	glfwSetFramebufferSizeCallback(window.getWindow(), framebuffer_size_callback);
	//==================

	//==================
	//Create our 2D quad
	Shader quadShader("QuadShader.vert", "QuadShader.frag");
	quadShader.use();

	float vertices[] = {
		// positions         // texture coords
		 1.0f,  1.0f, 0.0f,  0.0f, 0.0f,   // top right
		 1.0f, -1.0f, 0.0f,  0.0f, 1.0f,   // bottom right
		-1.0f, -1.0f, 0.0f,  1.0f, 1.0f,   // bottom left
		-1.0f,  1.0f, 0.0f,  1.0f, 0.0f    // top left 
	};
	unsigned int indices[] = {
		0, 1, 3,  // first Triangle
		1, 2, 3   // second Triangle
	};

	//send to the GPU
	VertexArray VAO;
	VertexBuffer VBO(vertices, sizeof(vertices));
	ElementBuffer EBO(indices, sizeof(indices));

	VertexBufferLayout layout;
	layout.push<float>(3);
	layout.push<float>(2);
	VAO.addBuffer(VBO, layout);
	//==================

	//==================
	//create quad texture
	Texture quadTexture(nullptr, WIDTH, HEIGHT);

	//bind OpenGL array to CUDA
	cudaArray_t texturePtr;
	cudaGraphicsResource* cudaResource;
	cudaGraphicsGLRegisterImage(&cudaResource, quadTexture.id(), GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
	cudaGraphicsMapResources(1, &cudaResource, 0);
	cudaGraphicsSubResourceGetMappedArray(&texturePtr, cudaResource, 0, 0);
	//==================


	//==================
	//Create CUDA Renderer
	Renderer renderer(WIDTH, HEIGHT);
	createScene(renderer);
	//=================


	//==================
	//Render our quad
	double lastTime = glfwGetTime();
	while (!window.close()) {
		processInput(window.getWindow());
		float currentFrame = (float)glfwGetTime();
		deltaTime = currentFrame - lastFrame;
		lastFrame = currentFrame;

		window.clear();

		updateTexture(renderer, &texturePtr);

		VAO.bind();
		quadTexture.bind();
		glDrawElements(GL_TRIANGLES, sizeof(indices), GL_UNSIGNED_INT, 0);
		quadTexture.unbind();
		VAO.unbind();

		window.swapBuffers();
		window.getEvents();
	}
	return 0;
}

bool lockFPS(uint32_t FPS) {
	static double lastTime = glfwGetTime();
	if (!(glfwGetTime() < lastTime + (1.0 / FPS))) {
		lastTime += (1.0 / FPS);
		return true;
	}
	return false;
}

void processInput(GLFWwindow* window) {
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);
	if (glfwGetKey(window, GLFW_KEY_M) == GLFW_PRESS) {
		if (glfwGetInputMode(window, GLFW_CURSOR) == GLFW_CURSOR_DISABLED) {
			glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
		}
		else {
			glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
		}
	}

	bool hasInput = false;
	Camera_Movement movement;
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
		movement = FORWARD;
		hasInput = true;
	}
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
		movement = BACKWARD;
		hasInput = true;
	}
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
		movement = RIGHT;
		hasInput = true;
	}
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
		movement = LEFT;
		hasInput = true;
	}
	if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
		movement = DOWN;
		hasInput = true;
	}
	if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) {
		movement = UP;
		hasInput = true;
	}
	if (glfwGetKey(window, GLFW_KEY_L) == GLFW_PRESS) {
		globalLight = true;
	}
	if (glfwGetKey(window, GLFW_KEY_K) == GLFW_PRESS) {
		globalLight = false;
	}

	if (hasInput) {
		camera.ProcessKeyboard(movement, deltaTime);
		samples = SAMPLE_MIN;
	}

}
void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
	glViewport(0, 0, width, height);
}

void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
	if (firstMouse) {
		lastX = (float)xpos;
		lastY = (float)ypos;
		firstMouse = false;
	}

	float xoffset = (float)xpos - lastX;
	float yoffset = lastY - (float)ypos;

	lastX = (float)xpos;
	lastY = (float)ypos;

	camera.ProcessMouseMovement(xoffset, yoffset);
	samples = SAMPLE_MIN;
}

float randFloat(float a, float b) {
	float random = ((float)rand()) / (float)RAND_MAX;
	float diff = b - a;
	float r = random * diff;
	return a + r;
}

void createScene(Renderer& renderer) {
	int temp = 3;
	for (int a = -temp; a < temp; a++) {
		for (int b = -temp; b < temp; b++) {
			float choose_mat = randFloat(0, 1.1f);
			vec3 center(a + randFloat(0, 1.0f), 0.201f, b + randFloat(0, 1.0f));
			glm::vec3 color = glm::vec3(randFloat(0.1f, 1.0f), randFloat(0.1f, 1.0f), randFloat(0.1f, 1.0f));
			if (choose_mat < 0.5f) {
				Material diffuse(color);
				renderer.addSphere(Sphere(center, 0.2f), diffuse);
			} else if (choose_mat < 1.0f) {
				Material metal(color, 1, randFloat(0.0f, 0.3f));
				renderer.addSphere(Sphere(center, 0.2f), metal);
			} else {
				Material emissive(color, 0, 0, 0, 0, 1);
				renderer.addSphere(Sphere(center, 0.2f), emissive);
			}
		}
	}
	Material dielect(glm::vec3(0), 0, 0, 1, 1.5f);
	renderer.addSphere(Sphere(glm::vec3(0, 1, -0.5f), 0.5), dielect);

	Material planeMat(vec3(0.6f),1,0.1f);
	renderer.addPlane(Plane(vec3(0.0f, 0.0f, -5.0f), vec3(0, 1, 0)), planeMat);

	renderer.addLight(Light(vec3(0),1.0f));

	vec3 scale(0.2f);
	{
		Model tree("Objects/tree.obj");
		glm::mat4 matrix(1.0f);
		matrix = glm::translate(matrix, glm::vec3(-4.0f, 0, 0));
		matrix = glm::scale(matrix, scale);
		tree.transform(matrix);
		renderer.addModel(tree, Material(vec3(0,0,1.0f)));
	}
	{
		Model tree("Objects/tree.obj");
		glm::mat4 matrix(1.0f);
		matrix = glm::translate(matrix, glm::vec3(0.0f,0.0f, -4.0f));
		matrix = glm::scale(matrix, scale);
		tree.transform(matrix);
		renderer.addModel(tree, Material(vec3(0,1.0f,0)));
	}
	{
		Model tree("Objects/tree.obj");
		glm::mat4 matrix(1.0f);
		matrix = glm::translate(matrix, glm::vec3(4.0f, 0, 0));
		matrix = glm::scale(matrix, scale);
		tree.transform(matrix);
		renderer.addModel(tree, Material(vec3(1.0f,0,0)));
	}
}

void updateTexture(Renderer& renderer, cudaArray_t* writeTo) {
	struct cudaResourceDesc description;
	memset(&description, 0, sizeof(description));
	description.resType = cudaResourceTypeArray;
	description.res.array.array = *writeTo;

	cudaSurfaceObject_t write;
	cudaCreateSurfaceObject(&write, &description);
	renderer.render(write, camera, QUASI_SAMPLE_N, globalLight);
}