#include <iostream>
#include "Renderer.h"
#include "GLFW.h"
#include "VertexArray.h"
#include "ElementBuffer.h"
#include "Shader.h"
#include "Texture.h"

using namespace std;
using namespace glm;

void createScene(Renderer& renderer);
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void processInput(GLFWwindow* window);
bool lockFPS(uint32_t FPS);

uint32_t WIDTH = 500;
uint32_t HEIGHT = 500;
float lastX = WIDTH / 2.0f;
float lastY = HEIGHT / 2.0f;
bool firstMouse = true;

Camera camera;
Renderer renderer(WIDTH, HEIGHT, camera);
int samples = SAMPLE_MIN;

float deltaTime = 0.0f;
float lastFrame = 0.0f;

//GLuint vbo;
//struct cudaGraphicsResource* cuda_vbo_resource;
//void* d_vbo_buffer = NULL;
//
//void createVBO(GLuint* vbo, struct cudaGraphicsResource** vbo_res,
//	unsigned int vbo_res_flags) {
//	assert(vbo);
//
//	// create buffer object
//	glGenBuffers(1, vbo);
//	glBindBuffer(GL_ARRAY_BUFFER, *vbo);
//
//	// initialize buffer object
//	unsigned int size = mesh_width * mesh_height * 4 * sizeof(float);
//	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
//
//	glBindBuffer(GL_ARRAY_BUFFER, 0);
//
//	// register this buffer object with CUDA
//	cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags);
//}
//
//void deleteVBO(GLuint* vbo, struct cudaGraphicsResource* vbo_res) {
//
//	// unregister this buffer object with CUDA
//	cudaGraphicsUnregisterResource(vbo_res);
//
//	glBindBuffer(1, *vbo);
//	glDeleteBuffers(1, vbo);
//
//	*vbo = 0;
//}
//
//void runCuda(struct cudaGraphicsResource** vbo_resource) {
//	// map OpenGL buffer object for writing from CUDA
//	float4* dptr;
//	cudaGraphicsMapResources(1, vbo_resource, 0);
//	size_t num_bytes;
//	cudaGraphicsResourceGetMappedPointer((void**)&dptr, &num_bytes, *vbo_resource);
//	//printf("CUDA mapped VBO: May access %ld bytes\n", num_bytes);
//
//	launch_kernel(dptr, mesh_width, mesh_height, g_fAnim);
//
//	// unmap buffer object
//	cudaGraphicsUnmapResources(1, vbo_resource, 0);
//}
//
//__global__ void simple_vbo_kernel(float4* pos, unsigned int width, unsigned int height, float time) {
//	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
//	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
//
//	// calculate uv coordinates
//	float u = x / (float)width;
//	float v = y / (float)height;
//	u = u * 2.0f - 1.0f;
//	v = v * 2.0f - 1.0f;
//
//	// calculate simple sine wave pattern
//	float freq = 4.0f;
//	float w = sinf(u * freq + time) * cosf(v * freq + time) * 0.5f;
//
//	// write output vertex
//	pos[y * width + x] = make_float4(u, w, v, 1.0f);
//}

int main() {
	//==================
	//Setup GLFW and OpenGL
	GLFW window(WIDTH, HEIGHT, "Test Application");

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
		 1.0f,  1.0f, 0.0f,  1.0f, 0.0f,   // top right
		 1.0f, -1.0f, 0.0f,  1.0f, 1.0f,   // bottom right
		-1.0f, -1.0f, 0.0f,  0.0f, 1.0f,   // bottom left
		-1.0f,  1.0f, 0.0f,  0.0f, 0.0f    // top left 
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

	//create our quad texture
	vector<vec3> imageTexture(WIDTH * HEIGHT);
	Texture texture(&imageTexture[0].x, WIDTH, HEIGHT);
	//==================

	//==================
	//Create CUDA Renderer
	Renderer renderer(WIDTH, HEIGHT, camera);
	createScene(renderer);
	renderer.render(&imageTexture[0].x, samples);
	texture.load(&imageTexture[0].x);
	//=================


	//==================
	//Render our quad
	double lastTime = glfwGetTime();
	while (!window.close()) {
		processInput(window.getWindow());
		float currentFrame = glfwGetTime();
		deltaTime = currentFrame - lastFrame;
		lastFrame = currentFrame;

		if (lockFPS(60)) {
			window.clear();

			renderer.render(&imageTexture[0].x, 10);
			texture.load(&imageTexture[0].x);

			VAO.bind();
			texture.bind();
			glDrawElements(GL_TRIANGLES, sizeof(indices), GL_UNSIGNED_INT, 0);
			texture.unbind();
			VAO.unbind();
			renderer.updateCamera(camera);
			//samples = glm::clamp(++samples, SAMPLE_MIN, SAMPLE_MAX);
		}
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

	if (hasInput) {
		camera.ProcessKeyboard(movement, deltaTime);
		//samples = SAMPLE_MIN;
	}

}
void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
	glViewport(0, 0, width, height);
}

void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
	if (firstMouse) {
		lastX = xpos;
		lastY = ypos;
		firstMouse = false;
	}

	float xoffset = xpos - lastX;
	float yoffset = lastY - ypos;

	lastX = xpos;
	lastY = ypos;

	camera.ProcessMouseMovement(xoffset, yoffset);
	//samples = SAMPLE_MIN;
}

void createScene(Renderer& renderer) {
	vec3 red = vec3(0.8f, 0, 0);
	vec3 gray = vec3(0.9f);
	vec3 purple = vec3(0.8f, 0.1f, 1.0f);
	vec3 green = vec3(0.3f, 1.0f, 0.7f);
	vec3 blue = vec3(0.5f, 0.7f, 1.0f);
	vec3 lightGreen = vec3(0.7f, 0.7f, 0.5f);

	Material redMarble(red,0,0,0,0,10.0f);
	Material greyMetal(gray, 1, 0.0f);
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