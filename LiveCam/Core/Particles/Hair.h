#pragma once

#include <string>
#include <vector>
#include <GL/glew.h>
#include <GL/glu.h>
#include <GLFW/glfw3.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <GLSL/GLSL.h>
#include <GLSL/Drawer.h>
#include <Graphics/GraphicsCommon.h>

#define K_ANC 4.0
#define DAMPING 1.5

struct HairStrand {
	Vec3 pos;
	Vec3 velocity;
	Vec3 spawn_pt;
	Vec3 spawn_dir;
};

class CollSphere {
public:
	float radius;
	Vec3 center;

	CollSphere();

	bool contains(const Vec3 &v) const;
	Vec3 project_surf(const Vec3 &v) const;
};

class Hair {
private:
	float hair_length;
	std::vector<HairStrand> hair;
	Mat4 xform;
	std::vector<CollSphere *> colliders;

public:
	Hair();
	~Hair();

	bool init(const Mesh *m, int num_spawns, float thresh = 0.4);
	void draw() const;

	void set_transform(Mat4 &xform);
	void update(float dt);
	void add_collider(CollSphere *cobj);
	Vec3 handle_collision(const Vec3 &v) const;
};
