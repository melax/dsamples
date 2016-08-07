//
// uses librealsense and depth camera
//
//  experimenting with direct point cloud direct interaction 
//


#include <cctype>  // std::tolower
#include <iostream>
#include <algorithm>  // std::max

#include <geometric.h>
#include <glwin.h>
#include <mesh.h>
#include "dcam.h"  
#include "physics.h"
#include "wingmesh.h"  // to easily whip up some content

#include "misc_gl.h"
#include "misc_image.h"



inline std::vector<float4> Planes(const std::vector<float3> &verts, const std::vector<int3> &tris) { std::vector<float4> planes; for (auto &t : tris) planes.push_back(PolyPlane({ verts[t[0]], verts[t[1]], verts[t[2]] }));  return planes; }

inline float4 mostabove(std::vector<float4> &planes, const float3 &v)  // returns plane which v is above by the largest amount
{
	assert(planes.size());
	return *std::max_element(planes.begin(), planes.end(), [&v](const float4 &a, const float4 &b) {return (dot(a, float4(v, 1)) < dot(b, float4(v, 1))); });
}

Shape AsShape(const WingMesh &m) { return Shape(m.verts, m.GenerateTris()); }


void draw(const Mesh &mesh)
{
	glBegin(GL_TRIANGLES);
	for (auto t : mesh.tris) for (int j = 0; j < 3; j++)
	{
		auto &v = mesh.verts[t[j]];
		glNormal3fv(qzdir(v.orientation));
		glTexCoord2fv(v.texcoord);
		glVertex3fv(v.position);
	}
	glEnd();
}



std::vector<float3> ObtainVoxelPointCloud(std::vector<float3> & dpts, float voxelSize, int minOccupants)
{
	std::vector<float3> points;
	// Structure for storing per-voxel data
	enum { HASH_SIZE = 1024, HASH_MASK = HASH_SIZE - 1 }; // Only works if HASH_SIZE is power of two
	struct Voxel { int3 coord; float3 point; int count; };
	Voxel voxelHash[HASH_SIZE];
	memset(voxelHash, 0, sizeof(voxelHash));

	const float inverseVoxelSize = 1.0f / voxelSize;
	static const int3 hashCoeff(7171, 3079, 4231);

	// For the pixels in our image
	for (auto &p : dpts)
	{
		// Obtain corresponding voxel
		auto fcoord = floor(p * inverseVoxelSize);
		auto vcoord = int3(static_cast<int>(fcoord.x), static_cast<int>(fcoord.y), static_cast<int>(fcoord.z));
		auto hash = dot(vcoord, hashCoeff) & HASH_MASK;
		auto & voxel = voxelHash[hash];

		// If we collide, flush existing voxel contents
		if (voxel.count && voxel.coord != vcoord)
		{
			if (voxel.count > minOccupants) points.push_back(voxel.point / (float)voxel.count);
			voxel.count = 0;
		}

		// If voxel is empty, store the point
		if (voxel.count == 0)
		{
			voxel.coord = vcoord;
			voxel.point = p;
			voxel.count = 1;
		}
		else // Otherwise just add position contribution
		{
			voxel.point += p;
			++voxel.count;
		}

	}

	// Flush remaining voxels
	for (auto it = std::begin(voxelHash); it != std::end(voxelHash); ++it)
	{
		if (it->count > minOccupants)
		{
			points.push_back(it->point / (float)it->count);
		}
	}
	return points;
}


void rbdraw(const RigidBody *rb)
{
	glPushMatrix();
	glMultMatrixf(rb->pose().matrix());  
	for (const auto &s : rb->shapes)
		gldraw(s.verts, s.tris);
	glPopMatrix();
}



int main(int argc, char *argv[]) try
{

	physics_driftmax = 0.0025f;

	GLWin glwin("point cloud push interaction");
	RSCam dcam;
	dcam.Init((argc == 2) ? argv[1] : NULL);
	Image<unsigned short> dimage(dcam.dcamera());

	glwin.ViewAngle = dcam.fov().y;
	float viewdist = 2.0f;
	float yaw = 120;
	int mousexold = 0;
	Mesh   mesh;

	bool   pause = false;
	bool   debuglines=false;
	int    center = 0;
	bool   chains = true;
	bool   usehull = false;
	std::vector<RigidBody*> rigidbodies;
	std::vector < std::pair<RigidBody*, RigidBody*>> links;
	for (float x = -0.2f; x < 0.2f; x+= 0.07f)  
		for(float z: {0.350f})
			for (float y = -0.2f; y <= 0.2f; y += 0.07f)
	{
				rigidbodies.push_back(new RigidBody({ AsShape(WingMeshDual(WingMeshCube(0.025f),0.028f)) }, { x,y,z }));
				//rigidbodies.push_back(new RigidBody({ AsShape(WingMeshCube(0.025f)                       ) }, { x,y,z }));
				links.push_back({(y > -0.2f)?rigidbodies[rigidbodies.size() - 2]:NULL , rigidbodies.back()});
	}

	//rigidbodies.push_back(new RigidBody({ AsShape(WingMeshCube(0.05f)) }, { 0,0,0.50f }));

	auto seesaw = new RigidBody({ AsShape(WingMeshBox({ 0.20f, 0.015f,  0.05f })) }, { 0,0,0.45f });
	rigidbodies.push_back(seesaw);

	glwin.keyboardfunc = [&](unsigned char key, int x, int y)->void
	{
		switch (std::tolower(key))
		{
		case 'q': case 27:  exit(0); break;   // 27 is ESC
		case ' ': pause = !pause; break;
		case 'c': chains = !chains; break;
		case 'd': debuglines = !debuglines; break;
		case 'h': usehull = !usehull; break;
		case 'r': for (auto &rb : rigidbodies) { rb->angular_momentum = rb->linear_momentum = float3(0.0f);rb->pose() = { rb->position_start,rb->orientation_start }; }  break;
		default:  std::cout << "unassigned key (" << (int)key << "): '" << key << "'\n";   break;
		}
	};

	if (dcam.dev->supports_option(rs::option::r200_lr_auto_exposure_enabled))
		dcam.dev->set_option(rs::option::r200_lr_auto_exposure_enabled, 1);

	while (glwin.WindowUp())
	{
		if (glwin.MouseState)
		{
			yaw += glwin.mousepos.x - mousexold;
		}
		mousexold = glwin.mousepos.x;
		viewdist *= powf(1.1f, (float)glwin.mousewheel);

		if (!pause)
			dimage = dcam.GetDepth();

		glPushAttrib(GL_ALL_ATTRIB_BITS);
		glViewport(0, 0, glwin.res.x, glwin.res.y);
		glClearColor(0.1f, 0.1f, 0.15f, 1);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glMatrixMode(GL_PROJECTION);
		glPushMatrix();
		glLoadIdentity();
		gluPerspective(glwin.ViewAngle, (double)glwin.aspect_ratio(), 0.01f, 50.0f);

		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();
		gluLookAt(0, 0, viewdist, 0, 0, 0, 0, -1, 0);
		glEnable(GL_DEPTH_TEST);

		glTranslatef(0, 0, 0.35f);
		glRotatef(yaw, 0, 1, 0);
		glTranslatef(0, 0, -0.35f);

		std::vector<float3> pts;
		std::vector<float3> outliers;
		std::vector<float3> vpts;
		glDisable(GL_BLEND);

		float2 wrange = { 0.20f,0.60f };


		auto dp_image_c = Transform(dimage, [](unsigned short s) {return byte3((unsigned char)clamp(255 - s / 4, 0, 255)); });
		drawimage(dp_image_c, { 0.78f,0.22f }, { 0.2f,-0.2f }, 3);

		float depth_scale = (dcam.dev) ? dcam.dev->get_depth_scale() : 0.001f;  // to put into meters    // if file assume file is mm


		for (auto p : rect_iteration(dimage.dim())) // p is int2 from 0,0 to w,h of dcam
		{
			float d = dimage.pixel(p) * depth_scale;  // d is in meters, whereas depth[i] is in camera units  mm for R200, .125mm for SR300 ivycam
			if (p.x<5 || p.x> dimage.dim().x - 5 || p.y<5 || p.y>dimage.dim().y - 5) continue;  // crop, seems to be lots of noise at the edges
			if (d > 1.0f)  // just too far
				continue;  
			float3 v = dimage.cam.deprojectz(asfloat2(p), d);
			if (d>wrange.x && d < wrange.y) 
				pts.push_back(v);
			else
				outliers.push_back(v);
		}

		vpts = ObtainVoxelPointCloud(pts, 0.0082f, 8);


		std::vector<std::pair<float3, float3>> lines;
		std::vector<std::pair<float3, float3>> glines;

		if (1)// && pts.size())
		{
			std::vector<LimitLinear>  linears;
			std::vector<LimitAngular> angulars;
			physics_gravity = { 0,  (float) chains,0 };  // ugg y is down

			if(!usehull) for(auto rb:rigidbodies) 
			{
				if (!rb->shapes[0].planes.size())
					rb->shapes[0].planes = Planes(rb->shapes[0].verts, rb->shapes[0].tris);
				auto planes = Transform(rb->shapes[0].planes, [&](float4 p) { return rb->pose().TransformPlane(p);});
				rb->gravscale = (float)chains;
				float separation = FLT_MAX;
				float3 pushpoint = float3(0, 0, 0);  //
				float4 pushplane;
				for (auto p : vpts)
				{
						auto plane = mostabove(planes, p);
						float sep;
						if ((sep = dot(plane, float4(p, 1))) < separation)
						{
							pushpoint = p;
							pushplane = plane;
							separation = sep;
						}
				}
				if (separation > 0.1f)
					continue;
				float3 closestpoint = ProjectOntoPlane(pushplane, pushpoint);
				pushplane = float4({ -pushplane.xyz(), -dot(-pushplane.xyz(),pushpoint) });
				linears.push_back(ConstrainAlongDirection(NULL, pushpoint, rb, rb->pose().inverse()*closestpoint, pushplane.xyz(), 0, 100.0f)); //  FLT_MAX));
				lines.push_back({ closestpoint,pushpoint });
				auto cp=Separated(rb->shapes[0].verts, rb->position, rb->orientation, { pushpoint }, { 0,0,0 }, { 0,0,0,1 }, 1);
				glines.push_back({ cp.p0w, cp.p1w });
			}
			Append(linears, ConstrainPositionNailed(NULL, seesaw->position_start, seesaw, { 0, 0, 0 }));
			Append(angulars, ConstrainAngularRange(NULL, seesaw, { 0, 0, 0, 1 }, { 0, 0,-20 }, { 0, 0,20 }));

			if (chains) for (auto link : links)
				Append(linears, ConstrainPositionNailed(link.first,link.first? float3(0, 0.035f, 0) : link.second->position_start-float3(0, -0.035f, 0) , link.second, { 0,-0.035f,0 }));
			if(!pause)
				
				if(usehull && vpts.size()>5) 
					PhysicsUpdate(rigidbodies, linears, angulars, { &vpts });
				else 
					PhysicsUpdate(rigidbodies, linears, angulars, std::vector<std::vector<float3>*>());
		}


		glColor3f(1, 1, 1);
		glwirefrustumz(dcam.deprojectextents(), { 0.1f,1.0f });  // draw the camera frustum volume

		glPushAttrib(GL_ALL_ATTRIB_BITS);
		glPointSize(1);
		glBegin(GL_POINTS);
		glColor3f(0, 1, 0.5f);
		for (auto p : pts)
			glVertex3fv(p);
		glColor3f(1, 0.15f, 0.15f);
		for (auto p : outliers)
			glVertex3fv(p);
		glEnd();

		glPointSize(3);
		glBegin(GL_POINTS);
		glColor3f(1, 1, 1);
		for (auto p : vpts)  // was: spts
			glVertex3fv(p);
		glEnd();

		glPopAttrib();

		if (debuglines)
		{
			glBegin(GL_LINES);
			glColor3f(0, 1, 1);
			if (0)for (auto line : lines)
				glVertex3fv(line.first), glVertex3fv(line.second);
			glColor3f(1, 1, 0);
			for (auto line : glines)
				glVertex3fv(line.first), glVertex3fv(line.second);
			glEnd();
		}
		if (usehull && vpts.size() > 5)
		{
			auto tris = calchull(vpts, 0);
			glBegin(GL_LINES);
			glColor3f(1, 1, 1);
			for (auto t : tris) for( int i : {0,1,1,2,2,0})
				glVertex3fv(vpts[t[i]]);
			glEnd();

		}
		if (chains)
		{
			glBegin(GL_LINES);
			glColor3f(1, 0, 1);
			for (auto link : links)
			{
				if(link.first) 
					glVertex3fv(link.first->pose()* float3(0, 0, 0)), glVertex3fv(link.first->pose()* float3(0, 0.035f, 0));
				glVertex3fv(link.second->pose()* float3(0, 0, 0)) , glVertex3fv(link.second->pose()* float3(0, -0.035f, 0));
			}
			glEnd();
		}
		glPushAttrib(GL_ALL_ATTRIB_BITS);
		glEnable(GL_LIGHTING);
		glEnable(GL_LIGHT0);
		glEnable(GL_TEXTURE_2D);
		glColor3f(0.5f, 0.5f, 0.5f);
		for (auto &rb : rigidbodies)
			rbdraw(rb);
		glPopAttrib();   // Restore state

		// Restore state
		glPopMatrix();  //should be currently in modelview mode
		glMatrixMode(GL_PROJECTION);
		glPopMatrix();
		glPopAttrib();
		glMatrixMode(GL_MODELVIEW);

		glwin.PrintString({ 0,0 }, "esc to quit.");
		glwin.PrintString({ 0,1 }, "[h] collision %s  ",(usehull)?"hull":"points");
		glwin.SwapBuffers();

	}
	return 0;
}
catch (const char *c)
{
	MessageBox(GetActiveWindow(), c, "FAIL", 0);
}
catch (std::exception e)
{
	MessageBox(GetActiveWindow(), e.what(), "FAIL", 0);
}
