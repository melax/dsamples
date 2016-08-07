//
//  DPCA - Principal Component Analysis on point cloud depth data.
//  uses Intel RealSense depth camera
//
//  Many basic use cases involving depth cameras can be implemented robustly with some very simple point cloud analysis.
//  This sample is meant to show how to extract the center of mass and covariance (or orientation) of a point cloud.
//  One mode shows how to also extract the curvatures on point cloud (or surface of object in view).
//  There is no segmentation, so experiment with this putting only one object or hand within the view frustum of the depth camera.
//  The range is limited, and can be adjusted dynamically, to avoid picking up unwanted things in the background.
//   
//  this program is a re-implementation of one of the examples from our "programming with depth data" developer lab session at IDF2013.
//  the video https://www.youtube.com/watch?v=a9vgHR8qyHg shows what was covered in that course
//

#include <cctype>  // std::tolower
#include <iostream>
#include <algorithm>  // std::max

#include <geometric.h>
#include <glwin.h>
#include <mesh.h>
#include "dcam.h"  
#include "mesh.h"
#include "misc_gl.h"
#include "misc.h"   
#include "tcpsocket.h"

// util drawing routines
// most of this code is duplicated from testcov and ploidfit from the sandbox repo

void glAxis(float len=1.0f)
{
	glPushAttrib(GL_ALL_ATTRIB_BITS);
	glLineWidth(3.0f);
	glBegin(GL_LINES);
	for (int i : {0, 1, 2})
	{
		float3 v(0, 0, 0);
		v[i] = 1.0f;
		glColor3fv(v);
		glVertex3fv({ 0, 0, 0 });
		glVertex3fv(v*len);
	}
	glEnd();
	glPopAttrib();
}
void glAxis(const Pose &pose, float len = 1.0f)
{
	glPushMatrix();
	glMultMatrixf(pose.matrix());
	glAxis(len);
	glPopMatrix();
}

std::vector<int3> gridtriangulation(int2 tess)
{
	std::vector<int3> tris;
	for (int y = 0; y < tess.y - 1; y++) for (int x = 0; x < tess.x - 1; x++)
	{
		tris.push_back({ (y + 1)* tess.x + x + 0, (y + 0)* tess.x + x + 0, (y + 0)* tess.x + x + 1 });
		tris.push_back({ (y + 0)* tess.x + x + 1, (y + 1)* tess.x + x + 1, (y + 1)* tess.x + x + 0 });  // note the {2,0} edge is the one that cuts across the quad
	}
	return tris;
}
Mesh sphere(int2 tess)
{
	std::vector<Vertex> verts;
	for (int y = 0; y < tess.y; y++) for (int x = 0; x < tess.x; x++)
	{
		float lat = 3.14159f * (y / (tess.y - 1.0f) - 0.5f);
		float lng = 3.14159f * 2.0f * x / (tess.x - 1.0f);
		float3 p(cos(lat)*cos(lng), cos(lat)*sin(lng), sin(lat));
		float3 u(-sin(lng), cos(lng), 0);
		verts.push_back({ p, quatfrommat({ u, cross(p, u), p }),{ x / (tess.x - 1.0f), y / (tess.y - 1.0f) } });
	}
	return{ verts, gridtriangulation(tess), Pose(), "",{ 1, 1, 1, 1 } };
}

Mesh scale(Mesh m, float3 r)
{
	auto stretch = [&r](Vertex v)->Vertex
	{
		float3 n = qzdir(v.orientation) / r;
		float3 u = qxdir(v.orientation) / r;
		return{ v.position* r, quatfrommat({ u, cross(n, u), n }), v.texcoord };
	};
	std::transform(m.verts.begin(), m.verts.end(), m.verts.begin(), stretch);
	return m;
}
Mesh ellipse(float3 r)
{
	return scale(sphere({ 23, 17 }), r);
}

bool show_ellipsoid_normals = false;  // just to make sure my mesh scale worked ok
inline void glellipsoid(const float3 &r)  // wire mesh version
{
	auto e = ellipse(r);
	glBegin(GL_LINES);
	glColor3f(0.5f, 0.5f, 0.5f);
	for (auto t : e.tris) for (auto i : { 0, 1, 1, 2 })  // just draw first 2 edges since {0,2} is the diagonal that cuts across the quad
		glVertex3fv(e.verts[t[i]].position);
	glColor3f(0.0f, 0.5f, 0.5f);
	for (auto p : e.verts) for (auto n : { 0.0f, 0.02f }) if (show_ellipsoid_normals)  // just for debugging
		glVertex3fv(p.position + qzdir(p.orientation)*n);
	glEnd();
}
inline void glellipsoid(const float3 &r, const Pose &pose) { glPushMatrix(); glMultMatrixf(pose.matrix()); glellipsoid(r); glPopMatrix(); }


void glWirePatch(std::function<float3(float2)> m)
{
	auto f = [](std::function<float3(float2)> m)
	{
		for (float x = 0; x <= 1.0f; x += 1.0f / 16)
		{
			glBegin(GL_LINE_STRIP);
			for (float y = 0; y <= 1.0f; y += 1.0f / 16)
				glVertex3fv(m({ x,y }));
			glEnd();
		}
	};
	f(m);
	f([m](float2 c) {return m({ c.y,c.x }); });
}

void glHeightField(float2 bmin, float2 bmax, std::function<float(float2)> h, float3 c = { 1,1,1 })
{
	glColor3fv(c);
	glWirePatch([&h, &bmin, &bmax](float2 c)->float3 {auto p = bmin + (bmax - bmin)*c; return{ p.x,p.y,h(p) }; });
}


// math support routine copied from ploidfit
float4 ParabloidFit(const std::vector<float3> &pts)  // least squares fitting quadratic patch centered at x,y==0,0
{
	float4x4 m;
	float4 b;
	for (auto &p : pts)
	{
		float4 v(p.x*p.x, p.y*p.y, p.x*p.y, 1.0f);
		m += outerprod(v, v);
		b += v*p.z;
	}
	return mul(inverse(m), b);   // returns 'h' best fits   z = h.x * x*x + h.y * y*y + h.z * x*y + h.w  hessian and zoffset 
}


int main(int argc, char *argv[]) try
{
	GLWin glwin("dpca - center of mass and covariance of depth data");
	RSCam dcam;
	dcam.Init((argc == 2) ? argv[1] : NULL); 
	glwin.ViewAngle = dcam.fov().y;
	Image<unsigned short> dimage(dcam.dcamera()); // will hold the depth data that comes from the depth camera
	Pose   camera    = { { 0, 0, 0 },linalg::normalize(float4{ 1, 0, -0.3f,  0 }) };
	float  viewdist  = 1.0f;
	float2 wrange    = { 0.125f,0.625f };
	bool   pause     = false;
	bool   recording = false;
	bool   hold      = false;
	int    mode      = 2;
	SOCKET server    = INVALID_SOCKET;
	int    tcp_port  = 12346;

	glwin.keyboardfunc = [&](unsigned char key, int x, int y)->void
	{
		switch (std::tolower(key))
		{
		case 'q': case 27:  exit(0)      ; break;   // 27 is ESC
		case ' ': pause     = !pause     ; break;
		case '0': case '1': case '2': case '3': case '4':  mode = (key - '0')%4; break;   // select current viewing mode 
		case '[': case'-': case '_':  wrange.y /= 1.125f     ; break;
		case ']': case'=': case '+':  wrange.y *= 1.125f     ; break;
		case 'b': dcam.addbackground(dimage.raster.data())   ; break;
		case 'h': server = start_server(tcp_port)            ; break;
		default:  std::cout << "unassigned key (" << (int)key << "): '" << key << "'\n";   break;
		}
	};

	if(dcam.dev && dcam.dev->supports_option(rs::option::r200_lr_auto_exposure_enabled))
		dcam.dev->set_option(rs::option::r200_lr_auto_exposure_enabled, 1);
	float depth_scale = (dcam.dev) ? dcam.dev->get_depth_scale() : 0.001f;  // to put into meters    // if file assume file is mm

	while (glwin.WindowUp())
	{
		if (glwin.MouseState)
		{
				camera.orientation = qmul(camera.orientation, qconj(VirtualTrackBall(float3(0, 0, 1), float3(0, 0, 0), glwin.OldMouseVector, glwin.MouseVector))); // equation is non-typical we are orbiting the camera, not rotating the object
		}
		viewdist *= powf(1.1f, (float) glwin.mousewheel);
		camera.position = float3(0, 0, sum(wrange) / 2.0f) + qzdir(camera.orientation) * viewdist;

		if (!pause)
			dimage = dcam.GetDepth();
		
		std::vector<float3> pointcloud,outliers;  
		for (auto p : rect_iteration(dcam.dim())) // p is int2 from 0,0 to w,h of dcam
		{
			float d = dimage.pixel(p) * depth_scale;  // d is in meters, whereas depth[i] is in camera units  mm for R200, .125mm for SR300 ivycam
			if (d > wrange.y*1.5f)  // way outside sample range
				continue;
			// stuff near edges our outside sample range put into outlier points
			(d<wrange.x || d > wrange.y || p.x<5 || p.x> dcam.dim().x - 5 || p.y<5 || p.y>dcam.dim().y - 5 ?outliers:pointcloud).push_back(dcam.deprojectz(asfloat2(p), d));
		}
		Pose pa;
		float3 va;
		std::tie<Pose, float3>(pa, va) = PrincipalAxes(pointcloud);
		auto sd = sqrt(va);  // standard deviation  (often use 2*sd for that 90%interval)

		if (server != INVALID_SOCKET)
		{
			do_server_thing(server, ToString() << "pose " << pa);
			// do_server_thing_persist(server, ToString() << "pose " << pa);    // if the frame rate is slow in a web page then you can try this call which attempts to re-use the same connection.  sometimes this fails though.
		}

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
		glEnable(GL_DEPTH_TEST);
		glMultMatrixf(camera.inverse().matrix());

		auto dp_image_c = Transform(dimage, [wrange,depth_scale](unsigned short s) {return byte3((unsigned char)clamp((int)((s*depth_scale-wrange.y)/(wrange.x-wrange.y)*255), 0, 255)); });
		//if(pointcloud.size())
		//	dp_image_c.pixel(int2(dp_image_c.cam.projectz(pa.position))) = { 0,255,0 };
		drawimage(dp_image_c, { 0.68f,0.22f }, { 0.3f,-0.2f },3);

		if (mode == 0)
			glcolorbox({ 0.2f,0.1f,0.02f },pa);
		else if(mode==1)
			glGridxy(sd.x*2.0f, { 0.75f,1.0f,1.0f }, pa);
		else if (mode == 2)
			glellipsoid(sd*2.0f, pa);
		else if (mode == 3 && pointcloud.size() > 10)
		{
			auto localcloud = Transform(pointcloud, [pa](const float3 &v) {return pa.inverse()*v; });
			auto h = ParabloidFit(localcloud);
			localcloud = Transform(localcloud, [&h](const float3 &p) { return p - float3(0.0f, 0.0f, h.w); });   // for (auto &p : points) p.z -= h.w; 
			float2x2 hess = { { h.x,h.z / 2.0f },{ h.z / 2.0f,h.y } };
			float ha = Diagonalizer(hess);
			float2x2 hr = { { cosf(ha),sinf(ha) },{ -sinf(ha),cosf(ha) } };  // axes of curvature
			auto hd = mul(mul(transpose(hr), hess), hr);   // diagonal entries are the maximum and minimum curvatures
			glPushMatrix();
			glMultMatrixf(pa.matrix());
			for (int i : {0, 1})  // showing two principal directions of curvature  (different than principal axes of the point cloud)
			{
				glPushAttrib(GL_ALL_ATTRIB_BITS);
				glLineWidth(2.0f);
				glBegin(GL_LINE_STRIP);
				glColor3fv([i]() {float3 c(0.5f); c[i] = 1.0f; return c; }());
				for (float d = -sd.x*2.5f; d <= sd.x*2.5f; d += sd.x / 32)
					glVertex3fv(float3(hr[i] * d, hd[i][i] * d*d));
				glEnd();
				glPopAttrib();
			}
			glHeightField(-sd.xy()*2.0f, sd.xy()*2.0f, [&hess](float2 p)->float {return dot(p, mul(hess, p)); });   // draws a quad patch using the hessian matrix
			glPopMatrix();
		}
		glAxis(pa,0.1f);

		glColor3f(1, 1, 1);
		glwirefrustumz(dcam.deprojectextents(), wrange );  // draw the camera frustum volume

		glPushAttrib(GL_ALL_ATTRIB_BITS);
		glPointSize(1);
		glBegin(GL_POINTS);
		glColor3f(0, 1, 0.5f);
		for (auto p : pointcloud)
			glVertex3fv(p);
		glColor3f(0.3f,0,0);
		for (auto p : outliers)
			glVertex3fv(p);
		glEnd();
		glPopAttrib();
		glColor3f(1, 1, 1);


		// Restore state
		glMatrixMode(GL_PROJECTION);
		glPopMatrix();
		glMatrixMode(GL_MODELVIEW);
		glPopMatrix();  
		glPopAttrib();

		glColor3f(1, 1, 1);
		const char *mode_description[] = { "Oriented box (fixed size)" , "Plane fit" , "Covariance" , "Hessian (principal curvatures)" };
		glwin.PrintString({ 0,0 }, "Camera %s ", pause ? "paused" : "running");
		glwin.PrintString({ 0,1 }, "Mode %d  %s ", mode, mode_description[mode]);
		glwin.PrintString({ 0,2 }, "keys 0-3 to change draw mode, -/+ move backplane, space to pause");
		if (server!=INVALID_SOCKET)
			glwin.PrintString({ 0,3 }, "HTTP server enabled localhost:%d",tcp_port);
		glwin.SwapBuffers();
		
	}
	return 0;
}
catch (const char *c)
{
	MessageBox(GetActiveWindow(),c, "FAIL",  0); 
}
catch (std::exception e)
{
	MessageBox(GetActiveWindow(), e.what(), "FAIL", 0);
}
