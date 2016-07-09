//
//  minimal gui support
//
//  enough of an immediate mode gui to break up opengl window into resizable panels. 
//  this is gl only and depends on glwin, opengl wrapper.
//  first quick attempt, the design/api needs some rethinking, sorry bout that.
//

#pragma once
#ifndef GL_GUI_H
#define GL_GUI_H

#include <assert.h>

#include "geometric.h"
#include "glwin.h"
#include "misc_gl.h"  // for drawimage()

//---------- mesh draw gl  --------------


class GUI  // simple immediate mode gui
{
	struct Partitioning : public std::vector<int2x2>
	{
		GUI &gui;
		Partitioning() = delete;
		Partitioning(const Partitioning& rhs):std::vector<int2x2>(rhs),gui(rhs.gui) { gui.bounds.push_back(gui.bounds.back()); }
		Partitioning(GUI &gui,std::vector<int2x2> regions) :std::vector<int2x2>(regions),gui(gui) { gui.bounds.push_back(gui.bounds.back()); }
		~Partitioning() { gui.pop(); }
	};
public:
	Partitioning partition_y(int n)
	{
		int2 subdims = this->dims() / int2(1, n) - int2(0, 1);
		std::vector<int2x2> regions;
		for (int i = 0; i < n;i++)
			regions.push_back({ {offset().x,offset().y + (subdims.y + 1)*i},subdims });
		return Partitioning(*this, regions);
	}
	GLWin &glwin;
	void * &selection;
	int2 mouse() { return{ glwin.mousepos.x, glwin.Height - glwin.mousepos.y }; }
	int2 mouse_prev() { return{ glwin.mousepos_previous.x, glwin.Height - glwin.mousepos_previous.y }; }
	std::vector<int2x2> bounds; //  offset, size;
	int2 &dims() { return bounds.back()[1]; }
	int2 &offset() { return bounds.back()[0]; }
	float2 tolocal(const int2& v) { return (asfloat2(v - offset())/ asfloat2(dims()));}
	float2 mousef() { return tolocal(mouse()); }
	float3 deproject(float2 p) { p = (p - float2(0.5f, 0.5f))* float2(2.0f*dims().x / dims().y, 2.0f); float fl = tan(glwin.ViewAngle / 2.0f*3.14f / 180.0f); return normalize(float3(p* fl, -1.0f)); }
	bool inview(void *c = NULL) { return  (selection) ? (selection == c) : within_range(mouse(), offset(), offset() + dims() - int2(1, 1)); }
	bool focus(void *c = NULL) { if (inview(c) && (glwin.downevent||glwin.MouseState))selection = c; return selection == c; }
	void set(int2x2 &b)
	{
		bounds.back() = b;
		set();
	}
	void set()
	{
		if (bounds.size() == 0)
			bounds.push_back({ { 0, 0}, {glwin.Width, glwin.Height } });
		bounds.back()[1] = max(int2(0,0), bounds.back()[1]);
		glViewport(bounds.back()[0].x, bounds.back()[0].y, bounds.back()[1].x, bounds.back()[1].y);
		glScissor (bounds.back()[0].x, bounds.back()[0].y, bounds.back()[1].x, bounds.back()[1].y);
	}
	void pop()
	{
		bounds.pop_back();
		set();
	}
	GUI(GLWin &glwin, void *&s, float4 c = { 0,0,0,0 }) :glwin(glwin), selection(s)
	{
		if (!glwin.MouseState) s = NULL;  // not sure
		bounds.push_back({ { 0, 0},{ glwin.Width, glwin.Height } });
		set();
		glClearColor(c.x,c.y,c.z,c.w);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glPushAttrib(GL_ALL_ATTRIB_BITS);
		glEnable(GL_SCISSOR_TEST);

		// Set up the viewport

		// Set up matrices
		glMatrixMode(GL_PROJECTION);
		glPushMatrix();
		glLoadIdentity();
		perspective();

		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();

	}
	~GUI()
	{
		glMatrixMode(GL_PROJECTION);
		glPopMatrix();
		glMatrixMode(GL_MODELVIEW);
		glPopMatrix();  //leave things in modelview mode
		glPopAttrib();
		glwin.PrintString({ 0, 0 }, "esc to quit.");
		glwin.SwapBuffers();
	}
	void start()
	{

	}
	std::vector<int2> stack;
	int2x2 splityc(int y,int r=1)
	{
		auto b = bounds.back();
		bounds.pop_back(); // wont need to ever go back to this.
		bounds.push_back({ { b[0].x,b[0].y + y + r},{b[1].x,b[1].y - y - r } });
		bounds.push_back({ { b[0].x,b[0].y },{b[1].x,y - r } });
		set();
		return b;
	}
	void splity(int &y, int r = 2)
	{
		auto b = splityc(y, r);
		bounds.push_back({ { b[0].x,b[0].y + y - r},{b[1].x,r * 2 } });
		if (focus(&y))
			y = mouse().y - b[0].y;  // mouse in local
		y = clamp(y, r, b[1].y - 1 - r);
		if (r)
		{
			set();
			glClearColor((float)(focus(&y)), (float)(inview(&y)), 0.5f, 1);
			glClear(GL_COLOR_BUFFER_BIT);
		}
		pop();
	}
	int2x2 splitync(int y, int r = 1)
	{
		auto b = bounds.back();
		bounds.pop_back(); // wont need to ever go back to this.
		bounds.push_back({ { b[0].x,b[0].y },{b[1].x,b[1].y - y - r } });
		bounds.push_back({ { b[0].x,b[0].y + b[1].y - y + r},{b[1].x,y - r } });
		set();
		return b;
	}
	void splityn(int &y, int r = 2)  // find a better way to combine with prev functino
	{
		auto b = splitync(y,r);
		bounds.push_back({ { b[0].x,b[0].y + b[1].y - y - r},{b[1].x,r * 2 }});
		if (focus(&y))
			y = b[1].y-(mouse().y - b[0].y);  // mouse in local
		y = clamp(y, r, b[1].y - 1 - r);
		if (r)
		{
			set();
			glClearColor((float)(focus(&y)), (float)(inview(&y)), 0.5f, 1);
			glClear(GL_COLOR_BUFFER_BIT);
		}
		pop();
	}
	int2x2 splitxc(int x, int r = 1)
	{
		auto b = bounds.back();
		bounds.pop_back(); // wont need to ever go back to this.
		bounds.push_back({ { b[0].x + x + r,b[0].y} ,{b[1].x - x - r,b[1].y }});
		bounds.push_back({ { b[0].x,b[0].y},{x - r,b[1].y } });
		set();
		return b;
	}
	void splitx(int &x, int r = 2)
	{
		auto b = splitxc(x, r);
		bounds.push_back({ { b[0].x + x - r,b[0].y},{r * 2,b[1].y } });
		if (focus(&x))
			x = mouse().x - b[0].x;  // mouse in local
		x = clamp(x, r, b[1].x - 1 - r);
		if (r)
		{
			set();
			glClearColor((float)(focus(&x)), (float)(inview(&x)), 0.5f, 1);
			glClear(GL_COLOR_BUFFER_BIT);
		}
		pop();
	}
	void ortho()
	{
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glOrtho(0.0, 1.0, 0.0, 1.0, -1.0, 1.0);
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
	}
	void perspective()
	{
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		gluPerspective(glwin.ViewAngle, (double)dims().x / dims().y, 0.01, 10);
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
	}
	void spinview(Pose &camera)
	{
		if (focus(&camera))
			camera.orientation = normalize(qmul(camera.orientation, quat_from_to(deproject(mousef()), deproject(tolocal(mouse_prev())))));
	}
	void trackball(float4 &q)
	{
		if (focus(&q))
			q = normalize(qmul( quat_from_to(deproject(tolocal(mouse_prev()))+float3(0,0,2), deproject(mousef()) + float3(0, 0, 2)),q));
	}
	void drawbar(float t,float3 c0,float3 c1 )
	{
		ortho();
		glBegin(GL_QUADS);
		glColor3fv(c0);
		glVertex3f(0, 1, -0.5f);
		glVertex3f(0, 0, -0.5f);
		glVertex3f(t, 0, -0.5f);
		glVertex3f(t, 1, -0.5f);
		glColor3fv(c1);
		glVertex3f(t, 1, -0.5f);
		glVertex3f(t, 0, -0.5f);
		glVertex3f(1, 0, -0.5f);
		glVertex3f(1, 1, -0.5f);
		glColor3f(1, 1, 1);
		glEnd();
	}
	void slider(int &i, const int2 range, std::string p = "")
	{
		if (range.y<=range.x)
			return;
		if (focus(&i))
		{
			i = clamp((int)(mousef().x * (range.y-range.x+1))+range.x, range.x, range.y);
		}
		drawbar((float)(i-range.x) / (range.y-range.x), float3(1,0,inview(&i)*0.5f), float3(0, 1, inview(&i)*0.5f));
		glwin.PrintString({ 0,-1 }, "%s%d/%d", p.c_str(), i,  range.y);
	}
	void slider(float &t, const float2 range = { 0.0f,1.0f }, std::string p = "")
	{
		if (focus(&t))
			t = clamp(mousef().x*(range.y-range.x)+range.x,range.x,range.y);
		drawbar((t-range.x) / (range.y-range.x), float3(1, 0, inview(&t)*0.5f), float3(0, 1, inview(&t)*0.5f));
		glwin.PrintString({ 0,-1 }, "%s%5.3f", p.c_str(), t);
	}
	void drawimage(const std::vector<byte3> &raster, const int2& dims) { ortho(); ::drawimage(raster, dims, { 0,0 }, { 1.0f,1.0f }); }
	void drawimage(std::pair<const std::vector<byte3> &, int2> im) { drawimage(im.first, im.second); }
	void drawimagem(const std::vector<byte3> &raster, const int2& dims) { ortho(); ::drawimage(raster, dims, { 0,1.0f }, { 1.0f,-1.0f }); }
	void drawimagem(std::pair<const std::vector<byte3> &, int2> im) { drawimagem(im.first, im.second); }
};


#endif // MISC_GL_H
