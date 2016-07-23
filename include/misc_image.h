//
// typical generic image utility routines
// 

#pragma once
#ifndef MISCIMAGE_H
#define MISCIMAGE_H

#include <iostream>
#include <fstream>
#include <string>
#include <exception>

#include "geometric.h"

class DCamera // generic depth camera interface  since other version depends on external library
{
	int2                             dim_       = { 320, 240 };
	float2                           focal_     = { 241.811768f, 241.811768f };  // focal length 1 pixel = 1 unit
	float2                           principal_ = { 162.830505f, 118.740089f };
public:
	float                            depth_scale = 0.001f;
	float                            get_depth_scale() { return depth_scale; }
	Pose                             pose;
	int2&                            dim()             { return dim_; }
	const int2&                      dim()       const { return dim_; }
	float2&                          focal()           { return focal_; } // focal length 1 pixel = 1 unit
	const float2&                    focal()     const { return focal_; }
	float2&                          principal()       { return principal_; }
	const float2&                    principal() const { return principal_; }
	DCamera() {}
	DCamera(int2 dim, float2 focal, float2 principal,float depth_scale) :dim_(dim), focal_(focal), principal_(principal),depth_scale(depth_scale) {}
	DCamera(int2 dim) :dim_(dim), focal_(asfloat2(dim)), principal_(asfloat2(dim)/2.0f) {}
	float3   deprojectz(const float2 &p, float d)       const { return float3((p.x - principal().x) / focal().x, (p.y - principal().y) / focal().y, 1.0f) * d; }
	float3   deprojectz(const int2 &p,unsigned short d) const { return deprojectz(asfloat2(p), (float)d); }
	float2   projectz(const float3 &v)                  const { return v.xy() / v.z * focal() + principal(); }
	std::vector<float2> projectz(const std::vector<float3> &pts) { return Transform(pts, [&](const float3 &v) {return this->projectz(v); }); }  // array version of projectz
	float2x2 deprojectextents()                         const { return float2x2(deprojectz(float2(0, 0), 1.0f).xy(), deprojectz(asfloat2(dim()), 1.0f).xy()); }  // upper left and lower right xy corners at plane z==1
	float2 fov() const { return{ atan2(principal().x + 0.5f, focal().x) + atan2(dim().x - principal().x - 0.5f, focal().x), atan2(principal().y + 0.5f, focal().y) + atan2(dim().y - principal().y - 0.5f, focal().y) }; }   // using ds4 conventions with that 0.5 offset
	bool intrinsicsimport(std::string filename)
	{
		std::ifstream camfile(filename);
		if (!camfile.is_open())
			return false;
		camfile >> dim() >> focal() >> principal() >> depth_scale;
		return true;
	}
	void intrinsicsexport(std::string filename)
	{
		std::ofstream camfile(filename, std::ios_base::trunc | std::ios_base::out);
		if (!camfile.is_open())
			throw "unable to open intrinsics output file";
		camfile << dim() << " " << focal() << "  " << principal() << " " << depth_scale <<" " << fov()*180.0f / 3.14159f << "  // w h fx fy px py depth_scale fovx fovy (degrees)\n";
	}
	template<class F> friend void visit_fields(DCamera & cam, F f);
};
inline std::ostream & operator << (std::ostream & out, const DCamera& dcam) { return out << dcam.dim() << " " << dcam.focal() << "  " << dcam.principal() << " " << dcam.depth_scale; }
template<class F> void visit_fields(DCamera & cam, F f) { f("dims", cam.dim_); f("focal", cam.focal_); f("principal", cam.principal_); f("depth_scale", cam.depth_scale); }

inline DCamera camcrop(const DCamera &c, int2 offset, int2 dim) { return{ dim, c.focal(), c.principal() - float2((float)offset.x, (float)offset.y) ,c.depth_scale}; }
inline DCamera camsub(const DCamera &c, int s) { return{ c.dim() / s, c.focal() / (float)s, c.principal() / (float)s ,c.depth_scale }; }
inline DCamera operator*(const DCamera &c, int s) { return{ c.dim() * s, c.focal() * (float)s, c.principal() * (float)s ,c.depth_scale }; }
inline DCamera operator/(const DCamera &c, int s) { return{ c.dim() / s, c.focal() / (float)s, c.principal() / (float)s ,c.depth_scale }; }

//-----------------
// using this 'greyscale' for double/float to/from byte representations with range -0.5 to 0.5 for float and 0 to 255 for byte representations  
inline unsigned char togreyscale(float  x)        { return (unsigned char)clamp((x + 0.5f)*255.0f, 0.0f, 255.0f); }
inline unsigned char togreyscale(double x)        { return togreyscale((float)x); }
inline unsigned char togreyscale(unsigned char x) { return x; }
inline float         greyscaletofloat(unsigned char c) { return c / 255.0f - 0.5f; } //centering transformation
inline double        greyscaletodouble(unsigned char c) { return c / 255.0f - 0.5f; } //centering tranformation



template<class T> std::vector<T> subimage(const T *src, int stride, int2 dim)
{
	std::vector<T> dst(dim.x*dim.y);
	for (int y = 0; y < dim.y; y++) for (int x = 0; x < dim.x; x++)
		dst[y*dim.x + x] = src[y*stride + x];
	return dst;
}
template<class T> std::vector<T> subimage(const std::vector<T> &src, int2 srcdim, int2 dim, int2 offset) { return subimage(src.data() + offset.x + offset.y*srcdim.x, srcdim.x, dim); }
template<class T> std::vector<T> subimage(const std::vector<T> &src, const DCamera &sc, const DCamera &dc) { return subimage(src, sc.dim(), dc.dim(), asint2(sc.principal() - dc.principal())); }


template<class T, class F>
std::vector<T> DownSample(F f, const std::vector<T> &src, int2 sdim)  // max,min,plus 2x2 subsampling
{
	int2 ddim = sdim / 2;
	std::vector<T> dst(ddim.x*ddim.y);
	auto s = [&src, &sdim](int2 p) { return src[p.y*sdim.x + p.x]; };
	for (int2 p(0, 0); p.y < sdim.y; p.y += 2) for (p.x = 0; p.x < sdim.x; p.x += 2)
		dst[p.y / 2 * ddim.x + p.x / 2] = f(f(s(p + int2(0, 0)), s(p + int2(1, 0))), f(s(p + int2(0, 1)), s(p + int2(1, 1))));
	return dst;
}

template<class T> std::vector<T> DownSampleMin(const std::vector<T> &src, int2 dim) { return DownSample([](T a, T b) {return std::min<T>(a, b); }, src, dim); }
template<class T> std::vector<T> DownSampleFst(const std::vector<T> &src, int2 dim) { return DownSample([](T a, T) {return a;                 }, src, dim); }

template<class T> std::vector<T> UpSample(const std::vector<T> &src, int2 sdim)
{
	std::vector<T> dst(src.size() * 4);
	for (int2 p(0, 0); p.y < sdim.y * 2; p.y++) for (p.x = 0; p.x < sdim.x * 2; p.x++)
		dst[p.y*sdim.x * 2 + p.x] = src[p.y / 2 * sdim.x + p.x / 2];
	return dst;
}


template <class T> struct Image
{
	DCamera cam;          // camera intrinsics including width,height in pixels of image 
	std::vector<T> raster;

	int2&      dim()               { return cam.dim(); }
	const int2 dim() const         { return cam.dim(); }
	T&         pixel(int2 p)       { return raster[p.y*dim().x + p.x]; }
	const T&   pixel(int2 p) const { return raster[p.y*dim().x + p.x]; }
	Image(int2 dim    ) :cam(dim), raster(dim.x, dim.y  ) {}
//	Image(int2 dim,T t) :cam(dim), raster(dim.x, dim.y,t) {}
	Image(DCamera cam)  :cam(cam), raster(cam.dim().x*cam.dim().y,T(0)) {}
	Image(DCamera cam, std::vector<T> data) :cam(cam), raster(data)  {}
	Image(int2    dim, std::vector<T> data) :cam(dim), raster(data)  {}
	Image(int2    dim, const T *data) :cam(dim), raster(data,data+dim.x*dim.y) {}
	operator std::pair<const std::vector<T>&, int2>() const { return { raster,dim() }; }
};
template<class T> Image<T> Crop(const Image<T> &src, int2 offset, int2 dim)
{
	return {DCamera(dim, src.cam.focal(), src.cam.principal() - asfloat2(offset),src.cam.depth_scale), subimage(src.raster, src.dim(), dim, offset) };
}

template <class T> Image<T> UpSample(const Image<T> &image)
{
	return  { image.cam * 2, UpSample(image.raster,image.dim()) };
}
template <class T> Image<T> DownSampleFst(const Image<T> &src)
{
	return{ src.cam / 2 , DownSampleFst(src.raster,src.dim()) };
}
template <class T> Image<T> DownSampleMin(const Image<T> &src)
{
	return{ src.cam / 2 , DownSampleMin(src.raster,src.dim()) };
}



template <class T> Image<T> Sample(const Image<T> &src, const DCamera &dstcam, T background = 0)  // uses point sampling
{
	Image<T> dst(dstcam);
	int2 pp;
	for (auto p : rect_iteration(dst.dim())) // for (int y = 0; y < dst.dim().y; y++) for (int x = 0; x < dst.dim().x; x++)
		dst.pixel(p) = within_range(pp = asint2(src.cam.projectz(dst.cam.pose*dst.cam.deprojectz(asfloat2(p), 1.f))), int2(0, 0), src.dim() - int2(1, 1)) ? src.pixel(pp) : background;
	return dst;
}

// Note that for depth image, the pixel values are distance from the image plane.
//  because the sampling may include a rotated camera, we here adjust the distance (sampled value) to be distance from the destination camera's image plane
template <class T> Image<T> SampleD(const Image<T> &src, const DCamera &dstcam, T background = 0)  // uses point sampling
{
	Image<T> dst(dstcam);
	float3 ppdir = dstcam.pose * dstcam.deprojectz(dstcam.principal(), 1.0f);
	int2 pp;
	for (auto p : rect_iteration(dst.dim())) // for (int y = 0; y < dst.dim().y; y++) for (int x = 0; x < dst.dim().x; x++)
		dst.pixel(p) = within_range(pp = asint2(src.cam.projectz(dst.cam.pose*dst.cam.deprojectz(asfloat2(p), 1.f))), int2(0, 0), src.dim() - int2(1, 1)) ? (T)dot(ppdir, src.cam.deprojectz(pp, src.pixel(pp))) : background;
	return dst;
}


template<typename F, typename S> auto Transform(const Image<S> &src, F f) ->  Image<decltype (f(S()))> { std::vector<decltype (f(S()))>  dst(src.raster.size()); std::transform(src.raster.begin(), src.raster.end(), dst.begin(), f); return Image<decltype (f(S()))>(src.cam, dst); }

template<class T> Image<unsigned char> togreyscale(const Image<T> &src)
{
	return Transform(src,[](T x) {return togreyscale(x); });
}

Image<byte3> torgb(const Image<unsigned char> &src)  // grey to rgb for drawing
{
	return  Transform(src, [](unsigned char p) {return byte3(p, p, p); });
}
Image<byte3> torgb(const Image<float > &src) { return torgb(togreyscale(src)); }
Image<byte3> torgb(const Image<double> &src) { return torgb(togreyscale(src)); }


template <class T> int2 imagefindmax(const T *image, int2 dim, std::function<bool(const T& a, const T&b)> f = [](const T& a, const T&b) {return a > b; })
{
	int2 best(0, 0);
	for (int2 p(0, 0); p.y < dim.y; p.y++) for (p.x = 0; p.x < dim.x; p.x++)
		if (f(image[dot(p, { 1, dim.x })], image[dot(best, { 1,dim.x })]))
			best = p;
	return best;
}
template <class T> int2 imagefindmax(const Image<T> &image, std::function<bool(const T& a, const T&b)> f = [](const T& a, const T&b) {return a > b; })
{
	return imagefindmax(image.raster.data(), image.dim(), f);
}



inline Image<unsigned char> Threshold(const Image<unsigned short> &depthdata, std::function<bool(unsigned short)> f)
{
	return Transform(depthdata, [&](unsigned short d)->unsigned char {return (f(d)) ? 255 : 0; });
}
inline Image<unsigned char> DistanceTransform(Image<unsigned char> image)  // just manhattan
{
	int2 dim = image.dim();
	auto cm = [dim](int2 p) {return clamp(p, int2(0,0), dim - int2(1, 1)); };
	for(auto p: rect_iteration(dim))
		image.pixel(p) = (unsigned char)std::min(255, std::min(std::min(image.pixel(cm(p - int2(1,0))) + 1, image.pixel(cm(p-int2(0,1))) + 1), image.pixel(p) + 0));
	for (auto r : rect_iteration(dim))
	{
		int2 p = dim - int2(1, 1) - r ;
		image.pixel(p) = (unsigned char)std::min(255, std::min(std::min(image.pixel(cm(p + int2(1, 0))) + 1, image.pixel(cm(p + int2(0, 1))) + 1), image.pixel(p) + 0));
	}
	return image;
}
template<class T> Image<T> SetBorder(Image<T> image, T v = T(0))
{
	int2 dim = image.dim();
	for (int y = 0; y < dim.y; y++)
		image.pixel({ 0,y }) = image.pixel({ dim.x - 1, y }) = v;
	for (int x = 0; x < dim.x; x++)
		image.pixel({ x,0 }) = image.pixel({ x, dim.y - 1 }) = v;
	return image;
}


Image<unsigned short> Clip(Image<unsigned short> depth, float4 plane,unsigned short val)
{
	for (int2 p(0, 0);p.y < depth.dim().y;p.y++) for (p.x = 0;p.x < depth.dim().x;p.x++)
		if (dot(depth.cam.deprojectz(p, depth.pixel(p)), plane.xyz()) + plane.w < 0)
			depth.pixel(p) = val;
	return depth;
}
inline bool within_range(float t, float2 r) { return t >= r.x && t < r.y; }
template<class T> std::vector<float3> PointCloud(const Image<T> &dimage, float2 filter_range, float depth_scale = 0.001f)
{
	std::vector<float3> pointcloud;
	float d;
	for (int2 p : rect_iteration(dimage.dim()))  // p.y and p.x iterate over dimensions of image
		if(within_range(d= (dimage.pixel(p)* depth_scale),filter_range))
			pointcloud.push_back(  dimage.cam.deprojectz(float2(p),d )  );  
	return pointcloud;
}
std::vector<float3> Mirror(std::vector<float3> points, float4 plane)
{
	for (auto &p : points)
		p += plane.xyz()* (dot(float4(p, 1), plane)*-2.0f);
	return points;
}



#if _MSC_VER > 1800
auto PlaneSplit(const std::vector<float3> &points,float4 plane, float epsilon=0.02f)
{
	struct result { std::vector<float3> under, coplanar, over; };
	std::vector<float3> b[3]; // under, coplanar, over;
	for (auto p : points)
	{
		float pd = dot(float4(p, 1), plane);
		b[(pd > -epsilon) + (pd > epsilon)].push_back(p);
	}
	return result{b[0], b[1], b[2]};
	//return make_tuple(move(b[0]), move(b[1]), move(b[2])); //  { under, coplanar, over };
}
auto MirrorPlaneSplit(const std::vector<float3> &points, float4 plane, float epsilon = 0.02f)
{
	auto r = PlaneSplit(points, plane, epsilon);
	r.under = Mirror(std::move(r.under), plane);
	return r;
}
#endif


//  Segment() - not a generic routine...
//  The following is specific to the depth camera tracking application 
// the implementation is a bit ad hoc.  its a first pass to get the plumbing all working.
static float2 g_com(0,0);   // for debugging only
static int2   g_entry(0, 0); // for debugging only
static int2   g_extreme(0, 0); // for debugging only

Image<unsigned short> Segment(const Image<unsigned short> &depth, int entry_options = 0xF, float2 wrange = { 0.1f,0.65f } , float depth_scale = 0.001f)
{
	auto dim = depth.dim();
	Image<unsigned short> depthsmall = DownSampleMin(DownSampleMin(depth));
	ushort2 wranged = ushort2(wrange / depth_scale);
	auto dt = DistanceTransform(Threshold(depthsmall, [wranged](unsigned short d) {return d >= wranged.x && d < wranged.y; }));

	int2 entry(0, 0);
	if (entry_options & 1) for (int2 p(0, dt.dim().y-1); p.x < dt.dim().x;p.x++) if (dt.pixel(p) > dt.pixel(entry))  entry = p;
	if (entry_options & 2) for (int2 p(0, 0           ); p.x < dt.dim().x;p.x++) if (dt.pixel(p) > dt.pixel(entry))  entry = p;
	if (entry_options & 4) for (int2 p(dt.dim().x-1, 0); p.y < dt.dim().y;p.y++) if (dt.pixel(p) > dt.pixel(entry))  entry = p;
	if (entry_options & 8) for (int2 p(0           , 0); p.y < dt.dim().y;p.y++) if (dt.pixel(p) > dt.pixel(entry))  entry = p;

	float avgdepth = 0;
	float2 com(0, 0);
	float wtotal = 0.0f;
	int count = 0;
	const int min_blob_radius = 2;  // note that this will be after downsampled for distance transform    had a camera that was having edge noise issues

	for(auto p: rect_iteration(dt.dim())) 
	{
		if (dt.pixel(p) >= min_blob_radius) // && p.y < firsty + 128 / 4)
		{
			float w = length(float2(p - entry));
			wtotal += w;
			com += asfloat2(p)*w;
			avgdepth += depthsmall.pixel(p)*w;   // although 32 bit float, avgdepth sum happens in unsigned short raster units at its depth_scale
			count++;
		}
	}
	if (count)
	{
		avgdepth *= depth_scale/wtotal;  // avgdepth convert to meters and divide by total weighting
		com /= wtotal;
	}
	float2 extreme = float2(entry);
	for (auto p : rect_iteration(dt.dim()))
	{
		if (dt.pixel(p) >= min_blob_radius) // && p.y < firsty + 128 / 4)
			if (dot(float2(p) - float2(entry), com - float2(entry)) > dot(float2(extreme) - float2(entry), com - float2(entry)))
				extreme = float2(p);
	}
	float scale = 2.0f;
	float angle = 0.0f;
	const float diam = 0.17f; //  diameter_of_interest =
	if (count)
	{
		scale = 2.0f / clamp(avgdepth / 0.450f, 0.55f, 2.0f);     // magic number alert, the 45cm comes from using it previously as a target distance before i had adaptive scale
		float exrad = dot(extreme - com, normalize(com - float2(entry)));
		com +=  normalize(com - float2(entry)) * (exrad - diam/2.0f/avgdepth * dt.cam.focal().x)  ;   // shift com to be half of diam ~20cm (ie 10cm) away from extreme point 
		angle = atan2((float)com.x - entry.x, (float)entry.y - com.y);
	}

	DCamera dstcam({ 64,64 }, float2( avgdepth*64.0f/diam) , { 32,32 },depth.cam.depth_scale);  

	float3 zd = linalg::normalize(dstcam.deprojectz(asfloat2(dstcam.dim()/2), 1.0f));
	float3 yd = linalg::normalize(ProjectOntoPlane({ zd,0.0f }, dt.cam.deprojectz(float2(entry), 1.0f) - dstcam.deprojectz(asfloat2(dstcam.dim() / 2), 1.0f) ));
	dstcam.pose.orientation = qmul(quat_from_to(dt.cam.deprojectz(dt.cam.principal(), 1.0f), dt.cam.deprojectz(com, 1.0f)), QuatFromAxisAngle(float3(0, 0, 1), angle)); //quatfrommat({ cross(yd, zd), yd, zd });
	dstcam.principal() = asfloat2(dstcam.dim())*0.5f;

	g_com = linalg::clamp(com, float2(0, 0), float2(dt.cam.dim() - int2(1, 1)));
	g_entry=entry;
	g_extreme=int2(extreme);

	return SampleD(depth, dstcam, (unsigned short)(4.0f/depth_scale));  // 4m away for anything outside of src depth image
}



#endif // MISCIMAGE_H

