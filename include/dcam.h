

#ifndef RS_CAM_H
#define RS_CAM_H

#include <assert.h>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <exception>

#include "geometric.h"
#include "misc.h"
#include "misc_image.h"

#include "../third_party/librealsense/include/librealsense/rs.hpp"
#include "../third_party/librealsense/include/librealsense/rsutil.h"

#ifdef _WIN64
#ifdef _DEBUG
#pragma comment(lib, "../third_party/librealsense/librealsense.vc14/realsense-s/obj/Debug-x64/realsense-sd.lib") 
#else  // release x64
#pragma comment(lib, "../third_party/librealsense/librealsense.vc14/realsense-s/obj/Release-x64/realsense-s.lib")   
#endif // _DEBUG vs release for x64
#else  // we have _WIN32
#ifdef _DEBUG
#pragma comment(lib, "../third_party/librealsense/librealsense.vc14/realsense-s/obj/Debug-Win32/realsense-sd.lib")  
#else  // release 32 
#pragma comment(lib, "../third_party/librealsense/librealsense.vc14/realsense-s/obj/Release-Win32/realsense-s.lib")   // pragma easier than using settings within project properties
#endif // _DEBUG vs release for Win32 
#endif // _WIN64 vs _WIN32

class DCam // generic intrinsics depth camera interface
{
public:
	rs::intrinsics zintrin;
	float          depth_scale = 0.001f;
	float          get_depth_scale(){ return depth_scale;  }
	const int2&    dim()      const { return *(reinterpret_cast<const int2*  >(&zintrin.width)); }
	int2&          dim()            { return *(reinterpret_cast<      int2*  >(&zintrin.width)); }
	const float2&  focal()    const { return *(reinterpret_cast<const float2*>(&zintrin.fx));    }
	float2&        focal()          { return *(reinterpret_cast<      float2*>(&zintrin.fx));    }
	const float2&  principal()const { return *(reinterpret_cast<const float2*>(&zintrin.ppx));   }
	float2&        principal()      { return *(reinterpret_cast<      float2*>(&zintrin.ppx));   }

	float2  fov()   const { return{ zintrin.hfov(), zintrin.vfov() }; }  // in degrees
	inline float3   deprojectz(float2 p, float d)             const { return (const float3 &)zintrin.deproject({p.x, p.y}, d); }
	inline float3   deprojectz(int2 p, unsigned short d)      const { return deprojectz(float2((float)p.x, (float)p.y), d); }
	inline float3   deprojectz(int x, int y, unsigned short d)const { return deprojectz(float2((float)x  , (float)y  ), d); }
	inline float3   deprojectz(int x, int y, float d)         const { return deprojectz(float2((float)x  , (float)y  ), d); }
	inline float2   project(const float3 &v)                  const { return (float2&) zintrin.project((const rs::float3&) v); }
	inline float2x2 deprojectextents()                        const { return float2x2(deprojectz(float2(0, 0), 1.0f).xy(), deprojectz(asfloat2(dim()), 1.0f).xy()); }   
	std::istream&   intrinsicsimport(std::istream &is)       { return  is >> dim()        >> focal()         >> principal()        >> depth_scale; }
	std::ostream&   intrinsicsexport(std::ostream &os) const { return  os << dim() << " " << focal() << "  " << principal() << " " << depth_scale; }
	friend std::istream& operator>>(std::istream &is,       DCam &dcam) { return dcam.intrinsicsimport(is); }
	friend std::ostream& operator<<(std::ostream &os, const DCam &dcam) { return dcam.intrinsicsexport(os); }
	void intrinsicsimport(std::string filename)
	{
		if (filename=="")
			return;
		std::ifstream fs(filename.c_str());
		if (!fs.is_open())
			return;
		fs >> *this;
	}
	void intrinsicsexport(std::string filename) const
	{
		std::ofstream camfile(filename.c_str(), std::ios_base::trunc | std::ios_base::out);
		if (!camfile.is_open())
			throw filename + "  unable to open intrinsics output file";
		camfile << *this << " " << fov() << "  // w h fx fy px py depth_scale fovx fovy (degrees)\n";
	}
	DCamera dcamera() { return { dim(), focal(), principal() ,get_depth_scale() };  }
};

class RSCam : public DCam   // wrapper allows for easy interchange between device and file stream
{
public:
	rs::context ctx;
	rs::device * dev;

	std::ifstream                    filein;
	std::vector<unsigned short>      fbuffer;
	bool                             enable_filter_depth = true;

	std::vector<uint16_t>            filtered_depth;
	std::vector<uint16_t>            background;
	void  addbackground(const unsigned short *dp,unsigned short fudge=3) 
	{
		background.resize(product(dim()),4096);
		for (int i = 0; i < dim().x*dim().y; i++)
			background[i] = std::min(background[i], (unsigned short) (dp[i]-fudge));
	}

	Image<unsigned short> FilterDS4(unsigned short *dp)  // modified in-place  very ds4 specific
	{
		Image<unsigned short> filtered_depth(dcamera(),std::vector<unsigned short>(dp, dp + product(dim())));
		dp = filtered_depth.raster.data();
		auto dleft = (const uint8_t *)dev->get_frame_data(rs::stream::infrared);
		// depth_scale = dev->get_depth_scale() ;   <- to do the math in meters use this term   and dont assume native is mm
		auto hasneighbor = [](unsigned short *p, int stride) ->bool	{ return (std::abs(p[stride] - p[0]) < 10 || std::abs(p[-stride] - p[0]) < 10); };
		for (int i = 0; i < dim().x*dim().y; i++)
		{
			if(dp[i]<30  || dleft[i] <8)  // ignore dark pixels - these could be background, but stereo match put them in foreground
			{
				dp[i] = 4096;
			}
		}
		for (int i = 0; i < dim().x*dim().y; i++) // for (int2 p: rect_iteration(dim()))
		{
			int x = i%dim().x, y = i / dim().x;
			if (x < 2 || x >= dim().x - 2 || y < 2 || y >= dim().y - 2)
				continue;
			if ( !hasneighbor(dp + i, 1) || !hasneighbor(dp + i, dim().x) || !hasneighbor(dp + i, 2) || !hasneighbor(dp + i, dim().x * 2))  // ignore flying pixels
			{
				dp[i] = 4096;
			}
		}
		for (int i = 0; i < dim().x*dim().y; i++)
		{
			if (background.size() == dim().x*dim().y)
				if (dp[i]>background[i])
					dp[i] = 4096;
		}
		return filtered_depth;
	}
	Image<unsigned short> FilterIvy(unsigned short *dp)  // modified in-place  very ds4 specific
	{
		Image<unsigned short> filtered_depth(dcamera(), std::vector<unsigned short>(dp, dp + product(dim())));
		dp = filtered_depth.raster.data();
		for (int i = 0; i < product(dim()); i++)
		{
			if (dp[i] == 0)
			{
				dp[i] = (unsigned short) (4.0f / dev->get_depth_scale());
			}
		}
		return filtered_depth;
	}
	Image<unsigned short>  GetFileDepth()
	{
		Image<unsigned short> image(dcamera(), std::vector<unsigned short>(product(dim())));
		filein.read((char*)image.raster.data(), sizeof(unsigned short)*image.raster.size());
		return image;
	}
	Image<unsigned short> ToImage(unsigned short *p)
	{
		Image<unsigned short> image(dcamera() , std::vector<unsigned short>(p, p + product(dim())));
		return image;
	}
	Image<unsigned short> GetDepth()
	{ 
		if (filein.is_open()) return GetFileDepth(); 
		dev->wait_for_frames();
		auto rawdepth = (uint16_t  *)dev->get_frame_data(rs::stream::depth);
		return (enable_filter_depth)? ((dev->get_stream_mode_count(rs::stream::infrared2))?FilterDS4(rawdepth):FilterIvy(rawdepth)): ToImage(rawdepth);  // filter is r200 specific 
	}
	std::string CamName() { return (dev) ? ((dev->get_stream_mode_count(rs::stream::infrared2)) ? "dscam" : "ivycam") : "filecam"; }

	inline bool Init()
	{
		if (ctx.get_device_count() == 0) throw std::runtime_error("No device found");
		dev = ctx.get_device(0);
		std::cout << "Found Device:  '" << dev->get_name() << "'" << std::endl;
		std::cout << "Firmware version: " << dev->get_firmware_version() << std::endl;
		std::cout << "Serial number: " << dev->get_serial() << std::endl;
		try
		{
			dev->enable_stream(rs::stream::depth, 320, 240, rs::format::z16, 60);
			dev->enable_stream(rs::stream::infrared, 0, 0, rs::format::y8, 0);
			//dev->enable_stream(rs::stream::infrared2, 0, 0, rs::format::y8, 0);
			dev->enable_stream(rs::stream::color, 640, 480, rs::format::rgb8, 30);
			dev->start();
		}
		catch (...)
		{
			dev->enable_stream(rs::stream::depth, 640, 480, rs::format::z16, 60);
			dev->enable_stream(rs::stream::infrared, 0, 0, rs::format::y8, 0);
			dev->start();
		}
		//rs_apply_ivcam_preset((rs_device*)dev, 6); // from leo: more raw IVCAM data (always 60Hz!)

		zintrin = dev->get_stream_intrinsics(rs::stream::depth);
		depth_scale = dev->get_depth_scale();
		//  leo suggested:    rs_apply_depth_control_preset((rs_device*)dev, 5);
		std::cout << "ds4 dims: " << *this << " " << fov() << "  // w h fx fy px py depth_scale fovx fovy (degrees)\n"; 


		return true;
	}
	inline bool Init(const char *filename_)
	{
		if (!filename_ || !*filename_)
			return Init();
		filein.open(filename_, std::ios_base::binary | std::ios_base::in);
		if (!filein.is_open())
			throw(std::exception((std::string("unable to open depth camera file:  ")+filename_).c_str()));
		intrinsicsimport(std::string(filename_) +"i");  // look for intrinsics file 
		fbuffer.resize(dim().x*dim().y);
		return true;
	}
	RSCam() :dev(nullptr)
	{
	}
};

#endif
