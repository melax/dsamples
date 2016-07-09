//
//  minimal program with reference code to show how to load and use a CNN trained using dclassified application
//  for more information please first see dclassified project.
//  
//  in windows desktop gui you can drag a .derp file onto the .exe, 
//  or, if using the command line, you can specify trained cnn file as first argument to .exe
//
//  its just text output 0-4 that should change depending on what the camera sees and what you trained the net for.
//

#include <iostream>
#include <algorithm>  // std::max
#include "dcam.h"
#include "misc_image.h"
//#include "misc.h"   // for freefilename()
#include "geometric.h"
#include "cnn.h"

#define NOMINMAX
#include <Windows.h>   // only used for messagebox


// note theres some duplicate code that could probably be moved into a common header...

CNN baby_cnn()
{
	CNN cnn({});
	cnn.layers.push_back(new CNN::LConv({ 16,16,2 }, { 5,5,2,16 }, { 12,12,16 }));
	cnn.layers.push_back(new CNN::LActivation<TanH>(12 * 12 * 16));
	cnn.layers.push_back(new CNN::LMaxPool({ 12,12,16 }));
	cnn.layers.push_back(new CNN::LFull(6 * 6 * 16, 32));
	cnn.layers.push_back(new CNN::LActivation<TanH>(32));
	cnn.layers.push_back(new CNN::LFull(32, 5));
	cnn.layers.push_back(new CNN::LSoftMax(5));
	cnn.Init(); // initializes weights
	return cnn;
}
CNN small_cnn()
{
	CNN cnn({});
	cnn.layers.push_back(new CNN::LConv({ 32,32,2 }, { 5,5,2,16 }, { 28,28,16 }));
	cnn.layers.push_back(new CNN::LActivation<TanH>(28 * 28 * 16));
	cnn.layers.push_back(new CNN::LMaxPool({ 28,28,16 }));
	cnn.layers.push_back(new CNN::LMaxPool({ 14,14,16 }));
	cnn.layers.push_back(new CNN::LFull(7 * 7 * 16, 32));
	cnn.layers.push_back(new CNN::LActivation<TanH>(32));
	cnn.layers.push_back(new CNN::LFull(32, 5));
	cnn.layers.push_back(new CNN::LSoftMax(5));
	cnn.Init();
	return cnn;
}
CNN reg_cnn()  // probably too big to learn quickly with small ground truth sample size
{
	CNN cnn({});
	cnn.layers.push_back(new CNN::LConv({ 64,64,2 }, { 5,5,2,16 }, { 60,60,16 }));
	cnn.layers.push_back(new CNN::LActivation<TanH>(60 * 60 * 16));
	cnn.layers.push_back(new CNN::LMaxPool({ 60,60,16 }));
	cnn.layers.push_back(new CNN::LMaxPool({ 30,30,16 }));
	cnn.layers.push_back(new CNN::LConv({ 15,15,16 }, { 8,8,16,256 }, { 8,8,64 }));
	cnn.layers.push_back(new CNN::LActivation<TanH>(8 * 8 * 64));
	cnn.layers.push_back(new CNN::LMaxPool({ 8,8,64 }));
	cnn.layers.push_back(new CNN::LFull(4 * 4 * 64, 64));
	cnn.layers.push_back(new CNN::LActivation<TanH>(64));
	cnn.layers.push_back(new CNN::LFull(64, 5));
	cnn.layers.push_back(new CNN::LSoftMax(5));
	cnn.Init();
	return cnn;
}

CNN(*cnntypes[])() = { baby_cnn,small_cnn,reg_cnn };
char *cnndescriptions[] = { "baby_cnn","small_cnn","reg_cnn" };

int main(int argc, char *argv[]) try
{
	if (argc != 2) 
		throw("usage test_cnn.exe trained_cnn_file.derp");

	std::cout << "creating network\n";


	int cnn_type=0;
	auto cnn = cnntypes[cnn_type]();
	int2 inputsize = ((CNN::LConv*)cnn.layers[0])->indims.xy();   // depending which is created, inputsize will be either 16x16 or 32x32


	int skip = 4;  // selection subregion will be center of input and of size 16*skip squared, with 128x128 max.  will subsample as necessary to fit cnn input size
	float2 drange = { 0.80f,0.30f };  // range of depth we care about.  note the .x > .y, this is so that closer things are more white and further things are more dark.


	Image<unsigned char> sample_dp(inputsize);
	Image<unsigned char> sample_ir(inputsize);
	std::vector<float> sample_in;
	std::vector<int> categories(5, 0);
	float3 catcolors[] = { { 0,0,1 },{ 0,1,0 },{ 1,0,0 },{ 1,1,0 },{ 1,0,1 } };
	std::vector<Image<byte3>> icons;
	auto sample_cl = Image<byte3>({ 1,1 }, std::vector<byte3>(1 * 1, { 127,127,127 }));
	for (auto c : categories)
		icons.push_back(sample_cl);


	std::ifstream cfile(argv[1], std::ios_base::binary | std::ios_base::in);
	if (!cfile.is_open())
		throw("unable to open file");
	cfile.read((char*)&cnn_type, sizeof(cnn_type));
	cfile.read((char*)&skip    , sizeof(skip));
	cfile.read((char*)&drange  , sizeof(drange));
	for (auto &icon : icons)
	{
		cfile.read((char*)&icon.cam, sizeof(icon.cam));
		icon.raster.resize(icon.cam.dim().x*icon.cam.dim().y);
		cfile.read((char*)icon.raster.data(), icon.raster.size()*sizeof(*icon.raster.data()));
	}
	cnn = cnntypes[cnn_type]();
	inputsize = ((CNN::LConv*)cnn.layers[0])->indims.xy();
	cnn.loadb(cfile);


	RSCam dcam;
    //dcam.enable_filter_depth = false;
	dcam.Init();
	float depth_scale = (dcam.dev) ? dcam.dev->get_depth_scale() : 0.001f;  // to put into meters    // if file assume file is mm
	if (dcam.dev->supports_option(rs::option::r200_lr_auto_exposure_enabled))
		dcam.dev->set_option(rs::option::r200_lr_auto_exposure_enabled, 1);

	void* current_selection = NULL;
	Pose camera;  // note this is the opengl camera, not the depth camera
	camera.orientation = normalize(float4(0, -3, 0, 1));
	int gv = 240*3/2; int gh = 320*3/2;
	int frame = 0;
	std::vector<float> errorhistory(128,1.0f);
	while (1)
	{
		auto dpimage = dcam.GetDepth();
		auto dleft = (const uint8_t *)dcam.dev->get_frame_data(rs::stream::infrared);
		Image<unsigned char> irimage(dcam.dcamera(), std::vector<unsigned char>(dleft, dleft + dcam.dim().x*dcam.dim().y));
		auto dp_image_g = Transform(dpimage, [&drange, depth_scale](unsigned short s) {return (unsigned char)clamp((int)(256.0f * ((s*depth_scale) - drange.x) / (drange.y - drange.x)), 0, 255); });
		auto dp_image_c = torgb(dp_image_g); 
		auto ir_image_c = torgb(irimage);    
		auto dcolor = (Image<byte3>({ 320, 240 }, (const byte3*)dcam.dev->get_frame_data(rs::stream::color_aligned_to_depth)));


		int2 swsize = int2(16,16) *skip, swoffset = irimage.dim() / 2 - swsize / 2;   // use 16,16 instead of input size here
		auto sample_raw = Crop(dpimage, swoffset, swsize);
		sample_dp = Crop(dp_image_g, swoffset, swsize);
		sample_ir = Crop(irimage, swoffset, swsize);
		sample_cl = Crop(dcolor, swoffset, swsize);

		if (sample_dp.cam.dim() != inputsize)
		{
			float2 scaleby = (asfloat2(inputsize) / asfloat2(sample_dp.cam.dim()));
			sample_dp = Sample(std::move(sample_dp), DCamera(inputsize, sample_dp.cam.focal()* scaleby, sample_dp.cam.principal()* scaleby,depth_scale));
			sample_ir = Sample(std::move(sample_ir), DCamera(inputsize, sample_ir.cam.focal()* scaleby, sample_ir.cam.principal()* scaleby,depth_scale));
		}
		sample_in = Transform(sample_dp, greyscaletofloat).raster;
		Append(sample_in, Transform(sample_ir, greyscaletofloat).raster);




		auto cnn_out = cnn.Eval(sample_in) ;
		int best = std::max_element(cnn_out.begin(),cnn_out.end())-cnn_out.begin();

		std::cout << "Category:  " << best << '\r'; 
		std::cout.flush();

	}
	return 0;
}
catch (const char *c)
{
	std::cerr << "Error: " << c << '\n';
	MessageBox(GetActiveWindow(), c, "FAIL", 0);
}
catch (std::exception e)
{
	std::cerr << "Error: " << e.what() << '\n';
	MessageBox(GetActiveWindow(), e.what(), "FAIL", 0);
}
