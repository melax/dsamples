

#include <cctype>  // std::tolower
#include <iostream>
#include <algorithm>  // std::max
#include <immintrin.h>  // rdtsc
#include "dcam.h"
#include "misc_image.h"
#include "misc.h"   // for freefilename()
#include "geometric.h"
#include "glwin.h"
#include "misc_gl.h"
#include "gl_gui.h"
#include "cnn.h"
#include "tcpsocket.h"


CNN baby_cnn()
{
	CNN cnn({});
	cnn.layers.push_back(new CNN::LConv({ 16, 16, 2 }, { 5, 5, 2, 16 }, { 12, 12, 16 }));
	cnn.layers.push_back(new CNN::LActivation<TanH>(12 * 12 * 16));
	cnn.layers.push_back(new CNN::LMaxPool(int3(12, 12, 16)));
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
	cnn.layers.push_back(new CNN::LConv({ 32, 32, 2 }, { 5, 5, 2, 16 }, { 28, 28, 16 }));
	cnn.layers.push_back(new CNN::LActivation<TanH>(28 * 28 * 16));
	cnn.layers.push_back(new CNN::LMaxPool(int3(28, 28, 16)));
	cnn.layers.push_back(new CNN::LMaxPool(int3(14, 14, 16)));
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
	cnn.layers.push_back(new CNN::LConv({ 64, 64, 2 }, { 5, 5, 2, 16 }, { 60, 60, 16 }));
	cnn.layers.push_back(new CNN::LActivation<TanH>(60 * 60 * 16));
	cnn.layers.push_back(new CNN::LMaxPool(int3(60, 60, 16)));
	cnn.layers.push_back(new CNN::LMaxPool(int3(30, 30, 16)));
	cnn.layers.push_back(new CNN::LConv({ 15, 15, 16 }, { 8, 8, 16, 256 }, { 8, 8, 64 }));
	cnn.layers.push_back(new CNN::LActivation<TanH>(8 * 8 * 64));
	cnn.layers.push_back(new CNN::LMaxPool(int3(8, 8, 64)));
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
	//hack();
	std::cout << "creating network\n";


	int cnn_type=0;
	auto cnn = cnntypes[cnn_type]();
	int2 inputsize = ((CNN::LConv*)cnn.layers[0])->indims.xy();   // depending which is created, inputsize will be either 16x16 or 32x32

	GLWin glwin((argc==2)? (std::string("using previously trained cnn: ")+argv[1]).c_str() : "librealsense simple CNN classifier util - alpha version", 1500, 937);

	RSCam dcam;
	//dcam.enable_filter_depth = false;
	dcam.Init();
	float depth_scale = (dcam.dev) ? dcam.dev->get_depth_scale() : 0.001f;  // to put into meters    // if file assume file is mm

	//glwin.ViewAngle = dcam.fov().y;
	float viewdist        = 0.75f;

	int skip = 4;  // selection subregion will be center of input and of size 16*skip squared, with 128x128 max.  will subsample as necessary to fit cnn input size
	float2 drange = { 0.80f,0.30f };  // range of depth we care about.  note the .x > .y, this is so that closer things are more white and further things are more dark.

	std::default_random_engine rng;
	int    training      = false;
	int    samplereview  = 0;
	int    currentsample = 0;
	bool   trainstarted  = false;
	bool   sinusoidal    = false;
	float  time          = 0.0f;

	std::vector<std::vector<float>> samples;
	std::vector<std::vector<float>> labels;
	std::vector<Image<byte3>>       snapshots;

	Image<unsigned char> sample_dp(inputsize);
	Image<unsigned char> sample_ir(inputsize);
	auto sample_cl = Image<byte3>({ 1,1 }, std::vector<byte3>(1*1, { 127,127,127 }));
	std::vector<float> sample_in;
	std::vector<int> categories(5,0);
	float3 catcolors[] = { {0,0,1},{0,1,0},{1,0,0},{1,1,0},{1,0,1} };
	std::vector<Image<byte3>> icons;
	for (auto c : categories)
		icons.push_back(sample_cl);
	auto addsample = [&](int c) 
	{
		if (samplereview) {samplereview = 0; return; } 
		if (icons[c].raster.size()<=1)icons[c] = sample_cl; 
		snapshots.push_back(sample_cl); 
		categories[c]++; 
		samples.push_back(sample_in); 
		labels.push_back(std::vector<float>(categories.size(), 0.0f)); 
		labels.back()[c] = 1.0f;
	};

	if (argc == 2)  // load everything
	{
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
		trainstarted = true;
		while (!cfile.eof())
		{
			DCamera cam;
			cfile.read((char*)&cam, sizeof(cam));
			Image<byte3> cl(cam,std::vector<byte3>(cam.dim().x*cam.dim().y));
			cfile.read((char*)cl.raster.data(), cl.raster.size()*sizeof(*cl.raster.data()));
			std::vector<float> sample(2*inputsize.x*inputsize.y);
			std::vector<float> label(5, 0.0f);
			cfile.read((char*)sample.data(), sample.size()*sizeof(float));
			cfile.read((char*)label.data() ,  label.size()*sizeof(float));
			samples.push_back(sample);
			labels.push_back(label);
			snapshots.push_back(sample_cl);
			categories[std::max_element(label.begin(), label.end()) - label.begin()]++;
		}
	}
	auto save_everything = [&](bool append_training_set)
	{
		std::ofstream cfile(freefilename("trained_net",".derp"), std::ios_base::binary);
		if (!cfile.is_open())
			throw("unable to open file");
		cfile.write((char*)&cnn_type, sizeof(cnn_type));
		cfile.write((char*)&skip    , sizeof(skip));
		cfile.write((char*)&drange  , sizeof(drange));
		for (auto &icon : icons)
		{
			cfile.write((char*)&icon.cam, sizeof(icon.cam));
			cfile.write((char*)icon.raster.data(), icon.raster.size()*sizeof(*icon.raster.data()));
		}
		cnn.saveb(cfile);
		for (unsigned int i = 0; append_training_set && i < samples.size(); i++)
		{
			cfile.write((char*)&snapshots[i].cam, sizeof(snapshots[i].cam));
			cfile.write((char*)snapshots[i].raster.data(), snapshots[i].raster.size()*sizeof(*snapshots[i].raster.data()));
			cfile.write((char*)samples[i].data(), samples[i].size()*sizeof(float));
			cfile.write((char*) labels[i].data(),  labels[i].size()*sizeof(float));
		}
	};
	float sint = 0.0f; 
	glwin.keyboardfunc = [&](unsigned char key, int x, int y)->void
	{
		switch (std::tolower(key))
		{
		case 'q': case 27:  exit(0);    break;   // 27 is ESC
		case '=': case '+': skip++;     break;
		case '-': case '_': skip--;     break;
		case 'r':  samplereview = !samplereview; break;
		case '\b': if (samplereview) { categories[std::max_element(labels[currentsample].begin(), labels[currentsample].end()) - labels[currentsample].begin()]--; snapshots.erase(snapshots.begin() + currentsample); labels.erase(labels.begin() + currentsample); samples.erase(samples.begin() + currentsample); } // no break here, fall through to clamp  currentsample
		case '[': case ']':  currentsample = clamp(currentsample + (key == ']') - (key == '['), 0, (int)samples.size() - 1); break;
		case 's':  if (sinusoidal) { addsample(clamp((int)(sint * 5), 0, 4)); for (int i = 0;i < 5;i++)labels.back()[i] = clamp(1.0f - pow(((i + 0.5f) / 5.0f) - sint, 2.0f)*4.0f); } sinusoidal = true; break;  // sinusoidal motion;
		case 'n':  addsample(0);        break;
		case 'p':  addsample(1);        break;
		case '0': case '1':case '2':case '3':case '4': addsample(key - '0');  break;
		case ')': icons[0] = sample_cl; break;   // shift-0 is ')' on my keyboard
		case '!': icons[1] = sample_cl; break;
		case '@': icons[2] = sample_cl; break;
		case '#': icons[3] = sample_cl; break;
		case '$': icons[4] = sample_cl; break;
		case 't': if (key == 'T') training++; else training = (!training)*5; break; // t toggles while shift-T will continue to increas amount of backprop iterations per frame
		case '\x13': case '\x14': save_everything(key=='\x14');  break;   // ctrl-s or crtl-t  (crtl-t causes all the training samples to be output as well)
		case '\x18': cnn = cnntypes[cnn_type](); std::cout << "recreating CNN\n";  break;
		default:  std::cout << "unassigned key (" << (int)key << "): '" << key << "'\n";   break;
		}
		
		skip = clamp(skip, 1, 8);
	};
	try {
		if (dcam.dev->supports_option(rs::option::r200_lr_auto_exposure_enabled))
			dcam.dev->set_option(rs::option::r200_lr_auto_exposure_enabled, 1);
	}
	catch (...) {}
	void* current_selection = NULL;
	Pose camera;  // note this is the opengl camera, not the depth camera
	camera.orientation = normalize(float4(3, 0, 1,0));
	int gv = 240*3/2; int gh = 320*3/2;
	int frame = 0;
	std::vector<float> errorhistory(128,1.0f);
	SOCKET server = INVALID_SOCKET;
	int enable_server=0;


	while (glwin.WindowUp())
	{

		viewdist *= powf(1.1f, (float)glwin.mousewheel);
		if (!samples.size())
			samplereview = 0;  // cant review samples if there aren't any, stay in live camera mode

		// CNN training  if it happens to be enabled
		__int64 timestart = __rdtsc();
		float mse = 0.0f;
		int traincount = training ? (1 << training) : 0;
		for (int i = 0; i < traincount && samples.size() != 0; i++)
		{
			trainstarted = true; 
			auto s = std::uniform_int<int>(0, (int)samples.size() - 1)(rng);
			mse += cnn.Train(samples[s], labels[s]);
		}
		mse = traincount ? mse / (float)traincount : 0;
		errorhistory[frame % (errorhistory.size() / 2)] = errorhistory[frame % (errorhistory.size() / 2) + (errorhistory.size() / 2)] = sqrt(mse);
		__int64 cycles_bprop = __rdtsc() - timestart;



		Image<unsigned short> dpimage = dcam.GetDepth();
		auto dleft = (const uint8_t *)dcam.dev->get_frame_data(rs::stream::infrared);
		Image<unsigned char> irimage(dcam.dcamera() , std::vector<unsigned char>(dleft, dleft + product(dcam.dim())));
		auto dcolor = (Image<byte3>({ dcam.dim() }, (const byte3*)dcam.dev->get_frame_data(rs::stream::color_aligned_to_depth)));
		if (dpimage.dim().x>320)  // since sr300 might be 640x480 and we wanted 320x240
		{
			dpimage = DownSampleFst(dpimage);
			irimage = DownSampleFst(irimage);
			dcolor  = DownSampleFst(dcolor );
		}
		auto dp_image_g = Transform(dpimage, [&drange, depth_scale](unsigned short s) {return (unsigned char)clamp((int)(256.0f * ((s*depth_scale) - drange.x) / (drange.y - drange.x)), 0, 255); });
		auto dp_image_c = torgb(dp_image_g); //  Transform(dpimage, [](unsigned short s) {return byte3((unsigned char)clamp(255 - s / 4, 0, 255)); });
		auto ir_image_c = torgb(irimage);  //, [](unsigned char  c) {return byte3(c);}                                        );

		std::vector<float3> pts;
		std::vector<float3> outliers;



		int2 swsize = int2(16,16) *skip, swoffset = irimage.dim() / 2 - swsize / 2;   // use 16,16 instead of input size here
		auto sample_raw = Crop(dpimage, swoffset, swsize);
		for (auto p : rect_iteration(sample_raw.dim()))
		{
			float3 v = sample_raw.cam.deprojectz(p, sample_raw.pixel(p))*depth_scale;
			((v.z > drange.y && v.z < drange.x) ? &pts : &outliers)->push_back(v);
		}
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

		if (samplereview)
		{
			currentsample = clamp(currentsample, 0, samples.size());
			sample_in = samples[currentsample];
			sample_cl = snapshots[currentsample];
			auto sd = Transform(samples[currentsample], [](float a) {return togreyscale(a); });
			sample_dp.raster = std::vector<unsigned char>(sd.begin(), sd.begin() + sd.size() / 2);
			sample_ir.raster = std::vector<unsigned char>(sd.begin() + sd.size() / 2,sd.end());
		}


		timestart = __rdtsc();
		auto cnn_out = trainstarted ? cnn.Eval(sample_in) :  std::vector<float>(categories.size(), 0.0f);
		__int64 cycles_fprop = __rdtsc() - timestart;
		int best = std::max_element(cnn_out.begin(),cnn_out.end())-cnn_out.begin();
		if(enable_server)
			do_server_thing(server,std::string("category ")+std::to_string(best));


		auto samplec = torgb(UpSample(UpSample(sample_dp)));

		for (auto im : { &dp_image_c,&ir_image_c ,&dcolor })  for (int i = 0; i < 7;i++) // draw red-blue box around focus subwindow
		{
			for (int x = 0; x < swsize.x; x++) for (auto y : { -1-i,swsize.y+i })
				im->pixel(swoffset + int2(x, y)) = byte3(255, 0, 0);
			for (int y = 0; y < swsize.y; y++) for (auto x : { -1-i,swsize.x+i })
				im->pixel(swoffset + int2(x, y)) = byte3(0, 0, 255);
		}


		GUI gui(glwin, current_selection);

		static int debugpanelheight = 1;
		gui.splity(debugpanelheight,3);
		if(gui.dims().y>10)
		{
			glwin.PrintString({ 1,-1 }, " training %s %d   er %5.3f",  training ? "ON" : "OFF", training, sqrt(mse));
			glwin.PrintString({ 1,-4 }, "rdtsc fprop  %9d   bprop %9d ", (int)cycles_fprop, (int)cycles_bprop);
			glwin.PrintString({ 1,-10 }, "x,y,b,d %d %d %d %d", glwin.mousepos.x, glwin.mousepos.y, glwin.MouseState, glwin.downevent);
		}
		gui.pop();
		gv = std::max(gv, std::min(gui.dims().y,100));
		gui.splityn(gv);   // gui section for live camera input feeds and setting depth range and sample window size
		{
			gh = gv * 4 / 3;
			gui.splitxc(gh+gh/2);
			if (samplereview)
			{
				gui.splitync(gui.dims().y-30);
				gui.splitxc(gui.dims().y);
				gui.drawimagem(snapshots[currentsample]);
				glwin.PrintString({ 0,-1 }, "Captured Sample Review Mode");
				gui.pop();
				auto cat = std::max_element(labels[currentsample].begin(), labels[currentsample].end()) - labels[currentsample].begin();
				gui.drawbar(0.5f, catcolors[cat], catcolors[cat]);
				glwin.PrintString({ 0,-1 }, (cat == best) ? "Success, CNN output matches this category" : "Fail, sample miscategorized");
				gui.pop();
				gui.slider(currentsample, { 0,(int)snapshots.size() - 1 }, "sample #"); // note the potential side effect of changing this value here
			}
			else
			{ 
				gui.splitx(gh);			
				gv = gh * 3 / 4;
					gui.drawimagem(dcolor);			
				gui.pop();
				gui.splityc(gv / 2, 1);
				gui.drawimagem(dp_image_c);
				if (gui.inview())
					glwin.PrintString({ 0,-1 }, "depth stream");
				if (gui.focus(&dp_image_c))
				{
					auto c =  (gui.mouse()-gui.offset()) * dp_image_c.dim() / gui.dims() - dp_image_c.dim() / 2;
					skip = clamp(std::max(abs(c.x), abs(c.y)) / (16 / 2), 1, 8);
				}
				gui.pop();
				gui.drawimagem(ir_image_c);
				if (gui.inview())
					glwin.PrintString({ 0,-1 }, "IR stream");
				if (gui.focus(&ir_image_c))
				{
					auto c = (gui.mouse() - gui.offset())* ir_image_c.dim() / gui.dims() - ir_image_c.dim() / 2;
					skip = clamp(std::max(abs(c.x), abs(c.y)) / (16 / 2), 1, 8);
				}
			}
			gui.pop();
			gui.splitxc( gv / 2);
			{  // draw the two 16x16 actual inputs   zoomed in.
				gui.splityc(gv / 2, 1);
				gui.drawimagem(samplec);
				if (gui.inview())
					glwin.PrintString({ 0,-1 }, "CNN input depth");
				gui.pop();
				gui.drawimagem(torgb(UpSample(UpSample(sample_ir))));
				if (gui.inview())
					glwin.PrintString({ 0,-1 }, "CNN input IR");
			}
			gui.pop(); // now the last bit on the right
			{
				static int slider_set = 92;
				slider_set = std::max(std::min(gui.dims().y,45), slider_set);
				gui.splity(slider_set);
				{
					//  gui.splityc(slider_set / 3 - 1);
					auto regions = gui.partition_y(3);
					gui.set(regions[0]); 
					{
						gui.slider(skip, { 1,8 }, "samplearea ");
						skip = clamp(skip, 1, 8);
					}
					//gui.pop();
					//gui.splityc(slider_set/3-1);
					gui.set(regions[1]);
					{
						gui.slider(drange.x, { 0.25f,2.0f }, "depthmax ");
						drange.x = clamp(drange.x, 0.25f, 2.0f);
						drange.y = std::min(drange.y, drange.x - 0.001f);
					}
					//gui.pop();
					gui.set(regions[2]);
					{
						gui.slider(drange.y, { 0.25f,2.0f }, "depthmin ");
						drange.y = clamp(drange.y, 0.25f, 2.0f);
						drange.x = std::max(drange.x, drange.y + 0.001f);  // drange  note x>y so brighter is closer
					}
				}
				gui.pop();
				{
					camera.orientation = qconj(camera.orientation);  // since trackball is normally used for adjusting another object's orientation, but here we're updating a the viewcamera looking at the object
					gui.trackball(camera.orientation);  // may update camera.orientation only if user mouse drags within subwindow area
					camera.orientation = qconj(camera.orientation);  // and back
					gui.perspective();
					glPushMatrix();
					camera.position = float3(0, 0, (drange.x + drange.y)*0.5f) + qzdir(camera.orientation) *  viewdist;
					glMultMatrixf(camera.inverse().matrix());

					glColor3f(1, 1, 1);
					glwirefrustumz(dcam.deprojectextents(), { 0.1f,1.0f });  // draw the depth camera frustum volume
					glColor3f(0, 1, 0.5f);
					glwirefrustumz(sample_dp.cam.deprojectextents(), drange);  // draw the sample camera frustum volume
					drawpoints(pts, { 0   ,1,0 });
					drawpoints(outliers, { 0.5f,0,0 });
					glPopMatrix();
				}
			}
		}
		gui.pop();  // end of the camera feeds

		static int cdata = 60;
		int midy = gui.dims().y; 
		gui.splitxc(gui.dims().x - midy*2);  // leaves a couple square sections on the right in the middle
		{
			auto cat_regions = gui.partition_y((int)categories.size());
			for (int i = 0; i < (int)categories.size(); i++)
			{
				//gui.splitync(midy / categories.size() - 1, 1);
				gui.set(cat_regions[i]);
				gui.splitxc(midy / categories.size());
				gui.drawimagem(icons[i]);
				glwin.PrintString({ 0,-1 }, "%d", i);
				if (gui.focus(&icons[i]))
					addsample(i);
				gui.pop();
				static int ncdata;
				auto bwidth = gui.dims().x;
				ncdata = bwidth - cdata;
				gui.splitx(ncdata);
				cdata = bwidth - ncdata;
				gui.drawbar(cnn_out[i], catcolors[i], catcolors[i] * 0.25f);
				glwin.PrintString({ 0,-1 }, "%5.2f", cnn_out[i]);
				gui.pop();
				{// the little text block at the end of each category row
					glwin.PrintString({ 0,-1 }, " %d ", categories[i]);  // puts the number of samples of this type as text output
				}
				//gui.pop();
			}
		}
		gui.pop(); // done left side of mid section
		gui.splitxc(midy);
		{ // middle of middle section
			static int slider_height = 30;
			slider_height = std::max(slider_height, 10);
			gui.splity(slider_height);
			gui.slider(training, { 0,6 }, training? "   Training: 2^":"     click here to crank up training");
			gui.pop();
			gui.splitxc(gui.dims().y-slider_height); // just to make it square
			if (training)
			{
				gui.ortho();
				glColor3f(1, 0, 0);
				glBegin(GL_QUAD_STRIP);
				for (unsigned int i = 0; i < errorhistory.size() / 2; i++)
				{
					glVertex2f((float)i / (errorhistory.size()/2), errorhistory[1 + i + frame % (errorhistory.size() / 2)]);
					glVertex2f((float)i / (errorhistory.size()/2), 0);
				}
				glEnd();
				glColor3f(1, 1, 1);
				glwin.PrintString({ 0,-1 }, "error: %5.2f", sqrt(mse));
			}
			else if (sinusoidal)
			{
				float sint = 0.5f + sin(2.0f * time/2*3.1415f) / 2;   
				gui.ortho();
				glColor3f(1, 0, 0);
				glBegin(GL_QUAD_STRIP);
				for (float x : {0.0f, 1.0f}) for (float dy : {-0.05f, 0.05f}) 
				{
					glVertex2f(x,1.0f-sint+dy);
				}
				glEnd();
				glColor3f(1, 1, 1);
				glwin.PrintString({ 0,-1 }, "normalized sin(k*time): %5.2f", sint);
			}
			else
			{ // sample status 
				gui.ortho();
				auto sizes = Transform(categories, [&](int c) {return (float)c / samples.size(); });
				float b = 0;
				float r = log2(samples.size() + 1.0f)/10.0f * 0.5f;
				for (unsigned int i = 0; i < sizes.size(); i++)
				{
					glColor3fv(catcolors[i]);
					glBegin(GL_TRIANGLE_FAN);
					glVertex2f(0.5f, 0.5f);
					for (float t = 0; t <= 1.0f; t += 1 / 32.0f)
						glVertex2f(0.5f + r* cos(2 * 3.14159f*(b + sizes[i] * t)), 0.5f + r* sin(2 * 3.14159f*(b + sizes[i] * t)));
					glEnd();
					b += sizes[i];
				}
				glColor3f(1, 1, 1);
				glwin.PrintString({ 0,-1 }, "samples collected: %d",samples.size());
			}
			gui.offset().y += gui.dims().y - 40;
			gui.dims().y = 37;
			gui.offset().x += 5;
			gui.dims().x -= 10;
			gui.set();
			gui.splitxc(gui.dims().x-90, 5);
			gui.dims().x = std::min(gui.dims().x, 180);
			gui.set();
			static float rslide = 0.0f;
			gui.slider(rslide, { 0.0f,1.0f }, samplereview? "Review Samples  ON     " : "Review Samples OFF    " );
			if (gui.selection == &rslide)
				samplereview = (rslide>0.5f);
			else
				rslide += samplereview ? 0.01f : -0.01f;
			rslide = clamp(rslide);
			gui.pop();
			gui.slider(enable_server, { 0,1 }, "HTTP ");
			if(enable_server && server==INVALID_SOCKET)
				server = start_server(12345);
			if (server == INVALID_SOCKET)
				enable_server = 0;
			gui.pop();
			// a vertical strip thats the same width as the slider for the training
		}
		gui.pop();
		if(trainstarted)
		{   // winner of cnn evaluation
			gui.drawbar(1.0f, catcolors[best], catcolors[best]);
			gui.offset() += int2(5, 5);
			gui.dims()   -= int2(10, 10);
			gui.set();
			gui.drawimagem(icons[best]);
			glwin.PrintString({ 0,-1 }, "%d  eval: %5.3f", best, cnn_out[best]);
		}
		else if (!samples.size())
		{
			// if we haven't done anything yet, this part of the GUI allows us to change the size of the CNN
			auto lg2 = [](int a)->int { int x = 0; while (a) { a >>= 1; x++; }return x; };
			gui.splitync(40);
			auto w_old = cnn_type;
			gui.slider(cnn_type, { 0,2 }, "Size of CNN, (16,32,64) squared samples ");
			if (cnn_type != w_old)
				cnn = cnntypes[cnn_type]();
			inputsize = ((CNN::LConv*)cnn.layers[0])->indims.xy();   
			gui.pop();
			gui.splitync(40);
			glwin.PrintString({ 0,-1 }, "Current CNN %s config uses %dx%d input", cnndescriptions[cnn_type], inputsize.x, inputsize.y);
			gui.pop();

		}
		gui.pop(); // done mid section

		frame++;
		time += 0.016f; // yo, which sys call should we use here?
		sint = 0.5f + sin(2.0f * time / 2.0f * 3.1415f) / 2;
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
