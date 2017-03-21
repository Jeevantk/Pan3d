#include <iostream>
#include <fstream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/utility.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/timelapsers.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"

using namespace cv;
using namespace std;
using namespace cv::detail;

int main(int argc,char* argv[])
{

	float conf_thresh=1.f;
	if(argc<3)
	{
		cout<<"Need Atleast Two images to Stitch"<<endl;
		cout<<argc<<endl;
		return -1;
	}

	int num_images=argc-1;

	//vetor to store all the imput images
	vector<Mat> images(num_images);


	Mat img;

	//Declares a pointer for Surf Feature Detection
	Ptr<FeaturesFinder> finder;

	finder = makePtr<SurfFeaturesFinder>();

	//Vector to store all the image features of all the images
	vector<ImageFeatures> features(num_images);

	


	cout<<"Reading the images and Computing the image features in the image"<<endl;
	for(int i=0;i<num_images;i++)
	{
		img=imread(argv[i+1]);

		if (img.empty())
        {
            cout<<"Can't open image " << argv[i+1]<<endl;
            return -1;
        }

		images.push_back(img);

		//computing the features of image i
		(*finder)(img, features[i]);

		features[i].img_idx = i;
	}

	finder->collectGarbage();

	cout<<"finding Matches in the images"<<endl;

	//Vector to store all the match imformation
	vector<MatchesInfo> pairwise_matches;

	Ptr<FeaturesMatcher> matcher;

	//initialising the matcher
	matcher=makePtr<BestOf2NearestMatcher>();

	(*matcher)(features, pairwise_matches);

	//Take only the images which we are sure from the same panaroma

	vector<int> indices = leaveBiggestComponent(features, pairwise_matches,conf_thresh);

	vector<Mat> img_subset;

	for(int i=0;i<indices.size();i++)
	{
		img_subset.push_back(images[indices[i]]);
	}

	images=img_subset;

	//Checking if we have atleast two images from the same subset

	num_images=images.size();
	if(num_images<2)
	{
		cout<<"Needs more images"<<endl;
		return -1;
	}

	//cout<<num_images<<endl;

	Ptr<Estimator> estimator;

	estimator=makePtr<HomographyBasedEstimator>();

	vector <CameraParams> cameras;

	if (!(*estimator)(features, pairwise_matches, cameras))
	{
		cout << "Homography estimation failed.\n";
		return -1;
	}

	for (int i = 0; i < cameras.size(); ++i)
	{
		Mat R;
		cameras[i].R.convertTo(R, CV_32F);
		cameras[i].R = R;

	}

	Ptr<detail::BundleAdjusterBase> adjuster;

	adjuster = makePtr<detail::BundleAdjusterRay>();

	adjuster->setConfThresh(conf_thresh);

	Mat_<uchar> refine_mask = Mat::zeros(3, 3, CV_8U);
	refine_mask(0,0) = 1;
	refine_mask(0,1) = 1;
	refine_mask(0,2) = 1;
	refine_mask(1,1) = 1;
	refine_mask(1,2) = 1;

	adjuster->setRefinementMask(refine_mask);

	if (!(*adjuster)(features, pairwise_matches, cameras))
	{
		cout << "Camera parameters adjusting failed.\n";
		return -1;
	}

	vector<double> focals;

	for(int i=0;i< cameras.size();i++)
	{
		focals.push_back(cameras[i].focal);
	}

	//for finding median focal length
	sort(focals.begin(), focals.end());

	float warped_image_scale;


	if (focals.size() % 2 == 1)
		warped_image_scale = static_cast<float>(focals[focals.size() / 2]);
	else
		warped_image_scale = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;

	vector<Mat> rmats;

	for(int i=0;i<cameras.size();i++)
	{
		rmats.push_back(cameras[i].clone());
	}

	WaveCorrectKind wave_correct = detail::WAVE_CORRECT_HORIZ;

	waveCorrect(rmats, wave_correct);

	for(int i=0;i<cameras.size();i++)
	{
		cameras[i].R=rmats[i];
	}
	









	return 1;
}