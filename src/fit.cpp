/************************************************************
		Parallel implementation of Cootes Fitting Algorithm using
		OpenMP.

		Based on the parallel algorithm designed in "Efficient parallel
		Implementation of Active Appeareance Model Fitting Algorithm
		on GPU" by Wang, Ma, Zhu et Sun.

		Based on the AAM Library of GreatYao:
		https://github.com/greatyao/aamlibrary

		@author Rafael Rodriguez (rodriguezsrafa@gmail.com)
		@date June 2017
*************************************************************/

#include "AAM_Parallel.h"

using namespace std;

static void usage()
{
	printf("Usage: fit model_file image_file\n");
	exit(0);
}

int main(int argc, char** argv)
{
	if(argc != 3)	usage();

	AAM_Parallel model;
	model.Read(argv[1]);
	cout<< "Read model file" << endl;
	char filename[100];
	strcpy(filename, argv[2]);
	cout << "Layers: "<< model.GetNumLayers() << endl;

// 	if(strstr(filename, ".avi"))
// 	{
// 		AAM_MovieAVI aviIn;
// 		AAM_Shape Shape;
// 		IplImage* image2 = 0;
// 		bool flag = false;
//
// 		aviIn.Open(filename);
// 		cvNamedWindow("Video",1);
// 		cvNamedWindow("AAMFitting",1);
//
// 		for(int j = 0; j < aviIn.FrameCount(); j ++)
// 		{
// 			printf("Tracking frame %04i: ", j);
//
// 			IplImage* image = aviIn.ReadFrame(j);
//
// 			if(j == 0 || flag == false)
// 			{
// 				flag = model.InitShapeFromDetBox(Shape, facedet, image);
// 				if(flag == false) goto show;
// 			}
//
// 			flag = model.Fit(image, Shape, 30, false);
// 			if(image2 == 0) image2 = cvCreateImage(cvGetSize(image), image->depth, image->nChannels);
// 			cvZero(image2);
// 			model.Draw(image2, Shape, 2);
// 			cvShowImage("AAMFitting", image2);
// show:
// 			cvShowImage("Video", image);
// 			cvWaitKey(1);
// 		}
// 		cvReleaseImage(&image2);
// 	}
//
// 	else
	{
		IplImage* image = cvLoadImage(filename, -1);

		// AAM_Shape Shape;
		// bool flag = flag = model.InitShapeFromDetBox(Shape, facedet, image);
		// if(flag == false) {
		// 	fprintf(stderr, "The image doesn't contain any faces\n");
		// 	exit(0);
		// }


		VJfacedetect facedet;
		facedet.LoadCascade("../resources/haarcascade_frontalface_alt2.xml");

		AAM_Shape Shape;
		AAM_Pyramid pyramid_model;
		bool flag = flag = pyramid_model.InitShapeFromDetBox(Shape, facedet, image);

		if(flag == false) {
			fprintf(stderr, "The image doesn't contain any faces\n");
			exit(0);
		}
		else
			cout << "Face detected" << endl;

		cvNamedWindow("Original");
		cvShowImage("Original", image);

		model.Fit(image, Shape, 1000, false);
		model.Draw(image);

		cvNamedWindow("Fitting");
		cvShowImage("Fitting", image);
		cvWaitKey(0);

		cvReleaseImage(&image);
	}

	return 0;
}
