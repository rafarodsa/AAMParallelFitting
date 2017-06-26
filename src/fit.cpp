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


	{
		IplImage* image = cvLoadImage(filename, -1);

		VJfacedetect facedet;
		facedet.LoadCascade("../resources/haarcascade_frontalface_alt2.xml");

		AAM_Shape Shape;
		ofstream out;
		AAM_Pyramid pyramid_model;
		bool flag = flag = model.__modelP.InitShapeFromDetBox(Shape, facedet, image);

		if(flag == false) {
			fprintf(stderr, "The image doesn't contain any faces\n");
			exit(0);
		}
		else
			cout << "Face detected" << endl;

		cvNamedWindow("Original");
		cvShowImage("Original", image);

		model.Fit(image, Shape, out, 1000, false, 0.003);
		model.Draw(image);

		cvNamedWindow("Fitting");
		cvShowImage("Fitting", image);
		cvWaitKey(0);

		cvReleaseImage(&image);
	}

	return 0;
}
