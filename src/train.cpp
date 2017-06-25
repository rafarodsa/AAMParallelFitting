/************************************************************
    Simplified Training algorithm for Cootes's AAM
    Based on the AAM Library of GreatYao:
    https://github.com/greatyao/aamlibrary

    @author Rafael Rodriguez
*************************************************************/

#include "AAM_Basic.h"

using namespace std;

static void usage()
{
	printf("Usage: build train_path image_ext point_ext model_file\n");
	exit(0);
}

int main(int argc, char** argv)
{

	if(5 != argc)	usage();

	file_lists imgFiles = AAM_Common::ScanNSortDirectory(argv[1], argv[2]);
	file_lists ptsFiles = AAM_Common::ScanNSortDirectory(argv[1], argv[3]);

	if(ptsFiles.size() != imgFiles.size()){
		fprintf(stderr, "ERROR(%s, %d): #Shapes != #Images\n",
			__FILE__, __LINE__);
		exit(0);
	}

	AAM_Pyramid model;
	model.Build(ptsFiles, imgFiles, 0, 16);
	// model.ReadModel(argv[4]);
	VJfacedetect facedet;
	facedet.LoadCascade("../resources/haarcascade_frontalface_alt2.xml");
	model.BuildDetectMapping(ptsFiles, imgFiles, facedet);

	model.WriteModel(argv[4]);

	return 0;
}
