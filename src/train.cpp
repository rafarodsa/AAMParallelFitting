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
	printf("Usage: train train_path image_ext point_ext model_file number_layers\n");
	exit(0);
}

int main(int argc, char** argv)
{

	if(6 != argc)	usage();

	file_lists imgFiles = AAM_Common::ScanNSortDirectory(argv[1], argv[2]);
	file_lists ptsFiles = AAM_Common::ScanNSortDirectory(argv[1], argv[3]);

	if(ptsFiles.size() != imgFiles.size()){
		fprintf(stderr, "ERROR(%s, %d): #Shapes != #Images\n",
			__FILE__, __LINE__);
		exit(0);
	}

<<<<<<< HEAD
	VJfacedetect facedet;
	facedet.LoadCascade("../resources/haarcascade_frontalface_alt2.xml");
	AAM_Pyramid model;
	model.Build(ptsFiles, imgFiles, 0, atoi(argv[5]));
	model.BuildDetectMapping(ptsFiles, imgFiles, facedet);


  ofstream os(argv[4], ios::out | ios::binary);
	if(!os){
		LOGW("ERROR(%s, %d): CANNOT create model \"%s\"\n", __FILE__, __LINE__, argv[4]);
		return false;
	}
=======
	AAM_Pyramid model;
	model.Build(ptsFiles, imgFiles, 0, 16);

  // ofstream os(argv[4], ios::out | ios::binary);
	// if(!os){
	// 	LOGW("ERROR(%s, %d): CANNOT create model \"%s\"\n", __FILE__, __LINE__, argv[4]);
	// 	return false;
	// }
>>>>>>> ParallelFitting
	model.WriteModel(argv[4]);

	return 0;
}
