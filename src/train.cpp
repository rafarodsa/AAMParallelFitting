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

  // ofstream os(argv[4], ios::out | ios::binary);
	// if(!os){
	// 	LOGW("ERROR(%s, %d): CANNOT create model \"%s\"\n", __FILE__, __LINE__, argv[4]);
	// 	return false;
	// }
	model.WriteModel(argv[4]);

	return 0;
}
