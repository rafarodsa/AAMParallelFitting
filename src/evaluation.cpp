/************************************************************
		Parallel implementation of Cootes Fitting Algorithm using
		OpenMP. Evaluation program: it takes a multilayer model
		and for each layer makes the fitting of a certain number
		of frames

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
	printf("Usage: evaluation model_file image_folder_path image_ext num_images outputfile.csv\n");
	exit(0);
}

int main(int argc, char** argv)
{
	if(argc != 6)	usage();

	cout<< "Read model file: "<< argv[1] << endl;
	AAM_Parallel model;
	model.Read(argv[1]);

	cout << "Layers: "<< model.GetNumLayers() << endl;

	file_lists imgFiles = AAM_Common::ScanNSortDirectory(argv[2], argv[3]);
	model.FitAll(imgFiles, argv[5], atoi(argv[4]), 30, model.GetNumLayers());

	return 0;
}
