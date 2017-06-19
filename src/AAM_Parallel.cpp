/*****************************************************************
  Parallel implementation of Cootes Fitting Algorithm using
  OpenMP.

  Based on the parallel algorithm designed in "Efficient parallel
  Implementation of Active Appeareance Model Fitting Algorithm
  on GPU" by Wang, Ma, Zhu et Sun.

  @author Rafael Rodriguez (rodriguezsrafa@gmail.com)
  @date June 2017
 ******************************************************************/
#include <fstream>
#include "AAM_Parallel.h"
#include "AAM_Basic.h"
#include "omp.h"


#define CVMAT_ELEM(Mat, i, j) CV_MAT_ELEM(*Mat, double, i,j)
#define MAX_CHANNELS 4

using namespace std;

AAM_Parallel::AAM_Parallel() {

}

AAM_Parallel::~AAM_Parallel() {

}

int AAM_Parallel::GetNumLayers() {
	return __modelP.__model.size();
}
void AAM_Parallel::SetModel(int layer) {
	if (layer >= __modelP.__model.size()) {
		fprintf(stderr, "Error: Layer %d is not available in the model\n", layer+1);
		exit(0);
	}
	__R = ((AAM_Basic*)(__modelP.__model[layer]))->__G;
	__model = &((AAM_Basic*)(__modelP.__model[layer]))->__cam;
}

void AAM_Parallel::Fit(IplImage* image, AAM_Shape& shape, int max_iter, bool showprocess, double epsilon) {
	int np = 8;
	double k_values[np] = {1,0.5,0.25,0.125,0.0625,0.03125,0.03125/2, 0.03125/4};
	bool converge = false;
	int iter = 0;
	double newError, error;
	IplImage* Drawimg = 0;
  double __update_c[__model->nModes()];
  double __update_q[4];

	CvMat* __s = cvCreateMat(1,__model->__shape.nPoints()*2, CV_64FC1);
	CvMat* __p = cvCreateMat(1,__model->__shape.nModes(), CV_64FC1);
	CvMat* __lambda = cvCreateMat(1,__model->__texture.nModes(), CV_64FC1);
	CvMat __qMat = cvMat(1, 4, CV_64FC1, __q);
	CvMat __sampledTexture = cvMat(1, __model->__texture.nPixels(), CV_64FC1, __texture);
	CvMat __cMat = cvMat(1,__model->nModes(),CV_64FC1, __c);

	VJfacedetect facedet;
	facedet.LoadCascade("haarcascade_frontalface_alt2.xml");
	cout << "Classifier file loaded" << endl;
	bool flag = flag = __modelP.InitShapeFromDetBox(shape, facedet, image);
	if(flag == false) {
		fprintf(stderr, "The image doesn't contain any faces\n");
		exit(0);
	}
	// else
		// cout << "Face detected" << endl;

	// cout << "Space allocated" << endl;
	shape.Point2Mat(__s);
	// cout << "shape to mat.." << endl;
	//shape parameter
	__model->__shape.CalcParams(__s, __p, &__qMat);
	// cout << "shape params computed" << endl;
	//texture parameter
	__model->__paw.CalcWarpTexture(__s, image, &__sampledTexture);
	__model->__texture.NormalizeTexture(__model->__MeanG, &__sampledTexture);
	__model->__texture.CalcParams(&__sampledTexture, __lambda);

	// cout << "texture sampled" << endl;
	//combined appearance parameter
	__model->CalcParams(&__cMat, __p, __lambda);

	// cout << "initial params computed" << endl;

	// EstimateParams(image);
	error = ComputeEstimationError(image, __c, __q);

	if(showprocess)
	{
		if(Drawimg == 0)	Drawimg = cvCloneImage(image);
		else cvCopy(image, Drawimg);
		// cout << "copied..." <<endl;
		Draw(Drawimg);
		// cout << "copied..." <<endl;
		AAM_Common::MkDir("result");
		char filename[100];
		sprintf(filename, "result/Iter.jpg");
		cvSaveImage(filename, Drawimg);
		// cout << "saved..." <<endl;
	}

  cout<< "Init error: " << error << endl;
	while (iter < max_iter && !converge) {
		ParamsUpdate(image);



		for (int k = 0; k < np; k++) {
			// cout << "k: " << k << endl;
			ComputeNewParams(k_values[k], __update_c, __update_q);
			newError = ComputeEstimationError(image, __update_c, __update_q);
			if (newError <= error) {
        for (int j = 0; j < __model->nModes(); j++)
          __c[j] = __update_c[j];
        for (int j = 0; j < 4; j++)
          __q[j] = __update_q[j];
				cout << "break..." <<endl;
				break;
			}
			// cout << "iter for new error: "<< k << endl;
		}
		cout << "Error " << iter<< ": " << newError << endl;
		if(showprocess)
		{
			if(Drawimg == 0)	Drawimg = cvCloneImage(image);
			else cvCopy(image, Drawimg);
			// cout << "copied..." <<endl;
			Draw(Drawimg);
			// cout << "copied..." <<endl;
			AAM_Common::MkDir("result");
			char filename[100];
			sprintf(filename, "result/Iter-%02d.jpg", iter);
			cvSaveImage(filename, Drawimg);
			// cout << "saved..." <<endl;
		}

		if (std::abs(newError - error) <= epsilon)
			converge = true;
		else
			error = newError;
		iter++;
	}

}

bool AAM_Parallel::Read(const std::string& filename) {
	// ifstream input(filename.c_str(), ios::in | ios::binary);
	//
	// if (!input) {
	// 	fprintf(stderr, "Cannot load model %s", filename.c_str());
	// 	return false;
	// }
	// __model->Read(input);
	// __R = cvCreateMat(__model->nModes()+4, __model->__texture.nPixels(), CV_64FC1);
	// ReadCvMat(input, __R);

	__modelP.ReadModel(filename);

	__R = ((AAM_Basic*)(__modelP.__model[0]))->__G;
	__model = &((AAM_Basic*)(__modelP.__model[0]))->__cam;

	__c = new double[__model->nModes()];
	__q = new double[4];
	__shape = new double[__model->__Qs->cols];
	__texture = new double [__model->__texture.nPixels()];
	__modelledTexture = new double[__model->__texture.nPixels()];
	__dif = new double[__model->__texture.nPixels()];
	__delta_c_q = new double[__model->nModes() + 4];

	return true;
}

void AAM_Parallel::ComputeModelledShape(IplImage* image, double* __uc, double* __uq) {
	CvMat* Qs = __model->__Qs;
	CvMat* So = __model->__MeanS;
	int k;
	double x, y;

	#pragma omp parallel
	{
		#pragma omp for schedule(dynamic,1) private(x,y,k)
		for (int i = 0 ; i < So->cols; i += 2) {
			k = 0;
			x = 0.0; y = 0.0;
			// this is done for every vertex
			while (k < __model->nModes()) {
				x += __uc[k] * CVMAT_ELEM(Qs,k,i);
				y += __uc[k] * CVMAT_ELEM(Qs,k,i+1);
				k++;
			}
			x += CVMAT_ELEM(So, 0, i);
			y += CVMAT_ELEM(So, 0, i+1);

			__shape[i] = (__uq[0] + 1)*x - __uq[1]*y + __uq[2];
			__shape[i+1] = __uq[1]*x + (__uq[0]+1)*y + __uq[3];

			// Ensure the vertex is inside the image.
			Clamp(__shape[i], 0.0, (double) (image->width - 1));
			Clamp(__shape[i+1], 0.0, (double) (image->height - 1));
		}
	}

}


void AAM_Parallel::ComputeModelledTexture(IplImage* image, double* __uc) {

	CvMat* Qg = __model->__Qg;
	CvMat* Go = __model->__MeanG;
	int j,k;
	int channels = image->nChannels, np = __model->__texture.nPixels();

	#pragma omp parallel
	{
    #pragma omp for schedule(dynamic,1)
		for (int i = 0; i < np; i++) {
			__modelledTexture[i] = CVMAT_ELEM(Go, 0, i);
			k = 0;
			while (k < __model->nModes()) {
				__modelledTexture[i] += (CVMAT_ELEM(Qg, k, i) * __uc[k]);
				k++;
			}
		}
	}
}


void AAM_Parallel::SampleTexture(IplImage* image) {
	int v1,v2,v3,tri_idx,k;
	std::vector<int> pixTri = __model->__paw.__pixTri;
	std::vector<std::vector<int>> tri = __model->__paw.__tri;
	std::vector<double> alpha = __model->__paw.__alpha;
	std::vector<double> beta = __model->__paw.__belta;
	std::vector<double> gamma = __model->__paw.__gamma;
	double x, y;

	int np = __model->__paw.__nPixels;
	int model_channels = __model->__texture.nPixels()/np;
	#pragma omp parallel
	{
		#pragma omp for schedule(dynamic,1) private(x,y,v1,v2,v3,tri_idx,k)
		for (int i = 0; i < __model->__paw.__nPixels; i++) {
			tri_idx = pixTri[i];
			v1 = tri[tri_idx][0];
			v2 = tri[tri_idx][1];
			v3 = tri[tri_idx][2];

			x = alpha[i]*__shape[v1*2] + beta[i]*__shape[v2*2] + gamma[i]*__shape[v3*2];
			y = alpha[i]*__shape[v1*2 + 1] + beta[i]*__shape[v2*2 + 1] + gamma[i]*__shape[v3*2 + 1];

			BilinearInterpolation(image, x, y, __texture+(i*model_channels));
		}
	}
}

void AAM_Parallel::NormalizingTexture(IplImage* image) {

	int channels = image->nChannels;
	double mean = 0.0, norm = 0.0, alpha = 0.0;
	const CvMat* mean_texture = __model->__MeanG;
	bool alpha0 = false;
	int np = __model->__paw.__nPixels;
	int model_channels = __model->__texture.nPixels()/np;
	channels = model_channels;
	// for (int i = 0; i < channels; i++) {
	// 	mean[i] = 0.0;
	// }

#pragma omp parallel
	{
    #pragma omp for schedule(dynamic,1) reduction(+:mean)
		for (int i = 0; i < __model->__texture.nPixels(); i++) {
				mean = mean + __texture[i];
		}

    #pragma omp single
		{
			mean /= __model->__texture.nPixels();
		}

		#pragma omp barrier

    #pragma omp for schedule(dynamic,1) reduction(+: norm)
		for (int i = 0; i < np; i++) {
			for (int k = 0; k < model_channels; k++) {
				__texture[i*model_channels + k] -= mean;   // unbias
				norm = norm + (__texture[i*model_channels + k]*__texture[i*model_channels + k]);
			}
		}

     #pragma omp single
		{
			norm = std::sqrt(norm);
		}

		#pragma omp barrier

		#pragma omp for schedule(dynamic,1) reduction(+: alpha)
		for (int i = 0; i < np; i++) {
			for (int k = 0; k < model_channels; k++) {
				__texture[i*model_channels + k] /= norm;   // normalize
				alpha = alpha + (__texture[i*model_channels + k] * CVMAT_ELEM(mean_texture,0,i*model_channels + k));
			}
		}
	}
	if (alpha != 0) {
		#pragma omp parallel for schedule(dynamic,1)
		for (int i = 0; i < np; i++) {
			for (int k = 0; k < model_channels; k++) {
				__texture[i*model_channels + k] /= alpha;
			}
		}
	}
}


double AAM_Parallel::ComputeEstimationError(IplImage* image, double* __uc, double* __uq) {
	double error = 0.0;
	ComputeModelledShape(image,__uc,__uq);
	SampleTexture(image);

	ComputeModelledTexture(image, __uc);
	CvMat __sampledTexture = cvMat(1, __model->__texture.nPixels(), CV_64FC1, __texture);
	CvMat __s = cvMat(1, __model->__shape.nPoints()*2, CV_64FC1, __shape);
	// __model->__paw.CalcWarpTexture(&__s, image, &__sampledTexture);
	// __model->__texture.NormalizeTexture(__model->__MeanG, &__sampledTexture);
	NormalizingTexture(image);
	int np = __model->__texture.nPixels();
	int model_channels = __model->__texture.nPixels()/np;
  #pragma omp parallel
	{
    #pragma omp for schedule(dynamic,1) reduction(+:error)
		for (int i = 0; i < np; i+=1 ) {
			__dif[i] = -__texture[i] + __modelledTexture[i];
			error = error + (__dif[i]*__dif[i]);
		}
	}

	return std::sqrt(error);
}

void AAM_Parallel::ParamsUpdate(IplImage* image) {
	int k = 0;
	double val;
	for (k = 0; k < __model->nModes() + 4; k++) {
		val = 0.0;
    #pragma omp parallel for schedule(dynamic,1) reduction(+:val)
		for (int i = 0; i < __model->__texture.nPixels(); i++) {
			val = val + (CVMAT_ELEM(__R, k, i) * __dif[i]);
		}
		__delta_c_q[k] = val;
	}
}

void AAM_Parallel::EstimateParams(IplImage* image) {
	double tx = image->width/2, ty = image->height/2;

	__q[0] = 0.1;
	__q[1] = 0.1;
	__q[2] = tx;
	__q[3] = ty;

	for (int i = 0; i < __model->nModes(); i++)
		__c[i] = 0;     //use the mean appeareance

}

void AAM_Parallel::ComputeNewParams(double k, double* __uc, double* __uq) {
	int i;
	#pragma omp parallel
	{
		#pragma for schedule(dynamic,1) nowait private(i)
		for (i = 0; i < __model->nModes(); i++) {
			__uc[i] = __c[i] +  k * __delta_c_q[i + 4];
		}

		#pragma for schedule(dynamic,1) private(i)
		for (i = 0; i < 4; i++) {
			__uq[i] = __q[i] + k * __delta_c_q[i];
      // cout << "q[" << i << "]= " << __q[i] << endl;
		}
	}
	CvMat c = cvMat(1,__model->nModes(),CV_64FC1,__uc);
	__model->Clamp(&c);
}


void AAM_Parallel::BilinearInterpolation(IplImage* image, double x, double y, double* pixel) {

	/*
    Taken from AAM Library of GreatYao
    https://github.com/greatyao/aamlibrary
	 */

	int X1 = cvFloor(x);
	int X2 = cvCeil(x);
	int Y1 = cvFloor(y);
	int Y2 = cvCeil(y);

	double s0 = x-X1,	t0 = y-Y1,	s1 = 1-s0,	t1 = 1-t0;

	char* imgdata = image->imageData;
	int step = image->widthStep;
	int off_g = 1;//(image->nChannels == 3) ? 1 : 0;
	int off_r = 2;//(image->nChannels == 3) ? 2 : 0;

	int ixB1 = image->nChannels*X1, ixG1= ixB1+off_g, ixR1 = ixB1+off_r;
	int ixB2 = image->nChannels*X2,	ixG2= ixB2+off_g,	ixR2 = ixB2+off_r;


	byte* p1 = (byte*)(imgdata + step*Y1);
	byte* p2 = (byte*)(imgdata + step*Y2);

	pixel[0] = s1*(t1*p1[ixB1]+t0*p2[ixB1])+s0*(t1*p1[ixB2]+t0*p2[ixB2]);
	pixel[1] = s1*(t1*p1[ixG1]+t0*p2[ixG1])+s0*(t1*p1[ixG2]+t0*p2[ixG2]);
	pixel[2] = s1*(t1*p1[ixR1]+t0*p2[ixR1])+s0*(t1*p1[ixR2]+t0*p2[ixR2]);
}


void AAM_Parallel::Clamp(double& x, double min, double max) {
	if (x < min) x = min;
	if (x > max) x = max;
}

void AAM_Parallel::Draw(IplImage* image) {
	AAM_Shape Shape;
	CvMat shape = cvMat(1,__model->__shape.nPoints()*2,CV_64FC1, __shape);
	Shape.Mat2Point(&shape);
	double minV, maxV;
	CvMat texture = cvMat(1,__model->__texture.nPixels(), CV_64FC1, __modelledTexture);
	cvMinMaxLoc(&texture, &minV, &maxV);
	cvConvertScale(&texture, &texture, 255/(maxV-minV), -minV*255/(maxV-minV));
	AAM_PAW paw;
	paw.Train(Shape, __model->__Points, __model->__Storage, __model->__paw.GetTri(), false);
	AAM_Common::DrawAppearance(image, Shape, &texture, paw, __model->__paw);

}
