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
#include "omp.h"


#define CVMAT_ELEM(Mat, i, j) CV_MAT_ELEM(*Mat, double, i,j)
#define MAX_CHANNELS 4

using namespace std;

AAM_Parallel::AAM_Parallel() {

}

AAM_Parallel::~AAM_Parallel() {

}

void AAM_Parallel::Fit(IplImage* image, int max_iter, bool showprocess, double epsilon) {
  int np = 5;
  double k_values[np] = {1,0.5,0.25,0.125,0.0625,0.03125};
  bool converge = false;
  int iter = 0;
  double newError, error;
  IplImage* Drawimg = 0;

  EstimateParams(image);
  error = ComputeEstimationError(image);
  while (iter < max_iter && !converge) {
    ParamsUpdate(image);



    for (int k = 0; k < np; k++) {
      ComputeNewParams(k_values[k]);
      newError = ComputeEstimationError(image);
      if (newError < error) {
        break;
      }
    }
    cout << "Error " << iter<< ": " << newError << endl;
    if(showprocess)
    {
      if(Drawimg == 0)	Drawimg = cvCloneImage(image);
      else cvCopy(image, Drawimg);
      Draw(Drawimg);
      AAM_Common::MkDir("result");
      char filename[100];
      sprintf(filename, "result/Iter-%02d.jpg", iter);
      cvSaveImage(filename, Drawimg);
    }

    if (newError <= epsilon)
      converge = true;
    else
      error = newError;
    iter++;
  }

}

bool AAM_Parallel::Read(const std::string& filename) {
  ifstream input(filename.c_str(), ios::in | ios::binary);

  if (!input) {
    fprintf(stderr, "Cannot load model %s", filename.c_str());
    return false;
  }
  __model.Read(input);
  __R = cvCreateMat(__model.nModes()+4, __model.__texture.nPixels(), CV_64FC1);
  ReadCvMat(input, __R);


  __c = new double[__model.nModes()];
  __q = new double[4];
  __shape = new double[__model.__Qs->cols];
  __texture = new double [__model.__texture.nPixels()];
  __modelledTexture = new double[__model.__texture.nPixels()];
  __dif = new double[__model.__texture.nPixels()];
  __delta_c_q = new double[__model.nModes() + 4];

  return true;
}

void AAM_Parallel::ComputeModelledShape(IplImage* image) {
  CvMat* Qs = __model.__Qs;
  CvMat* So = __model.__MeanS;
  int k;
  double x, y;

  #pragma omp parallel
  {
    #pragma omp for schedule(dynamic,1) private(x,y,k)
    for (int i = 0 ; i < So->cols; i += 2) {
      k = 0;
      x = 0.0; y = 0.0;
      // this is done for every vertex
      while (k < __model.nModes()) {
          x += __c[k] * CVMAT_ELEM(Qs,k,i);
          y += __c[k] * CVMAT_ELEM(Qs,k,i+1);
          k++;
      }
      x += CVMAT_ELEM(So, 0, i);
      y += CVMAT_ELEM(So, 0, i+1);

      //TODO check why the floor
      __shape[i] = __q[0]*x - __q[1]*y + __q[2];
      __shape[i+1] = __q[1]*x + __q[0]*y + __q[3];

      // Ensure the vertex is inside the image.
      Clamp(__shape[i], 0.0, (double) (image->width - 1));
      Clamp(__shape[i+1], 0.0, (double) (image->height - 1));
    }
  }

}


void AAM_Parallel::ComputeModelledTexture(IplImage* image) {

  CvMat* Qg = __model.__Qg;
  CvMat* Go = __model.__MeanG;
  int j,k;
  int channels = image->nChannels, np = __model.__texture.nPixels();
  #pragma omp parallel
  {
    #pragma omp for schedule(dynamic,1)
    for (int i = 0; i < np; i++) {
      for (j = 0; j < channels; j++) {
        __modelledTexture[i*channels + j] = CVMAT_ELEM(Go, 0, i*channels + j);
        k = 0;
        while (k < __model.nModes()) {
          __modelledTexture[i*channels + j] += CVMAT_ELEM(Qg, k, i*channels + j) * __c[k];
          k++;
        }
      }
    }
  }
}

void AAM_Parallel::SampleTexture(IplImage* image) {
  int v1,v2,v3,tri_idx;
  std::vector<int> pixTri = __model.__paw.__pixTri;
  std::vector<std::vector<int>> tri = __model.__paw.__tri;
  std::vector<double> alpha = __model.__paw.__alpha;
  std::vector<double> beta = __model.__paw.__belta;
  std::vector<double> gamma = __model.__paw.__gamma;
  double x, y;

  #pragma omp parallel
  {
    #pragma omp for schedule(dynamic,1) private(x,y,v1,v2,v3,tri_idx)
    for (int i = 0; i < __model.__paw.__nPixels; i++) {
      tri_idx = pixTri[i];
      v1 = tri[tri_idx][0];
      v2 = tri[tri_idx][1];
      v3 = tri[tri_idx][2];

      x = 0.0; y = 0.0;
      x = alpha[i]*__shape[v1*2] + beta[i]*__shape[v2*2] + gamma[i]*__shape[v3*2];
      y = alpha[i]*__shape[v1*2 + 1] + beta[i]*__shape[v2*2 + 1] + gamma[i]*__shape[v3*2 + 1];

      BilinearInterpolation(image, x, y, &__texture[i*image->nChannels]);
    }
  }

}

void AAM_Parallel::NormalizingTexture(IplImage* image) {

  int channels = image->nChannels;
  double mean[MAX_CHANNELS], norm = 0.0, alpha = 0.0;
  const CvMat* mean_texture = __model.__texture.GetMean();
  bool alpha0 = false;
  for (int i = 0; i < channels; i++) {
    mean[i] = 0.0;
  }

  #pragma omp parallel shared(mean)
  {
    #pragma omp for schedule(dynamic,1) reduction(+:mean[0:MAX_CHANNELS])
    for (int i = 0; i < __model.__texture.nPixels(); i++) {
      for (int k = 0; k < channels; k++)
        mean[k] = mean[k] + __texture[i*channels + k];
    }

    #pragma omp single
    for (int k = 0 ; k < channels; k++)
      mean[k] /= __model.__texture.nPixels();


    #pragma omp for schedule(dynamic,1) reduction(+: norm)
    for (int i = 0; i < __model.__texture.nPixels(); i++) {
      for (int k = 0; k < channels; k++) {
          __texture[i*channels + k] -= mean[k];   // unbias
          norm = norm + __texture[i*channels + k];
      }
    }

    #pragma single
    norm /= __model.__texture.nPixels();

    #pragma omp for schedule(dynamic,1) reduction(+: alpha)
    for (int i = 0; i < __model.__texture.nPixels(); i++) {
      for (int k = 0; k < channels; k++) {
          __texture[i*channels + k] /= norm;   // normalize
          alpha = alpha + (__texture[i*channels + k] * CVMAT_ELEM(mean_texture,0,i*channels + k));
      }
    }

    #pragma omp single
    {
      alpha0 = (alpha == 0);
    }
    if (!alpha0) {
      #pragma omp for schedule(dynamic,1)
      for (int i = 0; i < __model.__texture.nPixels(); i++) {
        for (int k = 0; k < channels; k++) {
            __texture[i*channels + k] /= alpha;
        }
      }
    }
  }
}


double AAM_Parallel::ComputeEstimationError(IplImage* image) {
  double error = 0.0;
  ComputeModelledShape(image);
  SampleTexture(image);
  NormalizingTexture(image);
  ComputeModelledTexture(image);

  #pragma omp parallel
  {
    #pragma omp for schedule(dynamic,1) reduction(+:error)
    for (int i = 0; i < __model.__texture.nPixels()*3; i++) {
      __dif[i] = __texture[i] - __modelledTexture[i];
      error = error + (__dif[i]*__dif[i]);
    }
  }

  return error;
}

void AAM_Parallel::ParamsUpdate(IplImage* image) {
  int k = 0;
  for (k = 0; k < __model.nModes(); k++) {
    #pragma omp parallel
    {
      #pragma omp for schedule(dynamic,1)
      for (int i = 0; i < __model.__texture.nPixels(); i++) {
        __delta_c_q[k] = CVMAT_ELEM(__R, k, i) * (-__dif[i*image->nChannels]);
      }
    }
  }
}

void AAM_Parallel::EstimateParams(IplImage* image) {
  double tx = image->width/2, ty = image->height/2;

  __q[0] = 1;
  __q[1] = 1;
  __q[2] = tx;
  __q[3] = ty;

  for (int i = 0; i < __model.nModes(); i++)
    __c[i] = 0;     //use the mean appeareance

}

void AAM_Parallel::ComputeNewParams(double k) {
  int i;
  #pragma omp parallel
  {
    #pragma for schedule(dynamic,1) nowait
    for (i = 0; i < __model.nModes(); i++) {
      __c[i] += k * __delta_c_q[i];
    }

    #pragma for schedule(dynamic,1)
    for (i = 0; i < 4; i++) {
      __q[i] += k * __delta_c_q[__model.nModes() + i];
    }
  }
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
  int off_g = (image->nChannels == 3) ? 1 : 0;
	int off_r = (image->nChannels == 3) ? 2 : 0;

  int ixB1 = image->nChannels*X1, ixG1= ixB1+off_g, ixR1 = ixB1+off_r;
  int ixB2 = image->nChannels*X2,	ixG2= ixB2+off_g,	ixR2 = ixB2+off_r;


  char* p1 = (char*)(imgdata + step*Y1);
  char* p2 = (char*)(imgdata + step*Y2);

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
  CvMat shape = cvMat(1,__model.__shape.nPoints()*2,CV_64FC1, __shape);
  Shape.Mat2Point(&shape);
  double minV, maxV;
  CvMat texture = cvMat(1,__model.__texture.nPixels()*image->nChannels, CV_64FC1, __modelledTexture);
  cvMinMaxLoc(&texture, &minV, &maxV);
  cvConvertScale(&texture, &texture, 255/(maxV-minV), -minV*255/(maxV-minV));
  AAM_PAW paw;
  paw.Train(Shape, __model.__Points, __model.__Storage, __model.__paw.GetTri(), false);
  AAM_Common::DrawAppearance(image,Shape, &texture, paw, __model.__paw);
}
