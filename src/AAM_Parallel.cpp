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

#define CVMAT_ELEM(Mat, i, j) *(Mat->db + i*Mat->cols + j)

using namespace std;

AAM_Parallel::AAM_Parallel() {

}

AAM_Parallel::~AAM_Parallel() {

}

void AAM_Parallel::Fit(const IplImage* image, int max_iter = 30, bool showprocess = false) {

}

bool AAM_Parallel::Read(const std::string& filename) {
  ifstream input(filename.c_str(), ios::in | ios:binary);
  if (!input) {
    fprintf(stderr, "Cannot load model %s", filename.c_str());
    return false;
  }
  __R = cvCreateMat(__model.nModes()+4, __model.texture.nPixels(), CV_64FC1);
  ReadCvMat(input, __R);
  __model.Read(input);
}

void AAM_Parallel::ComputeModelledShape(IplImage* image) {
  cvMat* Qs = __model.__Qs;
  cvMat* So = __model.__MeanS;
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
          x += __c[k] * CVMAT_ELEM(Qs,i,k);
          y += __c[k] * CVMAT_ELEM(Qs,i+1,k);
          k++;
      }
      x += CVMAT_ELEM(So, 0, i);
      y += CVMAT_ELEM(So, 0, i+1);

      //TODO check why the floor
      __shape[i] = std::floor(__q[0]*x - __q[1]*y + __q[2]);
      __shape[i+1] = std::floor(__q[0]*x + __q[1]*y + __q[3]);

      // Ensure the vertex is inside the image.
      Clamp(__shape[i], 0.0, (double) (image->width - 1));
      Clamp(__shape[i+1], 0.0, (double) (image->height - 1));
    }
  }

}

void ComputeModelledShape(IplImage* image) {

  cvMat* Qg = __model.__Qg;
  cvMat* Go = __model.__MeanG;
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
  std::vector<double> beta = __model.__paw.__beta;
  std::vector<double> gamma = __model.__paw.__gamma;
  double x, y;

  #pragma omp parallel
  {
    #pragma omp for schedule(dynamic,1) private(x,y,v1,v2,v3,tri_idx)
    for (int i = 0; i < __model.__texture.nPixels(); i++) {
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

void NormalizingTexture(IplImage* image) {

  int channels = image->nChannels;
  double mean[channels], norm = 0.0, alpha = 0.0;
  cvMat* mean_texture = __model.__texture.GetMean();

  for (int i = 0; i < channels; i++) {
    mean[i] = 0.0;
  }

  #pragma omp parallel
  {
    #pragma omp for schedule(dynamic,1) reduction(+: mean)
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
          alpha = alpha + (__texture[i*channels + k] * mean_texture[i*channel + k]);
      }
    }

    #pragma omp single
    if (alpha == 0) return

    #pragma omp for schedule(dynamic,1)
    for (int i = 0; i < __model.__texture.nPixels(); i++) {
      for (int k = 0; k < channels; k++) {
          __texture[i*channels + k] /= alpha;
      }
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
  char* imagedata = image->imageData;
  int step = image->widthStep;
  int off_g = (image->nChannels == 3) ? 1 : 0;
	int off_r = (image->nChannels == 3) ? 2 : 0;

  ixB1 = image->nChannels*X1; ixG1= ixB1+off_g;	ixR1 = ixB1+off_r;
  ixB2 = image->nChannels*X2;	ixG2= ixB2+off_g;	ixR2 = ixB2+off_r;

  p1 = (char*)(imgdata + step*Y1);
  p2 = (char*)(imgdata + step*Y2);

  pixel[0] = s1*(t1*p1[ixB1]+t0*p2[ixB1])+s0*(t1*p1[ixB2]+t0*p2[ixB2]);
  pixel[1] = s1*(t1*p1[ixG1]+t0*p2[ixG1])+s0*(t1*p1[ixG2]+t0*p2[ixG2]);
  pixel[2] = s1*(t1*p1[ixR1]+t0*p2[ixR1])+s0*(t1*p1[ixR2]+t0*p2[ixR2]);
}


void AAM_Parallel::Clamp(double& x, double min, double max) {
  if (x < min) x = min;
  if (x > max) x = max;
}
