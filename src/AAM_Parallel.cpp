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
    #pragma omp for schedule(dynamic,1) shared(Qs,So) private(x,y,k)
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
      __shape[i] = __q[0]*x - __q[1]*y + __q[2];
      __shape[i+1] = __q[0]*x + __q[1]*y + __q[3];
    }
  }

}
