/*****************************************************************
  Parallel implementation of Cootes Fitting Algorithm using
  OpenMP.

  Based on the parallel algorithm designed in "Efficient parallel
  Implementation of Active Appeareance Model Fitting Algorithm
  on GPU" by Wang, Ma, Zhu et Sun.

  @author Rafael Rodriguez (rodriguezsrafa@gmail.com)
  @date June 2017
******************************************************************/

#ifndef AAM_PARALLEL_H
#define AAM_PARALLEL_H

#include "AAM_CAM.h"
#include "AAM_Util.h"

class AAM_Parallel;

class AAM_Parallel {

public:
  AAM_Parallel();
  ~AAM_Parallel();

public:

  void Fit(IplImage* image, AAM_Shape& shape, int max_iter = 30, bool showprocess = false, double epsilon = 0.000003);
  bool Read(const std::string& filename);

  void Draw(IplImage* image);
private:
  double ComputeEstimationError(IplImage* image, double* __uc, double* __uq);
  void ComputeModelledShape(IplImage* image, double* __uc, double* __uq);
  void Clamp(double& x, double min, double max);
  void ComputeModelledTexture(IplImage* image, double* __uc);
  void SampleTexture(IplImage* image);
  void BilinearInterpolation(IplImage* image, double x, double y, double* pixel);
  void NormalizingTexture(IplImage* image);
  void EstimateParams(IplImage* image);
  void ParamsUpdate(IplImage* image);
  void ComputeNewParams(double k, double* __c, double* __q);

private:
  AAM_Pyramid __modelP;
  AAM_CAM* __model;
  CvMat* __R;

private:
  double* __c; //CAM parameters
  double* __q; //similarity transformation (sx,sy,tx,ty)
  double* __shape; // current shape S
  double* __texture; //current texture gs
  double* __modelledTexture;
  double* __dif;
  double* __delta_c_q;
};

#endif
