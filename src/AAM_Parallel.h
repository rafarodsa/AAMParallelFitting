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

class AAM_Parallel;

class AAM_Parallel {

public:
  AAM_Parallel();
  ~AAM_Parallel();

public:

  bool Fit(const IplImage* image, int max_iter = 30, bool showprocess = false);
  void Read(const std::string& filename);

private:
  double ComputeEstimationError(IplImage* image);
  void ComputeModelledShape(IplImage* image);
  void Clamp(double& x, double min, double max);
  void ComputeModelledTexture(IplImage* image);
  void SampleTexture(IplImage* image);
  void BilinearInterpolation(IplImage* image, int x, int y, double* pixel)
  void NormalizingTexture(IplImage* image);
  void EstimateParams(IplImage* image);
  void ParamsUpdate(IplImage* image);
  void ComputeNewParams(double k);

private:
  AAM_CAM __model;
  CvMat* __R;

private:
  double* __c; //CAM parameters
  double* __t; //similarity transformation (sx,sy,tx,ty)
  double* __shape; // current shape S
  double* __texture; //current texture gs
  double* __modelledTexture;
  double* __dif;
  double* __delta_c_q;
};

#endif
