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
  void Read(const string& filename);

private:
  void ComputeEstimationError();
  void ComputeShape();
  void ComputeModelledTexture();
  void SampleTexture();
  void NormalizingTexture();
  void EstimateParams();
  void ComputeNewParams();

private:
  AAM_CAM __model;

};

#endif
