
# Active Appearance Model C++ Library with Parallel Fitting implemented with OpenMP

This implementation was based on the AAM Library of GreatYao: https://github.com/greatyao/aamlibrary
and the parallel fitting algorithm was based on the parallel algorithm designed in "Efficient parallel
Implementation of Active Appeareance Model Fitting Algorithm on GPU" by Wang, Ma, Zhu et Sun for CUDA; in this project adapted to OpenMP

## Implementation

The parallel implementation can be found in __AAM_Parallel.c/.h__

## Dependencies
- opencv 1.0 or later
- cmake 2.6 or later
- gcc 7.0
- openmp 4.5

## How to build your program

  > mkdir build

  > cd build

  > cmake ..

  > make

### Prepare:
- For model training, you should have several pairs of images and annotations. AAMLibrary supports pts and asf format.
- Download the imm dataset from AAM-API's homepage [link: IMM Dataset](http://www2.imm.dtu.dk/pubdb/views/publication_details.php?id=922)
- Download helen dataset from this [link: Helen Dataset](http://ibug.doc.ic.ac.uk/resources/facial-point-annotations/)


### Training

- Train the Cootes's basic active appearance models using 16 layers
   > ./build train_images_path image_ext point_ext model_file


### Fitting

- Image alignment on an image
   > ./fit model_file image_file
