# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.8

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.8.1/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.8.1/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "/Users/rafaelrodriguez/Documents/Polimi/2Semestre/Advanced Algorithms and Parallel Programming/AAM_OpenMP"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/Users/rafaelrodriguez/Documents/Polimi/2Semestre/Advanced Algorithms and Parallel Programming/AAM_OpenMP"

# Include any dependencies generated for this target.
include src/CMakeFiles/simple_train.dir/depend.make

# Include the progress variables for this target.
include src/CMakeFiles/simple_train.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/simple_train.dir/flags.make

src/CMakeFiles/simple_train.dir/train.cpp.o: src/CMakeFiles/simple_train.dir/flags.make
src/CMakeFiles/simple_train.dir/train.cpp.o: src/train.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/Users/rafaelrodriguez/Documents/Polimi/2Semestre/Advanced Algorithms and Parallel Programming/AAM_OpenMP/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/simple_train.dir/train.cpp.o"
	cd "/Users/rafaelrodriguez/Documents/Polimi/2Semestre/Advanced Algorithms and Parallel Programming/AAM_OpenMP/src" && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/simple_train.dir/train.cpp.o -c "/Users/rafaelrodriguez/Documents/Polimi/2Semestre/Advanced Algorithms and Parallel Programming/AAM_OpenMP/src/train.cpp"

src/CMakeFiles/simple_train.dir/train.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/simple_train.dir/train.cpp.i"
	cd "/Users/rafaelrodriguez/Documents/Polimi/2Semestre/Advanced Algorithms and Parallel Programming/AAM_OpenMP/src" && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/Users/rafaelrodriguez/Documents/Polimi/2Semestre/Advanced Algorithms and Parallel Programming/AAM_OpenMP/src/train.cpp" > CMakeFiles/simple_train.dir/train.cpp.i

src/CMakeFiles/simple_train.dir/train.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/simple_train.dir/train.cpp.s"
	cd "/Users/rafaelrodriguez/Documents/Polimi/2Semestre/Advanced Algorithms and Parallel Programming/AAM_OpenMP/src" && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/Users/rafaelrodriguez/Documents/Polimi/2Semestre/Advanced Algorithms and Parallel Programming/AAM_OpenMP/src/train.cpp" -o CMakeFiles/simple_train.dir/train.cpp.s

src/CMakeFiles/simple_train.dir/train.cpp.o.requires:

.PHONY : src/CMakeFiles/simple_train.dir/train.cpp.o.requires

src/CMakeFiles/simple_train.dir/train.cpp.o.provides: src/CMakeFiles/simple_train.dir/train.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/simple_train.dir/build.make src/CMakeFiles/simple_train.dir/train.cpp.o.provides.build
.PHONY : src/CMakeFiles/simple_train.dir/train.cpp.o.provides

src/CMakeFiles/simple_train.dir/train.cpp.o.provides.build: src/CMakeFiles/simple_train.dir/train.cpp.o


# Object files for target simple_train
simple_train_OBJECTS = \
"CMakeFiles/simple_train.dir/train.cpp.o"

# External object files for target simple_train
simple_train_EXTERNAL_OBJECTS =

bin/simple_train: src/CMakeFiles/simple_train.dir/train.cpp.o
bin/simple_train: src/CMakeFiles/simple_train.dir/build.make
bin/simple_train: bin/libaamlibrary.dylib
bin/simple_train: /usr/local/lib/libopencv_videostab.2.4.13.dylib
bin/simple_train: /usr/local/lib/libopencv_ts.a
bin/simple_train: /usr/local/lib/libopencv_superres.2.4.13.dylib
bin/simple_train: /usr/local/lib/libopencv_stitching.2.4.13.dylib
bin/simple_train: /usr/local/lib/libopencv_contrib.2.4.13.dylib
bin/simple_train: /usr/local/lib/libopencv_nonfree.2.4.13.dylib
bin/simple_train: /usr/local/lib/libopencv_ocl.2.4.13.dylib
bin/simple_train: /usr/local/lib/libopencv_gpu.2.4.13.dylib
bin/simple_train: /usr/local/lib/libopencv_photo.2.4.13.dylib
bin/simple_train: /usr/local/lib/libopencv_objdetect.2.4.13.dylib
bin/simple_train: /usr/local/lib/libopencv_legacy.2.4.13.dylib
bin/simple_train: /usr/local/lib/libopencv_video.2.4.13.dylib
bin/simple_train: /usr/local/lib/libopencv_ml.2.4.13.dylib
bin/simple_train: /usr/local/lib/libopencv_calib3d.2.4.13.dylib
bin/simple_train: /usr/local/lib/libopencv_features2d.2.4.13.dylib
bin/simple_train: /usr/local/lib/libopencv_highgui.2.4.13.dylib
bin/simple_train: /usr/local/lib/libopencv_imgproc.2.4.13.dylib
bin/simple_train: /usr/local/lib/libopencv_flann.2.4.13.dylib
bin/simple_train: /usr/local/lib/libopencv_core.2.4.13.dylib
bin/simple_train: src/CMakeFiles/simple_train.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir="/Users/rafaelrodriguez/Documents/Polimi/2Semestre/Advanced Algorithms and Parallel Programming/AAM_OpenMP/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../bin/simple_train"
	cd "/Users/rafaelrodriguez/Documents/Polimi/2Semestre/Advanced Algorithms and Parallel Programming/AAM_OpenMP/src" && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/simple_train.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/simple_train.dir/build: bin/simple_train

.PHONY : src/CMakeFiles/simple_train.dir/build

src/CMakeFiles/simple_train.dir/requires: src/CMakeFiles/simple_train.dir/train.cpp.o.requires

.PHONY : src/CMakeFiles/simple_train.dir/requires

src/CMakeFiles/simple_train.dir/clean:
	cd "/Users/rafaelrodriguez/Documents/Polimi/2Semestre/Advanced Algorithms and Parallel Programming/AAM_OpenMP/src" && $(CMAKE_COMMAND) -P CMakeFiles/simple_train.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/simple_train.dir/clean

src/CMakeFiles/simple_train.dir/depend:
	cd "/Users/rafaelrodriguez/Documents/Polimi/2Semestre/Advanced Algorithms and Parallel Programming/AAM_OpenMP" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/Users/rafaelrodriguez/Documents/Polimi/2Semestre/Advanced Algorithms and Parallel Programming/AAM_OpenMP" "/Users/rafaelrodriguez/Documents/Polimi/2Semestre/Advanced Algorithms and Parallel Programming/AAM_OpenMP/src" "/Users/rafaelrodriguez/Documents/Polimi/2Semestre/Advanced Algorithms and Parallel Programming/AAM_OpenMP" "/Users/rafaelrodriguez/Documents/Polimi/2Semestre/Advanced Algorithms and Parallel Programming/AAM_OpenMP/src" "/Users/rafaelrodriguez/Documents/Polimi/2Semestre/Advanced Algorithms and Parallel Programming/AAM_OpenMP/src/CMakeFiles/simple_train.dir/DependInfo.cmake" --color=$(COLOR)
.PHONY : src/CMakeFiles/simple_train.dir/depend

