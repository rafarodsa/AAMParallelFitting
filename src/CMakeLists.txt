include_directories ("${PROJECT_SOURCE_DIR}/aamlibrary/src")
include_directories("${PROJECT_SOURCE_DIR}/src")

add_library(AAM_Parallel AAM_Parallel.cpp)

<<<<<<< HEAD
add_executable(train train.cpp)
target_link_libraries(train aamlibrary ${OpenCV_LIBS})
=======
add_executable(build train.cpp)
target_link_libraries(build aamlibrary ${OpenCV_LIBS})
>>>>>>> ParallelFitting
add_executable(fit fit.cpp)
target_link_libraries(fit aamlibrary ${OpenCV_LIBS} AAM_Parallel)
add_executable(evaluation evaluation.cpp)
target_link_libraries(evaluation aamlibrary ${OpenCV_LIBS} AAM_Parallel)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
