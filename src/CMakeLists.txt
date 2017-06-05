include_directories ("${PROJECT_SOURCE_DIR}/aamlibrary/src")

add_executable(simple_train train.cpp)
target_link_libraries(simple_train aamlibrary ${OpenCV_LIBS})

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
