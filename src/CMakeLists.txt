add_executable(trainer trainer.cc)
add_executable(predict predict.cc)

target_link_libraries(trainer OpenMP::OpenMP_CXX)
target_link_libraries(predict OpenMP::OpenMP_CXX)