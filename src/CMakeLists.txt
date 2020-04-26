add_executable(trainer trainer.cc)
add_executable(predict predict.cc)

target_link_libraries(trainer tbb)
target_link_libraries(predict tbb)