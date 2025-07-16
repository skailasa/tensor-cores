add_executable(sgemm main.cpp ${CPP_SOURCES})

target_include_directories(sgemm
  PRIVATE
    ${CMAKE_SOURCE_DIR}/include
  PUBLIC
    ${PROJECT_BINARY_DIR}  # so consumers can see Config.h if needed
)

target_link_libraries(sgemm
  PRIVATE
    gemm_with_openmp
)

if(NVIDIA)
  target_link_libraries(sgemm
  PRIVATE
    cuda_kernels
)

elseif(AMD)
  # target_link_libraries(${PROJECT_NAME}
  #   roc::rocthrust
  #   roc::hipsparse
  # )
elseif(CPU)
endif()

add_custom_target(run_sgemm
  COMMAND ${CMAKE_COMMAND} -E echo "🚀 Running SGEMM..."
  COMMAND ./sgemm
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/examples/sgemm
  DEPENDS sgemm
)