LIST(APPEND SRC
	# PluggableDevice
	${CMAKE_CURRENT_LIST_DIR}/platform.cpp
	${CMAKE_CURRENT_LIST_DIR}/stream_executor.cpp

	# Kernels
	${CMAKE_CURRENT_LIST_DIR}/kernels.cpp
	${CMAKE_CURRENT_LIST_DIR}/binary.cpp
	${CMAKE_CURRENT_LIST_DIR}/constant_op.cpp
	${CMAKE_CURRENT_LIST_DIR}/function_ops.cpp
	${CMAKE_CURRENT_LIST_DIR}/fill.cpp
	${CMAKE_CURRENT_LIST_DIR}/op_kernel.cpp
	${CMAKE_CURRENT_LIST_DIR}/resource_variable_ops.cpp
	${CMAKE_CURRENT_LIST_DIR}/unary_t.cpp
	${CMAKE_CURRENT_LIST_DIR}/unary_tt.cpp
	${CMAKE_CURRENT_LIST_DIR}/unary_tt_update.cpp
	${CMAKE_CURRENT_LIST_DIR}/shape_op.cpp
	${CMAKE_CURRENT_LIST_DIR}/training_ops.cpp
)
