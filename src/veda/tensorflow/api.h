#pragma once

#if _GLIBCXX_USE_CXX11_ABI != 0
#error "TF requires _GLIBCXX_USE_CXX11_ABI=0"
#endif

// https://stackoverflow.com/questions/2324658/how-to-determine-the-version-of-the-c-standard-used-by-the-compiler
#if __cplusplus != 201103L
#error "TF cannot be linked when not using C++11"
#endif

#define L_MODULE "VEDA-TensorFlow"
#include <tungl/c.h>
#include <veda/api.h>
#include <veda/tensors/api.h>
#undef CVEDA
#define CVEDA(...) veda::tensorflow::check(__VA_ARGS__, __FILE__, __LINE__)

// BUGFIX: https://stackoverflow.com/questions/55958530/custom-resource-in-tensorflow
#ifndef NDEBUG
#define NDEBUG 1
#include <tensorflow/core/platform/default/logging.h>
#undef NDEBUG
#endif
// /BUGFIX

#include <tensorflow/c/experimental/stream_executor/stream_executor.h>
#include <tensorflow/core/common_runtime/device.h>
#include <tensorflow/core/common_runtime/pluggable_device/pluggable_device.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/op_requires.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/graph/graph_node_util.h>
#include <tensorflow/core/kernels/constant_op.h>
#include <tensorflow/core/kernels/cwise_ops_common.h>
#include <tensorflow/core/kernels/function_ops.h>
#include <tensorflow/core/kernels/gather_functor.h>
#include <tensorflow/core/kernels/identity_op.h>
#include <tensorflow/core/kernels/resource_variable_ops.h>
#include <tensorflow/core/kernels/shape_ops.h>
#include <tensorflow/core/kernels/training_op_helpers.h>
#include <tensorflow/core/platform/casts.h>
#include <tensorflow/core/util/util.h>
#include <tensorflow/core/framework/op_kernel.h>

// Macros ----------------------------------------------------------------------
#define DEVICE_VE "VE"

#define REG1_( N, T, C, T0, O, ...)										REGISTER_KERNEL_BUILDER(Name(N).Device(DEVICE_VE).TypeConstraint<T0>(T) C, O<T0, __VA_ARGS__>)
#define REG2_( N, T, C, T0, T1, O, ...)									REG1_(N, T, C, T0, O, __VA_ARGS__); REG1_(N, T, C, T1, O, __VA_ARGS__)
#define REG3_( N, T, C, T0, T1, T2, O, ...)								REG1_(N, T, C, T0, O, __VA_ARGS__); REG2_(N, T, C, T1, T2, O, __VA_ARGS__)
#define REG4_( N, T, C, T0, T1, T2, T3, O, ...)							REG1_(N, T, C, T0, O, __VA_ARGS__); REG3_(N, T, C, T1, T2, T3, O, __VA_ARGS__)
#define REG5_( N, T, C, T0, T1, T2, T3, T4, O, ...)						REG1_(N, T, C, T0, O, __VA_ARGS__); REG4_(N, T, C, T1, T2, T3, T4, O, __VA_ARGS__)
#define REG6_( N, T, C, T0, T1, T2, T3, T4, T5, O, ...)					REG1_(N, T, C, T0, O, __VA_ARGS__); REG5_(N, T, C, T1, T2, T3, T4, T5, O, __VA_ARGS__)
#define REG7_( N, T, C, T0, T1, T2, T3, T4, T5, T6, O, ...)				REG1_(N, T, C, T0, O, __VA_ARGS__); REG6_(N, T, C, T1, T2, T3, T4, T5, T6, O, __VA_ARGS__)
#define REG8_( N, T, C, T0, T1, T2, T3, T4, T5, T6, T7, O, ...)			REG1_(N, T, C, T0, O, __VA_ARGS__); REG7_(N, T, C, T1, T2, T3, T4, T5, T6, T7, O, __VA_ARGS__)
#define REG9_( N, T, C, T0, T1, T2, T3, T4, T5, T6, T7, T8, O, ...)		REG1_(N, T, C, T0, O, __VA_ARGS__); REG8_(N, T, C, T1, T2, T3, T4, T5, T6, T7, T8, O, __VA_ARGS__)
#define REG10_(N, T, C, T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, O, ...)	REG1_(N, T, C, T0, O, __VA_ARGS__); REG9_(N, T, C, T1, T2, T3, T4, T5, T6, T7, T8, T9, O, __VA_ARGS__)

// TF Types --------------------------------------------------------------------
namespace tensorflow {
	constexpr std::array<DataType, 11> VE_TYPES = {{DT_UINT8, DT_UINT16, DT_INT8, DT_INT16, DT_INT32, DT_UINT32, DT_UINT64, DT_INT64, DT_FLOAT, DT_DOUBLE, DT_BOOL}};
	
//--------------------------------------------------------------------------
static inline size_t cnt(const Tensor& t) {
	size_t c = 1;
	for(auto d : t.shape())
		c *= d.size;
	return c;
}

//------------------------------------------------------------------------------
static inline size_t cnt(const Tensor* t) {
	return cnt(*t);
}

//------------------------------------------------------------------------------
template<typename T>
static inline VEDAdeviceptr ptr(const Tensor& t) {
	return (VEDAdeviceptr)t.flat<T>().data();
}

//------------------------------------------------------------------------------
template<typename T>
static inline VEDAdeviceptr	ptr(const Tensor* t) {
	return ptr<T>(*t);
}

//------------------------------------------------------------------------------
}

//------------------------------------------------------------------------------
struct SP_Stream_st {
	VEDAstream stream;
};

//------------------------------------------------------------------------------
struct SP_Timer_st {
	uint64_t start;
	uint64_t end;
};

#include "__ns.h"
#define HANDLE(DEVICE)	((VEDATensors_handle)DEVICE->device_handle)
#define GUARD(DEVICE)	veda::tensorflow::Guard __guard__(HANDLE(DEVICE))

//------------------------------------------------------------------------------
inline void check(VEDAresult res, const char* file, const int line) {
	if(__builtin_expect((res != VEDA_SUCCESS), 0)) {
		const char* err;
		vedaGetErrorName(res, &err);
		THROWAT(L_MODULE, file, line, "VEDA_ERROR: %s", err);
	}
}

//------------------------------------------------------------------------------
struct Guard {
	inline Guard(VEDATensors_handle hnd) {
		CVEDA(vedaCtxPushCurrent(hnd->ctx));
	}

	inline ~Guard(void) {
		VEDAcontext ctx;
		CVEDA(vedaCtxPopCurrent(&ctx));
	}
};

//------------------------------------------------------------------------------
VEDAcontext	vedaContext					(const ::tensorflow::OpKernelContext* ctx);
void		init_binary					(void);
void		init_constant_op			(void);
void		init_fill					(void);
void		init_function_ops			(void);
void		init_resource_variable_ops	(void);
void		init_unary_t				(void);
void		init_unary_tt				(void);
void		init_unary_tt_update		(void);
void		init_shape_op				(void);
//------------------------------------------------------------------------------
#include "__ns.h"

// TODO: #include "dense_update_functor.h"