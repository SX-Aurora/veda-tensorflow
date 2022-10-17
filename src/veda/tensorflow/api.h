#pragma once

#include <tensorflow/core/public/version.h>

#if TF_MAJOR_VERSION != 2
#error "Requires TF v2.X"
#endif

#if TF_MINOR_VERSION < 9
	#if _GLIBCXX_USE_CXX11_ABI != 0
	#error "TF < v2.9 requires _GLIBCXX_USE_CXX11_ABI=0"
	#endif
#else
	#if _GLIBCXX_USE_CXX11_ABI != 1
	#error "TF >= v2.9 requires _GLIBCXX_USE_CXX11_ABI=1"
	#endif
#endif

// https://stackoverflow.com/questions/2324658/how-to-determine-the-version-of-the-c-standard-used-by-the-compiler
#if TF_MINOR_VERSION <= 6
	#if __cplusplus != 201103L
	#error "TF < v2.7 cannot be linked when not using C++11"
	#endif
#elif TF_MINOR_VERSION <= 9
	#if __cplusplus != 201402L
	#error "TF >= 2.7 cannot be linked when not using C++14"
	#endif
#else
	#if __cplusplus != 201703L
	#error "TF >= 2.10 cannot be linked when not using C++17"
	#endif
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

#define REG1_( N, T, C, O, T0, ...)										REGISTER_KERNEL_BUILDER(Name(N).Device(DEVICE_VE).TypeConstraint<T0>(T) C, O<T0, __VA_ARGS__>)
#define REG2_( N, T, C, O, T0, T1, ...)									REG1_(N, T, C, O, T0, __VA_ARGS__); REG1_(N, T, C, O, T1, __VA_ARGS__)
#define REG3_( N, T, C, O, T0, T1, T2, ...)								REG1_(N, T, C, O, T0, __VA_ARGS__); REG2_(N, T, C, O, T1, T2, __VA_ARGS__)
#define REG4_( N, T, C, O, T0, T1, T2, T3, ...)							REG1_(N, T, C, O, T0, __VA_ARGS__); REG3_(N, T, C, O, T1, T2, T3, __VA_ARGS__)
#define REG5_( N, T, C, O, T0, T1, T2, T3, T4, ...)						REG1_(N, T, C, O, T0, __VA_ARGS__); REG4_(N, T, C, O, T1, T2, T3, T4, __VA_ARGS__)
#define REG6_( N, T, C, O, T0, T1, T2, T3, T4, T5, ...)					REG1_(N, T, C, O, T0, __VA_ARGS__); REG5_(N, T, C, O, T1, T2, T3, T4, T5, __VA_ARGS__)
#define REG7_( N, T, C, O, T0, T1, T2, T3, T4, T5, T6, ...)				REG1_(N, T, C, O, T0, __VA_ARGS__); REG6_(N, T, C, O, T1, T2, T3, T4, T5, T6, __VA_ARGS__)
#define REG8_( N, T, C, O, T0, T1, T2, T3, T4, T5, T6, T7, ...)			REG1_(N, T, C, O, T0, __VA_ARGS__); REG7_(N, T, C, O, T1, T2, T3, T4, T5, T6, T7, __VA_ARGS__)
#define REG9_( N, T, C, O, T0, T1, T2, T3, T4, T5, T6, T7, T8, ...)		REG1_(N, T, C, O, T0, __VA_ARGS__); REG8_(N, T, C, O, T1, T2, T3, T4, T5, T6, T7, T8, __VA_ARGS__)
#define REG10_(N, T, C, O, T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, ...)	REG1_(N, T, C, O, T0, __VA_ARGS__); REG9_(N, T, C, O, T1, T2, T3, T4, T5, T6, T7, T8, T9, __VA_ARGS__)

// -----------------------------------------------------------------------------
namespace tensorflow {
	constexpr std::array<DataType, 11> VE_TYPES = {{DT_UINT8, DT_UINT16, DT_INT8, DT_INT16, DT_INT32, DT_UINT32, DT_UINT64, DT_INT64, DT_FLOAT, DT_DOUBLE, DT_BOOL}};
	typedef VEDATensors_handle_struct VEDevice;
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
//------------------------------------------------------------------------------
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
template<typename T> VEDATensors_dtype dtype(void);
template<>	inline VEDATensors_dtype	dtype<bool>		(void)	{	return VEDA_TENSORS_DTYPE_S8;	}
template<>	inline VEDATensors_dtype	dtype<int8_t>	(void)	{	return VEDA_TENSORS_DTYPE_S8;	}
template<>	inline VEDATensors_dtype	dtype<int16_t>	(void)	{	return VEDA_TENSORS_DTYPE_S16;	}
template<>	inline VEDATensors_dtype	dtype<int32_t>	(void)	{	return VEDA_TENSORS_DTYPE_S32;	}
template<>	inline VEDATensors_dtype	dtype<int64_t>	(void)	{	return VEDA_TENSORS_DTYPE_S64;	}
template<>	inline VEDATensors_dtype	dtype<uint8_t>	(void)	{	return VEDA_TENSORS_DTYPE_U8;	}
template<>	inline VEDATensors_dtype	dtype<uint16_t>	(void)	{	return VEDA_TENSORS_DTYPE_U16;	}
template<>	inline VEDATensors_dtype	dtype<uint32_t>	(void)	{	return VEDA_TENSORS_DTYPE_U32;	}
template<>	inline VEDATensors_dtype	dtype<uint64_t>	(void)	{	return VEDA_TENSORS_DTYPE_U64;	}
template<>	inline VEDATensors_dtype	dtype<float>	(void)	{	return VEDA_TENSORS_DTYPE_F32;	}
template<>	inline VEDATensors_dtype	dtype<double>	(void)	{	return VEDA_TENSORS_DTYPE_F64;	}

//------------------------------------------------------------------------------
template<typename T> inline VEDATensors_tensor tf2veda(const ::tensorflow::Tensor* t) {
	return {t->dims(), t->shape().dim_sizes(), dtype<T>(), (VEDAdeviceptr)t->flat<T>().data()};
}

//------------------------------------------------------------------------------
template<typename T> inline VEDATensors_scalar tf2scalar(const ::tensorflow::Tensor* t)	{	return veda_tensors_scalar(t->scalar<T>()());	}
template<typename T> inline VEDATensors_scalar tf2scalar(const ::tensorflow::Tensor& t)	{	return tf2scalar<T>(&t);						}
template<typename T> inline VEDATensors_tensor tf2veda	(const ::tensorflow::Tensor& t)	{	return tf2veda<T>(&t);							}

//------------------------------------------------------------------------------
VEDATensors_handle	handle						(const ::tensorflow::OpKernelContext* ctx);
void				init_binary					(void);
void				init_broadcast_ops			(void);
void				init_constant_op			(void);
void				init_fill					(void);
void				init_function_ops			(void);
void				init_resource_variable_ops	(void);
void				init_shape_op				(void);
void				init_training_ops			(void);
void				init_unary_t				(void);
void				init_unary_tt				(void);
void				init_unary_tt_update		(void);

//------------------------------------------------------------------------------
#include "__ns.h"

#include "dense_update_functor.h"