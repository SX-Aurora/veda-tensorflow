#include <veda/tensorflow/api.h>

#include "__ns.h"
//------------------------------------------------------------------------------
device::ve::Handle*	vedaHandle(const ::tensorflow::OpKernelContext* ctx) {
	auto handle = dynamic_cast<device::ve::Handle*>(device::Type("ve")->handle(ctx->device()->tensorflow_gpu_device_info()->gpu_id));
	ASSERT(handle);
	return handle;
}

//------------------------------------------------------------------------------
#include "__ns.h"

extern "C" void TF_InitKernel(void) {
	using namespace sol::runtime::native::tensorflow::ve;

	// SEE: https://github.com/tensorflow/community/blob/master/rfcs/20190814-kernel-and-op-registration.md
	// SEE: https://github.com/tensorflow/tensorflow/blob/v2.5.0/tensorflow/c/kernels.cc#L48
	L_TRACE(">> TF_InitKernel");

	init_binary();
	init_constant_op();
	init_fill();
	init_function_ops();
	init_resource_variable_ops();
	init_shape_op();
	init_unary_t();
	init_unary_tt();
	init_unary_tt_update();
	
	L_TRACE("<< TF_InitKernel");
}
