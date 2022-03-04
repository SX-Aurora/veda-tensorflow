#include <veda/tensorflow/api.h>

using sol::runtime::native::tensorflow::ve::vedaHandle;

namespace tensorflow {
//------------------------------------------------------------------------------
template<typename T, int OP>
struct UnaryTTUpdate : public OpKernel {
	explicit UnaryTTUpdate(OpKernelConstruction* ctx) : OpKernel(ctx) {}

	void Compute(OpKernelContext* ctx) override {
		core::RefCountPtr<Var> variable;
		OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &variable));

		const Tensor& value = ctx->input(1);
		mutex_lock ml(*variable->mu());
		Tensor* var_tensor = variable->tensor();
		OP_REQUIRES(ctx, var_tensor->shape().IsSameSize(value.shape()),
			errors::InvalidArgument("Cannot update variable with shape ",
									var_tensor->shape().DebugString(),
									" using a Tensor with shape ",
									value.shape().DebugString(),
									", shapes must be equal."));
		OP_REQUIRES_OK(ctx, PrepareToUpdateVariable<VEDevice, T>(ctx, var_tensor, variable->copy_on_read_mode.load()));
		vedaHandle(ctx)->unaryTT(ptr<T>(var_tensor), ptr<T>(var_tensor), ptr<T>(value), cnt(var_tensor), cnt(var_tensor), cnt(value), OP, sol_dtype<T>());
	}
};

//------------------------------------------------------------------------------
}

#include "__ns.h"
//------------------------------------------------------------------------------
void init_unary_tt_update(void) {
	#define UnaryTTUpdate(N, O)	REG10_(N, "dtype",, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, ::tensorflow::UnaryTTUpdate, O)
	UnaryTTUpdate("AssignAddVariableOp",	UnaryOp::ADD);
	UnaryTTUpdate("AssignSubVariableOp",	UnaryOp::ADD);
}

//------------------------------------------------------------------------------
#include "__ns.h"
