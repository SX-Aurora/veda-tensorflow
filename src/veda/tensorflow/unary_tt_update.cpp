#include <veda/tensorflow/api.h>

namespace tensorflow {
//------------------------------------------------------------------------------
template<typename T, VEDATensors_unary_op OP>
struct UnaryTTUpdate : public OpKernel {
	explicit UnaryTTUpdate(OpKernelConstruction* ctx) : OpKernel(ctx) {}

	void Compute(OpKernelContext* ctx) override {
		using namespace veda::tensorflow;

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
		OP_REQUIRES_OK(ctx, PrepareToUpdateVariable<VEDATensors_handle_struct, T>(ctx, var_tensor, variable->copy_on_read_mode.load()));

		auto d_var_tensor	= tf2veda<T>(var_tensor);
		auto d_value		= tf2veda<T>(value);
		CVEDA(veda_tensors_unary_tt(handle(ctx), &d_var_tensor, &d_var_tensor, &d_value, OP));
	}
};

//------------------------------------------------------------------------------
}

#include "__ns.h"
//------------------------------------------------------------------------------
void init_unary_tt_update(void) {
	#define UnaryTTUpdate(N, O)	REG10_(N, "dtype",, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, ::tensorflow::UnaryTTUpdate, O)
	UnaryTTUpdate("AssignAddVariableOp",	VEDA_TENSORS_UNARY_ADD);
	UnaryTTUpdate("AssignSubVariableOp",	VEDA_TENSORS_UNARY_ADD);
}

//------------------------------------------------------------------------------
#include "__ns.h"
