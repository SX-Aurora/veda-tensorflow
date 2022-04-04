#include <veda/tensorflow/api.h>

namespace tensorflow {
//------------------------------------------------------------------------------
template<typename T, VEDATensors_unary_op OP>
struct UnaryT : public OpKernel {
	explicit UnaryT(OpKernelConstruction* ctx) : OpKernel(ctx) {}

	void Compute(OpKernelContext* ctx) override {
		using namespace veda::tensorflow;

		auto& input = ctx->input(0);
		Tensor* out = 0;
		OP_REQUIRES_OK(ctx, ctx->forward_input_or_allocate_output({0}, 0, input.shape(), &out));

		auto d_out		= tf2veda<T>(out);
		auto d_input	= tf2veda<T>(input);
		CVEDA(veda_tensors_unary_t(handle(ctx), &d_out, &d_input, OP));
	}
};

//------------------------------------------------------------------------------
}

#include "__ns.h"
//------------------------------------------------------------------------------
void init_unary_t(void) {
	#define UnaryT(N, O) REG10_(N, "T",, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, ::tensorflow::UnaryT, O)
	UnaryT("Sqrt",	VEDA_TENSORS_UNARY_SQRT);
	UnaryT("Abs",	VEDA_TENSORS_UNARY_ABS);
	UnaryT("Rsqrt",	VEDA_TENSORS_UNARY_RSQRT);
	UnaryT("Sin",	VEDA_TENSORS_UNARY_SIN);
	UnaryT("Cos",	VEDA_TENSORS_UNARY_COS);
	UnaryT("Tan",	VEDA_TENSORS_UNARY_TAN);
	UnaryT("Exp",	VEDA_TENSORS_UNARY_EXP);
	UnaryT("Log",	VEDA_TENSORS_UNARY_LOG);
}

//------------------------------------------------------------------------------
#include "__ns.h"
