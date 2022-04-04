#include <veda/tensorflow/api.h>

namespace tensorflow {
//------------------------------------------------------------------------------
template<typename T, VEDATensors_binary_op OP>
struct Binary : public OpKernel {
	explicit Binary(OpKernelConstruction* ctx) : OpKernel(ctx) {}

	void Compute(OpKernelContext* ctx) override {
		using namespace veda::tensorflow;

		auto& input_0 = ctx->input(0);
		auto& input_1 = ctx->input(1);

		Tensor* out = 0;
		if(input_0.shape() == input_1.shape())	OP_REQUIRES_OK(ctx, ctx->forward_input_or_allocate_output({0, 1}, 0, input_0.shape(), &out));
		else if(input_0.shape().dims() == 0)	OP_REQUIRES_OK(ctx, ctx->forward_input_or_allocate_output({1},    0, input_1.shape(), &out));
		else if(input_1.shape().dims() == 0)	OP_REQUIRES_OK(ctx, ctx->forward_input_or_allocate_output({0},    0, input_0.shape(), &out));
		THROWIF(out == 0, "Unsupported Binary"); // TODO: " << input_0.shape() << " and " << input_1.shape());

		auto d_out		= tf2veda<bool>(out);
		auto d_input_0	= tf2veda<T>(input_0);
		auto d_input_1	= tf2veda<T>(input_1);

		CVEDA(veda_tensors_binary(handle(ctx), &d_out, &d_input_0, &d_input_1, OP));
	}
};

//------------------------------------------------------------------------------
}

#include "__ns.h"
//------------------------------------------------------------------------------
void init_binary(void) {
	#define Binary(N, O) REG10_(N, "T",, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, ::tensorflow::Binary, O)
	Binary("Equal",			VEDA_TENSORS_BINARY_EQ);
	Binary("Greater",		VEDA_TENSORS_BINARY_GT);
	Binary("GreaterEqual",	VEDA_TENSORS_BINARY_GE);
	Binary("Less",			VEDA_TENSORS_BINARY_LT);
	Binary("LessEqual",		VEDA_TENSORS_BINARY_LE);
	Binary("NotEqual",		VEDA_TENSORS_BINARY_NE);
}

//------------------------------------------------------------------------------
#include "__ns.h"
