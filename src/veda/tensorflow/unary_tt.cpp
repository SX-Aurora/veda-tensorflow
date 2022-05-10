#include <veda/tensorflow/api.h>

namespace tensorflow {
//------------------------------------------------------------------------------
template<typename T, VEDATensors_unary_op OP>
struct UnaryTT : public OpKernel {
	explicit UnaryTT(OpKernelConstruction* ctx) : OpKernel(ctx) {}

	void Compute(OpKernelContext* ctx) override {
		using namespace veda::tensorflow;

		auto& input_0 = ctx->input(0);

		Tensor* out = 0;
		switch(ctx->num_inputs()) {
			case 1: {
				OP_REQUIRES_OK(ctx, ctx->forward_input_or_allocate_output({0}, 0, input_0.shape(), &out));
				auto d_out		= tf2veda<T>(out);
				auto d_input_0	= tf2veda<T>(input_0);
				CVEDA(veda_tensors_unary_tt(handle(ctx), &d_out, &d_input_0, &d_input_0, OP));
			} break;

			case 2: {
				auto& input_1 = ctx->input(1);

				if(input_0.shape() == input_1.shape())										OP_REQUIRES_OK(ctx, ctx->forward_input_or_allocate_output({0, 1}, 0, input_0.shape(), &out));
				else if(input_0.shape().dims() == 0 || input_0.shape().num_elements() == 1)	OP_REQUIRES_OK(ctx, ctx->forward_input_or_allocate_output({1},    0, input_1.shape(), &out));
				else if(input_1.shape().dims() == 0 || input_1.shape().num_elements() == 1)	OP_REQUIRES_OK(ctx, ctx->forward_input_or_allocate_output({0},    0, input_0.shape(), &out));
				THROWIF(out == 0, "Unsupported UnaryTT (%s)", veda_tensors_get_unary(OP));

				auto d_out		= tf2veda<T>(out);
				auto d_input_0	= tf2veda<T>(input_0);
				auto d_input_1	= tf2veda<T>(input_1);
				CVEDA(veda_tensors_unary_tt(handle(ctx), &d_out, &d_input_0, &d_input_1, OP));
			} break;

			default:
				FAIL();
		}
	}
};

//------------------------------------------------------------------------------
}

#include "__ns.h"
//------------------------------------------------------------------------------
void init_unary_tt(void) {
	#define UnaryTT(N, O) REG10_(N, "T",, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, ::tensorflow::UnaryTT, O)
	UnaryTT("Add",		VEDA_TENSORS_UNARY_ADD);
	UnaryTT("AddV2",	VEDA_TENSORS_UNARY_ADD);
	UnaryTT("Mul",		VEDA_TENSORS_UNARY_MUL);
	UnaryTT("RealDiv",	VEDA_TENSORS_UNARY_DIV);
	UnaryTT("Sub",		VEDA_TENSORS_UNARY_SUB);
	UnaryTT("Square",	VEDA_TENSORS_UNARY_MUL);
}

//------------------------------------------------------------------------------
#include "__ns.h"