#include <veda/tensorflow/api.h>

namespace tensorflow {
//------------------------------------------------------------------------------
template<typename T, int OP>
struct Binary : public OpKernel {
	explicit Binary(OpKernelConstruction* ctx) : OpKernel(ctx) {}

	void Compute(OpKernelContext* ctx) override {
		auto& input_0 = ctx->input(0);
		auto& input_1 = ctx->input(1);

		Tensor* out = 0;
		if(input_0.shape() == input_1.shape())	OP_REQUIRES_OK(ctx, ctx->forward_input_or_allocate_output({0, 1}, 0, input_0.shape(), &out));
		else if(input_0.shape().dims() == 0)	OP_REQUIRES_OK(ctx, ctx->forward_input_or_allocate_output({1},    0, input_1.shape(), &out));
		else if(input_1.shape().dims() == 0)	OP_REQUIRES_OK(ctx, ctx->forward_input_or_allocate_output({0},    0, input_0.shape(), &out));
		THROWIF(out == 0, "Unsupported Binary for " << input_0.shape() << " and " << input_1.shape());

		vedaHandle(ctx)->binary(ptr<bool>(out), ptr<T>(input_0), ptr<T>(input_1), cnt(out), cnt(input_0), cnt(input_1), OP, sol_dtype<T>());
	}
};

//------------------------------------------------------------------------------
}

#include "__ns.h"
//------------------------------------------------------------------------------
void init_binary(void) {
	#define Binary(N, O) REG10_(N, "T",, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, ::tensorflow::Binary, O)
	Binary("Equal",			BinaryOp::EQ);
	Binary("Greater",		BinaryOp::GT);
	Binary("GreaterEqual",	BinaryOp::GE);
	Binary("Less",			BinaryOp::LT);
	Binary("LessEqual",		BinaryOp::LE);
	Binary("NotEqual",		BinaryOp::NE);
}

//------------------------------------------------------------------------------
#include "__ns.h"
