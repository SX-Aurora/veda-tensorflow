#include <veda/tensorflow/api.h>

using sol::runtime::native::tensorflow::ve::vedaHandle;

namespace tensorflow {
//------------------------------------------------------------------------------
template<typename T, int OP>
struct UnaryTT : public OpKernel {
	explicit UnaryTT(OpKernelConstruction* ctx) : OpKernel(ctx) {}

	void Compute(OpKernelContext* ctx) override {
		auto& input_0 = ctx->input(0);

		Tensor* out = 0;
		switch(ctx->num_inputs()) {
			case 1: {
				OP_REQUIRES_OK(ctx, ctx->forward_input_or_allocate_output({0}, 0, input_0.shape(), &out));
				vedaHandle(ctx)->unaryTT(ptr<T>(out), ptr<T>(input_0), ptr<T>(input_0), cnt(out), cnt(input_0), cnt(input_0), OP, sol_dtype<T>());
			} break;

			case 2: {
				auto& input_1 = ctx->input(1);

				if(input_0.shape() == input_1.shape())	OP_REQUIRES_OK(ctx, ctx->forward_input_or_allocate_output({0, 1}, 0, input_0.shape(), &out));
				else if(input_0.shape().dims() == 0)	OP_REQUIRES_OK(ctx, ctx->forward_input_or_allocate_output({1},    0, input_1.shape(), &out));
				else if(input_1.shape().dims() == 0)	OP_REQUIRES_OK(ctx, ctx->forward_input_or_allocate_output({0},    0, input_0.shape(), &out));
				THROWIF(out == 0, "Unsupported UnaryTT (" << sol::runtime::UnaryOp(OP) << ") for " << input_0.shape() << " and " << input_1.shape());

				vedaHandle(ctx)->unaryTT(ptr<T>(out), ptr<T>(input_0), ptr<T>(input_1), cnt(out), cnt(input_0), cnt(input_1), OP, sol_dtype<T>());
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
	UnaryTT("Add",		UnaryOp::ADD);
	UnaryTT("AddV2",	UnaryOp::ADD);
	UnaryTT("Mul",		UnaryOp::MUL);
	UnaryTT("RealDiv",	UnaryOp::DIV);
	UnaryTT("Sub",		UnaryOp::SUB);
	UnaryTT("Square",	UnaryOp::MUL);
}

//------------------------------------------------------------------------------
#include "__ns.h"