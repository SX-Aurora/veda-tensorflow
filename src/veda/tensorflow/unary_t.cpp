#include <veda/tensorflow/api.h>

using sol::runtime::native::tensorflow::ve::vedaHandle;

namespace tensorflow {
//------------------------------------------------------------------------------
template<typename T, int OP>
struct UnaryT : public OpKernel {
	explicit UnaryT(OpKernelConstruction* ctx) : OpKernel(ctx) {}

	void Compute(OpKernelContext* ctx) override {
		auto& input = ctx->input(0);
		Tensor* out = 0;
		OP_REQUIRES_OK(ctx, ctx->forward_input_or_allocate_output({0}, 0, input.shape(), &out));
		vedaHandle(ctx)->unaryT(ptr<T>(out), ptr<T>(input), cnt(out), cnt(input), OP, sol_dtype<T>());
	}
};

//------------------------------------------------------------------------------
}

#include "__ns.h"
//------------------------------------------------------------------------------
void init_unary_t(void) {
	#define UnaryT(N, O) REG10_(N, "T",, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, ::tensorflow::UnaryT, O)
	UnaryT("Sqrt",	UnaryOp::SQRT);
	UnaryT("Abs",	UnaryOp::ABS);
	UnaryT("Rsqrt",	UnaryOp::RSQRT);
	UnaryT("Sin",	UnaryOp::SIN);
	UnaryT("Cos",	UnaryOp::COS);
	UnaryT("Tan",	UnaryOp::TAN);
	UnaryT("Exp",	UnaryOp::EXP);
	UnaryT("Log",	UnaryOp::LOG);
}

//------------------------------------------------------------------------------
#include "__ns.h"
