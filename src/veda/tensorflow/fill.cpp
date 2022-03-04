#include <veda/tensorflow/api.h>

using sol::runtime::native::tensorflow::ve::vedaHandle;

namespace tensorflow {
//------------------------------------------------------------------------------
template<typename T, typename I>
struct Fill : public OpKernel {
    explicit Fill(OpKernelConstruction* ctx) : OpKernel(ctx) {}

    void Compute(OpKernelContext* ctx) override {
		auto& input_0 = ctx->input(0);
		auto& input_1 = ctx->input(1);

		auto dims = input_0.flat<I>();
		TensorShape shape;
		OP_REQUIRES_OK(ctx, TensorShapeUtils::MakeShape(reinterpret_cast<const I*>(dims.data()), dims.size(), &shape));

		Tensor* out = 0;
		OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shape, &out));

		vedaHandle(ctx)->copy(ptr<T>(out), ptr<T>(input_1), cnt(out), cnt(input_1), sol_dtype<T>());
    }
};

//------------------------------------------------------------------------------
}

#include "__ns.h"
//------------------------------------------------------------------------------
void init_fill(void) {
	#define Fill(N,  I)	REG10_(N, "T", .TypeConstraint<I>("index_type").HostMemory("dims"), uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, ::tensorflow::Fill, I)
	Fill("Fill", int32_t);
	Fill("Fill", int64_t);
}

//------------------------------------------------------------------------------
#include "__ns.h"
