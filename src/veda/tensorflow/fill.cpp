#include <veda/tensorflow/api.h>

namespace tensorflow {
//------------------------------------------------------------------------------
template<typename T, typename I>
struct Fill final : public OpKernel {
    explicit Fill(OpKernelConstruction* ctx) : OpKernel(ctx) {}

    void Compute(OpKernelContext* ctx) override {
		using namespace veda::tensorflow; 

		auto& input_0	= ctx->input(0);
		auto& input_1	= ctx->input(1);
		auto dims		= input_0.flat<I>();
		TensorShape shape;
		OP_REQUIRES_OK(ctx, TensorShapeUtils::MakeShape(reinterpret_cast<const I*>(dims.data()), dims.size(), &shape));

		Tensor* out = 0;
		OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shape, &out));

		auto d_out		= tf2veda<T>(out);
		auto d_input_1	= tf2veda<T>(input_1);

		CVEDA(veda_tensors_copy(handle(ctx), &d_out, &d_input_1));
    }
};

//------------------------------------------------------------------------------
}

#include "__ns.h"
//------------------------------------------------------------------------------
void init_fill(void) {
	#define Fill(N, I)	REG10_(N, "T", .HostMemory("dims").TypeConstraint<I>("index_type"), ::tensorflow::Fill, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, I)
	Fill("Fill", int32_t);
	Fill("Fill", int64_t);
}

//------------------------------------------------------------------------------
#include "__ns.h"
