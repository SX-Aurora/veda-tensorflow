#include <veda/tensorflow/api.h>

namespace tensorflow {
//------------------------------------------------------------------------------
template<typename T, typename X>
inline T cast(const X x) {
	return *(const T*)&x;
}

//------------------------------------------------------------------------------
template<typename T, typename I>
struct Fill final : public OpKernel {
    explicit Fill(OpKernelConstruction* ctx) : OpKernel(ctx) {}

    void Compute(OpKernelContext* ctx) override {
		using namespace veda::tensorflow;
		Guard __guard__(device(ctx));

		ASSERT(ctx->input_memory_type(0) == HOST_MEMORY);
		ASSERT(ctx->input_memory_type(1) == HOST_MEMORY);

		auto& input_0	= ctx->input(0);
		auto& input_1	= ctx->input(1);
		auto dims		= input_0.flat<I>();

		// Bugfix for wrong tensor placement since v2.10.0, Issue #742
		I* dims_ptr = 0;
	#if TF_MINOR_VERSION > 9
		if(input_0.NumElements() && input_0.GetMemoryType() == AllocatorMemoryType::kDevice) {
			dims_ptr = new I[dims.size()];
			CVEDA(vedaMemcpyDtoH(dims_ptr, (VEDAdeviceptr)dims.data(), dims.size() * sizeof(I)));
		}
	#endif

		TensorShape shape;
		OP_REQUIRES_OK(ctx, TensorShapeUtils::MakeShape(reinterpret_cast<const I*>(dims_ptr ? dims_ptr : dims.data()), dims.size(), &shape));
		if(dims_ptr)
			delete[] dims_ptr;

		Tensor* out = 0;
		OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shape, &out));

	#if TF_MINOR_VERSION > 9
		// Bugfix for wrong tensor placement since v2.10.0, Issue #742
		if(input_1.GetMemoryType() == AllocatorMemoryType::kDevice) {
			auto d_out		= tf2veda<T>(out);
			auto d_input_1	= tf2veda<T>(input_1);
			CVEDA(veda_tensors_copy(handle(ctx), &d_out, &d_input_1));
		} else {
	#endif
			auto d_out = (VEDAdeviceptr)out->flat<T>().data();
			auto value = *input_1.flat<T>().data();

			vedaMemset<T>(d_out, value, out->NumElements());
	#if TF_MINOR_VERSION > 9
		}
	#endif
    }
};

//------------------------------------------------------------------------------
}

#include "__ns.h"
//------------------------------------------------------------------------------
void init_fill(void) {
	#define Fill(N, I)	REG10_(N, "T", .HostMemory("dims").HostMemory("value").TypeConstraint<I>("index_type"), ::tensorflow::Fill, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, I)
	Fill("Fill", int32_t);
	Fill("Fill", int64_t);
}

//------------------------------------------------------------------------------
#include "__ns.h"
