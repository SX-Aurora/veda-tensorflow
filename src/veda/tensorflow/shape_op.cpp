#include <veda/tensorflow/api.h>

#include "__ns.h"
//------------------------------------------------------------------------------
void init_shape_op(void) {
	using namespace ::tensorflow;
	REGISTER_KERNEL_BUILDER(Name("Shape").Device(DEVICE_VE).HostMemory("output").TypeConstraint<int32>("out_type").TypeConstraint("T", VE_TYPES),	ShapeOp<int32>);
	REGISTER_KERNEL_BUILDER(Name("Shape").Device(DEVICE_VE).HostMemory("output").TypeConstraint<int64>("out_type").TypeConstraint("T", VE_TYPES),	ShapeOp<int64>);
	REGISTER_KERNEL_BUILDER(Name("ShapeN").Device(DEVICE_VE).HostMemory("output").TypeConstraint<int32>("out_type").TypeConstraint("T", VE_TYPES),	ShapeNOp<int32>);
	REGISTER_KERNEL_BUILDER(Name("ShapeN").Device(DEVICE_VE).HostMemory("output").TypeConstraint<int64>("out_type").TypeConstraint("T", VE_TYPES),	ShapeNOp<int64>);
	REGISTER_KERNEL_BUILDER(Name("Size").Device(DEVICE_VE).HostMemory("output").TypeConstraint<int32>("out_type").TypeConstraint("T", VE_TYPES),	SizeOp<int32>);
	REGISTER_KERNEL_BUILDER(Name("Size").Device(DEVICE_VE).HostMemory("output").TypeConstraint<int64>("out_type").TypeConstraint("T", VE_TYPES),	SizeOp<int64>);
	REGISTER_KERNEL_BUILDER(Name("VariableShape").Device(DEVICE_VE).TypeConstraint<int32>("out_type").HostMemory("output").HostMemory("input"),		VariableShapeOp<int32>);
	REGISTER_KERNEL_BUILDER(Name("VariableShape").Device(DEVICE_VE).TypeConstraint<int64>("out_type").HostMemory("output").HostMemory("input"),		VariableShapeOp<int64>);
	REGISTER_KERNEL_BUILDER(Name("_VarHandlesOp").Device(DEVICE_VE).HostMemory("resources"),														ResourceHandlesOp<Var>);
	REGISTER_KERNEL_BUILDER(Name("Rank").Device(DEVICE_VE).HostMemory("output").TypeConstraint("T", VE_TYPES),										RankOp);
	REGISTER_KERNEL_BUILDER(Name("Identity").Device(DEVICE_VE).TypeConstraint("T", VE_TYPES),														IdentityOp);
}

//------------------------------------------------------------------------------
#include "__ns.h"
