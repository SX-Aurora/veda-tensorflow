#include <veda/tensorflow/api.h>

namespace tensorflow {
	template <>	const VEDevice& OpKernelContext::eigen_device() const {
		return *veda::tensorflow::handle(this);
	}
}