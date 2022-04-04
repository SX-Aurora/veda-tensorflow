#include <veda/tensorflow/api.h>

namespace tensorflow {
	template <>	const VEDATensors_handle_struct& OpKernelContext::eigen_device() const {
		return *veda::tensorflow::handle(this);
	}
}