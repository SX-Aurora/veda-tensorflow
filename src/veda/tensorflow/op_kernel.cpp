#include <veda/tensorflow/api.h>

namespace tensorflow {
	template <>	const sol::runtime::device::ve::Handle& OpKernelContext::eigen_device() const {
		return *sol::runtime::native::tensorflow::ve::vedaHandle(this);
	}
}