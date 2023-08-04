# VEDA TensorFlow

VEDA TensorFlow is a library to add device support for the NEC SX-Aurora TSUBASA
into TensorFlow using the Pluggable Device API.

[![Github](https://img.shields.io/github/v/tag/sx-aurora/veda-tensorflow?display_name=tag&sort=semver)](https://github.com/sx-aurora/veda-tensorflow)
[![PyPI](https://img.shields.io/pypi/v/veda-tensorflow)](https://pypi.org/project/veda-tensorflow)
[![License](https://img.shields.io/pypi/l/veda-tensorflow)](https://pypi.org/project/veda-tensorflow)
![Python Versions](https://img.shields.io/pypi/pyversions/veda-tensorflow)
![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)
![Maintenance](https://img.shields.io/pypi/dm/veda-tensorflow)

## Release Notes
<table>
<tr><th>Version</th><th>Comment</th></tr>

<tr><td>v7</td><td>
<ul>
	<li>Added TF v2.13.* support</li>
	<li>Added TF v2.12.* support</li>
	<li>Fixed &lt;v2.10.* support</li>
</ul>
</td></tr>

<tr><td>v6</td><td>
<ul>
	<li>Added TF v2.11.* support</li>
	<li>Added TF v2.10.* support</li>
	<li>Upgraded to VEDA CPP API</li>
</ul>
</td></tr>

<tr><td>v5</td><td>
<ul>
	<li>Added TF v2.9.* support</li>
</ul>
</td></tr>

<tr><td>v4</td><td>
<ul>
	<li>Added BroadcastTo operation</li>
	<li>Increased <code>host_memory_allocate</code> alignment to be 64, as lower values keep failing in <code>isAligned()</code></li>
</ul>
</td></tr>

<tr><td>v3</td><td>
<ul>
	<li>Bugfixes for loss functions</li>
	<li>Added missing optimizers: SGD, Adadelta, Adagrad, Adam, and Adamax</li>
	<li>Fixed possible segfault in PluggableDevice <code>host_memory_allocate</code></li>
</ul>
</td></tr>

<tr><td>v2</td><td>
<ul>
	<li>Minor changes to enable TF v2.7.1 and v2.8.0</li>
	<li>Fixed vedaInit error checking to ignore if already initialized</li>
</ul>
</td></tr>

<tr><td>v1</td><td>
Initial Release
</td></tr>

</table>

## F.A.Q.
### I get the error message: "Internal: platform is already registered with name: "NEC_SX_AURORA"

This error is caused by the combination of RH-Python38 package and using a
VirtualEnv. Due to [improper checking for symlinks in
TensorFlow](https://github.com/tensorflow/tensorflow/issues/55497) the device
support library gets loaded and initialized twice causing this error message.

You can use the following workaround as long as the bug is not resolved in
TensorFlow.

```python
# BEGIN BUGFIX
import sys
import os

sys.path = list(set(os.path.realpath(p) for p in sys.path))

import site
getsitepackages = site.getsitepackages
def getsitepackages_(prefixes=None):
    return list(filter(lambda x: 'lib64' not in x, getsitepackages(prefixes)))
site.getsitepackages = getsitepackages_
# END BUGFIX

import tensorflow
...
```

### I get the error message "tensorflow.python.framework.errors_impl.InvalidArgumentError: 'visible_device_list' listed an invalid Device id '1' but visible device count is 1"
This is a known problem within TF due to [TF throws: "'visible_device_list'
listed an invalid Device id" when using non-GPU
PluggableDevices](https://github.com/tensorflow/tensorflow/issues/60895) when
using CUDA and VE devices at the same time. The VE devices get added to list of
GPUs, ultimately creating invalid devices indices.

Either you need to manually patch your TF installation (see the TF issue), or
use `VEDA_VISIBLE_DEVICES=100` or `CUDA_VISIBLE_DEVICES=` to disable either the
CUDA or VE devices.