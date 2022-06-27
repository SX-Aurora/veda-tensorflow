# VEDA TensorFlow

VEDA TensorFlow is a library to add device support for the NEC SX-Aurora TSUBASA
into TensorFlow using the Pluggable Device API.

## Release Notes
<table>
<tr><th>Version</th><th>Comment</th></tr>

<tr><td>v4</td><td>
<ul>
	<li>Added TF v2.9.* support</li>
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