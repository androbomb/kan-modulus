# Efficient Kolmogorov - Arnold Network for nVidia Modulus & nVidia Modulus Sym

This repository contains an efficient implementation of Kolmogorov-Arnold Network (KAN). The original implementation of KAN is available [here](https://arxiv.org/pdf/2404.19756).

The ```KANLinear``` is based on the Efficient KAN by Blealtan Cao [@Blealtan](https://github.com/Blealtan/efficient-kan/commits?author=Blealtan) , ["An efficient pure-PyTorch implementation of Kolmogorov-Arnold Network (KAN)."](https://github.com/Blealtan/efficient-kan/tree/master).

It was needed a small change in the ```KANLinear``` to handle the batched tensor in modulus. 

**Addition of June 7th, 2024**: added Chebyshev and Jacobi KAN for nVidia Modulus, based on [#SynodicMonth](https://github.com/SynodicMonth) and [@SpaceLearner](https://github.com/SpaceLearner) GituHub repository [[1](https://github.com/SynodicMonth/ChebyKAN/), [2](https://github.com/SpaceLearner/JacobiKAN/tree/main)], adapted to work with Modulus. 

## Code
The code is contained in a single python file, ```kan.py```, in the ```src``` folder.

**Addition of June 7th, 2024**: two new files, ```chebyshev_kan.py``` and ```jacobi_kan.py``` offering `cKANArch` and  `jKANArch` modulus model class. 
in `examples` there are avilable also the modulus code for using (and testing) the two classes.

**Addition of October, 21th, 2024**: three new files, ```rbf_layer.py```. ```rbf_arch.py``` and ```rbf_kan.py``` are added. 
- ```rbf_layer.py```: introduces two layers, the `pytorch` Radial Basis Function Network Layer, and its adaptation to be used as RBF-KAN (also dubbed FastKAN, from [ArXiv:2405.06721](https://arxiv.org/abs/2405.06721); 
- ```rbf_arch.py``` introduces the RBF network Modulus Arc `RBFArch`. This fixes a small bug the stanrdard RBF implemention in `Nvidia Modulus SYM` has. 
- ```rbf_kan.py```: implements the `RBFKANLayer` to create the `RBFKANArch` for usange in `Modulus Sym`.

in `examples` there are avilable also the modulus exampple code for using (and testing) the two new architectures `RBFArch` and `RBFKANLayer`.

**Changes of September, 24th, 2025**: All code refactored to work with PhysicsNemo 25:06. Added `Apptainer` `.def` file for reproducibility.


## Examples
There are two PDE examples in the ```examples``` folder, Heat Equation and Burgers Equation. 


## Using Apptainer 

### launch python script with singularity
To launch it, use
```bash
apptainer exec --nv container_physicsnemo.sif python your.py
```
The `--nv` is crucial to see the nVidia drivers. 

If we need to write on cache we need to bind the cache to a writable directory, via
```bash
apptainer exec --nv --bind /tmp/:/home/private/.cache container_physicsnemo.sif python your.py
```

### Creating the Environment using Singularity Apptainer

First step, is to define the `.def` singularity file:
```
bootstrap: docker
from: nvcr.io/nvidia/physicsnemo/physicsnemo:25.06
stage: build

%environment
    export LC_ALL=C
    export HYDRA_FULL_ERROR=1
    export CUDA_LAUNCH_BLOCKING=1
```
If you need to customise your instance, like installing packages via `pip`, use the `%post` keyword, e.g.:
```
%post
    pip3 install ipykernel plotly torchmetrics torchvision pydicom albumentations pyyaml SimpleITK einops
```

After that, we need to build the `.sif` container, as
```bash
apptainer build container_physicsnemo.sif container_physicsnemo.def 
```
We can test is by running it interatively
```bash
apptainer run container_physicsnemo.sif
```

Finally, we can use  `.sif` container to create an ipykernel; to do so, we need to create the folder 
```bash
mkdir -p ~/.local/share/jupyter/kernels/physicsnemo2506
```
and, within it, create the file `kernel.json`
```json
{
    "display_name": "PhysicsNemo2506",
    "argv": [
        "/usr/bin/apptainer",
        "run",
        "--nv",
        "--bind",
        "/home",
        "/path/to/container_physicsnemo.sif",
        "python3",
        "-m",
        "ipykernel_launcher",
        "-f",
        "{connection_file}"
    ],
    "language": "python",
    "metadata": {
        "debugger": true
    },
    "env": {
        "LD_LIBRARY_PATH": ":/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/.singularity.d/libs:/usr/local/lib/python3.12/dist-packages/nvidia/cudnn"
    }
}
```
Notice that we added in the `env` section the `LD_LIBRARY_PATH` env var, adding to the original one the path to cudnn as `/usr/local/lib/python3.12/dist-packages/nvidia/cudnn`
For more details, see [the AI-INFN platform guide](https://ai-infn.baltig-pages.infn.it/wp-1/docs/environments/apptainer/#make-gpus-available-to-the-kernel).
