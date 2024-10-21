# Efficient Kolmogorov - Arnold Network for nVidia Modulus & nVidia Modulus Sym

This repository contains an efficient implementation of Kolmogorov-Arnold Network (KAN). The original implementation of KAN is available [here](https://arxiv.org/pdf/2404.19756).

The ```KANLinear``` is based on the Efficient KAN by Blealtan Cao [@Blealtan](https://github.com/Blealtan/efficient-kan/commits?author=Blealtan) , ["An efficient pure-PyTorch implementation of Kolmogorov-Arnold Network (KAN)."](https://github.com/Blealtan/efficient-kan/tree/master).

It was needed a small change in the ```KANLinear``` to handle the batched tensor in modulus. 

**Addition of Juve 7th, 2024**: added Chebyshev and Jacobu KAN for nVidia Modulus, based on [#SynodicMonth](https://github.com/SynodicMonth) and [@SpaceLearner](https://github.com/SpaceLearner) GituHub repository [[1](https://github.com/SynodicMonth/ChebyKAN/), [2](https://github.com/SpaceLearner/JacobiKAN/tree/main)], adapted to work with Modulus. 

## Code
The code is contained in a single python file, ```kan.py```, in the ```src``` folder.

**Addition of June 7th, 2024**: two new files, ```chebyshev_kan.py``` and ```jacobi_kan.py``` offering `cKANArch` and  `jKANArch` modulus model class. 
in `examples` there are avilable also the modulus code for using (and testing) the two classes.

**Addition of October, 21th, 2024**: three new files, ```rbf_layer.py```. ```rbf_arch.py``` and ```rbf_kan.py``` are added. 
- ```rbf_layer.py```: introduces two layers, the `pytorch` Radial Basis Function Network Layer, and its adaptation to be used as RBF-KAN (also dubbed FastKAN, from [ArXiv:2405.06721](https://arxiv.org/abs/2405.06721); 
- ```rbf_arch.py``` introduces the RBF network Modulus Arc `RBFArch`. This fixes a small bug the stanrdard RBF implemention in `Nvidia Modulus SYM` has. 
- ```rbf_kan.py```: implements the `RBFKANLayer` to create the `RBFKANArch` for usange in `Modulus Sym`.

in `examples` there are avilable also the modulus exampple code for using (and testing) the two new architectures `RBFArch` and `RBFKANLayer`.

## Examples
There are two PDE examples in the ```examples``` folder, Heat Equation and Burgers Equation. 

