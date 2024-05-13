# Efficient Kolmogorov - Arnold Network for nVidia Modulus & nVidia Modulus Sym

This repository contains an efficient implementation of Kolmogorov-Arnold Network (KAN). The original implementation of KAN is available [here](https://arxiv.org/pdf/2404.19756).

The ```KANLinear``` is based on the Efficient KAN by Blealtan Cao [@Blealtan](https://github.com/Blealtan/efficient-kan/commits?author=Blealtan) , ["An efficient pure-PyTorch implementation of Kolmogorov-Arnold Network (KAN)."](https://github.com/Blealtan/efficient-kan/tree/master).

It was needed a small change in the ```KANLinear``` to handle the batched tensor in modulus. 

## Code
The code is contained in a single python file, ```kan.py```, in the ```src``` folder.

## Examples
There are two examples in the ```examples``` folder. 

