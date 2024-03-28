# Self-Attention and Nadaraya-Watson Kernel Regression

Here, we show connections between the **Transformer** and the **Kernel Regression**. We show how the dot-product between queries $\mathbf{q}_i$ and keys $\mathbf{k}_i$ can be swapped out with miscellaneous kernel operations $\alpha(\cdot, \cdot)$, chief among them being the _Nadaraya-Watson kernel_ $K$. We also empirically show how Self-attention variants can successfully learn on sequential data like periodic and aperiodic functions.

> This is a class project for _MA4270: Data Modelling and Computation_ by Rishabh Anand (A0220603Y) and Ryan Chung Yi Sheng (A0219702J).
