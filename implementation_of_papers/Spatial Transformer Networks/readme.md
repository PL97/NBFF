paper link: [spatial Transformer Networks](https://arxiv.org/abs/1506.02025)


## idea overview

parametrize transformation matrix and sampling grid


## transformation
For example 2D affine transformation can be parametrized as a affine transofrmation matrix
$$
A=
\begin{bmatrix}
\theta_{11}, \theta_{12}, \theta_{13}\\
\theta_{21}, \theta_{22}, \theta_{23}\\
\end{bmatrix}
$$

Affine transformation: 
$$
\begin{pmatrix}
x'\\
y'
\end{pmatrix}
= A^T
\begin{pmatrix}
x\\
y
\end{pmatrix}
$$ 

The transformation can be more constrained if putting extra constrain on $\theta_{ij}$


## sampling
for a particular "pixel", the value is defined as below
$$
V_i^c=\sum_n^H \sum_m^W U_{n m}^c k\left(x_i^s-m ; \Phi_x\right) k\left(y_i^s-n ; \Phi_y\right) \forall i \in\left[1 \ldots H^{\prime} W^{\prime}\right] \forall c \in[1 \ldots C]
$$
where $k$ is a generic sampling kernel. The sampling kernel should be easy to calcuate the gradient/sub_gradient wrt $x$ and $y$, e.g. interger samping kernel

$$
V_i^c=\sum_n^H \sum_m^W U_{n m}^c \delta\left(\left\lfloor x_i^s+0.5\right\rfloor-m\right) \delta\left(\left\lfloor y_i^s+0.5\right\rfloor-n\right)
$$