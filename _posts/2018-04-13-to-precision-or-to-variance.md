---
layout: post
title: "To precision or to variance?"
author: "Scott Roy"
categories:
tags: [conditional distribution of multivariate normal, covariance matrix, precision matrix, marginal distribution of multivariate normal]
image: normal.png
---

Multivariate normal distributions are really nice.  (Invertible) affine transformations, marginalization, and conditioning all preserve a multivariate normal distribution.

In this post, I want to discuss marginalization and conditioning.  In particular, I want to point out that computing the marginal distribution is easy when we parametrize with the covariance matrix, and computing the conditional distribution is easy when we parametrize with the precision matrix.

Suppose $$(X, Y)$$ is distributed multivariate normal

$$
\begin{bmatrix} X \\ Y \end{bmatrix} \sim N \left( \begin{bmatrix} \mu_x \\ \mu_y \end{bmatrix}, \begin{bmatrix}  \Sigma_{xx} & \Sigma_{xy} \\ \Sigma_{yx} & \Sigma_{yy} \end{bmatrix} \right).
$$

The marginal distribution of $$X$$ is really easy to compute:

$$
X \sim N \left( \mu_x, \Sigma_{xx} \right).
$$

The conditional distribution $$X \vert Y$$ is again multivariate normal, but with mean and covariance given by

$$
\begin{aligned}
\mu_{x\vert y} &= \mu_x + \Sigma_{xy} \Sigma_{yy}^{-1} (Y - \mu_y) \\
\Sigma_{x\vert y} &= \Sigma_{xx \cdot y} := \Sigma_{xx} - \Sigma_{xy} \Sigma_{yy}^{-1} \Sigma_{yx}.
\end{aligned}
$$

What do these formulas look like when we parametrize by precision matrix?  (The precision matrix $$K$$ is the inverse of the covariance matrix: $$K = \Sigma^{-1}$$.)

If we write

$$
K = \begin{bmatrix} K_{xx} & K_{xy} \\ K_{yx} & K_{yy}, \end{bmatrix}
$$

then we have the following relations:

$$
\begin{aligned}
K_{xx} &= \Sigma_{xx \cdot y}^{-1} \\
K_{yy} &= \Sigma_{yy \cdot x}^{-1} := (\Sigma_{yy} - \Sigma_{yx} \Sigma_{xx}^{-1} \Sigma_{xy})^{-1} \\
K_{xy} &= -\Sigma_{xx}^{-1} \Sigma_{xy} K_{yy} \\
K_{yx} &= -\Sigma_{yy}^{-1} \Sigma_{yx} K_{xx}.
\end{aligned}
$$

In terms of $$K$$, the marginal distribution of $$X$$ has precision matrix $$K_{xx \cdot y}$$, and the conditional distribution $$X \vert Y$$ has precision matrix $$K_{xx}$$.  Notice the duality!  The marginal distribution $$X$$ has covariance $$\Sigma_{xx}$$ and precision $$K_{xx \cdot y}$$, and the conditional distribution $$X \vert  Y$$ has covariance $$\Sigma_{xx \cdot y}$$ and precision $$K_{xx}$$.

Choosing the parametrization (covariance or precision) depends on whether your application requires computing marginal or conditional distributions.  In Gaussian Markov random fields, for example, conditional distributions are important, so using a precision parametrization makes life easier.