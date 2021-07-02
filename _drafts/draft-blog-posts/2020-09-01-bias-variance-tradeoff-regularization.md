---
layout: post
title: "The bias-variance tradeoff and regularization"
author: "Scott Roy"
categories:
tags:  
image: 
---

The key ideas of the bias-variance tradeoff and regularization are nicely illustrated with a simple example about estimating a mean.

Suppose we observe $$n$$ i.i.d. samples $$x_1, x_2, \ldots, x_n$$ from some distribution and we wish to estimate its mean $$\mu$$.  The obvious estimate is

$$\hat{\mu}_n = \frac{1}{n} \sum_{i=1}^n x_i,$$

the sample mean.  Since the sample mean is unbiased, its mean square error equals its variance:

$$\textbf{MSE}(\hat{\mu}) = \textbf{Var}(\hat{\mu}) = \frac{\sigma^2}{n} (\sigma^2 := \textbf{Var}(x_i)).$$

The most obvious way to reduce variance is to get more data (this is also true with complicated models such as neural networks).  That said, we *can* reduce the variance without getting more data by shrinking the sample mean by a factor $$t$$ (with $$0 \leq t < 1$$):

$$\hat{\mu}^{S}_t := t \hat{\mu}.$$


Notice that $$\textbf{Var}(\hat{\mu}^{S}_t) = t^2 \textbf{Var}(\hat{\mu})$$ so shrinking the sample mean does reduce the variance, but of course it also introduces some bias:

$$\textbf{Bias}(\hat{\mu}^{S}_t) = \text{E}(\hat{\mu}^{S}_t) - \mu = -(1-t) \mu.$$

The bias and variance are traded off in the mean square error:

$$\begin{aligned} 
\textbf{MSE}(\hat{\mu}^{S}_t) &= \textbf{Var}(\hat{\mu}^{S}_t) + \textbf{Bias}(\hat{\mu}^{S}_t)^2 \\
&= \frac{t^2 \sigma^2}{n} + (1-t)^2 \mu^2.
\end{aligned}$$

By minimizing over $$t$$, we see the optimal amount to shrink the sample mean is

$$t^* = \frac{\mu^2}{\mu^2 + \frac{\sigma^2}{n}}.$$

Notice that $$t^* \to 1$$ as $$n \to \infty$$, i.e., with lots of data, we don't shrink the sample mean.  We also do not shrink the sample mean if the true mean is large ($$t^* \to 1$$ as $$\mu^2 \to \infty$$).  The shrinkage estimate $$\hat{\mu}^{S}_t$$ can be interpreted as a posterior mode with zero-mean Gaussian prior.  It therefore makes sense that if the true mean is far from zero, we should not shrink.  We typically shrink toward the origin (zero), but we can also shrink toward any prespecified point.  Doing so would also decrease the variance, but increase the bias.

## Shrinkage and regularization

In machine learning, we do not typically reduce variance by directly shrinking an estimate, but rather indirectly shrink it with regularization.  Regularization is defined for an estimate expressed as the minimizer of a loss function.  We can write the sample mean as the minimizer of the square loss

$$\hat{\mu} = \text{argmin}_\mu\ \frac{1}{n} \sum_{i=1}^n (x_i - \mu)^2,$$

and define a regularized sample mean as the minimizer of the regularized square loss

$$\hat{\mu}^R_{\lambda} = \text{argmin}_\mu\ \frac{1}{n} \sum_{i=1}^n (x_i - \mu)^2 + \lambda \mu^2.$$

Explicitly we can write $$\hat{\mu}^R_{\lambda} = \left(\frac{1}{1+\lambda}\right) \hat{\mu}$$ and so the regularized sample mean is a shrinkage of the sample mean, albeit parametrized by $$\lambda \in [0, \infty]$$ instead of $$t \in [0, 1]$$.

We already know the optimal amount to shrink is

$$t^* = \frac{\mu^2}{\mu^2 + \frac{\sigma^2}{n}} = \frac{1}{1 + \frac{\sigma^2}{n \mu^2}}.$$

It follows that $$\lambda^* = \frac{\sigma^2}{n \mu^2}$$ is the optimal amount to regularize.  We again point out that as we get more data ($$n \to \infty$$), we should regularize/shrink less ($$\lambda^* \to 0$$).

___
#### Side blurb
If we don't normalize by $$n$$ in the square loss and instead define $$\hat{\mu}^R_{\lambda} = \text{argmin}_\mu\ \sum_{i=1}^n (x_i - \mu)^2 + \lambda \mu^2,$$ the optimal amount of regularization $$\lambda^* = \frac{\sigma^2}{\mu^2}$$ is independent of $$n$$.  That said, we typically do normalize by $$n$$.  Doing so not only alleviates numerical issues caused by the sum getting too large, but also means the optimal amount of regularization $$\frac{\sigma^2}{n\mu^2}$$ is likely in $$[0,1]$$ for moderate to large sample sizes.  This is desirable since we usually search for $$\lambda$$ using a uniform grid on the log scale over a small range {1, 0.1, 0.01, 0.001, etc}.  

---

# MSE between $$\hat{Y}$$ and $$Y$$

Until now we've measured MSE between the estimate $$\hat{\mu}$$ and the parameter $$\mu$$.  In prediction tasks, though, we are interested in MSE between the prediction $$\hat{y}$$ and the ground truth $$y$$.  Viewing $$\hat{y} = \hat{\mu}$$ as a prediction (e.g., from a linear model with only an intercept term), we analyze the MSE of the shrunk estimate $$t \hat{y}$$:

$$\begin{aligned}
\textbf{E}((t\hat{\mu} - y)^2)) &= \textbf{E}(t^2 \hat{\mu}^2 - 2 t \hat{\mu} y + y^2) \\
&= \left( \frac{\sigma^2}{n} + \mu^2 \right) t^2 - 2 \mu^2 t + \sigma^2 + \mu^2.
\end{aligned}$$  

(We used the assumption that $$y$$ is independent of $$\hat{\mu}$$, the property that $$\textbf{E}(XY) = \textbf{E}(X) \textbf{E}(Y)$$ for independent random variables, and the relation $$\textbf{Var}(X) = \textbf{E}(X^2) - \textbf{E}(X)^2$$.)

The optimal shrinkage factor is still $$t^* = \frac{\mu^2}{\mu^2 + \frac{\sigma^2}{n}}$$, but the mean error no longer tends to zero as $$n$$ becomes large.

In general the MSE of a model decomposes into a bias term, a variance term, and an irreducible variance term.  Let $$f(x) = \textbf{E}(y \vert x)$$ be the conditional mean and $$\sigma^2_x = \textbf{Var}(y \vert x)$$ be the variance about the conditional mean.  In ML, we build a model $$\hat{f}_D$$ to estimate $$f$$ based on training data $$D = (x_1, y_1), \ldots, (x_n, y_n)$$.  The mean square error decomposes into three parts:

$$\begin{aligned}
\textbf{E}((\hat{f}_D(x) - y)^2 \vert x) &= \textbf{E}((\hat{f}_D(x) - f(x) + f(x) - y)^2 \vert x) \\
&= \textbf{E}((\hat{f}_D(x) - f(x))^2 \vert x) + \textbf{E}((f(x) - y)^2 \vert x) \\
&= \textbf{E}((\hat{f}_D(x) - \textbf{E}(\hat{f}_D(x) \vert x) + \textbf{E}(\hat{f}_D(x) \vert x) - f(x))^2 \vert x) + \sigma^2_x  \\
&= \textbf{E}((\hat{f}_D(x) - \textbf{E}(\hat{f}_D(x) \vert x))^2 \vert x) + (\textbf{E}(\hat{f}_D(x) \vert x) - f(x))^2 + \sigma^2_x \\
&= \textbf{Var}(\hat{f_D}(x) \vert x) + \textbf{Bias}(\hat{f_D}(x) \vert x) + \sigma^2_x. 
\end{aligned}$$

(The variance and bias are computed over datasets $$D$$.)

There is not always a tradeoff between bias and variance.  Indeed, there are actions that can reduce or increase variance without affecting bias and vice versa.  For example, we can often reduce variance by simply getting more data, which does not increase bias.  We can similarly reduce variance (without affecting bias) by averaging independently trained models, as is done in random forests.  Adding irrelevant predictors to a linear regression will not hurt bias, but can hurt variance and decrease MSE.  If we have lots and lots of data, we can reduce bias (without introducing variance) by using a more complicated model or reducing regularization.

The irreducible variance $$\sigma^2_x$$ has to do with how stochastic $$y$$ is conditional on $$x$$.  We can reduce it by getting more predictive features.


# Ridge regression

Ridge regression (L2-regularized linear regression) shrinks the coefficient vector along the principal component directions of the data.  As such, we can view ridge regression as a smoothed feature selection technique.  Consider data $$(x_1, y_1), (x_2, y_2), \ldots, (x_n, y_n)$$ with outcome $$y_i \in \mathbb{R}$$ and covariates $$x_i \in \mathbb{R}^p$$.  The regularized coefficient vector in ridge regression is the minimizer of the regularized loss

$$\hat{\beta}_{\lambda} = \text{argmin}_\beta\ \frac{1}{n} \sum_{i=1}^n (\beta^T x_i - y_i)^2 + \lambda \vert \beta \vert^2.$$

Explicitly we have

$$\hat{\beta}_{\lambda} = (X^T X + n \lambda I)^{-1} X^T y,$$

where $$X$$ is an $$n \times p$$ matrix with rows $$x_i$$ and $$y$$ is an $$n \times 1$$ vector with components $$y_i$$.  This reduces to the well-known least squares solution $$\hat{\beta} = (X^T X)^{-1} X^T y$$ when $$\lambda = 0$$.

The coefficient vector is better analyzed in the "PCA basis."  Suppose $$n \geq p$$, the columns of $$X$$ are indpendent, and the the columns of $$X$$ and the vector $$y$$ are centered.  Let $$X = U \Sigma V^T$$ be the SVD of $$X$$ with  $$U \in \mathbb{R}^{n \times p}$$, $$\Sigma \in \mathbb{R}^{p \times p}$$, and $$V \in \mathbb{R}^{p \times p}$$.  It follows that

$$\frac{1}{n} X^T X = V \left( \frac{1}{n} \Sigma^2 \right) V^T$$

is the eigendecomposition of the sample covariance matrix and the columns of $$V$$ define the orthogonal PCA basis.

We can write

$$\begin{aligned}
\hat{\beta}_{\lambda} &= (X^T X + n \lambda I)^{-1} X^T y \\
&= V \textbf{Diag}\left( \frac{\Sigma_{jj}}{\Sigma_{jj}^2 + n\lambda} \right) U^T y \\
&= V \Sigma^{-1} \textbf{Diag} \left( \frac{\Sigma_{jj}^2}{\Sigma_{jj}^2 + n\lambda} \right) U^T y 
\end{aligned}$$

It follows that ridge regression shrinks the coefficent corresponding to $$j$$th PCA direction by

$$\frac{\Sigma_{jj}^2}{\Sigma_{jj}^2 + n \lambda} = \frac{\frac{\Sigma_{jj}^2}{n}}{\frac{\Sigma_{jj}^2}{n} + \lambda} = \frac{\text{(variance in $j$th PCA direction)}}{\text{(variance in $j$th PCA direction)} + \lambda}.$$

Directions with small variance are shrunk more than directions with large variance.

Since regularization acts by shrinking each PCA direction by some amount, we might ask if we can achieve lower MSE by directly shrinking the PCA directions.  Let $$\alpha$$ be the coefficients of $$\beta$$ with respect to the PCA basis given by the columns of $$V$$, i.e., $$\beta = V \alpha$$.  Similarly let $$\hat{\beta} = V \hat{\alpha}$$.  Let $$T = \textbf{Diag}(t_1, \ldots, t_p)$$ be a matrix of shrinkage factors.  The MSE of the estimator $$T \hat{\alpha}$$ is:

$$\begin{aligned}
\textbf{MSE}(T \hat{\alpha}) &= \sum_{j=1} \textbf{E} \left( (t_j \hat{\alpha} - t_j \alpha_j)^2 \right) + (1-t_j)^2 \alpha_j^2 \\
&= t_j^2 \text{Var}(\hat{\alpha}_j) + (1-t_j)^2 \alpha_j^2
\end{aligned}$$

The optimal shrinkage is $$t_j^* = \frac{\alpha_j^2}{\alpha_j^2 + \textbf{Var}(\hat{\alpha}_j)}$$.


To compare with the shrinkage from regularization, we compute

$$\begin{aligned}
\textbf{Var}(\hat{\alpha}) &= \textbf{Var}(V^T \hat{\beta}) \\
&= V^T \textbf{Var}(\hat{\beta}) V \\
&= \sigma^2 V^T (X^T X)^{-1} V \\
&= \sigma^2 V^T V \Sigma^{-2} V^T V \\
&= \sigma^2 \Sigma^{-2},
\end{aligned}$$

which implies $$\textbf{Var}(\hat{\alpha}_j) = \frac{\sigma^2}{\Sigma_{jj}^2}$$.  It follows that we can achieve smaller mean square error by directly shrinking the coefficient estimate (regularization would require $$\lambda = \frac{\sigma^2}{n \alpha_j^2}$$, which is not possible unless $$\alpha_j$$ are constant).  That said, we could achieve the same result if we regularize by $$\sum_{j=1}^p \lambda_j \beta_j^2$$ instead.

## References
* *The elements of statistical learning* by Trevor Hastie, Robert Tibshirani, and Jerome Friedman
