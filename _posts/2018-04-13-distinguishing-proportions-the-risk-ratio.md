---
layout: post
title: "Distinguishing proportions: the risk ratio"
author: "Scott Roy"
categories:
tags: [confidence interval, difference in proportions, propagation of error, ratio of proportions, risk ratio]
image: payperclick.jpg
---

Suppose I conduct an experiment to determine whether to use font A or font B in an online ad.  After running the experiment, I find that there is a 1% chance that a user clicks on the font A ad, and a 0.8% chance that the user clicks on the font B ad.

We can can compare these click rates on an absolute scale (font A increases the click rate by 0.2%) or compare on a relative scale (the click rate for font A is 1.25 times higher).  The relative comparison (the probability of clicking on a font A ad over the probability of clicking on a font B ad) is called the risk ratio in some fields.  To determine statistical significance, we can see if the difference is significantly different than 0 or if the ratio is significantly different than 1.  I'd highly recommend comparing the difference to 0 (in fact, we'll handle the ratio by taking the log and turning it into a difference), but the ratio is nice for reporting purposes and so I'll discuss computing a confidence interval for it.

Suppose I show $$n$$ impressions of the ad with font A, and let $$X_1, \ldots, X_n$$ be the outcomes ($$X_1 = 1$$ means the user clicked on the ad and $$X_1=0$$ means the user did not click on the ad).  Similarly, let $$Y_1, \ldots, Y_m$$ be the outcomes of the $$m$$ impressions of the ad with font B I show.

The estimated probability of clicking on the font A ad is $$\hat{p}_A = \sum X_i / n$$.  By the central limit theorem, $$\hat{p}_A$$ is approximately distributed $$N(p_A,\ p_A(1-p_A) / n)$$, where $$p_A$$ is the true probability.  Similarly, $$\hat{p}_B$$ is approximately distributed $$N(p_B,\ p_B(1-p_B) / m)$$.

Thus the difference is approximately distributed

$$
\hat{p}_A - \hat{p}_B\ \dot\sim\ N(p_A - p_B,\ p_A(1-p_A) / n + p_B(1-p_B) / m).
$$

Under the null hypothesis, we assume $$p_A = p_B$$.  If we let $$p$$ denote this common value, we have

$$
\frac{\hat{p}_A - \hat{p}_B}{\sqrt{p(1-p) \left(\frac{1}{n} + \frac{1}{m}\right)}} \ \dot\sim\ N(0, 1).
$$

Slutsky's theorem lets us replace $$p$$ with the pooled estimate $$\hat{p} = (\sum X_i + \sum Y_i) / (n + m)$$.  In summary, to test if the difference $$\hat{p}_A - \hat{p}_B$$ is significantly different than $$0$$, we compute the $$Z$$-score

$$
\frac{\hat{p}_A - \hat{p}_B}{\sqrt{\hat{p}(1-\hat{p}) \left(\frac{1}{n} + \frac{1}{m}\right)}}
$$

and see how extreme it is for a draw from standard normal.

As an aside, an alternative statistic to use to test significance is

$$
Z = \frac{\hat{p}_A - \hat{p}_B}{\sqrt{\hat{p}_A(1-\hat{p}_A) / n + \hat{p}_B(1-\hat{p}_B) / m}}.
$$

One nice feature of this statistic is that the test for being significant at level $$\alpha$$ using it is equivalent to the $$(1-\alpha)$$ confidence interval for the difference $$\hat{p}_A - \hat{p}_B$$ containing 0.

How can we find a confidence interval for the risk ratio?  The distribution of the ratio of two independent normals is complicated (unless both normals have zero mean, in which case the ratio is distributed Cauchy).  The trick is to turn the ratio into a difference by taking a log, use propagation of error, and then transform back.

Propagation of error approximately describes how the mean and variance of a random variable change under a transformation.  More precisely, $$\textbf{E}(f(X)) \approx f(\textbf{E}(X))$$ and $$\textbf{Var}(f(X)) \approx f'(\textbf{E}(X))^2\ \textbf{Var}(X)$$.  Propagation of error is often used in conjunction with the central limit theorem: if $$\sqrt{n}(X_n - \mu)$$ converges in distribution to $$N(0, \sigma^2)$$, then $$\sqrt{n}(f(X_n) - f(\mu))$$ converges to $$N(0, f'(\mu)^2 \sigma^2)$$.

By propagation of error, $$\log \hat{p}_A$$ is approximately distributed $$N(\log(p_A), \frac{1-p_A}{n p_A})$$.  The standard error for the difference in log hat probabilities is thus

$$
SE = \sqrt{\frac{1-\hat{p}_A}{n \hat{p}_A} +\frac{1-\hat{p}_B}{m\hat{p}_B}},
$$

and the $$95\%$$ confidence interval for the difference is $$\pm 1.96\ SE$$.  We exponentiate to get the confidence interval for the risk ratio:

$$
\frac{\hat{p}_A}{\hat{p}_B}\ e^{\pm 1.96\ SE}.
$$

Note the confidence interval is asymmetric!