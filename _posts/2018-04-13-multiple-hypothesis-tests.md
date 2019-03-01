---
layout: post
title: "Multiple Hypothesis Tests"
author: "Scott Roy"
categories:
tags: [Bonferroni correction, false discovery rate, multiple hypothesis tests, p-value, q-value, type I error, type II error]
image: plant.jpg
---

In hypothesis testing, rejecting the null hypothesis typically means we found something interesting --- a drug performs better than the placebo, there is a difference between two fertilizers, the new textbook helps students achieve better test scores than the current one, etc. Rejecting the null hypothesis when it is true amounts to a false discovery.  On the other hand, accepting the null hypothesis when it is false means you failed to find an interesting result.  Here are some common synonyms:

* Asserting something interesting that's not = False Positive = False Discovery = Type I Error
* Failing to detect something interesting = False Negative = Miss = Type II Error

Performing multiple hypothesis tests inflates false positive rates. Imagine 500 science labs around the world try to determine if playing music to pea plants makes them grow faster.  Each lab uses the same experiment design and performs a t-test with significance level $$ \alpha = 0.01$$.  Assuming the null that music has no effect on pea growth, each lab will reject the null with probability $$ 0.01$$.  Although each individual lab is unlikely to reject the null, there is a greater than $$ 99\%$$ chance that some lab will reject the null:

$$
\begin{aligned}
\textbf{P}(\text{at least one false positive}) &= 1 - \textbf{P}(\text{500 true negatives}) \\
&= 1 - (0.99)^{500} \\
&= 0.993.
\end{aligned}
$$

There are a couple ways to address this inflation of false positives:

* Use a Bonferroni correction to control the family-wise error rate (FWER)
* Control the false-discovery rate (FDR)

### Bonferroni correction

For $$m$$ hypothesis tests, the family-wise error rate is the false positive rate for the family of tests: the probability that one of the $$m$$ tests rejects the null when it is true.  To control the FWER at $$\alpha$$, we set the significance level for each individual test at $$ \alpha / m$$.  Alternatively, we can use significance level $$\alpha$$ for each individual test, but form Bonferroni-adjusted $$ p$$-values by multiplying each $$p$$-value by $$m$$.  The Bonferroni correction only works for a moderate number of tests --- if $$m$$ is too large, each individual test will be too cautious and never reject the null.  You'll never find anything interesting!

### False-discovery rate

A more modern approach for multiple hypothesis testing is to control the false-discovery rate .  We accept that some hypothesis tests will result in false positives, but control the fraction of false-positives to all positives. The test works as follows.

Choose a false-discovery rate $$\alpha$$, and compute the q-values $$q_i = (i/m) \alpha$$.
Look at the ordered $$p$$-values $$ p_{(1)} < p_{(2)} < \ldots < p_{(m)}$$ corresponding to (after relabeling) the null hypotheses $$ H_1, H_2, \ldots, H_m$$, and find the largest index $$j$$ with $$p_{(k)} \leq q_k$$ for $$k = 1\ldots j$$.
Reject the first $$j$$ hypotheses $$H_1, H_2, \ldots, H_j$$.
This procedure caps the FDR at $$\alpha$$, and unlike the Bonferroni correction, we will typically reject some hypotheses.