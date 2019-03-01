---
layout: post
title: "The correspondence between correlation and p-values"
author: "Scott Roy"
categories: journal
tags: []
image: contingency22.png
---

Suppose $$ X \in \{0,1\}^N$$ is a binary vector and $$ Y \in \mathbb{R}^N$$ is some real-valued vector.  One way to measure association between $$ X$$ and $$ Y$$ is the (sample) correlation

$$ \begin{aligned} r &= \frac{\sum_{i=1}^N (X_i - \bar{X})(Y_i - \bar{Y})}{\sqrt{\sum_{i=1}^N (X_i - \bar{X})^2 \cdot \sum_{i=1}^N (Y_i - \bar{Y})^2}} \end{aligned}$$ 

Since $$ X$$ is binary, we can view it as defining two groups over $$ Y$$ .  Measuring the "difference" of $$ Y$$ between these groups by computing a p-value is an alternative way of thinking of the association between $$ X$$ and $$ Y$$ .  We show that these two approaches are closely related.  In fact, if we have several vectors $$ X_1, \ldots, X_p$$ , then ranking these vectors by correlation with $$ Y$$ is equivalent to ranking these vectors by the p-values defined by their groupings.

We quickly define some notation.  The vector $$ X$$ splits the observations $$ Y$$ into two groups: $$ \{ Y_i : X_i = 0 \}$$ and $$ \{ Y_i : X_i = 1 \}$$ .  Let $$ N_0$$ and $$ N_1$$ be the respective sizes of these groups.  We reindex the observations $$ Y$$ using notation from ANOVA.  We let $$ Y_{ij}$$ denote the $$ j$$ th observation ($$ j = 1\ldots N_j$$ ) from the $$ i$$ th group ($$ i = 0, 1$$ ).  The notation $$ \bar{Y}_{i\cdot}$$ denotes the mean of $$ Y$$ over the $$ i$$ th group and $$ \bar{Y}_{\cdot \cdot}$$ denotes the overall mean of $$ Y$$ .

The correlation can be written as

$$ \begin{aligned} r &= \frac{\bar{Y}_{1\cdot} - \bar{Y}_{0\cdot}}{\sqrt{(N-1) S^2 \left(\frac{1}{N_1} + \frac{1}{N_0} \right)}}, \end{aligned}$$ 

where $$ S^2 = \frac{1}{n-1} \sum_{i=1}^N (Y_i - \bar{Y})^2$$ is the sample variance.  This resembles the two sample t-statistic (which hints at the connection to p-values), but has the sample variance $$ S^2$$ instead of the pooled variance

$$ \begin{aligned} S^2_p &= \frac{\sum_{i=0}^1 \sum_{j=1}^{N_j} (Y_{ij} - \bar{Y}_{i\cdot})^2}{N-2}. \end{aligned}$$ 

We can see the connection to p-values immediately if $$ Y \in \{0,1\}^N$$ is also binary and we perform a difference of proportions test with the statistic

$$ \begin{aligned} T &= \frac{p_1 - p_0}{\sqrt{p(1-p) \left( \frac{1}{N_1} + \frac{1}{N_0} \right)}} \\ &= \frac{\bar{Y}_{1 \cdot} - \bar{Y}_{0 \cdot}}{ \sqrt{\left(\frac{N-1}{N}\right) S^2 \left( \frac{1}{N_1} + \frac{1}{N_0} \right)}} \\ &= r / \sqrt{N}. \end{aligned}$$ 

When $$ Y$$ is not binary, we need to relate the sample variance to the pooled variance.  We use the following sum of squares partition (derivation in next section):

$$ \begin{aligned} (N-1) S^2 \left( \frac{1}{N_1} + \frac{1}{N_0} \right) &= (N-2) S_p^2 \left( \frac{1}{N_1} + \frac{1}{N_0} \right) + (\bar{Y}_{1\cdot} - \bar{Y}_{0\cdot})^2. \end{aligned}$$ 

Using this partition to rewrite the denominator in the correlation expression and dividing numerator and denominator by $$ \sqrt{S_p^2 \left( \frac{1}{N_1} + \frac{1}{N_0} \right)}$$ yields

$$ \begin{aligned} r &= \frac{\bar{Y}_{1\cdot} - \bar{Y}_{0\cdot}}{\sqrt{(N-1) S^2 \left(\frac{1}{N_1} + \frac{1}{N_0} \right)}} \\ &= \frac{(\bar{Y}_{1\cdot} - \bar{Y}_{0\cdot}) / \sqrt{S_p^2 \left( \frac{1}{N_1} + \frac{1}{N_0} \right)}}{\sqrt{N-2 + (\bar{Y}_{1\cdot} - \bar{Y}_{0\cdot})^2 / \left(S_p^2 \left( \frac{1}{N_1} + \frac{1}{N_0} \right)\right)}} \\ &= \frac{T}{\sqrt{N-2 + T^2}}, \end{aligned}$$ 

where

$$ \begin{aligned} T &= \frac{\bar{Y}_{1\cdot} - \bar{Y}_{0\cdot}}{\sqrt{S_p^2 \left(\frac{1}{N_1} + \frac{1}{N_0} \right)}} \end{aligned}$$ 

is the two sample t-statistic.

To summarize, if $$ Y$$ is binary and $$ T$$ is a difference of proportions statistic, then

$$ \begin{aligned} r &= \sqrt{N} \end{aligned}T$$ .

On the other hand, if $$ T$$ is a two sample t-statistic, then

$$ \begin{aligned} r &= \frac{T}{\sqrt{N-2 + T^2}} \end{aligned}$$ .

In both cases, the correlation is a (strictly) increasing function of the statistic.  This means that if we have multiple groups defined by binary vectors $$ X_1, \ldots, X_p$$ and we want to know which vector defines groups with the most statistically significant differentiation in $$ Y$$ , we can just find the vector $$ X_i$$ with the highest (absolute) correlation with $$ Y$$ .

Partitioning the sum of squares
Partitioning the variation is fundamental in ANOVA and regression analysis and is a simple consequence of the Pythagorean theorem.  Define the following three vectors

$$ Y = \begin{bmatrix} \begin{pmatrix} Y_{11} \\ Y_{12} \\ \vdots \\ Y_{1 N_1} \end{pmatrix} \\ \begin{pmatrix} Y_{21} \\ Y_{22} \\ \vdots \\ Y_{2 N_2} \end{pmatrix} \end{bmatrix} \quad Y_{\text{trt}} = \begin{bmatrix} \begin{pmatrix} \bar{Y}_{1 \cdot} \\ \bar{Y}_{1 \cdot} \\ \vdots \\ \bar{Y}_{1 \cdot} \end{pmatrix} \\ \begin{pmatrix} \bar{Y}_{2 \cdot} \\ \bar{Y}_{2 \cdot} \\ \vdots \\ \bar{Y}_{2 \cdot} \end{pmatrix} \end{bmatrix} \quad  \bar{Y}_{\cdot \cdot} = \begin{bmatrix} \begin{pmatrix} \bar{Y}_{\cdot \cdot} \\ \bar{Y}_{\cdot \cdot} \\ \vdots \\ \bar{Y}_{\cdot \cdot} \end{pmatrix} \\ \begin{pmatrix} \bar{Y}_{\cdot \cdot} \\ \bar{Y}_{\cdot \cdot} \\ \vdots \\ \bar{Y}_{\cdot \cdot} \end{pmatrix} \end{bmatrix}$$ 

The vectors $$ (Y_{\text{trt}} - \bar{Y}_{\cdot \cdot})$$ and $$ (Y - Y_{\text{trt}})$$ are orthogonal.  Applying the Pythagorean theorem to the decomposition $$ Y - \bar{Y}_{\cdot \cdot} = (Y - Y_{\text{trt}}) + (Y_{\text{trt}} - \bar{Y}_{\cdot \cdot})$$ gives the sum of squares decomposition used above.

Generalizing to non-binary X
We can generalize the above results further to the case where $$ X$$ is non-binary.  Suppose we regress $$ Y \sim 1 + X$$ and get slope $$ \hat{\beta}$$ .  To make matters simple, let $$ SXX = \sum_{i=1}^N (X_i - \bar{X})^2$$ be the sum of squares for $$ X$$ (and similarly for $$ SYY$$ ), $$ SXY = \sum_{i=1}^N (X_i - \bar{X}) (Y_i - \bar{Y})$$ , and $$ RSS = \sum_{i=1}^N (Y_i - \hat{Y}_i)^2$$ be the residual sum of squares.

The coefficient $$ \hat{\beta} = \frac{SXY}{SXX}$$ and the correlation between $$ X$$ and $$ Y$$ is

$$ \begin{aligned} r = \hat{\beta} \sqrt{\frac{SXX}{SYY}}. \end{aligned}$$ 

To test whether $$ \hat{\beta}$$ is nonzero, we see how many standard errors it is from 0.  The standard error of $$ \hat{\beta}$$ is $$ \text{se}(\hat{\beta}) = \sqrt{\frac{RSS}{(N-2) SXX}}$$ and the test statistic is

$$ \begin{aligned} T &= \frac{\hat{\beta}}{\text{se}(\hat{\beta})} = \hat{\beta} \sqrt{\frac{SXX}{RSS / (N-2)}}. \end{aligned}$$ 

This reduces to the two-sample t-statistic when $$ X$$ is binary and follows a t-distribution with $$ N-2$$ degrees of freedom.

Dividing numerator and denominator in the expression for $$ r$$ by $$ \sqrt{RSS / (N-2)}$$ (after rewriting $$ SYY = RSS + \hat{\beta}^2 SXX$$ ) we get

$$ \begin{aligned} r &= \hat{\beta} \sqrt{\frac{SXX}{SYY}} \\ &= \frac{\hat{\beta} \sqrt{\frac{SXX}{RSS / (N-2)}}}{\sqrt{N-2 + \hat{\beta}^2 \frac{SXX}{RSS / (N-2)} }} \\ &= \frac{T}{\sqrt{N-2 + T^2}},  \end{aligned}$$ 

as before.

Connection to testing nonzero correlation
To test if the sample correlation $$ r$$ is nonzero, the test statistic is

$$ \begin{aligned} T &= r \sqrt{\frac{N-2}{1-r^2}} \end{aligned}$$ 

and follows a t-distribution with $$ N-2$$ degrees of freedom.  As shown in the previous section, this is equivalent to testing whether the regression coefficient is 0 in the simple linear regression $$ Y \sim 1 + X$$ .

To get confidence intervals for the correlation coefficient (or to test against non-zero values) usually Fisher's transformation is used.

Generalizing to multivariate regression and F-Statistics
In linear models, we often analyze the sample correlation between the response $$ Y$$ and the predicted values $$ \hat{Y}$$ to evaluate the model's fit. EX 6.8 in weisberg.

Chi-squared test and phi coefficient
For two binary variables $$ X$$ and $$ Y$$ , we can summarize their relation with a contingency table:

[]

In the table $$ O_{ij}$$ denotes the number of observations where $$ X = i$$ and  $$ Y = j$$ .  The phi coefficient measures the association between $$ X$$ and $$ Y$$ and is defined as:

$$ \begin{aligned} \phi &= \frac{O_{00} O_{11} - O_{01} O_{10}}{\sqrt{O_{0\cdot} O_{\cdot 0} O_{1 \cdot} O_{\cdot 1}}} \end{aligned}$$ .

The phi coefficient is thus a product of counts where $$ X$$ and $$ Y$$ agree minus a product of counts where they disagree, normalized by row and column counts $$ O_{i\cdot}$$ and $$ O_{\cdot j}$$ so that the value is between -1 and 1.

The phi coefficient is equal to the (regular) correlation between $$ X$$ and $$ Y$$ .  More interesting is that the phi coefficient is determined by the chi-squared statistic to test independence of $$ X$$ and $$ Y$$ :

$$ \begin{aligned} \phi &= \sqrt{\frac{\chi^2}{n}} \end{aligned}$$ .

As a reminder

$$ \begin{aligned} \chi^2 &= \sum \frac{(O_{ij} - E_{ij})^2}{E_{ij}} \end{aligned}$$ ,

where

$$ \begin{aligned} E_{ij} &= n \left( \frac{O_{i \cdot}}{n} \right) \left( \frac{O_{\cdot j}}{n} \right) := n r_i c_j \end{aligned}$$ 

is the expected number observations in cell $$ (i,j)$$ under the assumption that $$ X$$ and $$ Y$$ are independent.  The above relation between the phi coefficient and the chi-squared statistic is easy to show by expanding the chi-squared statistic.  The following relation is useful while expanding:

$$ \begin{aligned} O_{i \cdot} O_{\cdot j} &= n O_{ij} + s_{ij} \Delta \end{aligned}$$ ,

where $$ s_{ij}$$ is either 1 (if $$ i \neq j$$ ) or -1 ($$ i = j$$ ) and $$ \Delta = O_{00} O_{11} - O_{01} O_{10}$$ is the determinant of the contingency table.

Tying this section back to the theme of this post: suppose we (individually) test the relation between $$ k$$ binary variables $$ X_1, X_2, \ldots, X_k$$ and a binary response $$ Y$$ .  Then the following quantities all have the same ordering (are monotonic functions of each other):

p-values computed using a difference of proportions test
p-values computed using a two-sample t-test
p-values computed using a chi-squared independence test
The correlations (phi coefficients)
As far as a finding the predictor that is "most statistically significantly" associated with $$ Y$$ (meaning the one with the smallest p-value), we could simply select the one with the largest correlation.

Selecting the covariate most correlated with $$ Y$$ is a common variable selection technique.  When the variables are discrete, mutual information is another common technique

G-test and mutual information
The chi-squared test discussed in the previous section was introduced by Karl Pearson in 1900.  We can also use a likelihood ratio test to test if the observed count data comes from two independence categorical variables.  The resulting statistic is the so-called G-statistic:

$$ \begin{aligned} G &= 2 \sum O_{ij} \log \left( \frac{O_{ij}}{E_{ij}} \right) \end{aligned}$$ .

By sight, we can recognized this is $$ G = 2 n MI(X, Y)$$ , where $$ MI(X,Y)$$ is the mutual information between $$ X$$ and $$ Y$$ (the Kullback-Leibler divergence of the product of the marginal distributions from the joint distribution).  The G-test and chi-squared test are very similar in practice.  In fact the chi square statistic is the second order taylor expansion about 1 replacing the log with its second order tayor expansion about 1 .



References

Applied Linear Regression by Sanford Weisberg (Chapter 2 and exercises 2.10 and 2.12)


