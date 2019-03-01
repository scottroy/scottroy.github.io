---
layout: post
title: "Interpreting regression coefficients"
author: "Scott Roy"
categories:
tags: [interpreting regression coefficients, statistical significance]
image: regression.png
---

Suppose we regress a response $$Y$$ on covariates $$X_j$$ for $$j = 1 \ldots p$$.  In linear regression, we get the model

$$Y = \beta_0 + \beta_1 X_1 + \ldots \beta_p X_p.$$

How do we interpret $$\beta_j = 0$$?  Does it mean that the $$j$$th covariate is uncorrelated with the response? The answer is no!  It means the $$j$$th covariate is uncorrelated with the response after we control for the effects of the other covariates.  A neat way to see this is to note the following way to compute the coefficient $$\beta_j$$.  For notational convenience, we assume $$j = 1$$.

Regress $$Y$$ against the covariates $$X_2, \ldots, X_p$$, and compute the residuals.  These residuals describe the part of the response $$Y$$ not explained by regression on the covariates $$X_2, \ldots, X_p$$ .
Regress $$X_1$$ against the covariates $$X_2, \ldots, X_p$$, and get the residuals.  These residuals describe the part of the regressor $$X_1$$ not explained by the covariates $$X_2, \ldots, X_p$$.
We form an added-variable plot for $$X_1$$ after $$X_2, \ldots, X_p$$ by plotting the residuals from step 1 against the residuals from step 2.  The slope of the regression line in the added-variable plot, which describes the relation between $$Y$$ and $$X_1$$ after controlling for the other covariates, is equal to the coefficient $$\beta_1$$.
For a concrete example, suppose we regress a person's income against their height and age, and find that $$\beta_{\text{height}}$$ is not significantly different from 0.  We should interpret this as there is no relationship between income and height, after we adjust for age.
