---
layout: post
title: "Multilevel models"
author: "Scott Roy"
categories:
tags:  
image: 
---


## Introduction

A multilevel model is one in which modeled variables depend on each other in a way that can be described by a directed acyclic graph.  The simplest multilevel models are hierarchical (which correspond to trees).  As an introductory example, suppose I have 30 friends and I want to measure how interested each is in romanic comedies.  For each friend $$i$$, I have ratings $$X_{ij}$$ they gave different romcom movies $$j \in M_i$$.  Not all my friends have seen the same movies, so the set of rated movies $$M_i$$ can vary by friend.  A simple hierarchical model is to suppose that each friend's interest in the romcom genre $$R_i$$ is normally distributed

$$R_i \sim N(R, \sigma^2)$$

and conditional on general interest $$R_i$$, a friend's ratings are also normally distributed:

$$X_{ij} | R_i \sim N(R_i, \sigma_i^2).$$

This is a multilevel model in which each friend's interest $$R_i$$ depends on $$R$$, and each friend's ratings $$X_{ij}$$ depend on the interest $$R_i$$.  The dependency graph for this model is shown below.  ADD GRAPH.

What is the MLE estimate for $$R_i$$?  Trick question!  $$R_i$$ is a random variable, not a parameter, so we don't estimate it with MLE.  We instead compute a posterior mean:

$$\textbf{E}[R_i | X_{ij}, \sigma_i^2, R, \sigma^2] = \left( \frac{\sigma^2}{\sigma^2 + \frac{\sigma_i^2}{|M_i|}} \right) \left(\frac{1}{|M_i|} \sum_{j \in M_i} X_{ij} \right) + \left( \frac{\frac{\sigma_i^2}{|M_i|}}{\sigma^2 + \frac{\sigma_i^2}{|M_i|}} \right) R$$

We'll walk through the math later, but for now focus on the intuition: a friend's expected interest in romcoms is a weighted average of their movie ratings $$X_{ij}$$ and $$R$$, the average interest in romcoms among my friends.  The weight given to each source depends on its certainty relative to the other source.  This is called partial pooling.  A friend who has many similar romcom ratings will have average posterior romcom interest closer to their movie ratings, whereas a friend with few or unpredictable ratings will have average posterior interest closer to $$R$$.

Hierarchical models are everywhere.  Another example is student test scores, where students belong to classrooms, classrooms belong to schools, schools belong to school districts, and school districts belong to states.  We can add more richness by recognizing that in addition to classrooms, students belong to differnet socio-economic backgrounds.  This additional connection creates a dependency graph that isn't a tree.  ADD FIGURE

Let's give one last example about modeling app crashes.  Publishers create apps and host them on an app catalog (e.g., Google Play Store) for devices to download.  Suppose we want to estimate reliability of each app and publisher.  For each app, we assume we have observed reliability data for the app from every device with it installed.  We model

$$\begin{aligned}
P_i &\sim N(P, \sigma_P^2) \\
A_j\ \vert\ P_i &\sim N(P_i, \sigma_A^2) \\
D_k &\sim N(D, \sigma_D^2) \\
Y_{jk}\ \vert\ P_i, D_j\ &\sim N(A_j + D_k, \sigma_Y^2),
\end{aligned}$$

where $$P_i$$ is publisher reliability, $$A_j$$ is app reliability, $$D_k$$ is general reliability of a device, and $$Y_{jk}$$ is observed reliability of app $$j$$ on device $$k$$.  We introduce $$D_k$$ because low-end or malware-infected devices may have lower general reliability, which influences $$Y_{jk}$$ beyond the quality of app $$j$$ itself.  The model is graphically illustrated below.

# Multilevel models

Models are termed fixed-effect, mixed-effect, or random-effect, depending on what is treated as a parameter or random variable in the model.  It is best to just carefully specify a model, rather than focus too much on which of these categories the model belongs to.

## Fixed-effect

In a fixed effect model, only the error term is treated as a random variable.  The covariates are treated as constants and the coefficients are treated as unknown, but fixed, parameters.  Ordinary least squares is a fixed effect model:

$$Y = \beta_0 + \beta_1 X_1 + \ldots + \beta_p X_p + \epsilon,$$

where $$\epsilon \sim N(0, \sigma^2)$$.  We could also view the covariates as random variables that have been conditioned on:

$$Y \vert X \sim N(\beta_0 + \beta_1 X_1 + \ldots + \beta_p X_p, \sigma^2)$$.

The parameters are typically found with maximum likelihood estimation.

## Mixed-effect

In a mixed-effect model, we have

$$Y = \beta_0 + \sum_{i=1}^p X_i \beta_i + \sum_{j=1}^q W_j Z_j + \epsilon$$,

where $$\beta_i$$ are fixed effects (unknown parameters), $$Z_j$$ are random effects with $$Z_j \sim N(0, \sigma_Z^2)$$, and $$\epsilon$$ is some random error.  In this form, the random effects are often interpreted as part of the error  term.  We can also specify the model as

$$\begin{aligned}
Z_j &\sim N(0, \sigma^2_Z) \\
Y \vert Z_j &\sim N(\beta_0 + \sum_{i=1}^p X_i \beta_i + \sum_{j=1}^q W_j Z_j, \sigma^2_Y).
\end{aligned}$$

The mean of $$Z_j$$ is 0 because it is considered part of the error term here, but it could be specified otherwise.  If the model is intercept varying, we could always move a nonzero random effect mean to the fixed-effect intercept.

The model is fit with MLE again.  Notice that $$(Y, Z_j)$$ are jointly multivariate normal, and so the marginal distribution of $$Y$$ is also multivariate normal.  We can also estimate $$E(Z_j \vert Y)$$.



## Random-effect


Sometimes the term mixed effect model in which unmodeled variables are mixed with modeled ones.  It is possible to model all unmodeled variables in a mixed-effect model.  By putting priors with no variance on these variables, we can recover the mixed-effect scheme.

This is typically how repeated measurement data is handled in regression scenarios, in which the error term is not independent.

---

### Parameters and variables

Multilevel models can be confusing because they contain both parameters (known and unknown), random variables, and constants.  In fully Bayesian models, there are only known (hyper)parameters and variables.  Mixed-effect models have constants (fixed effects), random variables (random effects), and unknown parameters (coefficients and variances).  We can estimate the parameters in a mixed-effect model with MLE, but the (unobserved) random variables may require EM.

In a Bayesian approach, we'd make the data constants and unknown parameters random variables and have corresponding known hyperparameters for them.  Constants can be embedded as is in a Bayesian model by making concentrated priors with zero or near zero variance and mean equal to the constant.

---

# Linear models with unit indicators
