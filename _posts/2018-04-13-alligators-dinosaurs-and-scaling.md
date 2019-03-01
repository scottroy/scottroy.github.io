---
layout: post
title: "Alligators, Dinosaurs, and Scaling"
author: "Scott Roy"
categories:
tags: [dimension of data]
image: alligator.jpg
---

I was walking around a book store in Fremont and found an elementary math education book.  While flipping through the book, I found the following neat problem.

Suppose you find a dinosaur skeleton that looks like an alligator, except that it is twice as big.  A typical alligator weighs 500 pounds.  Estimate how much the dinosaur weighed?

The answer is not 1000 pounds!  This is not how things scale.  If I double a length, areas increase by 4, and volumes increase 8 times (imagine doubling the side length of a square or cube).  In $$d$$ dimensions, if lengths are scaled by $$s$$, volumes will scale by $$s^d$$.

Weight is proportional to volume, and the dinosaur had $$2^3 = 8$$ times the volume as the alligator.  Thus the dinosaur weighed 4000 pounds.  Another way to see why 1000 pounds is incorrect is to consider the following: if I scale the alligator by 2, not only is the length doubled, but the width and height are also doubled.  Three dimensions doubled, which leads to an 8-fold increase in volume.

We can use these ideas to determine the dimension of a dataset.  High-dimensional data often lies in a low-dimensional manifold.  The dimension of this manifold is the "true" dimension of the data.  Imagine I make a ball around one of the data points, and count the number of points in the ball.  If I double the radius of the ball, the number of points will increase by $$2^d$$, where $$d$$ is the "true" dimension of the data.  We can estimate $$d$$ with

$$
\hat{d} = \log_2 \left( \frac{\text{# points in big ball}}{\text{# points in small ball}} \right).
$$
