---
layout: post
title: "Fun with GPS"
author: "Scott Roy"
categories:
tags: [GPS, identifiability]
image: GPS.png
---

In this post, I want to discuss the number of GPS satellites you need to determine your location.  I illustrate with an example.

Suppose Joe is lost in London, but knows where certain landmarks are, e.g., he knows where the library is, and where the museum and Buckingham palace are.  Joe stops me in the street to find his bearings.  I tell him, "Oh, you're 5000 feet from the library."  Joe now knows that he lies on a circle of radius 5000 around the library.  I then tell Joe that he is 3500 feet from the museum.  At this point Joe knows that he is 1) on a circle of radius 5000 around the library, and 2) on a circle of radius 3500 around the museum.  These two circles meet at 2 points, so Joe can narrow his location down to 2 possibilities.  If I then tell Joe he is 1000 feet from Buckingham palace, he will know exactly where he is.  To summarize, if I tell Joe his distance from 3 landmarks, he can determine his exact location.

How does this work in 3 dimensions (e.g., if we want to determine our elevation)?  In three dimensions, 2 spheres intersect in a circle, 3 spheres intersect at two points, and 4 spheres intersect at 1 point.  By knowing your distance to 4 landmarks, you can determine your exact position in space.

So you need 4 GPS satellites!  Not quite.  There are twenty something GPS satellites so that at any point on earth, at least 4 are visible.