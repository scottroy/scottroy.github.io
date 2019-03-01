---
layout: post
title: "Randomized response"
author: "Scott Roy"
categories:
tags: [data privacy, randomized response]
image: privacy.jpg
---

Randomized response is a technique, introduced in the mid-1960s, to survey people about sensitive topics.  Suppose a teacher decides to survey his students on whether they cheat.  Rather than directly asking each student "Have you cheated?," a question cheaters are unlikely to answer truthfully, the teacher can employ randomized response.  Randomized response creates a privacy layer between the interviewer and the interviewee.  Here's how it works.  The teacher asks each student if he has cheated, but before the student answers, the teacher tells the student to flip a coin and keep the outcome of the flip secret.  If the coin comes up heads, the teacher tells the student to answer truthfully, but if the coin comes up tails, the student is instructed to answer "Yes."  So, if the student answers "Yes" to the question "Have you cheated?," it is either because 1) the student has cheated, or 2) the coin showed tails.  The teacher has no way to know why the student answered yes (there is a "privacy" layer).  If the student answers "No," the teacher knows the student did not cheat.

Let's say there are 300 people in the class, and 140 students answer "No."  What percent of the class cheated?  The coin randomly divides the class into two groups of (approximately) 150 students each: the "tails" students all answered yes, and the "heads" students told the truth.  The 140 "No"s are the number of "heads" students that have not cheated, which means about 10 in 150 "heads" students have cheated.  This puts the rate of cheating in the class (which matches the rate of cheating among "heads" students) at 1 in 15 students.

To summarize, randomized response allows us to estimate a quantity of interest (e.g., probability of cheating or mean income), without knowing the truth about any one individual.  There are variants of the technique.  For example, rather than flip a coin, the responder can roll a dice.  Rather than being instructed to say "Yes" or the truth, the interviewee could be told to tell the truth or lie.  The math behind the variants is a good exercise with Bayes rule.
