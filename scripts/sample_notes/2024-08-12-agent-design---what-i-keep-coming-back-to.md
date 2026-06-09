---
date: 2024-08-12
title: Agent design — what I keep coming back to
tags: [ai, agents, workflow]
---

# Agent design — what I keep coming back to

The thing about agents is that the design question is really a question about delegation. What can a model decide on its own, and what does it have to ask? I keep drawing the same picture: a loop of tool call, observation, decision. Most agent frameworks just hide the loop.

The interesting part is what to do when the model is uncertain. Either you let it ask (interactive, slow) or you let it guess (autonomous, risky). Real work is somewhere in the middle, and the boundary shifts as the model gets more reliable.
