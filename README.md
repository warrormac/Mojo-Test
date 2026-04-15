# Mojo Integration for Optimized Times

**Date:** 2025-05-26  
**Author:** Andre Alvarez  
**Project:** Fast-S3Data / Search-Validator-API  
**Context:** Evaluating Mojo language as a performance accelerator

# Summary

After technical exploration, I have **decided to postpone Mojo integration** into the `Search-Validator-API and Embedder-API` module. While Mojo offers promising speed improvements and an elegant syntax, **current language and tooling limitations make it impractical to integrate today**.

This memo explains the rationale behind the decision and why optimization is not yet a necessity.

# What was Evaluated

I targeted the semantic outlier detection logic in: search-validator-api/core/[heuristics.py](http://heuristics.py). Because the logic is numerically intensive and a good candidate for acceleration  
This involves:

* Mean vector calculation  
* L2 distance computation  
* Threshold-based outlier flagging

And I wrote the logic in Mojo, got it running. But when the code was ready to compile it into a .so for python, the problem of Mojo not supporting exporting functions to be called from python or other languages appeared. That means: 

* No `export fn`  
* No shared library generation (`.so`)  
* No compatibility with `ctypes`, `cffi`, or direct embedding

# **No Standard Library Support Yet**

Basic modules like `std` and `builtins` (for `List`, `F64`, etc.) are still evolving and not accessible in production-grade builds. Workarounds exist but are not stable.

This meant that i would have to develop a work around that would look like this:   
	

Python → JSON → Mojo → JSON → Python

And this would introduce the following to the API, which isn't worth the tradeoff right now:

* More I/O overhead than it removes  
* Loss of real-time validation in pipelines  
* Extra error-handling and coupling logic

# Why Search-Validator-API Doesn’t Need Mojo Yet

1. Already Optimized with Fast Model  
   

   Currently the APIs are using the hugging face model “sentence-transformers/static-retrieval-mrl-en-v1”. Which is highly optimized and already takes care of the heavy lifting in both APIs.  
2. Validation Logic is Not a Bottleneck

   Validation logic (semantic, heuristic, topological) runs quickly even on large chunk sets

3. Finally, introducing an external compute tool (Mojo) right now would:  
     
   1. Add complexity to onboarding and deployment  
   2. Require non-trivial cross-language devops work  
   3. Provide only modest returns at current scale  
      
