
Below are the reviewer comments for the first round of review for the paper [[NPE for RW]].
Reviewer 1 has also provided a PDF



---


1. **Are the objectives and the rationale of the study clearly stated?**  Please provide suggestions to the author(s) on how to improve the clarity of the objectives and rationale of the study. Please number each suggestion so that author(s) can more easily respond.  
  
	Reviewer #1: Yes, but I have some issues with that. The fundamental issue concerns modelling and simulation in general; its application to biology is just one of several potential applications. As you will see from the paper, the focus is on the method itself rather than its application to biological models.  I expand on this topic further in the additional material.  
  
	Reviewer #2: The lack of a clear goal for the study is one of our main concerns. The study provides neither new biological results nor a new method (it extensively uses an existing Python package). The scope is rather narrow, as the study uses a very specific random walk model that does not strictly require simulation-based inference to infer parameters (some of the authors recently published an article that uses other methods to infer these two parameters). Furthermore, the manuscript does not provide a practical guide on how to use simulation-based inference for random walk models in biology, as the examined model is not of broad biological interest. The study also fails to demonstrate the strengths and challenges of simulation-based inference for a variety of random walk models, nor does it outline the best practices (e.g. it misses the opportunity to show how to use diagnostic methods for simulation-based inference). It is therefore hard to evaluate a manuscript that is well-executed and without obvious major technical issues, but which lacks novelty, generality, or practical guidance.  
  
2. **If applicable, is the application/theory/method/study reported in sufficient detail to allow for its replicability and/or reproducibility?**  Please provide suggestions to the author(s) on how to improve the replicability/reproducibility of their study. Please number each suggestion so that the author(s) can more easily respond.  
  
	Reviewer #1: Mark as appropriate with an X:  
	Yes [x] No [] N/A []  
	Provide further comments here:  
  
	Reviewer #2: Mark as appropriate with an X:  
	Yes [X] No [] N/A []  
	Provide further comments here:  
  
	i. The code is particularly well written and can be used to reproduce the study and learn more about applying simulation-based inference. There are however some files missing (e.g., notebooks/example_results/results_extracted.pkl' for the demo.py ) to run notebooks mentioned in the code repository out-of-the-box.  
	  
	ii. It is unclear from the text whether training and parameter inference were performed only once, or if repeating the experiment with different seeds would lead to similar results.  
	  
	iii. If applicable, are statistical analyses, controls, sampling mechanism, and statistical reporting (e.g., P-values, CIs, effect sizes) appropriate and well described?  
  
3. **Please clearly indicate if the manuscript requires additional peer review by a statistician.** Kindly provide suggestions to the author(s) on how to improve the statistical analyses, controls, sampling mechanism, or statistical reporting. Please number each suggestion so that the author(s) can more easily respond.  
  
	Reviewer #1: Mark as appropriate with an X:  
	Yes [x] No [] N/A []  
	Provide further comments here:  
	  
	Reviewer #2: Mark as appropriate with an X:  
	Yes [X] No [] N/A []  
	Provide further comments here:  
	  
	The analysis does not report diagnostic of the simulation-based inference (although the authors provide posterior predictive checks, which are important, they are not a complete diagnostic workflow) -- which is not a problem here because parameters were known from another study, but which would, in general, be a problem. Reference 15 of the article provides guidelines for such diagnostics.  
  
4. **Could the manuscript benefit from additional tables or figures, or from improving or removing (some of the) existing ones?**  Please provide specific suggestions for improvements, removals, or additions of figures or tables. Please number each suggestion so that author(s) can more easily respond.  
  
	Reviewer #1: Yes, because I have the impression that the scope of the journal is not being met.  
	  
	Reviewer #2: 1. Section 4.3.2 states: “Figure 6 shows the joint posterior distribution inferred from the 2D spatial data, overlaid with the result from the 1D column count analysis for direct comparison.” However, this does not seem to be the case for Figure 6.  
  
	i. It would be helpful to include a summary table of the hyperparameters in the paper itself, rather than only in the code repository.  
	  
	ii. If applicable, are the interpretation of results and study conclusions supported by the data?  
  
5. **Please provide suggestions (if needed) to the author(s) on how to improve, tone down, or expand the study interpretations/conclusions.** Please number each suggestion so that the author(s) can more easily respond.  
  
	Reviewer #1: Mark as appropriate with an X:  
	Yes [x] No [] N/A []  
	Provide further comments here: The problem is that only one example was chosen to demonstrate the practical application of the method. For such a topic, I would expect this method to be applied to a series of examples, including a comparison and benchmarking.  
	  
	Reviewer #2: Mark as appropriate with an X:  
	Yes [] No [] N/A [X]  
	Provide further comments here:  
  
6. **Have the authors clearly emphasized the strengths of their study/theory/methods/argument?**  Please provide suggestions to the author(s) on how to better emphasize the strengths of their study. Please number each suggestion so that the author(s) can more easily respond.  
  
	Reviewer #1: Yes  
	  
	Reviewer #2: Because of the lack of clearly established goals, it is currently unclear what would be the current strengths of the manuscript.  
  
7. **Have the authors clearly stated the limitations of their study/theory/methods/argument?**  Please list the limitations that the author(s) need to add or emphasize. Please number each limitation so that author(s) can more easily respond.  
  
	Reviewer #1: Not really! For such a topic, I would expect this method to be applied to a series of examples, including a comparison and benchmarking.  
	  
	Reviewer #2: It is not clear why authors chose NPE specifically? What about other algorithms? (e.g., NRE) - this choice should probably be at least discussed or justified.  
	Section 4.3.2 states: “As summarised in Table 2, leveraging the full spatial data yields a posterior with a precision that is highly comparable to that obtained using curated 1D summary statistics, cf. Section 3.5. This result demonstrates that the CNN can automatically learn features from the raw data that are as informative as the carefully chosen summary statistics.” However, it does not seem to illustrate an improvement in the precision of parameter inference. It would be much more convincing to present a case in which switching from 1D to 2D summary statistics actually impacts the quality of the parameter estimate. Moreover, the authors used manually selected summary statistics at different levels of aggregation in both cases, 1D histogram versus 2D histogram.  
  
8. **Does the manuscript structure, flow or writing need improving (e.g., the addition of subheadings, shortening of text, reorganization of sections, or moving details from one section to another)?**  Please provide suggestions to the author(s) on how to improve the manuscript structure and flow. Please number each suggestion so that author(s) can more easily respond.  
  
	Reviewer #1: I provided a comment on this topic in the additional material.  
	  
	Reviewer #2: The manuscript is very well written and suitable for a quantitative and theoretical biology public.  We however found that the writing could benefit, at times, from more precision and fewer colloquialisms. For example, in this paragraph: "The fundamental idea behind normalizing flows is deceptively simple: construct a complex target distribution by transforming samples from a simple base distribution (typically a standard Gaussian) through a series of invertible, differentiable mappings. This approach can be understood as a sophisticated generalization of the familiar inverse transform sampling method, extended to high dimensions through neural networks", wording such as "deceptively simple" or "familiar inverse transform sampling" could be avoided and enhanced.  
  
9. **Could the manuscript benefit from language editing?**  
  
	Reviewer #1: No  
	
	Reviewer #2: No  

10. **Additional points**

Reviewer #1: I provided additional material for this review, in which I addressed some general aspects, as well as some of the points highlighted in the attached paper (JTB-D-25-00973_review-commented.pdf).  
  
Reviewer #2: The manuscript Rapid parameter inference for spatiotemporal stochastic biological models using neural posterior estimation, considered for publication in the Journal of Theoretical Biology, provides an overview of the application of a simulation-based inference approach for the inference of the parameters of a random walk model depicting a barrier assay experiment in migrating cells. This review is a joint effort, written by two researchers, including a PhD candidate.  
Overall, we recognise several very strong aspects of the manuscript. First, it is very well written, and particularly suitable for a quantitative and theoretical biology audience. It presents a methodologically important perspective that simulation-based inference could be highly relevant for parameter inference in biological modelling. The study also provides a high-quality code to implement the method discussed in the manuscript.  
  
As covered in the other sections of this review, the goal for the manuscript is currently unclear. To us, the manuscript would benefit from a clearer direction with either more novelty in the research results or more general guidance for using simulation-based inference for biological models. A suggestion for such a direction could be, for instance, to explore the strengths and weaknesses of using simulation-based inference methods for biological random-walk models in general.  
  
2. P7: the definition of "amortized" could come earlier for clarity, the word being mentioned twice earlier (p3, in Introduction 1 and Table 1)  
3. As stated in the SBI package documentation, reference 63 (SBI package) should be updated to [https://joss.theoj.org/papers/10.21105/joss.07754](https://joss.theoj.org/papers/10.21105/joss.07754)  
4. Paragraph 1 of Section 3.2.1 is unclear.  
5. The last paragraph of Section 2.1: "More efficient variants of ABC, such as Sequential Monte Carlo ABC [60, 64], have been proposed to try to circumvent some of the above issues." It is unclear why the authors mention alternative methods without discussing them.  
6. The Section 2.3 label is unclear: "Example Results."  
7. The first paragraph of Section 3.1: "NPE [43, 24, 14] offers a fundamentally different approach to inference for simulator-based models." This statement is incorrect or unclear. NPE is a simulation-based inference method that produces approximation of a posterior distribution over parameters.  
8. Section 3.3: The section is about SNPE, yet the closing sentence says: "In this work we will exclusively use the sequential version of NPE". It is unclear why the authors devoted an entire section to describing an approach that is not employed in the paper. On the other hand, the "Neural Spline Flows" method is not detailed in the next section, 3.4, even though it is actually used to produce results.  
9. Section 3.5.1: " In the absence of strong prior information about parameter values, these uniform priors represent a minimally informative choice that allows the data to dominate the posterior inference." We believe this statement is imprecise, as uniform priors are not necessarily the minimally informative choice in general.  
10. Table 1: Cost per Inference vs. Amortized - are they the same thing?  
11. Table 2: It would be great to include the results from the alternative models (ABC, MCMC, etc) and the true values.  
12. Table 3: It would be informative to separate the training of the network and the generation of the simulation for training, since simulations are usually the bottleneck for complex simulators rather than for training the network.  
  
___________________________  
  
