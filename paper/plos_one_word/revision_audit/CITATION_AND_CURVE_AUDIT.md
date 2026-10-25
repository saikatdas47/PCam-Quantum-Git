# Citation and curve review

Executed 24 October 2026. This is a targeted literature search and claim check. It is not a systematic review or proof that no similar work exists. No models were retrained. Measured scores, thresholds and statistics were preserved.

## Graph checks

All 13 saved test score files were checked. Each contains 32,768 rows with 16,377 positives and 16,391 negatives. ROC endpoints, monotonic coordinates and AUROC integration passed. At 11 thresholds per file, false positive rate and sensitivity matched the confusion counts. At 11 precision recall thresholds per file, precision and recall matched the confusion counts. See curve_checks.json. Full raw scores drive every plotted curve.

ROC uses false positive rate on the horizontal axis and sensitivity on the vertical axis. Precision recall uses sensitivity on the horizontal axis and precision on the vertical axis. These are standard axes. Precision need not be monotonic. Small jumps follow individual ranked predictions and the first predictions can have precision zero or one. At recall one, precision reaches the test prevalence of 0.4998 when all rows are predicted positive. The conventional recall zero endpoint has precision one. The empirical curves are not smoothed. Different optimization seeds are displayed separately and are not treated as patient samples.

The previous combined figure mixed 13 overlapping curves. The revision has separate ROC and precision recall figures. Four panels each compare the three seeds of one head against the same classical reference. No methodology figure is embedded or regenerated.

## Sentence review

The following review checks claims in Introduction and Related work. References support the specific external claim. Statements describing our experiment are supported by exported configuration, predictions and manifests. Interpretations are identified as such and do not assert priority.

1. PatchCamelyon (PCam) provides small tissue images for testing models that detect metastatic tissue [1,2].

Support: [1] Veeling BS, Linmans J, Winkens J, Cohen T, Welling M. Rotation equivariant CNNs for digital pathology. MICCAI. 2018. Author version: https://arxiv.org/abs/1806.03962., [2] Veeling BS and PCam contributors. PatchCamelyon dataset repository. https://github.com/basveeling/pcam. Accessed 24 October 2026.

2. Its fixed training, validation and test partitions allow studies to use a common benchmark.

Support: Own study description or bounded interpretation of the adjacent source. No claim that previous work lacks these controls.

3. The task concerns tissue patches.

Support: Own study description or bounded interpretation of the adjacent source. No claim that previous work lacks these controls.

4. A patch score alone does not establish performance for whole slides or patients.

Support: Own study description or bounded interpretation of the adjacent source. No claim that previous work lacks these controls.

5. This distinction matters when interpreting results from an image classifier.

Support: Own study description or bounded interpretation of the adjacent source. No claim that previous work lacks these controls.

6. A pretrained image encoder can provide features for a smaller classifier.

Support: Own study description or bounded interpretation of the adjacent source. No claim that previous work lacks these controls.

7. ResNet introduced residual connections for image recognition [3].

Support: [3] He K, Zhang X, Ren S, Sun J. Deep residual learning for image recognition. CVPR. 2016. Author version: https://arxiv.org/abs/1512.03385.

8. Work on feature transfer showed that transfer depends on the source task and network layer [4].

Support: [4] Yosinski J, Clune J, Bengio Y, Lipson H. How transferable are features in deep neural networks? Advances in Neural Information Processing Systems. 2014. https://arxiv.org/abs/1411.1792.

9. Mari et al. extended transfer learning to models that place a quantum circuit after a pretrained classical network [5].

Support: [5] Mari A, Bromley TR, Izaac J, Schuld M, Killoran N. Transfer learning in hybrid classical-quantum neural networks. Quantum. 2020;4:340. doi:10.22331/q-2020-10-09-340.

10. This approach makes small circuit experiments possible without encoding every image pixel into qubits.

Support: Own study description or bounded interpretation of the adjacent source. No claim that previous work lacks these controls.

11. It also raises a direct question.

Support: Own study description or bounded interpretation of the adjacent source. No claim that previous work lacks these controls.

12. Does the circuit improve predictions once the image features are fixed?

Support: Own study description or bounded interpretation of the adjacent source. No claim that previous work lacks these controls.

13. Reviews of quantum learning describe both possible uses and limits in training and evaluation [6,7].

Support: [6] Cerezo M, Arrasmith A, Babbush R, Benjamin SC, Endo S, Fujii K, et al. variational quantum algorithms. Nature Reviews Physics. 2021;3:625–644. doi:10.1038/s42254-021-00348-9., [7] Cerezo M, Verdon G, Huang HY, Cincio L, Coles PJ. Challenges and opportunities in quantum machine learning. Nature Computational Science. 2022;2:567–576. doi:10.1038/s43588-022-00311-3.

14. Circuit output depends on how data enter the circuit [8].

Support: [8] Schuld M, Sweke R, Meyer JJ. Effect of data encoding on the expressive power of variational quantum-machine-learning models. Physical Review A. 2021;103:032430. doi:10.1103/PhysRevA.103.032430.

15. Some circuit families also have very small gradients, although this cannot be assumed for every small model [9].

Support: [9] McClean JR, Boixo S, Smelyanskiy VN, Babbush R, Neven H. Barren plateaus in quantum neural network training landscapes. Nature Communications. 2018;9:4812. doi:10.1038/s41467-018-07090-4.

16. Classical models can remain competitive even on some tasks designed around quantum features [10].

Support: [10] Huang HY, Broughton M, Mohseni M, Babbush R, Boixo S, Neven H, et al. power of data in quantum machine learning. Nature Communications. 2021;12:2631. doi:10.1038/s41467-021-22539-9.

17. Broad benchmark results further show why useful classical controls and circuits without entangling gates should be tested [11].

Support: [11] Bowles J, Ahmed S, Schuld M. Better than classical? The subtle art of benchmarking quantum machine learning models. arXiv:2403.07059v2. 2024. https://arxiv.org/abs/2403.07059.

18. These findings motivate a controlled comparison rather than an assumed quantum benefit.

Support: Own study description or bounded interpretation of the adjacent source. No claim that previous work lacks these controls.

19. The evaluation also needs to separate model selection from final testing [12].

Support: [12] Cawley GC, Talbot NLC. On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation. Journal of Machine Learning Research. 2010;11:2079–2107. https://www.jmlr.org/papers/v11/cawley10a.html.

20. Patches from one slide can be related, so uncertainty should account for slide groups [13].

Support: [13] Obuchowski NA. Nonparametric analysis of clustered ROC curve data. Biometrics. 1997;53(2):567–578. doi:10.2307/2533958.

21. Here we compare four small heads and a classical reference on the same eight features from a frozen ResNet50.

Support: Own study description or bounded interpretation of the adjacent source. No claim that previous work lacks these controls.

22. We retain the official training rows, choose the reference within training data and lock the fitted models before test scoring.

Support: Own study description or bounded interpretation of the adjacent source. No claim that previous work lacks these controls.

23. Our contribution is the resulting comparison of input mapping and CNOT gates, with paired slide uncertainty, separate seed variation and saved predictions.

Support: Own study description or bounded interpretation of the adjacent source. No claim that previous work lacks these controls.

24. The question is whether this tested circuit helps this representation under the stated training schedule.

Support: Own study description or bounded interpretation of the adjacent source. No claim that previous work lacks these controls.

25. Mari et al. established the use of pretrained classical features with a small trainable quantum head [5].

Support: [5] Mari A, Bromley TR, Izaac J, Schuld M, Killoran N. Transfer learning in hybrid classical-quantum neural networks. Quantum. 2020;4:340. doi:10.22331/q-2020-10-09-340.

26. This is the starting point of our study, not a new architecture.

Support: Own study description or bounded interpretation of the adjacent source. No claim that previous work lacks these controls.

27. Work on data encoding further links circuit outputs to Fourier functions of the inputs [8].

Support: [8] Schuld M, Sweke R, Meyer JJ. Effect of data encoding on the expressive power of variational quantum-machine-learning models. Physical Review A. 2021;103:032430. doi:10.1103/PhysRevA.103.032430.

28. This gives a reason to include a classical trigonometric reference alongside neural controls.

Support: Own study description or bounded interpretation of the adjacent source. No claim that previous work lacks these controls.

29. It does not imply that our reference can reproduce every circuit function.

Support: Own study description or bounded interpretation of the adjacent source. No claim that previous work lacks these controls.

30. Two studies are particularly close to the pathology task.

Support: Own study description or bounded interpretation of the adjacent source. No claim that previous work lacks these controls.

31. Goyal et al. studied quantum edge and frequency preprocessing with image classifiers and fusion on a modified PCam dataset [14].

Support: [14] Goyal N, Uehara G, Spanias A. Quantum-enhanced cancer detection for histopathologic images. IEEE ICIP. 2025. doi:10.1109/ICIP55913.2025.11084595.

32. Their intervention changes image processing.

Support: Own study description or bounded interpretation of the adjacent source. No claim that previous work lacks these controls.

33. Baral et al. studied classical and hybrid classifiers under adversarial attacks [15].

Support: [15] Baral B, Bhalgamiya B, Majumder R, Dutta Roy D, Dutta Roy T. Adversarial attacks on hybrid classical-quantum deep learning models for histopathological cancer detection. APL Machine Learning. 2025;3:036106. doi:10.1063/5.0270673.

34. They used a balanced subset of 20,000 PCam images and an 80:20 split.

Support: Own study description or bounded interpretation of the adjacent source. No claim that previous work lacks these controls.

35. They also included a classical bottleneck model and evaluated their selected hybrid model with three random seeds.

Support: Own study description or bounded interpretation of the adjacent source. No claim that previous work lacks these controls.

36. Classical controls and repeated seeds therefore already exist in this literature.

Support: Own study description or bounded interpretation of the adjacent source. No claim that previous work lacks these controls.

37. Our experiment instead holds the frozen representation fixed and uses the official partitions for a direct comparison of classifier heads.

Support: Own study description or bounded interpretation of the adjacent source. No claim that previous work lacks these controls.

38. Bowles et al. compared quantum and classical models across benchmark tasks and examined circuits without entanglement [11].

Support: [11] Bowles J, Ahmed S, Schuld M. Better than classical? The subtle art of benchmarking quantum machine learning models. arXiv:2403.07059v2. 2024. https://arxiv.org/abs/2403.07059.

39. Their results support checking whether a circuit component adds value.

Support: Own study description or bounded interpretation of the adjacent source. No claim that previous work lacks these controls.

40. Huang et al. showed that access to data can allow classical learners to compete even in some settings built around quantum problems [10].

Support: [10] Huang HY, Broughton M, Mohseni M, Babbush R, Boixo S, Neven H, et al. power of data in quantum machine learning. Nature Communications. 2021;12:2631. doi:10.1038/s41467-021-22539-9.

41. Neither paper predicts the result of our PCam experiment.

Support: Own study description or bounded interpretation of the adjacent source. No claim that previous work lacks these controls.

42. Singh et al. studied MedMNIST models trained on classical hardware and evaluated on real quantum hardware [16].

Support: [16] Singh G, Jin H, Merz KM Jr. Benchmarking MedMNIST dataset on real quantum hardware. Scientific Reports. 2026;16:9017. doi:10.1038/s41598-026-35605-3.

43. Our exact simulation does not test device noise, finite shots or hardware speed.

Support: Own study description or bounded interpretation of the adjacent source. No claim that previous work lacks these controls.

44. These studies address related but different questions.

Support: Own study description or bounded interpretation of the adjacent source. No claim that previous work lacks these controls.

45. Their reported accuracies cannot be ranked against our AUROC without matching data, splits and evaluation rules.

Support: Own study description or bounded interpretation of the adjacent source. No claim that previous work lacks these controls.

46. We add a focused PCam result using shared features, the full official training partition, a reference chosen within training data and uncertainty paired by slide.

Support: Own study description or bounded interpretation of the adjacent source. No claim that previous work lacks these controls.

47. The contribution is the evidence from this controlled setting and its saved outputs.

Support: Own study description or bounded interpretation of the adjacent source. No claim that previous work lacks these controls.

48. We do not claim the first quantum PCam classifier or the first CNOT ablation.

Support: Own study description or bounded interpretation of the adjacent source. No claim that previous work lacks these controls.

## Source access and limits

Primary publisher or author records were checked for all 24 entries. Detailed relevant methods and results were inspected for the closest studies, including Mari, Bowles, Goyal, Baral and Singh. Baral explicitly includes a classical bottleneck and three seed evaluation of the selected hybrid model. The paper does not claim these as new contributions.

For foundational methods, sources were checked for the specific cited idea and bibliographic identity. Some access was limited to an abstract or bibliographic record, especially Youden and Obuchowski. Efron and Holm publisher pages did not expose useful text during this retrieval. Efron's bibliographic record and author retrospective, and Holm's archived original paper hosted by the University of São Paulo, provided additional checks. These references identify the general method. They do not certify our custom cluster bootstrap or jackknife implementation. External code was not exhaustively audited or reproduced.

Official reference titles, DOI strings and bibliographic punctuation retain their source form. Authored manuscript prose uses simple wording and avoids semicolons and decorative dashes. Mathematical negative signs remain necessary for reporting negative effects.

## Contribution boundary

The contribution is a focused empirical comparison on fixed ResNet50 features, full official training rows, training selected controls, protected test access, paired slide uncertainty and reusable predictions. Quantum transfer learning, PCam hybrid classification, classical bottleneck controls, repeated seeds and circuits without entangling gates have prior work. This revision makes no first use or state of the art claim.
