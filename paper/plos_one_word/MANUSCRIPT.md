# A comparison of simulated quantum and classical classification heads on frozen ResNet50 features for PatchCamelyon

Saikat Das

Department of Computer Science and Engineering, Ahsanullah University of Science and Technology, Dhaka, Bangladesh

Corresponding author: Saikat Das. Email: saikatdasmain47@gmail.com

## Abstract

Hybrid quantum classical image classifiers require controls that distinguish the contribution of a quantum circuit from that of the underlying image representation. We evaluated whether a small simulated quantum classifier improved patch level metastatic tissue classification under a fixed eight component representation and a shared neural training schedule. An ImageNet pretrained ResNet50 encoder was frozen, and its 2,048-dimensional features were reduced to eight training fitted principal components. We compared standard and arctangent input multilayer perceptrons, two layer rotation circuits with and without controlled NOT gates, and a classical reference selected using slide separated training folds. Final neural heads were fitted using three initialization seeds and all 262,144 official training patches. Models and analysis settings were locked before scoring the 32,768-patch official test set. Uncertainty was estimated using 20,000 paired whole slide bootstrap resamples across 129 test slides. Mean test area under the receiver operating characteristic curve was 0.8657 for the controlled NOT circuit and 0.8776 for the selected Fourier Nyström reference, a difference of −0.01191 (conditional 95% interval −0.02359 to −0.00038). The controlled NOT versus no controlled NOT difference was −0.00848. Its interval after adjustment for multiple comparisons included zero. A post hoc slide jackknife sensitivity test for the primary comparison gave p=0.0526, showing that exclusion of zero depended on the uncertainty method. These results provide no evidence of a quantum head improvement in the evaluated configuration. The eight components retained 58.80% of training feature variance. Conclusions concern this compressed representation, one dataset and exact classical simulation.

## Introduction

PatchCamelyon (PCam) provides small tissue images for testing models that detect metastatic tissue [1,2]. Its fixed training, validation and test partitions allow studies to use a common benchmark. A positive label means tumor is present in the central 32 by 32 pixels [2]. Patch performance does not establish performance for whole slides or patients.

A pretrained image encoder can provide features for a smaller classifier. ResNet introduced residual connections for image recognition [3]. Work on feature transfer showed that transfer depends on the source task and network layer [4]. Mari et al. extended transfer learning to models that place a quantum circuit after a pretrained classical network [5]. This approach makes small circuit experiments possible without encoding every image pixel into qubits. It also raises a direct question. Does the circuit improve predictions once the image features are fixed?

Reviews of quantum learning describe both possible uses and limits in training and evaluation [6,7]. Circuit output depends on how data enter the circuit [8]. Some circuit families also have very small gradients, although this cannot be assumed for every small model [9]. Classical models can remain competitive even on some tasks designed around quantum features [10]. Broad benchmark results further show why useful classical controls and circuits without entangling gates should be tested [11]. These findings motivate a controlled comparison rather than an assumed quantum benefit.

The evaluation also needs to separate model selection from final testing [12]. Patches from one slide can be related, so uncertainty should account for slide groups [13]. Here we compare four small heads and a classical reference on the same eight features from a frozen ResNet50. We retain the official training rows, choose the reference within training data and lock the fitted models before test scoring. Our contribution is the resulting comparison of input mapping and CNOT gates, with paired slide uncertainty, separate seed variation and saved predictions. In this pathology setting, many patches share a slide. We therefore examine whether apparent head differences remain meaningful after pairing uncertainty by slide. The question concerns an eight component constraint and a shared schedule.

## Related work

Mari et al. established the use of pretrained classical features with a small trainable quantum head [5]. This is the starting point of our study, not a new architecture. Work on data encoding further links circuit outputs to Fourier functions of the inputs [8]. This gives a reason to include a classical trigonometric reference alongside neural controls. It does not imply that our reference can reproduce every circuit function.

Two studies are particularly close to the pathology task. Goyal et al. studied quantum edge and frequency preprocessing with image classifiers and fusion on a modified PCam dataset [14]. Their intervention changes image processing. Baral et al. studied classical and hybrid classifiers under adversarial attacks [15]. They used a balanced subset of 20,000 PCam images and an 80:20 split. They also included a classical bottleneck model and evaluated their selected hybrid model with three random seeds. Classical controls and repeated seeds therefore already exist in this literature. Our experiment instead holds the frozen representation fixed and uses the official partitions for a direct comparison of classifier heads.

Bowles et al. compared quantum and classical models across benchmark tasks and examined circuits without entanglement [11]. Their results support checking whether a circuit component adds value. Huang et al. showed that access to data can allow classical learners to compete even in some settings built around quantum problems [10]. Neither paper predicts the result of our PCam experiment. Singh et al. studied MedMNIST with classical training and real device inference [16]. Goyal et al. later proposed configurable quantum edge processing for histopathology and evaluated circuits on real processors [17]. This June 2026 preprint studies image preprocessing and hardware behavior. Our head comparison uses exact simulation.

These studies address related but different questions. Their reported accuracies cannot be ranked against our AUROC without matching data, splits and evaluation rules. We add a focused PCam result using shared features, the full official training partition, a reference chosen within training data and uncertainty paired by slide. This study connects head comparisons to slide dependence, shared schedule effects and exact image overlap in an official pathology benchmark. It does not introduce a new architecture or claim the first quantum PCam classifier.

## Materials and methods

### Study design and executed scope

This computer study used existing public images. The effective final scope was one frozen ResNet50 encoder, eight transformed features, four neural heads and seeds 101, 202 and 303. A documented resource driven amendment reduced an earlier plan containing additional backbones, dimensions, seeds and controls before reported official test scoring. Historical source configurations retain that larger plan. The executed SCOPE.json and final specifications define the reported experiment. The test lock is an internal timestamped artifact lock, not external preregistration. Development results were available before final test evaluation. No test driven model winner was selected. Fig 1 summarizes the executed study workflow.

### Dataset and split handling

We used the official training, validation and test partitions and retained their repeated rows (Table 1). Images were 96 by 96 RGB patches. A positive PCam label indicates at least one tumor pixel in the central 32 by 32 region [1,2]. Whole slide identifiers did not overlap between partitions. A post hoc audit hashed every original patch and checked raw label order against the saved manifests. There were 220,025 unique training images, 28,108 unique validation images and 29,383 unique test images. No exact images were shared between training and validation or between validation and test. Training and test shared 33 image hashes, involving 35 training rows and 34 test rows. We retained the locked full test analysis and added a separate descriptive sensitivity excluding those 34 test rows. This does not establish absence of near duplicates or patient overlap.

**Table 1. Official dataset partitions.** Counts are rows and distinct WSI identifiers, not patients.

| Split | Patches | Negative | Positive | WSIs |
| --- | --- | --- | --- | --- |
| train | 262144 | 131072 | 131072 | 216 |
| valid | 32768 | 16399 | 16369 | 54 |
| test | 32768 | 16391 | 16377 | 129 |

Training only selection used a fixed, label stratified 16,384-row subset. Three StratifiedGroupKFold partitions used WSI groups, with sample and fold seeds 71911 and 71912. Validation slides were divided using seed 71914 into 27 threshold selection slides (17,090 patches) and 27 diagnostic slides (15,678 patches). Reported development metrics use the diagnostic half. The entire official test partition was scored after lock. Dataset size was determined by the benchmark, not a prospective power calculation. No retrospective power claim is made.

### Frozen representation and transformations

The encoder was the pinned ImageNet ResNet50 variant resnet50.tv_in1k, with 23,508,032 frozen parameters and a saved weight state checksum. Images were resized directly to 224×224 with bilinear antialiased interpolation, converted to tensors and normalized using means (0.485, 0.456, 0.406) and standard deviations (0.229, 0.224, 0.225). No random augmentation, crop, stain normalization or encoder fine tuning was used. Global average pooled outputs had 2,048 dimensions. Feature chunks and per image identifiers were shared across heads.

For each training fold, randomized principal component analysis [18] with eight components, 24 oversamples, seven power iterations and seed 71911 was fitted exclusively to that fold's fitting rows. Each principal component was standardized using fitting row statistics. The final transform was refitted on all official training rows. Held out data were transformed without refitting. The standard MLP received z. The arctangent MLP and both RY circuits received a=arctan(z), component by component, without an extra scaling factor. Eight components were a resource constraint that allowed the declared circuit comparison. They were not chosen because eight was proven sufficient. The saved final PCA retained 58.8005% of training feature variance. The three fitting fold transforms retained 58.9259%, 58.7760% and 58.7407% (S3 Table). Explained variance does not measure retained information about the tissue label. A full feature classical baseline was not part of the locked experiment, so the predictive cost of compression remains unknown.

### Neural models and circuit ablation

Both perceptrons used an 8→32→1 architecture with a hyperbolic tangent hidden activation and affine output, totaling 321 trainable parameters. They differed only in standard versus arctangent input mapping. Circuits encoded each component with an RY rotation from the all zero state, followed by two trainable RY layers. The CNOT on model applied an ordered nearest neighbor chain, wires 0→1 through 6→7, after each trainable layer. CNOT off omitted those gates. Eight Pauli Z expectation values fed an affine scalar readout. Each circuit had 16 rotation parameters and nine readout parameters, totaling 25. Off/on models shared initial rotation and readout values for a given seed. The comparison isolates these gates within the declared circuit and optimizer. It does not measure entanglement itself or match capacity to the perceptron.

The simulator represented an exact real 256-amplitude statevector in PyTorch float64. There were no finite shots, device noise, quantum hardware measurements or stochastic shot effects. With no CNOT gates, successive rotations on a wire combine, making this control a classically evaluable trigonometric model. It is therefore a structural circuit ablation. With CNOT off, the logit is b+sum over j of w_j cos(a_j+theta_1j+theta_2j).

After test inspection, a separate NumPy implementation constructed full Kronecker rotation matrices and computational basis CNOT matrices. Across four and eight wires, both gate settings and six cases per setting, expectations agreed within 1.23e-15. PyTorch gradients for all input angles and rotation weights agreed with central finite differences of the matrix reference within 1.89e-9, using step 1e-6. Wire zero was the most significant bit. CNOT control and target order were checked explicitly. The gate free cosine expression agreed within 8.89e-16. These checks validate the retained circuit function. They are not an external replication or quantum device test. Full checks and tolerances are in Supplementary methods and S6 Table.

### Analytical computational complexity

For batch size B, q wires and L layers, this statevector implementation uses O(B·2^q) state storage and O(B·L·q·2^q) gate operations per forward pass, with additional automatic differentiation storage during training. At q=8 it tracks 256 real amplitudes per patch. An 8→32→1 MLP instead has O(B·q·32) leading dense layer work. These are implementation level scaling descriptions, not measured FLOP counts. The gate free RY function could be evaluated directly with O(B·q) trigonometric operations. The present off implementation retains the common statevector path. Its two rotation weights on each wire enter only through their sum, so nominal parameter count does not imply independently identifiable capacity. Neither parameter count nor simulation time establishes quantum advantage.

### Optimization and training only selection

All neural heads used the same training loop, binary cross entropy with logits, Adam [19] (betas 0.9 and 0.999, epsilon 10^−8, zero weight decay), batch size 128 and 100 epochs. All neural head fits used CPU float64 with two PyTorch threads. A common learning rate of 0.003 was selected from 0.001, 0.003 and 0.01 by averaging fold AUROCs across the four heads. This experiment compares a shared schedule. It does not compare the best schedule for each architecture. Deterministic epoch permutations used seed + 10,000×epoch, with epoch indexed from zero. Final runs each performed 204,800 updates. There was no validation based early stopping or best epoch selection. Rotation weights began as independent normal values with mean zero and SD 0.1. Circuit readout weights and bias used the PyTorch Linear uniform range from minus 1/sqrt(8) to plus 1/sqrt(8). MLP layers used the same Linear rule with the appropriate input width. The CV initialization seed was 71911. Final seeds were 101, 202 and 303. CNOT pairs shared initial values. S4 Table and the accompanying CSV give all existing head and learning rate fold results. Parameter initialization rules and reconstructed values are supplied with the code, with their reconstruction runtime identified.

Fourteen nonneural classical candidates comprised six logistic regressions, three ExtraTrees variants, three histogram gradient boosting variants and two Fourier Nyström variants. Standard and arctangent MLPs at the selected common learning rate also entered the classical reference selection pool. The winning reference was chosen by mean training fold AUROC, with a deterministic tie rule. Its training fold mean AUROC was 0.884710. It used the arctangent mapping, 256 training sampled Nyström landmarks [20] and the kernel K(a,b)=∏_j{[1+cos(a_j−b_j)]/2} over eight coordinates. Eigenvalues were floored at 10^−12. An SGD logistic classifier used alpha 0.0001, maximum 1,000 iterations, tolerance 10^−4 and seed 71911. The reference was refitted on all training rows. Its optimization and feature map budget differ from those of neural heads.

### Threshold selection and test access

Each final model's probability threshold maximized Youden's index [21] on threshold selection slides. Ties used proximity to 0.5, then the smaller threshold. This procedure selects an operating threshold. It is not probabilistic calibration. No Platt or isotonic calibration was fitted. Diagnostic validation rows did not determine these thresholds. Final models, transforms, selection, validation outputs and scientific code were checksummed before test scoring.

Saved checkpoints allowed interrupted runs to continue. Recovery details and numerical replay checks are in Supplementary methods. The internal lock protected fitted models, transforms, selection and thresholds before test scoring. It does not establish independent replication.

### Use of artificial intelligence tools

OpenAI Codex assisted with language editing, manuscript organization, code development and debugging, and analysis and figure scripts. File hashes, metric recomputation and circuit validation were used to check outputs. The author is responsible for the work.

### Outcomes and statistical analysis

The primary outcome was the difference between mean per seed test AUROC of CNOT on and the training selected classical reference. This is the mean of seed specific AUROCs, not the AUROC of an averaged score ensemble. Secondary contrasts were CNOT on minus off, CNOT on minus arctangent MLP, CNOT on minus standard MLP, and arctangent minus standard MLP. ROC and precision recall plots describe performance across thresholds [22]. Average precision summarized the ranking of predicted scores. Brier score and log loss evaluated predicted probabilities. Accuracy, sensitivity, specificity, precision and F1 used thresholds fixed on the threshold selection validation slides.

Within slide dependence motivates cluster aware ROC inference [13]. We do not claim to implement the specific estimator in that reference. Bootstrap resampling follows the general approach of Efron [23], with slides as the resampling units here. The locked analysis used 20,000 paired WSI bootstrap replicates (seed 71913). Each replicate resampled 129 slides with replacement and included all patches with their slide's multiplicity. Identical resampling weights were applied to every model and seed. AUROC was patch weighted within each resample, rather than an unweighted mean of slide specific AUROCs. Percentile intervals used the 2.5th and 97.5th percentiles for the primary contrast. Four secondary comparisons used Bonferroni percentile limits at 0.625% and 99.375%, giving nominal family wise 95% coverage. Training seed sample SD was reported separately. These intervals condition on the fitted models, training data, selection and three retained seeds. They do not include variability from retraining on new data or establish patient independence.

For post hoc sensitivity, we deleted each WSI in turn, recomputed model AUROCs and estimated the contrast standard error as sqrt[(G−1)/G Σ(θ_(−g)−mean θ_(−g))²], G=129. We used approximate two sided t tests with 128 degrees of freedom and Holm correction [24] across the four secondary sensitivity tests. This jackknife approximation was not the locked primary test. We did not run patch independent DeLong, patch level McNemar or t tests treating three seeds as independent patient samples. No equivalence margin or smallest important effect was specified. All inferential results are interpreted without post hoc power claims.

Analyses used the saved Python 3.13/PyTorch 2.11/scikit learn 1.6.1/SciPy 1.16.3/NumPy 2.1.3 environment for the experiment. Manuscript reanalysis software versions are recorded separately. Scikit learn provides the classical analysis tools [25]. Sources and exact configuration hashes accompany predictions. No rows were excluded as outliers from the locked test analysis. The exact overlap exclusion and slide influence checks were later descriptive analyses. Overlap between individual model intervals does not test a paired model difference. Inconclusive secondary contrasts do not establish equivalence.

## Results

### Integrity and completeness

A separate analysis script in the same project recomputed all 13 test AUROC and average precision values from saved predictions. It also reconstructed the five nominal contrast intervals using the saved 20,000 bootstrap arrays. A direct weighted AUROC calculation from original scores matched the first 32 saved draws per model. The full resampling procedure was not independently repeated. These are computational consistency checks by the same workflow, not validation by another researcher. The retained circuit passed the separate matrix and gradient tests described above. File integrity counts and checks are in Supplementary methods.

### Test discrimination and operating characteristics

The selected classical reference achieved AUROC 0.8776. Neural mean AUROCs ranged from 0.8657 for CNOT on to 0.8765 for the standard MLP (Table 2). Seed variability and conditional test set uncertainty describe different sources of variation and are displayed separately. Table 2 reports mean performance across seeds. It does not describe a pooled ensemble. S1 Table gives operating metrics and S2 Table gives exact confusion counts. Figs 2 and 3 show the ROC and precision recall curves.

**Table 2. Official test performance.** Neural results are means across three fixed seeds. The selected classical reference is one deterministic fitted candidate. AUROC intervals use paired WSI draws averaged across the declared seeds. AP denotes average precision. Threshold dependent metrics use threshold selection slide thresholds.

| Head | Fits | AUROC mean (seed SD) | Conditional 95% CI | AP | Brier |
| --- | --- | --- | --- | --- | --- |
| Standard MLP | 3 | 0.8765 (0.0016) | 0.8458 to 0.9055 | 0.8784 | 0.1523 |
| Arctangent MLP | 3 | 0.8729 (0.0015) | 0.8407 to 0.9032 | 0.8751 | 0.1524 |
| RY, CNOT off | 3 | 0.8742 (0.0012) | 0.8453 to 0.9004 | 0.8671 | 0.1463 |
| RY, CNOT on | 3 | 0.8657 (0.0020) | 0.8342 to 0.8946 | 0.8649 | 0.1502 |
| Selected Fourier Nyström reference | 1 | 0.8776 | 0.8486 to 0.9046 | 0.8743 | 0.1459 |

### Primary comparison and ablations

The primary AUROC difference was −0.011911 (95% conditional percentile interval −0.023594 to −0.000379). Thus the locked bootstrap comparison favored the classical reference, but its upper limit lay close to zero. Post hoc jackknife sensitivity gave t=-1.956, p=0.0526. Exact computed values are in Table 4. The estimated difference favored the classical reference. Exclusion of zero depended on the uncertainty procedure. These procedures need not agree near zero. We interpret the effect and its uncertainty without treating a p value threshold as a separate scientific conclusion.

The CNOT on/off difference was −0.008475. Its nominal interval excluded zero, but the four comparison adjusted interval included zero (Table 3 and Fig 4). The other adjusted secondary intervals also included zero. Removing CNOT gates did not provide a family adjusted conclusive gain. Equally, the results do not support adding them in this configuration. Differences between MLP mappings are not evidence that dimensional compression itself is harmless, because a full dimensional control was not executed in the final scope.

**Table 3. Locked AUROC contrasts.** Differences are AUROC units. Four secondary intervals are Bonferroni adjusted, not unadjusted significance claims.

| Contrast | AUROC difference | Nominal 95% CI | Family adjusted 95% CI |
| --- | --- | --- | --- |
| on selected | -0.011911 | [-0.023594, -0.000379] | Primary comparison |
| on off | -0.008475 | [-0.016990, -0.000628] | [-0.019470, 0.001427] |
| on atan | -0.007233 | [-0.025644, 0.014327] | [-0.029973, 0.022183] |
| on standard | -0.010747 | [-0.028909, 0.010240] | [-0.033098, 0.017242] |
| atan standard | -0.003514 | [-0.006798, -0.000352] | [-0.007650, 0.000600] |

### Training diagnostics and elapsed time

Mean neural training compute times were 268.65 seconds for standard MLP, 266.63 seconds for arctangent MLP, 5,447.13 seconds for CNOT off and 7,374.70 seconds for CNOT on. The descriptive CNOT on/standard MLP ratio was approximately 27.45. The classical reference fit took 31.57 seconds, including its feature map fit, and is not on the same timing boundary. Neural timers include mini batch indexing, forward/backward passes and optimizer updates, and exclude setup, epoch order generation, diagnostics, checkpoint writes and frozen feature extraction. Runs span Colab sessions and some checkpoints contain multiple runtime segments. These times characterize observed classical execution, not standardized hardware speed comparisons or quantum hardware cost.

Training diagnostic losses used the first 1,024 training rows. Before test access, a fit was blocked if its epoch 81 loss minus epoch 100 loss exceeded 0.02. All twelve fits passed this rule. Fig 5 shows the saved training traces and observed neural compute times. A small recent decline does not prove convergence. Circuit losses remained higher than MLP losses. The traces cannot separate limited capacity from unsuitable optimization. The existing CV grid also shows that 0.001 gave the highest mean fold AUROC for both MLPs, whereas 0.01 did so for both circuits (S4 Table). The common selected rate remained 0.003. No new schedule was selected from test results. Full traces, update counts and measured timing are retained in the analysis files.

### Image overlap and slide influence

The exact audit found 3,385 repeated test rows beyond the first occurrence of each image. The 34 test rows whose hashes also occurred in training contained 33 positive labels and one negative label. Excluding these rows left 32,734 test patches. The mean CNOT on AUROC was then 0.865606 and the classical AUROC was 0.877485. Their difference was minus 0.011879, compared with minus 0.011911 in the locked analysis. This post hoc point estimate does not add an interval or replace the original analysis.

Test slides contributed between 98 and 716 patches, with median 140 and quartiles 120 and 387. The largest slide contributed 2.19% of test rows and the largest five contributed 10.31%. Removing one slide at a time gave primary differences from minus 0.015786 to minus 0.009775. The largest change arose on removing camelyon16_test_099, which contributed 677 patches. All leave one slide point estimates favored the classical reference. These descriptive checks do not quantify patient independence or remove all sources of bias. Counts and influence values are supplied in S5 and S7 Tables.

## Discussion

Within this fixed representation and optimization schedule, a simulated entangling head did not improve discrimination over the selected classical reference. The primary point estimate was negative, consistent with the broader benchmarking concern that a quantum component should be justified against classical alternatives [11]. The intervals concern these retained fitted models and seeds. They do not measure variation from repeating the full training and selection process. No clinically important margin was defined. The point estimate favored the classical reference, while exclusion of zero depended on the uncertainty procedure.

The off/on intervention changes CNOT connectivity while retaining initialization, readout size, optimizer, batch order and update count. Its adjusted interval includes zero, so the study does not establish a general causal penalty from entanglement. A shared schedule improves control over computational treatment but may disadvantage a model whose optimal schedule differs. The selected classical pool also has different optimization budgets. These features define the experiment and should not be concealed as fully equal hyperparameter optimization.

The study differs from quantum image preprocessing on a modified PCam subset [14]. Here the image representation is held fixed, whereas preprocessing studies alter the information reaching the image model. Reported accuracies from such work are contextual evidence, not a head to head SOTA comparison. The present experiment cannot claim a new best PCam classifier. Its empirical value is the controlled intervention, official row coverage, cluster aware paired uncertainty and reusable prediction evidence. The setting adds two practical lessons. Shared image features do not remove dependence among patches from one slide. Official split names also do not guarantee that every image is pixel distinct across training and test. In this case, excluding the identified overlap changed the point difference little, while the shared schedule and eight component constraint still limit what can be concluded about model capability.

Limitations include one dataset and one frozen ImageNet representation, severe eight component compression, three neural seeds, unequal parameter counts between MLP and circuit heads, no independent external cohort, no patient identifiers, repeated native patches, the observed exact train and test overlap, no exhaustive near duplicate audit and approximate conditional WSI inference. The seed SD does not capture training set or selection uncertainty. Exact noiseless simulation cannot predict finite shot or noisy hardware behavior. Resource driven scope changes and recovery history must be disclosed. The work evaluates patch classification, not slide diagnosis, patient prognosis, clinical deployment or quantum speedup.

## Conclusions

For official PatchCamelyon patches represented by eight training transformed frozen ResNet50 features, the tested simulated CNOT on head showed no demonstrated benefit over a training selected classical reference. Its mean test AUROC was lower. Exclusion of zero depended on the uncertainty procedure. The post hoc exact overlap exclusion left a similar point difference. Adjusted circuit and mapping ablations were inconclusive. The results support reporting a bounded controlled evaluation rather than a quantum advantage claim.

## Data and code availability

Original PCam data are available at https://github.com/basveeling/pcam#download and the maintainers' mirror at https://zenodo.org/records/2546921 [2]. Study code and derived evidence are prepared for release at https://github.com/saikatdas47/PCam-Quantum-Git. The release includes sources, configurations, split and slide IDs, fitted transforms and models, predictions, bootstrap arrays, analysis scripts, figures, tables, environment records and checksums. Upload is pending. Raw images are obtained separately from PCam maintainers. Complete frozen feature arrays must be regenerated for new training.

## Declarations

Funding: The author received no specific funding for this work.

Competing interests: The author declares no competing interests.

Author contributions: Saikat Das: Conceptualization, methodology, software, validation, formal analysis, investigation, resources, data curation, visualization, project administration, writing of the original draft, and review and editing of the manuscript.

Ethics approval and consent to participate: Not applicable. This secondary analysis used publicly available PCam images and metadata. No participants were recruited and no new human tissue was collected. No new ethics approval was obtained.

## References

1. Veeling BS, Linmans J, Winkens J, Cohen T, Welling M. Rotation equivariant CNNs for digital pathology. Medical Image Computing and Computer Assisted Intervention. MICCAI 2018. LNCS 11071. pp. 210–218. doi:10.1007/978-3-030-00934-2_24.
2. Veeling BS and PCam contributors. PatchCamelyon dataset repository. https://github.com/basveeling/pcam. Accessed 24 October 2026.
3. He K, Zhang X, Ren S, Sun J. Deep residual learning for image recognition. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016. pp. 770–778. https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html.
4. Yosinski J, Clune J, Bengio Y, Lipson H. How transferable are features in deep neural networks? Advances in Neural Information Processing Systems. 2014. https://arxiv.org/abs/1411.1792.
5. Mari A, Bromley TR, Izaac J, Schuld M, Killoran N. Transfer learning in hybrid classical-quantum neural networks. Quantum. 2020;4:340. doi:10.22331/q-2020-10-09-340.
6. Cerezo M, Arrasmith A, Babbush R, Benjamin SC, Endo S, Fujii K, et al. Variational quantum algorithms. Nature Reviews Physics. 2021;3:625–644. doi:10.1038/s42254-021-00348-9.
7. Cerezo M, Verdon G, Huang HY, Cincio L, Coles PJ. Challenges and opportunities in quantum machine learning. Nature Computational Science. 2022;2:567–576. doi:10.1038/s43588-022-00311-3.
8. Schuld M, Sweke R, Meyer JJ. Effect of data encoding on the expressive power of variational quantum-machine-learning models. Physical Review A. 2021;103:032430. doi:10.1103/PhysRevA.103.032430.
9. McClean JR, Boixo S, Smelyanskiy VN, Babbush R, Neven H. Barren plateaus in quantum neural network training landscapes. Nature Communications. 2018;9:4812. doi:10.1038/s41467-018-07090-4.
10. Huang HY, Broughton M, Mohseni M, Babbush R, Boixo S, Neven H, et al. Power of data in quantum machine learning. Nature Communications. 2021;12:2631. doi:10.1038/s41467-021-22539-9.
11. Bowles J, Ahmed S, Schuld M. Better than classical? The subtle art of benchmarking quantum machine learning models. arXiv:2403.07059v2. 2024. https://arxiv.org/abs/2403.07059.
12. Cawley GC, Talbot NLC. On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation. Journal of Machine Learning Research. 2010;11:2079–2107. https://www.jmlr.org/papers/v11/cawley10a.html.
13. Obuchowski NA. Nonparametric analysis of clustered ROC curve data. Biometrics. 1997;53(2):567–578. doi:10.2307/2533958.
14. Goyal N, Uehara G, Spanias A. Quantum-enhanced cancer detection for histopathologic images. IEEE International Conference on Image Processing. 2025. pp. 1342–1347. doi:10.1109/ICIP55913.2025.11084595.
15. Baral B, Bhalgamiya B, Majumder R, Dutta Roy D, Dutta Roy T. Adversarial attacks on hybrid classical-quantum deep learning models for histopathological cancer detection. APL Machine Learning. 2025;3:036106. doi:10.1063/5.0270673.
16. Singh G, Jin H, Merz KM Jr. Benchmarking MedMNIST dataset on real quantum hardware. Scientific Reports. 2026;16:9017. doi:10.1038/s41598-026-35605-3.
17. Goyal N, Uehara G, Spanias A. Configurable Algorithms for Histopathologic Cancer Detection on Quantum Hardware. arXiv:2606.21752v1. 2026. Preprint. https://arxiv.org/abs/2606.21752.
18. Jolliffe IT, Cadima J. Principal component analysis: a review and recent developments. Philosophical Transactions of the Royal Society A. 2016;374:20150202. doi:10.1098/rsta.2015.0202.
19. Kingma DP, Ba J. Adam: A Method for Stochastic Optimization. ICLR. 2015. https://arxiv.org/abs/1412.6980.
20. Williams CKI, Seeger M. Using the Nyström Method to Speed Up Kernel Machines. Advances in Neural Information Processing Systems 13. 2001. pp. 682–688. https://proceedings.neurips.cc/paper/2000/file/19de10adbaa1b2ee13f77f679fa1483a-Paper.pdf.
21. Youden WJ. Index for rating diagnostic tests. Cancer. 1950;3(1):32–35. https://pubmed.ncbi.nlm.nih.gov/15405679/.
22. Saito T, Rehmsmeier M. The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets. PLOS ONE. 2015;10(3):e0118432. doi:10.1371/journal.pone.0118432.
23. Efron B. Bootstrap Methods: Another Look at the Jackknife. Annals of Statistics. 1979;7(1):1–26. doi:10.1214/aos/1176344552.
24. Holm S. A Simple Sequentially Rejective Multiple Test Procedure. Scandinavian Journal of Statistics. 1979;6(2):65–70. https://www.jstor.org/stable/4615733.
25. Pedregosa F, Varoquaux G, Gramfort A, Michel V, Thirion B, Grisel O, et al. Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research. 2011;12:2825–2830. https://www.jmlr.org/papers/v12/pedregosa11a.html.

## Figure captions

Five figure captions are embedded in the Word manuscript. Fig 1 is the executed methodology. Figs 2 and 3 show ROC and precision recall curves. Fig 4 shows paired contrasts. Fig 5 shows training diagnostics and elapsed compute time. Editable methodology and 400 DPI figure files are in paper/PLOS_ONE_submission/figures.
