# Research plan

## Question and controls

Does an eight wire simulated quantum head improve patch classification on the same compressed frozen ResNet50 features used by classical heads?

I compare standard and arctangent MLP inputs to separate the mapping effect. I compare the same two layer RY circuit with CNOT off and on to isolate the gate effect. A training selected classical reference gives another comparator.

## Final scope

Use all 262,144 official training patches. Validation and test each have 32,768 patches. Keep the encoder frozen. Fit scaling and PCA on training data only. Final neural fits use eight components and seeds 101, 202 and 303.

Model selection uses 16,384 training rows in three slide separated folds. The shared neural learning rate is selected from 0.001, 0.003 and 0.01. The retained rate is 0.003. Neural heads share Adam, 100 epochs and batch size 128. Classical selection has a different candidate and optimization budget.

Threshold selection uses 17,090 validation rows. Diagnostic validation uses 15,678 different rows, separated by slide. Fitted artifacts and comparisons are locked before test access.

The primary contrast is mean seed AUROC of CNOT on minus the selected classical reference. Four secondary contrasts examine CNOT on versus off and both MLPs, and arctangent versus standard MLP. Use 20,000 paired whole slide bootstrap draws across 129 test slides. Adjust secondary intervals for four comparisons. Report seed SD separately. Measure only elapsed training compute with the retained timing boundary.

## Order

1. Check software and encoder fixtures.
2. Audit official files and slide metadata.
3. Extract training and validation features.
4. Complete training fold selection.
5. Fit final heads and save predictions.
6. Select thresholds and verify diagnostic validation.
7. Lock fitted artifacts and comparisons.
8. Extract test features and score fixed models.
9. Calculate metrics, paired intervals, seed variation and elapsed time.
10. Preserve arrays, models, configuration, logs and hashes.

The earlier broader scope was reduced for resources before official test scoring. This is a recorded amendment, not preregistration. Later circuit checks, pixel audit and jackknife tests are post hoc checks or sensitivities. The negative result is retained.

`final_scope.json` records the completed comparison. `selection_protocol.json` retains historical selection definitions, including dimensions and seeds considered before the reduction. The final scope governs interpretation.
