# Methodology

Official patches are resized to 224 by 224 and use the retained pretrained preprocessing. A frozen ImageNet ResNet50 produces 2,048 features. Training fitted scaling and eight component PCA define the shared representation. The final PCA uses all training rows. Fold transforms use only their training subsets.

Two MLPs compare standard and arctangent inputs. Two simulated circuits compare CNOT off and on. Each circuit has eight wires, input RY encoding, two trainable RY layers and an ordered neighboring CNOT chain when enabled. A linear readout uses eight Pauli Z expectations. MLPs have 321 parameters and circuits have 25. Paired circuits share initial rotations and readout within each seed.

Simulation uses real state vectors and PyTorch gradients. It is exact and noiseless. No quantum device, measurement shots or shot randomness is used. The classical reference is a training selected Fourier Nystrom map with 256 landmarks and a logistic SGD readout.

Neural fitting uses three seeds and a common 100 epoch Adam schedule at learning rate 0.003, batch size 128. Thresholds use dedicated validation slides. Threshold selection is not probability calibration. Fixed artifacts are locked before test scoring.

Test reporting includes AUROC, average precision, probability error and threshold based metrics on the same rows. Paired slide resampling and separate seed SD describe different uncertainty. Pixel overlap exclusion and slide jackknife are later sensitivity checks.

`workflow.drawio` is editable. SVG, PNG and 400 DPI TIFF exports describe the executed reduced scope.
