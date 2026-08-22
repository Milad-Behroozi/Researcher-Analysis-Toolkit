# Provenance — EIS datasets

## Synthetic EIS benchmark

The synthetic EIS datasets were generated using `generate_synthetic_eis.py` and were not obtained experimentally. They were prepared to evaluate user-defined equivalent-circuit fitting, automatic topology ranking, and fitting behavior at controlled noise levels for which the generating circuit and parameter values were known.

Three equivalent-circuit configurations were used:

* `R0-p(R1,CPE1)`, representing a single distributed relaxation;
* `R0-p(R1,CPE1)-p(R2,CPE2)`, representing two distributed relaxation processes;
* `R0-p(R1,CPE1)-W1`, representing a distributed relaxation followed by a semi-infinite Warburg contribution.

For every circuit, spectra were calculated at 50 logarithmically spaced frequencies between 10⁶ and 10⁻² Hz in descending order. Separate datasets were generated with 0, 1, 3, and 5% independent Gaussian noise applied to the real and imaginary impedance components. A fixed random-number seed of `20260723` was used to provide reproducible datasets.

The generating circuit, assigned element values, frequency settings, noise definition, and dataset-specific noise fractions are documented in `benchmark_metadata.json`. The files follow the naming convention:

* `EIS_single_relaxation_noiseX.csv`;
* `EIS_two_relaxation_noiseX.csv`;
* `EIS_diffusion_noiseX.csv`;

where `X` denotes the nominal noise percentage.

The complete noise series can be used to evaluate user-defined circuit fitting and the influence of added noise. The following datasets were used for the automatic topology-ranking results reported in Table S4 of the Supporting Information:

* `EIS_single_relaxation_noise0.csv`;
* `EIS_two_relaxation_noise3.csv`;
* `EIS_diffusion_noise5.csv`.

The noisy datasets should be assessed using fit quality, parameter-bound diagnostics, uncertainty and correlation estimates, model complexity, and physical plausibility together. Because these datasets were generated from predefined models within the same software framework, they represent an internal numerical benchmark rather than independent validation of equivalent-circuit identification.

## Experimental EIS dataset

`EIS_cerium_oxide_based_composite_minus0p80V_vs_AgAgCl_1M_KOH.csv` contains the experimental impedance spectrum used for Figure 3 and Table 2 of the main article. The measurement was performed for a cerium oxide–based composite electrode in 1.0 M KOH at −0.80 V versus Ag/AgCl using a sinusoidal perturbation amplitude of 5 mV.

Only data points containing complete numerical values for both the real and imaginary impedance components were retained, and no interpolation was applied. The processed dataset contains 87 complete frequency points spanning 9.99 × 10⁵ to 6.813 × 10⁻² Hz.

This experimental dataset is provided as a representative application of candidate-circuit fitting and diagnostic comparison. Unlike the synthetic datasets, it does not have a predefined ground-truth equivalent circuit.
