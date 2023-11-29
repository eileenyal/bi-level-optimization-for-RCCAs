# bi-level-optimization-for-RCCAs

### DISCLAIMER: USE AT YOUR OWN RISK. THERE IS NO IMPLIED WARRANTY WHATSOEVER!
## Overview
This repository contains a collection of MATLAB scripts for analyzing various properties of Refractory Complex Concentrated Alloys (RCCAs). Addtionally, it includes a MATLAB script for building Back-propagation Neural Network(BPNN) and a collection of python scripts for Machine Learning(ML) Model, including gradient boosting regressor (GBR), multi-layer perceptron regressor (MLP), support vector regression (SVR), linear support vector regression (Linear SVR), k-nearest neighbor regressor (KNN), histogram gradient boosting regression (HGBoost), random forest regression (RF), bootstrap aggregating regression (Bagging), adaptive boosting regression (AdaBoost). Furthermore, bi-level optimization and pareto optimization, applied for obtaining the optimum alloy composition, are also included.
## parameters calculation
we need use the formula to calculate these paramters. The parameters cover a range of topics, including atomic size difference, local size mismatch, atomic size parameter γ, enthalpy of mixing of the alloy, configurational entropy of mixing , average melting point, comprehensive effect of the melting point difference, solid-solution formation parameter Ω, shear modulus difference, local modulus mismatch, modulus mismatch parameter η, valence electron concentration (VEC), average number of itinerant electrons per atom(e⁄a), electronegativity difference, local electronegativity mismatch, average energy cohesive (Brewer) , disordered solid solution formation parameter Λ, lattice distortion energy, energy factor, Painalgin factor, sixth power of work function, density, Vickers hardness.
## model selection
we need use pymodels and bpnn to choose the most suitable ML models.
## feature selection
we need use the best ML models with all possible feature inputs to select the best feature set.
## bi-level optimization
we need use the best ML model with the best feature set as the mapping function to implement the mutiple obejective optimization through NSGAII in PlateEMO. And then we should define the objective funtion to search optimum properties with Pareto solutions and utilize the GA to search the desired composition  that approximates the Pareto solutions. 
## Notes
This repository is provided "as is," and there are no guarantees or warranties associated with the code.
Users are advised to exercise caution and thoroughly review the scripts before using them.
The repository is actively maintained, and updates may be available to enhance or improve the scripts.
