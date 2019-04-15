# DFT-LR-Bayesian-Regression-Python3
A bayesian regression using molecular dataset based on the (F. A. Faber et al., J. Chem. Theory Comput 13, 5255-5264 2017.) literature

# SupplementaryMaterials dataset is required for this program to run. The link I used was recently made unavailable.
*** Possible to find dataset (qm9) from this article: Ramakrishnan, Raghunathan; Dral, Pavlo; Dral, Pavlo O.; Rupp, Matthias; Anatole von Lilienfeld, O. (2017): Quantum chemistry structures and properties of 134 kilo molecules. figshare. Collection ***

# Background
Bayesian Regression model is essentially a lienar model with a L2 penalty on the coefficients. Unlike ridge regression where the key to that penalty is a regularization hyperparameter which must be set, in BR the optimal regularizer is estimated from the data. In this case our CM vector or Coulomb Matrix is a 2D matrix. Each row or col represents an atom. The composition of the matrix is relatively simple, the Coulomb classic mutual exclusion between every two atoms. So then we are left with:

  M_i,j = 0.5*Z^(2.4)                for I = J
          Z_I * Z_J / abs(R_I - M_J) for I != J
          
Here off-diagonal elements correspond to the Coulomb repulsion between atoms I and J, while diagonal elements encode a polynomial fit of atomic energies to nuclear charge. (M. Rupp et al., Phys. Rev. Lett. 108, 058301, 2012.)

I used SK_learn algorithms for defining training and test sets as well as performing the Bayesian Ridge Regression on the CM vector. Code is commented generally in order establish where/how I split the data and apply skleearn and numpy libraries.

# Results
My Bayesian Regression data was run for all 13 properties of the CM feature vector generated from feature.py file. My results compare my results to the Faber et al. paper where my inspiration came from to run this program. The error of my results does not exceed 3% which is very good, and most likely means I ran my program correctly. The CM-BR.txt file included in this repository is available for analysis, as well as a picture ResultsBayesRegression for easy understanding of waht is being calculated in the .txt file.

Overall, my Master's degree research involves DFT so this was a fun side project to run and see how close regression techniques could come to accurately determine DFT calculations done using traditional DFT methods.
