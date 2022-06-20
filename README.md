# FAKMCT
Fuzzy Adaptive Resonance Theory K-Means Clustering Technique (FAKMCT)

## Authors

Alfi Nurrahmah

## Maintainer

Alfi Nurrahmah <221810140@stis.ac.id>

## Functions

fakmct : A set of function for clustering data observation with hybrid method Fuzzy ART and K-Means

## Examples 

```{r}
library(fakmct)
# Using dataset iris
## load data
data.inputs = iris[,-5]
true.labels = as.numeric(unlist(iris$Species))

## run model data
ex.iris<-fakmct(data.inputs, alpha = 0.3, rho = 0.5, beta = 1, max_epochs = 50, max_clusters = 5)
ex.iris$labels
ex.iris$size
ex.iris$centroids
ex.iris$params

## plot data
plot(data.inputs, col = ex.iris$labels, pch = true.labels,
     main = paste0("Dataset: Iris"))

# Using data IPM 2019

## load simulate data IPM
data("simulatedataIPM")
dt <- simulatedataIPM

## run model data IPM
mod.fakm<-fakmct(dt, alpha = 0.3, rho = 0.5, beta = 0.1, max_epochs = 50, max_clusters = 5)
mod.fakm$labels
mod.fakm$size
mod.fakm$centroids
mod.fakm$params

## plot data IPM
plot(dt, col = mod.fakm$labels, pch=mod.fakm$labels, main = paste0("Dataset Human Development Index (IPM)"))

```

## References

  - Carpenter, Gail A., Stephen Grossberg, and David B. Rosen. (1991) Fuzzy ART: Fast stable learning and categorization of analog patterns by an 
    adaptive resonance system. Neural Networks, Vol 4, Issue 6, 759-771, ISSN 0893-6080, https://doi.org/10.1016/0893-6080(91)90056-B.
  - McQueen, J. (1967) Some Methods for Classification and Analysis of Multivariate Observations. Computer and Chemistry, 4, 257-272.
  - Sengupta S., Ghosh T., Dan P. K. (2011). Fuzzy ART K-Means Clustering Technique: a hybrid neural network approach to cellularmanufacturing systems. 
    International Journal of Computer Integrated Manufacturing, 24 (10), 927–938. https://doi.org/10.1080/0951192X.2011.602362
  - Othman Z. A., Muftah A., Suhaila A., Saadat, Z., al Hashmi, M. (2011). Improvement Anomaly Intrusion Detection using Fuzzy-ART 
    Based on K-means based on SNC Labeling. Jurnal Teknologi Maklumat Multimedia, 10, 1–11.
  - L. Steinmeister and D. C. Wunsch II. (2021) FuzzyART: An R Package for ART-based Clustering. http://dx.doi.org/10.13140/RG.2.2.11823.25761
