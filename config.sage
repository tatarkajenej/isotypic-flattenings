# Field arithmetic
PRIME = 1009
FIELD = FiniteField(PRIME)

# Sparseness
TENSOR_SPARSENESS = False   # for the data of a tensor
BASAL_FLATTENING_SPARSENESS = False   # for flattenings that look nice with respect to a basis, i.e. have a decent chance of being reasonably sparse, if the tensor looks nice
SAMPLED_FLATTENING_SPARSENESS = False   # for flattenings that are randomly sampled, i.e. are not likely to have low density

# Parallelization
PARALLEL_NCPUS = 4

# Padding to avoid false-positive rank drops
# TESTING_PAD = 20

# Interpolation
PNMATRIX_IDEAL_STOP = 3
