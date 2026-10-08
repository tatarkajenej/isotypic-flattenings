# Configurable parameters for the various modules.
# -- notably includes: the finite field arithmetic and parallelization.
load("config.sage")

# Small utility functions.
# -- notably includes: some custom polynomial functions and an auxiliary function for parallelization.
load("utils.sage")

# Custom support for evaluating large polynomial minors and similar.
load("interpolation.sage")

# The main Tensor and StructTensor classes.
# -- includes a variety of I/O methods and miscellaneous Tensor invariants like centroids
# -- also has a bunch of mostly deprecated code for early flattenings (Koszul, quadratic...)
load("tensor.sage")

# Support for apolarity.
load("apolar.sage")

# Character tables of Specht modules.
# -- supports and double checks parts of the computations in stratification.sage
load("chartables.sage")

# Implementation of obstruction designs, adapted from the Hauenstein-Ikenmeyer-Landsberg proof of brk(M_2) >= 7
load("obstruction_designs.sage")
