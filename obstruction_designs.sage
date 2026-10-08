#################################################
# STORING A RANK DECOMPOSITION & CACHING MINORS #
#################################################

class MinorCache(object):
    """
        Object to cache minors of a matrix.
    """
    def __init__(self, matrix, use_cache=True):
        self.matrix = Matrix(FIELD, matrix)
        self.use_cache = bool(use_cache)

        if self.use_cache:
            self.cache = {}
        else:
            self.get_minor = self.make_minor

    def nrows(self):
        return self.matrix.nrows()
    def ncols(self):
        return self.matrix.ncols()
    def __getitem__(self, index):
        return self.matrix.__getitem__(index)
    def __str__(self):
        return self.matrix.__str__()
    __repr__ = __str__

    @staticmethod
    def random(shape, use_cache=True):
        return MinorCache(random_matrix(FIELD, *shape), use_cache=use_cache)

    def get_minor(self, indices):
        """
            Return the determinant of the matrix made by taking the vectors specified by indices as columns and restricting to the first len(indices) coordinates.

            Do this by first trying to utilize the cache, only computing if a suitable cached result is not available.
        """
        cannonical, sign = sort_and_sign(indices)
        try:
            result = self.cache[cannonical]
        except KeyError:
            self.cache[cannonical] = self.make_minor(cannonical)
            result = self.cache[cannonical]
        if sign:
            return -result
        else:
            return result

    def make_minor(self, indices):
        """
            Actually compute a queried minor. Always len(indices) many rows from the top.
        """
        # check if there is a repetition, then we would know to return 0
        for i in range(len(indices)-1):
            if indices[i] == indices[i+1]:  # here we soft-assume that indices came in sorted, which is the case when make_minor is called from get_minor
                return 0
                break
        else:
            return self.matrix.matrix_from_rows_and_columns(range(len(indices)), indices).determinant()

    def precompute(self, upto):
        """
            Using dynamic programming, precompute all minors up to the given size in the given minor. Do this by Laplace-expanding with respect to the newest row.
        """
        self.cache[()] = FIELD(1)
        for r in range(1, upto+1):
            for minor in increasing_tuples(r, self.ncols()):
                minor_value = FIELD(0)
                sign = ((r-1) % 2 == 0)
                for i in range(r):  # Laplace expansion
                    term = self.matrix[r-1, minor[i]] * self.cache[minor[:i]+minor[i+1:]]
                    if sign:
                        minor_value -= term
                    else:
                        minor_value += term
                    sign = not sign
                self.cache[minor] = minor_value

    def only_keep_sizes(self, remaining_sizes):
        """
            Delete minors of all sizes except the given ones from the cache for this factor.
        """
        self.cache = {key:value for key, value in self.cache.items() if len(key) in remaining_sizes}

    def get_relevant_minors(self, part):
        """
            Compute and keep exactly those minors relevant to HighestWeightVector's with this partition.
        """
        part = Partition(part).conjugate()
        self.precompute(part[0])
        self.only_keep_sizes(part)

    def deform(self, action_matrix):
        """
            Return a new MinorCache object where all your columns are deformed by the given matrix. 'None' can be given to indicate an identity matrix and skip the computation.
        """
        if action_matrix is None:
            return self
        else:
            return MinorCache(action_matrix * self.matrix, use_cache=self.use_cache)


class TensorDecomposition(object):
    """
        Object to store a decomposition of a tensor into rank 1 tensors and query certain determinants from the data of this decomposition, caching the answers.
    """
    def __init__(self, number_of_factors, data, use_cache=True):
        """
            number_of_factors = number_of_factors of the underlying tensor space
            data = one matrix for each factor, i-th columns in each together encode the i-th term (as a consequence, all of the matrices have to have the same width). Those matrices may already be MinorCache's
            use_cache = should we actually bother with caching and looking up data? if set to False, minors are just recomputed whenever queried
        """
        assert isinstance(number_of_factors, (Integer, int))
        self.factors = number_of_factors
        self.length = None
        self.shape = []

        # read and validate the data
        self.use_cache = bool(use_cache)
        self.caches = []
        for factor_matrix in data:
            if self.length is None:
                self.length = factor_matrix.ncols()
            else:
                assert self.length == factor_matrix.ncols()
            self.shape.append(factor_matrix.nrows())
            if isinstance(factor_matrix, MinorCache):
                self.caches.append(factor_matrix)
            else:
                self.caches.append(MinorCache(factor_matrix, use_cache=self.use_cache))

    def __str__(self):
        return "Tensor decomposition built from\n" + "\nx\n".join([str(cache) for cache in self.caches])
    __repr__ = __str__

    @cached_method
    def to_tensor(self):
        """
            Recover the actual tensor that this decomposition represents.

            Only possible for 3-factor tensors, sorry.
        """
        if self.factors != 3:
            raise ValueError("Sorry, I can currently only handle a Tensor object being a 3-factor tensor.")
        result = Tensor(self.shape)
        for r in range(self.length):
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    for k in range(self.shape[2]):
                        result[i,j,k] += self.caches[0][i,r] * self.caches[1][j,r] * self.caches[2][k,r]
        return result

    @staticmethod
    def random(rank, shape):
        """
            Return a random decomposition of the given rank in the given shape.
        """
        return TensorDecomposition(len(shape), [random_matrix(FIELD, a, rank) for a in shape])

    def get_minor(self, factor, indices):
        """
            Return the determinant of the matrix made by taking the vectors specified by indices in the given factor as columns and restricting to the first len(indices) coordinates.
        """
        return self.caches[factor].get_minor(indices)

    def precompute(self, factor, upto):
        """
            Using dynamic programming, precompute all minors up to the given size in the given minor. Do this by Laplace-expanding with respect to the newest row.
        """
        self.caches[factor].precompute(upto)

    def only_keep_sizes(self, factor, remaining_sizes):
        """
            Delete minors of all sizes except the given ones from the cache for this factor.
        """
        self.caches[factor].only_keep_sizes(remaining_sizes)

    def get_relevant_minors(self, partitions):
        """
            Something on from the Kronecker space given by these partitions will be computing using your data, prepare exactly the relevant minors.
        """
        assert len(partitions) == self.factors
        for f in range(self.factors):
            self.caches[f].get_relevant_minors(partitions[f])

    def precompute_dumb(self, factor, size):
        """
            Brute force, unclever precompute. For the purposes of comparing the performance.
        """
        self.caches[factor].precompute_dumb(size)

    def deform(self, action_matrices):
        """
            Return a new TensorDecomposition object where all of the vectors are deformed by a matrix given in the list, one per factor. 'None' can be given in the list to indicate an identity matrix and skip the computation.
        """
        return TensorDecomposition(self.factors, [self.caches[f].deform(action_matrices[f]) for f in range(self.factors)], use_cache=self.use_cache)


class DecompositionProjectionLedger(object):
    """
        Object to remember a tensor decomposition and also do computations with its (random) deformations.
    """

    def __init__(self, decomposition, dest_shape):
        assert isinstance(decomposition, TensorDecomposition)
        self.original = decomposition
        self.factors = self.original.factors
        self.length = self.original.length
        self.orig_shape = self.original.shape
        assert len(dest_shape) == self.factors
        self.dest_shape = dest_shape
        self.use_cache = self.original.use_cache
        self.caches = [[] for _ in range(self.factors)]
        self.sample_counts = [0] * self.factors

    def make_projections(self, sample_counts, parallel=False, partitions=None):
        """
            Make random projection of the original decomposition. The optional partitions argument can be a list of partitions for whose KroneckerSpace this is intended to be used, then the relevant minors will be precomputed during the parallel process.

            sample_counts may be a list (one for each factor) or a single integer (used for all factors)
        """
        if isinstance(sample_counts, (Integer, int)):
            sample_counts = [sample_counts] * self.factors
        if partitions is None:
            partitions = [None] * self.factors
        self.sample_counts = tuple(sample_counts)

        self.caches = []
        for f in range(self.factors):
            if parallel:
                parallel_input = []
                step, extra = divmod(self.sample_counts[f], PARALLEL_NCPUS)
                begin = 0
                end = 0
                for i in range(PARALLEL_NCPUS):
                    end += step
                    if i < extra:
                        end += 1
                    if begin != end:
                        parallel_input.append((f, end-begin, partitions[f]))
                    begin = end
                assert begin == self.sample_counts[f]
                parallel_output = self.parallel_projection_envelope(parallel_input)
                new_cache_list = []
                for data in parallel_output:
                    new_cache_list.extend(data[1])
                self.caches.append(new_cache_list)
            else:
                self.caches.append(self.parallel_projection_envelope(f, self.sample_counts[f], partitions[f]))

    @parallel(ncpus=PARALLEL_NCPUS)
    def parallel_projection_envelope(self, factor, amount, partition=None):
        """
            Create amount many MinorCache's that are a random projection of self.original on the factor-th factor to self.dest_shape[factor]. If partition is not None, precompute the relevant minors.
        """
        result = []
        with seed():    # reseed RNG, so that the threads don't run with the same random samples
            for _ in range(amount):
                new_cache = self.original.caches[factor].deform(random_matrix(FIELD, self.dest_shape[factor], self.orig_shape[factor]))
                if partition is not None:
                    new_cache.get_relevant_minors(partition)
                result.append(new_cache)
        return result

    def get_slice(self, indices):
        """
            Return a projected TensorDecomposition corresponding to the given sequence of indices. Give the TensorDecomposition object your MinorCache's, so that the minors need not be computed again.
        """
        return TensorDecomposition(self.factors, [self.caches[f][indices[f]] for f in range(self.factors)], use_cache=self.use_cache)

    def diagonal_slice(self, domain_factors, domain_index, codomain_index):
        """
            Slice in a format that will be useful in KroneckerSpace.flattening.
        """
        indices = []
        for f in range(self.factors):
            if f in domain_factors:
                indices.append(domain_index)
            else:
                indices.append(codomain_index)
        return self.get_slice(indices)

    def clean(self):
        """
            Delete all data that is not fixed.
        """
        del self.caches
        del self.sample_counts
        self.caches = [[] for _ in range(self.factors)]
        self.sample_counts = [0] * self.factors

####################################
# ENCODING A HIGHEST WEIGHT VECTOR #
####################################

def make_tableau(pi, sigma):
    """
        Turn a partition and a permutation into a tableua described as a list of columns.
    """
    d = sum(pi)
    if sigma in Permutations(d):
        sigma = permutation_to_0list(sigma)
    else:
        sigma = list(sigma)
    col_lengths = []
    i = len(pi)
    j = 0
    while i > 0:
        i -= 1
        while j < pi[i]:
            col_lengths.append(i+1)
            j += 1

    tab = []
    for c in col_lengths[::-1]:
        col = sigma[:c]
        sigma = sigma[c:]
        tab.append(col)
    tab.reverse()
    return tab

class HighestWeightVector(object):
    """
        Object to encode a highest weight vector in a Weyl module.

        (In the notation of section 4.3 in Hauenstein-Ikenmeyer-Landsberg, this is the tensor sigma * F_{A, pi})
    """
    def __init__(self, *args):
        """
            either:
                len(args) == 1 and args[0] is a list of lists, describing a tableau with entries 0, ..., d-1 as a list of COLUMNS
            or:
                len(args) == 2, args[0] is a partition pi and args[1] is a permutation sigma
                we will then write sigma into the diagram of pi columnwise starting from the last
        """
        if len(args) == 1:
            # this is the primary form of input
            tab = args[0]
            self.d = 0
            all_entries = set()
            for col in tab:
                self.d += len(col)
                all_entries.update(col)
            assert all_entries == set(range(self.d))
            self.blocking = tab
        elif len(args) == 2:
            # here we just cook up the tableau and pass it to the previous case
            self.__init__(make_tableau(*args))
        else:
            raise ValueError("Unrecognized input format in HighestWeightVector.__init__")

        self.height = max([len(block) for block in self.blocking])

    def permutation(self):
        """
            Recover the permutation written in the tableau. Only for debug purposes.
        """
        result = []
        for col in self.blocking[::-1]:
            result.extend(col)
        return result

    def evaluate_on_decomposable(self, cache):
        """
            Evaluate self on a decomposable tensor given by the product of the column vectors of the given cache.
        """
        result = FIELD(1)
        for block in self.blocking:
            result *= cache.get_minor(block)
        return result

    def evaluate_on_expansion(self, decomposition, factor_index, expansion_tuple):
        """
            Evaluate self on a decomposable tensor stemming from one term in the expansion of a power of a rank decomposition.
        """
        result = FIELD(1)
        for block in self.blocking:
            indices = [expansion_tuple[i] for i in block]
            result *= decomposition.get_minor(factor_index, indices)
        return result

    def to_simplified_symmetrizer(self):
        """
            Realize self as an element of the group algebra of the approriate quotient of S_n. Represent it as a dictionary {string with repetitions : coefficient}

            (For debug.)
        """
        old = {(None,)*self.d : FIELD(1)}
        for block in self.blocking:
            # build the det of this block
            this = {}
            index = [None] * self.d
            for perm in Permutations(len(block)):
                sign = perm.sign()
                perm = permutation_to_0list(perm)
                for i in range(len(block)):
                    index[block[i]] = perm[i]
                this[tuple(index)] = FIELD(sign)

            # multiply what you got so far by it
            new = {}
            for a in this:
                for b in old:
                    index = [None] * self.d
                    for i in range(self.d):
                        assert a[i] is None or b[i] is None
                        if a[i] is not None:
                            index[i] = a[i]
                        if b[i] is not None:
                            index[i] = b[i]
                    index = tuple(index)
                    new[index] = new.get(index, FIELD(0)) + this[a] * old[b]

            new = {key:value for key, value in new.items() if value != 0}
            old = new

        return old

def get_Specht_basis(pi, verbose=False):
    """
        Create a list of HighestWeightVector's that correspond to a basis of the Specht module.
    """
    specht_dim = Specht_module_dimension(pi)
    d = sum(pi)
    permutations = Permutations(d)
    h = len(pi)
    test_decomposables = []
    for _ in range(specht_dim):
        new = MinorCache.random((h,d))
        new.get_relevant_minors(pi)
        test_decomposables.append(new)
    test_matrix = Matrix(FIELD, specht_dim)
    result = []
    r = 0   # this should be held to r == test_matrix.rank()
    while r < specht_dim:
        if verbose:
            print("-- have %i so far, need %i" % (r, specht_dim))
        # get a random new one
        sigma = permutations.random_element()
        if verbose:
            print("-- -- trying", sigma)
        new = HighestWeightVector(pi, sigma)
        # see if rank increased
        for j in range(specht_dim):
            test_matrix[r, j] = new.evaluate_on_decomposable(test_decomposables[j])
        if test_matrix.rank() > r:
            result.append(new)
            r += 1
        elif verbose:
            print("-- -- FAILED")
    return result


##############################
# BINOMIAL EXPANSION PRUNING #
##############################

class GraphColoringList(object):
    """
        A list of colorings of a graph, given as tuples of colors for each vertex. Also remembers the graph and the number of colors.
    """
    def __init__(self, graph, number_of_colors, data=[]):
        self.graph = graph
        self.colors = number_of_colors
        self.colorings = list(data)

    def append(self, item):
        self.colorings.append(item)

    def __len__(self):
        return self.colorings.__len__()

    def __iter__(self):
        return self.colorings.__iter__()

    def __getitem__(self, index):
        return self.coloring.__getitem__(index)


class HighestWeightPolynomial(object):
    """
        Object to manage evaluations of a certain highest-weight element in an isotypic component of Sym^d(A x B x C)* on tensors presented through rank decompositions, with pruning of the resulting binomial expansion.
    """
    def __init__(self, *args, verbose=False):
        """
            either:
                len(args) == 1 and args[0] is either
                    a list of tableaux
                or:
                    a list of HighestWeightVector's
            or:
                len(args) == 2, args[0] is a list of partitions and args[1] is a list of permutations
        """
        self.verbose = bool(verbose)
        if len(args) == 1:
            tabs = args[0]
            assert len(tabs) > 0
            if isinstance(tabs[0], HighestWeightVector):
                self.HW_vectors = tabs
            else:
                self.HW_vectors = [HighestWeightVector(tab) for tab in tabs]
        elif len(args) == 2:
            partitions, permutations = args
            self.HW_vectors = [HighestWeightVector(partitions[i], permutations[i]) for i in range(len(self.partitions))]
        else:
            raise ValueError("Unrecognized input format in HighestWeightPolynomial.__init__")

        self.factors = len(self.HW_vectors)
        self.d = None
        for F in self.HW_vectors:
            if self.d is None:
                self.d = F.d
            else:
                assert self.d == F.d

        # build the pruning graph
        self.pruning_graph = self.make_pruning_graph()

    def make_pruning_graph(self):
        """
            Graph on vertices 0, 1, ..., d-1, where two vertices are connected if and only if they appear in a common block in one of the factors. The interpretation is that if these two indices took the same term from a rank decomposition, the value of our polynomial would vanish because of a repetition in one of the constituent determinants.
        """
        vertices = list(range(self.d))
        edges = set()
        for F in self.HW_vectors:
            for block in F.blocking:
                for i in range(len(block)):
                    for j in range(i):
                        edges.add((block[i], block[j]))
        return Graph([vertices, edges], format="vertices_and_edges")

    def enumerate_colorings(self, r):
        """
            Create lists of admissible colorings of the pruning graph by r colors.

            These will correspond to terms in the expansion of a power of a rank r decompositions that survive the pruning and may yield nonzero contributions.

            (For now, this method has no parallelization options, but the library function all_graph_colorings seems to be performing remarkably well, so I leave it)
        """
        result = []
        q = r
        chrom = self.pruning_graph.chromatic_number()
        temp_coloring = [None] * self.d
        while q >= chrom:
            new_additions = []
            for coloring in sage.graphs.graph_coloring.all_graph_colorings(self.pruning_graph, q):   # this is a generator enumerating all coloring *that use all q colors*, and returning them as dicts {color : [vertices]}
                color_map = list(range(q))   # we need to account for the extra colorings that result from using a different subset of colors; here the i-th item is the color to be used in place of what is originally the color i
                copies_made = 0
                while True:
                    # add the coloring
                    for c in coloring:
                        for i in coloring[c]:
                            temp_coloring[i] = color_map[c]
                    new_additions.append(tuple(temp_coloring))
                    copies_made += 1
                    # increment color_map
                    i = q
                    while i > 0:
                        i -= 1
                        if color_map[i] < i + r - q:    # there is space
                            color_map[i] += 1           # so increment this position
                            for j in range(i+1, q):     # and reset all of the following ones
                                color_map[j] = color_map[j-1] + 1
                            break
                    else:   # we didn't exit the loop via a break, which means we're at the maximal value
                        break
            result.extend(new_additions)
            q -= 1
        return GraphColoringList(self.pruning_graph, r, result)

    def evaluate(self, decomposition, coloring_list=None, parallel=False):
        """
            Evaluate self on the d-th power of a tensor presented as a rank decomposition
        """
        rank = decomposition.length
        if coloring_list is None:
            if self.verbose:
                print("-- computing the pruning")
            expansion_tuples = self.enumerate_colorings(rank)
            if self.verbose:
                print("-- done, %i terms remain in the expansion" % len(expansion_tuples))
        else:
            assert isinstance(coloring_list, GraphColoringList)
            assert coloring_list.colors == rank
            assert coloring_list.graph == self.pruning_graph
            expansion_tuples = coloring_list.colorings

        if parallel:
            coloring_batches = []
            step, extra = divmod(len(expansion_tuples), PARALLEL_NCPUS)
            begin = 0
            end = 0
            for i in range(PARALLEL_NCPUS):
                end += step
                if i < extra:
                    end += 1
                coloring_batches.append(expansion_tuples[begin:end])
                begin = end
            assert begin == len(expansion_tuples)

            self._temp_decomposition = decomposition
            parallel_output = self.parallel_evaluate_envelope(coloring_batches) # this generator yields results in random order, but we will just be summing, so we don't care
            result = FIELD(0)
            for data in parallel_output:
                result += data[1]
        else:
            result = FIELD(0)
            for et in expansion_tuples:
                result += self.evaluate_on_expansion(decomposition, et)

        return result

    @parallel(ncpus=PARALLEL_NCPUS)
    def parallel_evaluate_envelope(self, colorings_batch):
        result = FIELD(0)
        for et in colorings_batch:
            result += self.evaluate_on_expansion(self._temp_decomposition, et)
        return result

    def evaluate_on_expansion(self, decomposition, expansion_tuple):
        """
            Evaluate the polynomial on a particular term of an expansion.
        """
        result = FIELD(1)
        for t in range(self.factors):
            result *= self.HW_vectors[t].evaluate_on_expansion(decomposition, t, expansion_tuple)
        return result

    def evaluate_on_tensor(self, tensor):
        """
            Evaluate on a tensor that does not come with a decomposition. Do this by reducing to a mask and then evaluating the mask.
        """
        mask, mask_vars = self.to_mask(return_var_dict=True)
        subst = {mask_vars[key] : tensor[key] for key in mask_vars}
        return FIELD(mask.substitute(subst))


def find_zero_pattern(tableaux, d=None):
    """
        Look for a zero pattern in this choice of partitions and permutations; return the first you find if there exists one and None if there doesn't.

        input: list of tableaux filled with 0, ..., d-1, described as a list of COLUMNS
                (d can be inferred if not supplied explicitly)
    """
    if d is None:
        for tab in tableaux:
            if d is None:
                d = sum([len(row) for row in tab])
            else:
                assert d == sum([len(row) for row in tab])

    entry_to_column = []
    for t in range(len(tableaux)):
        new = [None] * d
        for j in range(len(tableaux[t])):
            for a in tableaux[t][j]:
                new[a] = j
        entry_to_column.append(new)

    for i in range(d):
        for j in range(i):
            # have_zero_pattern = True
            for t in range(len(entry_to_column)):
                if entry_to_column[t][i] != entry_to_column[t][j]:
                    break   # these two entries don't constitute a zero pattern
            else:   # the previous loop didn't break
                return (j,i) # which means we didn't find a counterexample to the zero pattern
                break
    return None    # each pair of entries checked all factors without issue and found a counterexample to the zero pattern in some, meaning there is no zero pattern


##############################################
# MANAGING THE ISOTYPIC COMPONENT AS A WHOLE #
##############################################

def standard_tableau_to_my_format(tab):
    """
        Take a standard tableau, i.e. a list of rows filled with 1, ..., d and turn it into the corresponding list of columns filled with 0, ..., d-1
    """
    return [[x-1 for x in row] for row in tab.conjugate()]

class KroneckerSpace(object):
    """
        An object to generate and hold a list of obstruction designs giving a basis of a Kronecker space, then compute isotypic flattenings with it.
    """
    def __init__(self, partitions, testing_secant=None, verbose=False, parallel_init=False, clean_afterwards=True):
        self.verbose = bool(verbose)
        self.parallel_init = bool(parallel_init)

        self.d = None
        self.partitions = []
        self.heights = []
        for part in partitions:
            part = Partition(part)
            self.partitions.append(part)
            self.heights.append(len(part))
            if self.d is None:
                self.d = part.size()
            else:
                assert self.d == part.size()
        self.len = len(self.partitions)
        self.length = self.len
        assert self.len >= 1
        self.Kronecker_coef = SpechtTable(self.d).Kronecker_coef(*self.partitions)
        self.standard_tableaux = [[standard_tableau_to_my_format(tab) for tab in StandardTableaux(part)] for part in self.partitions]
        self.Specht_bases = [[HighestWeightVector(tab) for tab in tablist] for tablist in self.standard_tableaux]
        # randomly generate the right number of linearly independent basis elements for the Kronecker space
        # first, tensors we will test linear the linear dependence on
        # determine from which secant we will get the testing tensors
        if testing_secant is not None:
            assert isinstance(testing_secant, (Integer, int))
            assert testing_secant >= 0
            self.r = testing_secant
        else:
            self.r = ceil(prod(self.heights) / (sum(self.heights)+1-len(self.heights))) # take the expected generic brk; if there happens to be a defectivity, there will remain a possibility that we don't find enough independent elements of the Kronecker space -- deal with it :-/
        if self.verbose:
            print("-- using testing_secant =", self.r)

        self.test_tensors = []
        for _ in range(self.Kronecker_coef):
            new = TensorDecomposition.random(self.r, self.heights)
            new.get_relevant_minors(self.partitions)
            self.test_tensors.append(new)
        self.basis_polys = []
        independent = 0 # how many independent generators we have so far
        self.test_matrix = Matrix(FIELD, len(self.test_tensors))
        # while independent < len(self.test_tensors):
        while independent < self.Kronecker_coef:
            if self.verbose:
                print("-- got %i so far, need %i" % (independent, self.Kronecker_coef))
            # take one new polynomial
            new_poly = self.random_element()
            if self.verbose:
                print("-- -- computing test coloring")
            test_coloring = new_poly.enumerate_colorings(self.r)
            # evaluate it
            for j in range(len(self.test_tensors)):
                self.test_matrix[independent, j] = new_poly.evaluate(self.test_tensors[j], test_coloring, parallel=self.parallel_init)
            # see if it increased rank
            if self.test_matrix.rank() > independent:
                independent += 1
                self.basis_polys.append(new_poly)
            elif self.verbose:
                print("-- -- FAILED")

        if clean_afterwards:
            self.clean()

    def random_element(self):
        while True:
            hw_vectors = []
            for f in range(self.len):
                i = randint(0, len(self.standard_tableaux[f])-1)
                hw_vectors.append(self.Specht_bases[f][i])
            if self.len % 2 == 0 or find_zero_pattern([wp.blocking for wp in hw_vectors]) is None:
                break
        return HighestWeightPolynomial(hw_vectors, verbose=self.verbose)


    #### TODO: Match convention on where Kronecker goes with the paper
    def flattening(self, decomposition, domain_factors, domain_count, codomain_count, colorings=None, parallel=False):
        """
            Return the flattening on a tensor given as a decomposition. domain_factors = indices of factors that will go on the domain (~columns); the Kronecker space itself will always go on the domain (~columns)

            decomposition can either be a DecompositionProjectionLedger with sufficiently many projections, or a TensorDecomposition, at which point a DecompositionProjectionLedger will be created.
        """
        if isinstance(decomposition, DecompositionProjectionLedger):
            for f in range(self.len):
                if f in domain_factors:
                    assert decomposition.sample_counts[f] >= domain_count
                else:
                    assert decomposition.sample_counts[f] >= codomain_count
        else:
            if self.verbose:
                print("-- preparing projections")
            decomposition = DecompositionProjectionLedger(decomposition, self.heights)
            sample_counts = []
            for f in range(self.len):
                if f in domain_factors:
                    sample_counts.append(domain_count)
                else:
                    sample_counts.append(codomain_count)
            decomposition.make_projections(sample_counts, parallel=parallel, partitions=self.partitions)

        self._temp_r = decomposition.length
        if colorings is None:
            if self.verbose:
                print("-- preparing colorings")
            if parallel:
                kronecker_batches = generate_batches(self.Kronecker_coef)
                parallel_output = list(self.parallel_coloring_envelope(kronecker_batches))
                parallel_output.sort()
                colorings = []
                for data in parallel_output:
                    colorings.extend(data[1])
            else:
                colorings = self.parallel_coloring_envelope((0, self.Kronecker_coef))

        if self.verbose:
            print("-- building the matrix")
        self._temp_decomposition = decomposition
        self._temp_domain_factors = domain_factors
        self._temp_domain_count = domain_count
        # self._temp_codomain_count = codomain_count
        self._temp_colorings = colorings
        if parallel:
            codomain_batches = generate_batches(codomain_count)
            parallel_output = list(self.parallel_flattening_envelope(codomain_batches))
            parallel_output.sort()  # we sort to keep the order consistent with the caches in the DecompositionProjectionLedger; even though for rank etc., we wouldn't care if the column blocks were permuted
            flattening_pieces = [data[1] for data in parallel_output]
            result = concatenate_matrix_rows(codomain_count, domain_count * self.Kronecker_coef, flattening_pieces)
            self.clean_temps()
        else:
            result = self.parallel_flattening_envelope((0, domain_count))
        return result


    @parallel(ncpus=PARALLEL_NCPUS)
    def parallel_coloring_envelope(self, kronecker_range):
        begin, end = kronecker_range
        result = []
        for k in range(begin, end):
            result.append(self.basis_polys[k].enumerate_colorings(self._temp_r))
        return result

    @parallel(ncpus=PARALLEL_NCPUS)
    def parallel_flattening_envelope(self, codomain_range):
        begin, end = codomain_range
        result = Matrix(FIELD, end-begin, self.Kronecker_coef * self._temp_domain_count)
        for k in range(self.Kronecker_coef):
            for i in range(end-begin):
                for j in range(self._temp_domain_count):
                    result[i, k*self._temp_domain_count + j] = self.basis_polys[k].evaluate(
                        self._temp_decomposition.diagonal_slice(self._temp_domain_factors, j, begin+i),
                        self._temp_colorings[k],
                        parallel=False
                    )
        return result

    def metaflattening(self, decomposition, domain_factors, domain_count, codomain_count, colorings=None, parallel=False):
        flat = self.flattening(decomposition, domain_factors, domain_count, codomain_count, colorings=colorings, parallel=parallel)
        return PnMatrix(Tensor([flat[:, k*domain_count : (k+1)*domain_count] for k in range(self.Kronecker_coef)]))

    def clean_temps(self):
        """
            Delete all termporary data.
        """
        keys = list(self.__dict__.keys())
        for key in keys:
            if begins_with(key, "_temp"):
                del self.__dict__[key]

    def clean(self):
        """
            Delete now-unnecessary data used during __init__.
        """
        del self.Specht_bases
        del self.r
        del self.test_tensors
        del self.test_matrix
        del self.parallel_init
