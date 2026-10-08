##############################
# PARALLELIZATION MANAGEMENT #
##############################

def generate_batches(N, batch_number=PARALLEL_NCPUS):
    """
        Return a list that splits range(N) as evenly as possible into batch_number disjoint parts, denoted as singletons of pairs ((begin,end),) denoting intervals
    """
    result = []
    assert isinstance(N, (Integer, int))
    N = Integer(N)
    step, extra = divmod(N, batch_number)
    begin = 0
    end = 0
    for i in range(batch_number):
        end += step
        if i < extra:
            end += 1
        if begin != end:
            result.append(((begin, end),))
        begin = end
    assert begin == N
    return result

#################
# INDEXING SETS #
#################

def position_to_multiindex(pos, shape):
    result = []
    a = pos
    for d in shape:
        a, new = divmod(a, d)
        result.append(new)
    return result

def tuples_of_sum_aux(length, sum):
    """
        Yield all tuples of nonnegative integers of the given length and maximal sum.
    """
    assert length >= 0
    if length == 0:
        if sum == 0:
            yield ()
    else:
        for first in range(0, sum+1):
            for tup in tuples_of_sum_aux(length-1, sum-first):
                yield (first,) + tup

@cached_function
def tuples_of_sum(length, sum):
    return list(tuples_of_sum_aux(length, sum))

@cached_function
def tuples_of_sum_at_most(length, sum):
    result = []
    for s in range(0, sum+1):
        result.extend(tuples_of_sum(length, s))
    return result

def increasing_tuples(k, n, start=0):
    """
        Yield all increasing k-tuples among elements of range(start, n).
    """
    if n >= start:
        if k == 0:
            yield ()
        elif k > 0:
            for first in range(start, n):
                for partial_tuple in increasing_tuples(k-1, n, first+1):
                    yield (first,) + partial_tuple
#
# def nondecreasing_tuples(k, n, start=0):
#     """
#         Yield all nondecreasing k-tuples among elements of range(start, n).
#     """
#     if n >= start:
#         if k == 0:
#             yield ()
#         elif k > 0:
#             for first in range(start, n):
#                 for partial_tuple in nondecreasing_tuples(k-1, n, first):
#                     yield (first,) + partial_tuple

# def random_subset(S, k):
#     """
#         Return a random subset of S of size k.
#     """
#     indices = list(range(len(S)))
#     shuffle(indices)
#     return set([S[index] for index in indices[:k]])


################
# PERMUTATIONS #
################

def permutation_to_0list(perm):
    """
        Turn a Sage permutation object (element of Permutations(n)) to a list of indices 0, ..., n-1
    """
    return [perm[i]-1 for i in range(len(perm))]

def sort_and_sign(L):
    """
        Given a list, return a sorted tuple and the sign of the permutation that made it sorted.
    """
    aux = list(zip(L, range(len(L))))
    aux.sort()

    # get the sign
    visited = [False] * len(L)
    i = 0
    sign = False
    while i < len(L):
        if not visited[i]:
            visited[i] = True
            j = aux[i][1]
            # sign = not sign # sign will flip on each next step of a cycle, but we want odd cycles to not change anything, so we put this extra flip at the start to make it work
            while j != i:
                visited[j] = True
                sign = not sign
                j = aux[j][1]
        i += 1

    return tuple(data[0] for data in aux), sign

###########
# STRINGS #
###########

def begins_with(string, substring):
    if not isinstance(string, str) or len(substring) > len(string):
        return False
    else:
        for i in range(len(substring)):
            if string[i] != substring[i]:
                return False
                break
        else:
            return True


######################
# LINEAR ALGEBRA QoL #
######################

def concatenate_matrix_columns(nrows, ncols, pieces):
    """
        Return the matrix of given sizes that consists of pieces layed out left to right.

        If the sizes don't match up somewhere, will throw an error.
    """
    result = Matrix(FIELD, nrows, ncols)
    begin = 0
    end = 0
    for piece in pieces:
        end += piece.ncols()
        result[:, begin:end] = piece
        begin = end
    assert begin == ncols
    return result

def concatenate_matrix_rows(nrows, ncols, pieces):
    """
        Return the matrix of given sizes that consists of pieces layed out top to bottom.

        If the sizes don't match up somewhere, will throw an error.
    """
    result = Matrix(FIELD, nrows, ncols)
    begin = 0
    end = 0
    for piece in pieces:
        end += piece.nrows()
        result[begin:end, :] = piece
        begin = end
    assert begin == nrows
    return result

def my_str_matrix(M):
    return str(list(M)).replace("[", "[\n\t").replace("]", "\n]").replace("), (", "),\n\t(").replace("(", "[").replace(")", "]")


#######################
# RINGS & POLYNOMIALS #
#######################

@cached_function
def monomials_of_degree(polyring, k):
    """
        In newer versions of Sage, this is already a method of polynomial rings. But for compatibility with older versions, let's separate it out like this.
    """
    return [polyring({tup:1}) for tup in tuples_of_sum(polyring.ngens(), k)]

@cached_function
def monomials_of_degree_at_most(polyring, k):
    return [polyring({tup:1}) for tup in tuples_of_sum_at_most(polyring.ngens(), k)]

def degree_ideal(polyring, k):
    """
        Return the ideal of polyring generated by degree k monomials.

        (Point of this function is that writing m**k is sloppy, creates the ideal object with an absurd number of generators.)
    """
    return polyring.ideal(monomials_of_degree(polyring, k))

def hilbert_function(ideal):
    """
        Return the local Hilbert function of an ideal as a finite tuple of integers.

        Presume the ideal defines a zero-dimensional scheme supported at the origin
    """
    polyring = ideal.ring()
    m = polyring.ideal(polyring.gens())
    for g in ideal.gens():
        assert g in m   # want a scheme supported at origin only

    # maybe this could be improved with a binary search, but I'm too lazy to do it now
    k = 1
    while True:
        for m in monomials_of_degree(polyring, k):
            if m not in ideal:
                break
        else: # m^k is contained in ideal
            break
        k += 1

    basis = ideal.normal_basis()
    result = [len(basis)]

    k -= 1
    while k > 0:
        I = ideal + degree_ideal(polyring, k)
        basis = [b for b in basis if I.reduce(b) == b]  # this line really assumes all these ideals are being created with the same term order, so if that were not the case for some weird reason, this gets screwed up
        result[-1] -= len(basis)
        result.append(len(basis))
        k -= 1

    result.reverse()
    return tuple(result)
