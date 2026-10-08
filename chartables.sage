class SpechtTable(object):
    """
        Object to perform computations with character tables of representations of symmetric groups.
    """

    def __init__(self, k):
        assert isinstance(k, (int, Integer))
        self.k = k
        self.symgroup = SymmetricGroup(self.k)
        self.partitions = list(Partitions(self.k))
        self.size = len(self.partitions)

        self.chartable = self.symgroup.character_table().change_ring(QQ)    # a priori the matrix comes over a cyclotomic field object...
        # now bring it in line with our indexing: columns should be char vectors of Spechts, in the order in self.partitions, rows should be conjugacy classes corresponding to partitions in that same order
        self.chartable.permute_rows(Permutation(list(range(self.size, 0, -1))))
        self.chartable = self.chartable.transpose()
        self.inverse_matrix = self.chartable.inverse()

    def __len__(self):
        return self.size

    def __iter__(self):
        return iter(self.partitions)

    def __getitem__(self, index):
        return self.partitions[index]

    def decompose_character_vector(self, char):
        return self.inverse_matrix * vector(QQ, char)

    def decompose_product(self, *args):
        args = list(args)
        for t in range(len(args)):
            if not isinstance(args[t], (int, Integer)):
                args[t] = Partition(args[t])
            if isinstance(args[t], Partition):
                args[t] = self.partitions.index(args[t])

        char = self.character_Hadamard(*args)
        coefs = self.inverse_matrix * char
        result = {}
        for k in range(self.size):
            if coefs[k] != 0:
                result[self.partitions[k]] = coefs[k]
        return result

    def character_Hadamard(self, *indices):
        return vector(QQ, [prod(self.chartable[t,i] for i in indices) for t in range(self.size)])

    def Kronecker_coef(self, *args):
        args = list(args)
        for t in range(len(args)):
            if not isinstance(args[t], (int, Integer)):
                args[t] = Partition(args[t])
            if isinstance(args[t], Partition):
                args[t] = self.partitions.index(args[t])

        char = self.character_Hadamard(*args)
        return Integer(self.inverse_matrix[0] * char)    # read of the coefficient of the trivial representation

    def find_higher_Kroneckers(self):
        """
            Return all instances of a Kronecker_coef bigger than 1 that you can see.
        """
        result = []
        for i in range(len(self)):
            for j in range(i+1):
                char = self.character_Hadamard(i,j)
                these_Kroneckers = self.inverse_matrix[:j+1] * char
                for k in range(j+1):
                    c = these_Kroneckers[k]
                    if c > 1:
                        result.append((self[i], self[j], self[k], c))
        return result

def Weyl_module_dimension(n, partition):
    return Integer(SymmetricFunctions(QQ).schur()(partition).eval_at_permutation_roots([1]*n))

def Specht_module_dimension(partition):
    result = sum(partition).factorial()
    for i in range(len(partition)):
        for j in range(partition[i]):
            hook = partition[i] - j
            for ii in range(i+1, len(partition)):
                if partition[ii] > j:
                    hook += 1
                else:
                    break
            result /= hook
    return Integer(result)

def Weyl_dim_poly(partition, polyring=None):
    if polyring is None:
        polyring = PolynomialRing(QQ, "n")
    n = polyring.gen()

    giambelli = Matrix(polyring, len(partition))
    for i in range(len(partition)):
        for j in range(len(partition)):
            index = partition[i] + j - i
            giambelli[i,j] = binomial(n+index-1, index)
    return giambelli.determinant()
