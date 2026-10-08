class Tensor(object):
    """
        Object to hold a tensor as a list of matrices, compute Koszul flattenings, possibly change into different flattenings. The default interpretation is the written information is the flattening A* --> Hom(C*, B) and A, B, C are arithmetic spaces

        Should have a subclass for structure tensors of finite-dimensional commutative algebras.
    """

    def __init__(self, data=None):
        """
            data can be:
                * another Tensor object
                * a matrix of linear forms
                * a 3-tuple describing the shape
                * a list of matrices
                * 3-nested list of FIELD elements
        """
        self.matrices = []
        if isinstance(data, Tensor):
            self.matrices = [Matrix(M) for M in data.matrices]
            self.shape = data.shape
        elif isinstance(data, sage.structure.element.Matrix):
            self.matrices = []
            x = data.base_ring().gens()
            subst = {xx : 0 for xx in x}
            for xx in x:
                subst[xx] = 1
                self.matrices.append(data.substitute(subst))
                subst[xx] = 0
            self.shape = (len(x), data.nrows(), data.ncols())
        else:
            # now we presume data is a list or some similar collection
            if isinstance(data[0], Integer) or isinstance(data[0], int):
                # initialize empty
                self.shape = tuple(data)
                self.matrices = [Matrix(FIELD, self.shape[1], self.shape[2], sparse=TENSOR_SPARSENESS) for _ in range(self.shape[0])]
            else:
                self.matrices = [Matrix(FIELD, M, sparse=TENSOR_SPARSENESS) for M in data]   # this should work with both matrices or nested lists
                self.shape = (len(self.matrices), self.matrices[0].nrows(), self.matrices[0].ncols())

        self.has_linform_matrix = False
        self.has_dict = False

    def __getitem__(self, pos):
        """
            pos = collection supporting indices 0, 1, 2
        """
        return self.matrices[pos[0]][pos[1:]]

    def __setitem__(self, pos, val):
        """
            pos = collection supporting indices 0, 1, 2
        """
        self.matrices[pos[0]][pos[1:]] = val

    def to_nested_list(self):
        """
            Return representation of selfed as a 3-nested list of FIELD elements
        """
        return [[[self[a,b,c] for c in range(self.shape[2])] for b in range(self.shape[1])] for a in range(self.shape[0])]
    as_nested_list = to_nested_list

    def to_flat_list(self):
        """
            Return the list of entries in the canonical way. This loses information about shape.
        """
        return [self[i,j,k] for i in range(self.shape[0]) for j in range(self.shape[1]) for k in range(self.shape[2])]

    def as_dict(self):
        """
            Return the dictionary {(i,j,k) |--> T[i,j,k]}
        """
        if not self.has_dict:
            self.make_dict()
        return self.dict

    def make_dict(self):
        self.dict = {}
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                for k in range(self.shape[2]):
                    if self[i,j,k] != 0:
                        self.dict[i,j,k] = self[i,j,k]
        self.has_dict = True

    def __str__(self):
        layers = [str(M) for M in self.matrices]
        divider = "\n" + ("-" * max([len(layers[i].split("\n")[0]) for i in range(len(self.matrices))])) + "\n"
        return divider.join(layers)

    def __repr__(self):
        return repr(self.to_nested_list())

    def __eq__(self, other):
        if not isinstance(other, Tensor):
            return False
        if self.shape != other.shape:
            return False
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                for k in range(self.shape[2]):
                    if self[i,j,k] != other[i,j,k]:
                        return False
        return True

    def __add__(self, other):
        assert isinstance(other, Tensor)
        assert self.shape == other.shape
        result = Tensor(self.shape)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                for k in range(self.shape[2]):
                    result[i,j,k] = self[i,j,k] + other[i,j,k]
        return result

    def __neg__(self):
        result = Tensor(self.shape)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                for k in range(self.shape[2]):
                    result[i,j,k] = -self[i,j,k]
        return result

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        assert isinstance(other, Tensor)
        return self.__add__(other.__neg__())

    def __rsub__(self, other):
        return self.__neg__().__add__(other)

    def __mul__(self, scalar):
        assert scalar in FIELD
        result = Tensor(self.shape)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                for k in range(self.shape[2]):
                    result[i,j,k] = scalar * self[i,j,k]
        return result

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    @staticmethod
    def random(a, b, c):
        """
            Create a tensor of given shape filled with random entries
        """
        matrices = []
        for _ in range(a):
            new_matrix = Matrix(FIELD, b, c)
            new_matrix.randomize()
            matrices.append(new_matrix)
        return Tensor(matrices)

    @staticmethod
    def from_polynomial_string(string, shape, letters="xyz", index_from_one=True, subscript_string="_"):
        """
            Read a tensor from a Macaulay2 printout of it as a trilinear polynomial.
        """
        polyring = PolynomialRing(FIELD, [letters[i]+subscript_string+str(j) for i in range(3) for j in range(int(index_from_one), shape[i]+int(index_from_one))])
        variables = [[polyring(letters[i]+subscript_string+str(j)) for j in range(int(index_from_one), shape[i]+int(index_from_one))] for i in range(3)]
        for i in range(3):
            for j in range(shape[i])[::-1]:
                string = string.replace(str(variables[i][j]), "variables[%i][%i]" % (i,j))
        polynomial = eval(string)

        return Tensor([[[FIELD(polynomial.monomial_coefficient(variables[0][i]*variables[1][j]*variables[2][k])) for k in range(shape[2])] for j in range(shape[1])] for i in range(shape[0])])

    @staticmethod
    def from_polynomial(f):
        """
            Recover the symmetric tensor corresponding to a polynomial.
        """
        polyring = f.parent()
        n = polyring.ngens()
        result = Tensor((n,n,n))
        for m in f.monomials():
            assert m.degree() == 3
            c = FIELD(f.monomial_coefficient(m))
            vars = []
            exps = m.exponents()[0]
            for i in range(n):
                vars.extend([i] * exps[i])
            for i,j,k in [(0,1,2), (0,2,1), (1,0,2), (1,2,0), (2,0,1), (2,1,0)]:
                result[vars[i],vars[j],vars[k]] += c
        return result

    @staticmethod
    def direct_sum(*summands):
        """
            Return a direct sum of several tensors.
        """
        shape = [0, 0, 0]
        for T in summands:
            assert isinstance(T, Tensor)
            for i in range(3):
                shape[i] += T.shape[i]

        result = Tensor(shape)
        A_offset = 0
        B_offset = 0
        C_offset = 0
        for T in summands:
            for a in range(T.shape[0]):
                for b in range(T.shape[1]):
                    for c in range(T.shape[2]):
                        result[A_offset+a, B_offset+b, C_offset+c] = T[a,b,c]
            A_offset += T.shape[0]
            B_offset += T.shape[1]
            C_offset += T.shape[2]

        return result

    def permute(self, permutation):
        """
            Act on the tensor with a permutation from S_3, given as some collection supporting indices 0, 1 and 3
        """
        assert Set(permutation) == Set([0,1,2])
        def aux_perm(indices):
            return (indices[permutation[0]], indices[permutation[1]], indices[permutation[2]])
        return Tensor([[[self[aux_perm((a,b,c))] for c in range(self.shape[permutation.index(2)])] for b in range(self.shape[permutation.index(1)])] for a in range(self.shape[permutation.index(0)])])

    def act_on_first(self, action_matrix):
        """
            Act with a linear map on the first factor.

            In action_matrix, each row specifies one linear combination
        """
        return Tensor([sum([action_matrix[i,j] * self.matrices[j] for j in range(self.shape[0])]) for i in range(action_matrix.nrows())])

    def act_on_second(self, action_matrix):
        """
            Act with a linear map on the second factor
        """
        return Tensor([action_matrix * mat for mat in self.matrices])

    def act_on_third(self, action_matrix):
        """
            Act with a linear map on the third factor
        """
        return Tensor([mat * action_matrix.transpose() for mat in self.matrices])

    def act_on(self, which, action_matrix):
        if which == 0:
            return self.act_on_first(action_matrix)
        elif which == 1:
            return self.act_on_second(action_matrix)
        elif which == 2:
            return self.act_on_third(action_matrix)
        else:
            raise ValueError("Unrecognized factor: %s" % repr(which))

    def deform_to(self, shape):
        """
            Deform to the given shape by acting by random matrices.
        """
        A = tuple(random_matrix(FIELD, shape[i], self.shape[i]) for i in range(3))
        return self.act_on_first(A[0]).act_on_second(A[1]).act_on_third(A[2])

    def as_linform_matrix(self):
        """
            Return a matrix of linear forms that represents the flattening A --> Hom(B*, C).

            (To get the spaces of other flattenings, use permute(...) first.)
        """
        if not self.has_linform_matrix:
            self.make_linform_matrix()
        return self.linform_matrix

    def make_linform_matrix(self):
        polyring = PolynomialRing(FIELD, "x", self.shape[0])
        self.linform_matrix = sum([polyring.gens()[i] * self.matrices[i].change_ring(polyring) for i in range(self.shape[0])])
        self.has_linform_matrix = True

    def kronecker(self, other):
        """
            Return kronecker product of self with other. Other must be a Tensor
        """
        assert isinstance(other, Tensor)
        return Tensor([self.matrices[i].tensor_product(other.matrices[j]) for i in range(self.shape[0]) for j in range(other.shape[0])])

    def is_concise(self, factor=None):
        """
            Is this tensor concise w.r.t. to the given factor? If factor=None, return the conjunction of all three concisenesses.
        """
        if factor is None:
            return self.is_concise(0) and self.is_concise(1) and self.is_concise(2)
        else:
            perm = [0,1,2]
            perm[factor] = 0
            perm[0] = factor
            assert self.permute(perm).shape[0] == self.shape[factor]
            return self.permute(perm).flattening().rank() == self.shape[factor]

    def generic_rank(self):
        """
            Return the rank of a generic matrix in A* --> B x C.
        """
        M = self.as_linform_matrix()
        x = M.base_ring().gens()
        sample = M.substitute({xx:FIELD.random_element() for xx in x})
        return sample.rank()


def id_tensor(n):
    """
        Return the n x n x n identity tensor
    """
    result = Tensor((n,n,n))
    for i in range(n):
        result[i,i,i] = 1
    return result

def random_tensor_of_rank(rank, shape):
    """
        Within the given shape, create a random tensor of given rank.
    """
    result = Tensor(shape)
    for _ in range(rank):
        a = [FIELD.random_element() for __ in range(shape[0])]
        b = [FIELD.random_element() for __ in range(shape[1])]
        c = [FIELD.random_element() for __ in range(shape[2])]
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    result[i,j,k] += a[i] * b[j] * c[k]
    return result
generic_tensor_of_rank = random_tensor_of_rank

def random_partsym_of_rank(rank, shape):
    """
        Within the given shape, create a random partially symmetric tensor of given rank.
    """
    assert shape[1] == shape[2]
    result = Tensor(shape)
    for _ in range(rank):
        a = [FIELD.random_element() for __ in range(shape[0])]
        b = [FIELD.random_element() for __ in range(shape[1])]
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    result[i,j,k] += a[i] * b[j] * b[k]
    return result

def generic_brk(shape):
    return ceil(
        (shape[0]*shape[1]*shape[2]) / (shape[0]+shape[1]+shape[2]-2)
    )


class StructTensor(Tensor):
    """
        Object to hold a structure tensor of a finite-dimensional commutative algebra
    """

    def __init__(self, ideal):
        self.ideal = ideal
        assert self.ideal.dimension() == 0
        self.polyring = self.ideal.ring()
        assert self.polyring.base_ring() == FIELD
        self.basis = ideal.normal_basis()[::-1]  # always monomials
        assert self.basis[0] == 1
        self.dim = len(self.basis)

        super().__init__((self.dim, self.dim, self.dim))    # initialize as empty

        for i in range(self.dim):
            for j in range(self.dim):
                res = self.ideal.reduce(self.basis[i]*self.basis[j])
                for k in range(self.dim):
                    self[k,i,j] = res.monomial_coefficient(self.basis[k])

    def as_multiplication_table(self):
        """
            Return the "multiplication table" flattening, i.e. A* --> A* otimes A*
        """
        return self

    def as_selfaction(self):
        """
            Return the "self-action" flattening, i.e. A --> Hom(A,A), or equivalently A --> A* otimes A
        """
        return self.permute((2,1,0))
