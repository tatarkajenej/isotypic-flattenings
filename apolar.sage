def derivative_scalar(alpha, beta):
    """
        Return alpha! / beta!
    """
    result = 1
    for i in range(len(beta)):
        x = alpha[i]
        while x > beta[i]:
            result *= x
            x -= 1
    return result


class Apolarizer(object):
    """
        Object to compute general apolar ideals with no precomputations
    """

    def __init__(self, polyring):
        self.polyring = polyring
        assert self.polyring.base_ring() == FIELD
        self.n = self.polyring.ngens()

    def apolar_ideal(self, gens):
        """
            Return the apolar ideal of a vector subspace of self.polyring, as an ideal of self.polyring
        """

        d = max([g.degree() for g in gens])
        size = binomial(self.n + d, self.n)
        big_matrix = Matrix(FIELD, size, size * len(gens))

        basis_as_monomials = []
        basis_as_exponents = []
        for dd in range(d+1):
            for m in monomials_of_degree(self.polyring, dd):
                basis_as_monomials.append(m)
                basis_as_exponents.append(tuple(m.exponents()[0]))

        for g in range(len(gens)):
            assert gens[g] in self.polyring
            # construct the full catalecticant matrix of g and write it as a block in big_matrix
            for exponents, coef in gens[g].dict().items():
                for i in range(size):
                    e = basis_as_exponents[i]
                    m = tuple(exponents[t] - e[t] for t in range(self.n))
                    for t in range(self.n):
                        if m[t] < 0:    # e doesn't actually divide exponents
                            break
                    else:   # here we verified divisibility, so let's extract the contribution to the catalecticant
                        big_matrix[basis_as_exponents.index(m), i + g*size] += coef * derivative_scalar(exponents, e)

        ker = big_matrix.kernel()
        return self.polyring.ideal([sum([basis_as_monomials[i] * v[i] for i in range(size)]) for v in ker.gens()]) + degree_ideal(self.polyring, d+1)

    def __call__(self, I):
        return self.apolar_ideal(I)


def random_polynomial_in_degree(polyring, degree, homogeneous=False):
    """
        Randomly sample a polynomial from the given specification
    """
    if homogeneous:
        allowed_monomials = tuples_of_sum(polyring.ngens(), degree)
    else:
        allowed_monomials = tuples_of_sum_at_most(polyring.ngens(), degree)
    polydict = {}
    for m in allowed_monomials:
        polydict[m] = FIELD.random_element()
    return polyring(polydict)

def apolar_algebra(number_of_vars, dual_gen_degrees, homogeneous=False):
    """
        Generate the apolar algebra to randomly sampled dual generators in the given list of degrees.
    """
    n = number_of_vars
    R = PolynomialRing(FIELD, "x", n)
    apolar = Apolarizer(R)
    gens = [random_polynomial_in_degree(R, d, homogeneous=homogeneous) for d in dual_gen_degrees]
    return apolar(gens)
