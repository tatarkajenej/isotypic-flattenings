def vandermonde(samples, field=FIELD):
    result = Matrix(field, len(samples))
    for i in range(len(samples)):   # this could be parallelized, but it already seems so fast in cases of interest that I won't bother for now
        x = field(samples[i])
        entry = field(1)
        for j in range(len(samples)):
            result[i,j] = entry
            entry *= x
    return result


class Evaluator(object):
    """
        Base class for black box objects that take in arguments and evaluate to scalars. To be used for interpolation.
    """
    def __init__(self, nargs, func):
        assert isinstance(nargs, (Integer, int))
        self.func = func

    def __call__(self, *args):
        return self.func(*args)

    def fix_first(self, value):
        """
            Return an evaluator on one fewer arguments where the original first argument is set to a fixed value.
        """
        assert self.n > 0
        def new_func(*args):
            return self.func(value, *args)
        return Evaluator(self.n-1, new_func)

    def fix_last(self, value):
        """
            Return an evaluator on one fewer arguments where the original last argument is set to a fixed value.
        """
        assert self.n > 0
        def new_func(*args):
            return self.func(*args, value)
        return Evaluator(self.n-1, new_func)

class MatrixDetEvaluator(Evaluator):
    """
        Evaluator for determinants of matrices filled with affine forms
    """
    def __init__(self, polymatrix):
        if isinstance(polymatrix, MatrixDetEvaluator):
            self.n = polymatrix.n
            self.absolute_coef = Matrix(polymatrix.absolute_coef)
            self.linear_coefs = [Matrix(mat) for mat in polymatrix.linear_coefs]
        elif isinstance(polymatrix, tuple): # we received the absolute and linear coefs as (absolute, [linear])
            self.absolute_coef = polymatrix[0]
            self.linear_coefs = polymatrix[1]
            self.n = len(self.linear_coefs)
        else:
            polyring = polymatrix.base_ring()
            vars = polyring.gens()
            self.n = polyring.ngens()
            subs = {x:0 for x in vars}
            self.absolute_coef = polymatrix.substitute(subs).change_ring(FIELD)
            self.linear_coefs = []
            for x in vars:
                subs[x] = 1
                self.linear_coefs.append(polymatrix.substitute(subs).change_ring(FIELD) - self.absolute_coef)
                subs[x] = 0

    def __call__(self, *args):
        matrix = Matrix(self.absolute_coef)
        for i in range(self.n):
            matrix += FIELD(args[i]) * self.linear_coefs[i]
        return matrix.determinant()

    def fix_first(self, value):
        assert self.n > 0
        result = MatrixDetEvaluator(self)
        result.absolute_coef += value * result.linear_coefs.pop(0)
        result.n -= 1
        return result

    def fix_last(self, value):
        assert self.n > 0
        result = MatrixDetEvaluator(self)
        result.absolute_coef += value * result.linear_coefs.pop()
        result.n -= 1
        return result


class MutableLinearArray(object):
    """
        Object to hold and compute with and mutable multidimensional array of scalars, i.e. a potentially-high-order tensor.
    """
    def __init__(self, shape):
        # initialize as empty
        for s in shape:
            assert isinstance(s, (Integer, int))
        self.shape = shape
        self.len = prod(self.shape)
        self.n = len(self.shape)
        self.entries = {}

    def __getitem__(self, key):
        return self.entries.get(key, FIELD(0))
    def __setitem__(self, key, value):
        self.entries[key] = value
    def update(self, other):
        self.entries.update(other)
    def __len__(self):
        return self.len

    @parallel(ncpus=PARALLEL_NCPUS)
    def parallel_action_envelope(self, data_range):
        begin, end = data_range
        return self._act(self._temp_factor, self._temp_action_matrix, begin, end)

    def _act(self, factor, action_matrix, begin, end):
        result = {}
        start_point = position_to_multiindex(begin, tuple(self.shape[i] for i in range(self.n) if i != factor))
        pre = start_point[:factor]
        post = start_point[factor:]
        tup_post = tuple(post)
        index = begin
        while index < end:
            # do stuff
            axis = Matrix(FIELD, self.shape[factor], 1)
            for a in range(self.shape[factor]):
                axis[a,0] = self[tuple(pre)+(a,)+tup_post]
            axis = action_matrix * axis
            for a in range(self.shape[factor]):
                result[tuple(pre)+(a,)+tup_post] = axis[a,0]
            # increment index and pre
            index += 1
            for j in range(len(pre)):
                pre[j] += 1
                if pre[j] == self.shape[j]:
                    pre[j] = 0
                    # continue
                else:
                    break
            else:   # we've reached the end of pre
                # so now we go increment post
                for j in range(len(post)):
                    post[j] += 1
                    if post[j] == self.shape[factor+j+1]:
                        post[j] = 0
                        # continue
                    else:
                        tup_post = tuple(post)
                        break
                else:   # we've reached the end of post
                    if index < end:
                        raise IndexError("MutableLinearArray._act overflowed its array indexing but didn't reach end.")
        return result

    def act_on(self, factor, action_matrix, parallel=False):
        """
            Apply the linear function induced by the matrix to "axes" of the array in the designated direction.
        """
        comp_len = Integer(len(self)/self.shape[factor])
        if parallel:
            self._temp_action_matrix = action_matrix
            self._temp_factor = factor
            axis_batches = generate_batches(comp_len)
            parallel_output = list(self.parallel_action_envelope(axis_batches))
            self.entries = {}
            for data in parallel_output:
                # for key in data[1]:
                #     # assert key not in self.entries
                #     if key in self.entries:
                #         raise KeyError("Duplicate key %s on reading batch %s" % (str(key), str(data[0][0])))
                #     self.entries[key] = data[1][key]
                self.entries.update(data[1])
        else:
            self.entries = self._act(factor, action_matrix, 0, comp_len)

    def all_keys(self, begin=None, end=None):
        """
            Generator over all valid keys of this array.
        """
        if begin is None:
            begin = 0
        if end is None:
            end = len(self)
        index = begin
        mutkey = self.pos_to_key(index, return_cls=list)
        while index < end:
            yield tuple(mutkey)
            index += 1
            # increment mutkey
            for j in range(self.n):
                mutkey[j] += 1
                if mutkey[j] == self.shape[j]:
                    mutkey[j] = 0
                    # continue
                else:
                    break
            else:
                if index < end:
                    raise ValueError("Something's wrong, inner loop in all_keys exited without a break but index isn't at the end yet. Probably a faulty batch range.")

    def pos_to_key(self, pos, return_cls=tuple):
        return return_cls(position_to_multiindex(pos, self.shape))
    def key_to_pos(self, key):
        result = sum((key[i]*self.shape[i]) for i in range(self.n))


class PolynomialInterpolator(object):
    """
        Object handling the interpolation of a polynomial in several variables by Vandermonding with respect to one variable at a time.
    """
    array_cls = MutableLinearArray

    def __init__(self, polyring, evaluation):
        self.polyring = polyring
        assert self.polyring.base_ring() == FIELD
        self.n = self.polyring.ngens()
        self.evaluation = evaluation    # black box object, something that's callable (with the right number of arguments) and spits out numbers

    @parallel(ncpus=PARALLEL_NCPUS)
    def parallel_eval_envelope(self, key_range):
        result = {}
        for key in self._temp_array.all_keys(*key_range):
            result[key] = self.evaluation(*key)
        return result

    def interpolate(self, deg_bounds, parallel=False):
        """
            Interpolate your black box function, assuming it is a polynomial of degree at most deg_bound. Do this by interpolating several specializations in the last variable.
        """
        if isinstance(deg_bounds, (Integer, int)):
            deg_bounds = (deg_bounds,) * self.n
        assert len(deg_bounds) == self.n

        # prepare univariate Vandermonde matrices
        self._temp_vandermondes = []
        self._temp_vandermonde_inverses = []
        for i in range(self.n):
            assert 0 <= deg_bounds[i] < PRIME   # we can only interpolate up to degrees less than or equal to the size of our field
            try:
                previous = deg_bounds.index(deg_bounds[i], 0, i)
                self._temp_vandermondes.append(self._temp_vandermondes[previous])
                self._temp_vandermonde_inverses.append(self._temp_vandermonde_inverses[previous])
            except ValueError: # deg_bounds[i] has not appeared yet
                self._temp_vandermondes.append(vandermonde(list(FIELD)[:deg_bounds[i]+1]))  # we always sample on [0, ..., d]
                self._temp_vandermonde_inverses.append(self._temp_vandermondes[-1].inverse())

        # prepare the initial array of evaluations
        self._temp_array = self.array_cls(tuple(d+1 for d in deg_bounds))
        if parallel:
            key_batches = generate_batches(len(self._temp_array))
            parallel_output = self.parallel_eval_envelope(key_batches)
            for data in parallel_output:
                self._temp_array.update(data[1])
        else:
            for key in self._temp_array.all_keys():
                self._temp_array[key] = self.evaluation(*key)

        # act with vandermonde inverse
        for i in range(self.n):
            self._temp_array.act_on(i, self._temp_vandermonde_inverses[i], parallel=parallel)

        # read off polyomial
        return self.polyring(self._temp_array.entries)

    __call__ = interpolate

    def fix_first(self, value):
        new_polyring = PolynomialRing(FIELD, self.polyring.gens()[:-1])
        new_eval = self.evaluation.fix_first(value)
        return PolynomialInterpolator(new_polyring, new_eval)

    def fix_last(self, value):
        new_polyring = PolynomialRing(FIELD, self.polyring.gens()[:-1])
        new_eval = self.evaluation.fix_last(value)
        return PolynomialInterpolator(new_polyring, new_eval)


class PnMatrix(object):
    """
        Object to effectively compute minors of a matrix filled with multivariate affine forms.
    """
    def __init__(self, tensor):
        assert isinstance(tensor, Tensor)
        linform = tensor.as_linform_matrix()
        self.polyring = PolynomialRing(FIELD, linform.base_ring().gens()[1:])
        self.matrix = linform.substitute({linform.base_ring().gen(0):FIELD(1)}).change_ring(self.polyring)
        self.vars = self.polyring.gens()
        self.n = len(self.vars)
        self.absolute_coef = tensor.matrices[0]
        self.linear_coefs = tensor.matrices[1:]

    def __str__(self):
        return "P^%i interpolator of\n" % self.n + self.matrix.__str__()
    __repr__ = __str__

    def __getitem__(self, key):
        return self.matrix[key]

    def nrows(self):
        return self.matrix.nrows()
    def ncols(self):
        return self.matrix.ncols()

    def minor(self, rows, cols, parallel=False):
        """
            Compute the indicated minor by evaluating at points. An apriori monomial support must be given, unless it can be inferred from equidegree.
        """
        assert len(rows) == len(cols)
        self._temp_size = len(rows)
        assert self._temp_size < len(FIELD)    # if there are fewer points in the FIELD than the degree of our minor, we would have a hard time interpolating...

        # self.handle_sampling(size, resample, parallel=parallel)
        self._temp_submatrix_absolute = self.absolute_coef.matrix_from_rows_and_columns(rows, cols)
        self._temp_submatrix_linear = [M.matrix_from_rows_and_columns(rows, cols) for M in self.linear_coefs]

        return self._compute_minor(parallel=parallel)

    def _compute_minor(self, parallel=False):
        self._temp_minor_eval = MatrixDetEvaluator((self._temp_submatrix_absolute, self._temp_submatrix_linear))
        self._temp_interpolator = PolynomialInterpolator(self.polyring, self._temp_minor_eval)
        return self._temp_interpolator(self._temp_size, parallel=parallel)

    def random_deformed_minor(self, size, parallel=False):
        left_deform = random_matrix(FIELD, size, self.nrows())
        right_deform = random_matrix(FIELD, self.ncols(), size)

        self._temp_size = size
        self._temp_submatrix_absolute = left_deform * self.absolute_coef * right_deform
        self._temp_submatrix_linear = [left_deform * M * right_deform for M in self.linear_coefs]

        return self._compute_minor(parallel=parallel)

    def minor_ideal(self, size, parallel=False):
        """
            Try to compute the ideal of self.polyring generated by all minors.

            Computing all minors is usually infeasible, so we try to do this in a randomized fashion, adding minors one at a time.
        """
        I = self.polyring.ideal(0)
        attempts_since_change = 0
        while True:
            minor = self.random_deformed_minor(size, parallel=parallel)
            I_new = I + self.polyring.ideal(minor)
            if self.n > 1:
                I_new = I_new.intersection(self.polyring.ideal(1))
            if I_new == I:
                attempts_since_change += 1
                if attempts_since_change >= PNMATRIX_IDEAL_STOP:
                    break
            else:
                attempts_since_change = 0
                I = I_new
        return I

    def determinant(self, parallel=False):
        return self.minor(range(self.nrows()), range(self.ncols()), parallel=parallel)

    def generic_rank(self):
        random_sub = self.absolute_coef
        for l in self.linear_coefs:
            random_sub += FIELD.random_element() * l
        return random_sub.rank()

    def stack_rows(self):
        """
            Put all of the coefficients on top of each other, as if the parameter space were tensored with the codomain.
        """
        result = Matrix(FIELD, self.nrows() * (self.n+1), self.ncols())
        for i in range(self.n+1):
            if i < self.n:
                addition = self.linear_coefs[i]
            else:
                addition = self.absolute_coef
            result[i*self.nrows():(i+1)*self.nrows(), :] = addition
        return result

    def stack_cols(self):
        """
            Put all of the coefficients next to each other, as if the parameter space were tensored with the domain.
        """
        result = Matrix(FIELD, self.nrows(), self.ncols() * (self.n+1))
        for i in range(self.n+1):
            if i < self.n:
                addition = self.linear_coefs[i]
            else:
                addition = self.absolute_coef
            result[:, i*self.ncols():(i+1)*self.ncols()] = addition
        return result
