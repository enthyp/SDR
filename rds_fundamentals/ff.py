# TODO: rest of the code from notebook can be used for tests
from abc import ABC, abstractmethod
from itertools import zip_longest


#######
# UTILS
#######
def base_repr(n, base):
    r = []

    while n != 0:
        r.append(n % base)
        n //= base

    return r


def base_val(seq, base):
    return sum(s * base ** i for i, s in enumerate(seq))


def ext_euclidean(a, b):
    # a > b
    # r_{i - 1}, r_i starting from i = 1
    r_i_m_1, r_i = a, b

    # a * s_i + b * t_i = r_i for all i, definition of s_i and t_i so smart! 
    s_i_m_1, t_i_m_1 = 1, 0
    s_i, t_i = 0, 1  

    while r_i != 0:
        # r_{i + 1} = r_{i - 1} - q_i * r_i
        r_i_p_1 = r_i_m_1 % r_i
        q_i = r_i_m_1 // r_i
        
        s_i_p_1 = s_i_m_1 - q_i * s_i
        t_i_p_1 = t_i_m_1 - q_i * t_i

        r_i_m_1, r_i = r_i, r_i_p_1
        s_i_m_1, s_i = s_i, s_i_p_1
        t_i_m_1, t_i = t_i, t_i_p_1

    return r_i_m_1, s_i_m_1, t_i_m_1


# a, b: lists of coefficients in given finite field [c_0, ..., c_k], c_k != 0
def add_poly(a, b, field):
    zipped = zip_longest(a, b, fillvalue=0)  # again, 0 is zero
    return [field.add(a, b) for (a, b) in zipped]


def subtract_poly(a, b, field):
    zipped = zip_longest(a, b, fillvalue=0)  # again, 0 is zero
    return [field.subtract(a, b) for (a, b) in zipped]


def mult_poly(a, b, field):
    deg_a, deg_b = len(a) - 1, len(b) - 1
    result = [0] * (deg_a + deg_b + 1)  # using the fact that number zero actually represents zero in all our finite fields

    for i in range(deg_a + 1):
        for j in range(deg_b + 1):
            coef = field.mult(a[i], b[j])
            position = i + j
            result[position] = field.add(result[position], coef)

    return result


def div_poly(a, b, field):
    # careful - highest degree coefficient must be non-zero
    deg_a, deg_b = len(a) - 1, len(b) - 1    
    quotient, remainder = [], a.copy()

    # required to divide a's coefficients by b's highest coefficient
    # division always works, because it's a field
    b_coef_inv = field.mult_inv(b[-1])

    for i in range(deg_a - deg_b + 1):
        multiplier = field.mult(remainder[-(i + 1)], b_coef_inv)
        quotient.append(multiplier)  # needs to be reversed at the end

        for j in range(deg_b + 1):
            coef = field.mult(multiplier, b[deg_b - j])
            remainder[-(i + 1) - j] = field.subtract(remainder[-(i + 1) - j], coef)

    # remove trailing zeros from remainder (crucial for extended Euclidean algorithm to work
    # since we pass it to div_poly again)
    non_zero_indexes = [i for i, r_i in enumerate(remainder) if r_i != 0]
    return quotient[::-1] if quotient else [0], remainder[:non_zero_indexes[-1] + 1] if non_zero_indexes else [0]


def ext_euclidean_poly(a, b, field):
    deg_a, deg_b = len(a) - 1, len(b) - 1
    assert deg_a > deg_b
    # r_{i - 1}, r_i starting from i = 1
    r_i_m_1, r_i = a, b

    # a * s_i + b * t_i = r_i for all i
    # again, using the fact that 0 and 1 represent neutral elements of our finite fields!
    s_i_m_1, t_i_m_1 = [1], [0]
    s_i, t_i = [0], [1]

    # iterate until we get a pair where one polynomials divides the other
    while any(element != 0 for element in r_i):
        # r_{i + 1} = r_{i - 1} - q_i * r_i
        q_i, r_i_p_1 = div_poly(r_i_m_1, r_i, field)

        s_i_p_1 = subtract_poly(s_i_m_1, mult_poly(q_i, s_i, field), field)
        t_i_p_1 = subtract_poly(t_i_m_1, mult_poly(q_i, t_i, field), field)
        
        r_i_m_1, r_i = r_i, r_i_p_1
        s_i_m_1, s_i = s_i, s_i_p_1
        t_i_m_1, t_i = t_i, t_i_p_1

    return r_i_m_1, s_i_m_1, t_i_m_1  # TODO: might need to eliminate trailing zeroes


#######################
# FINITE FIELD BUILDERS
#######################
# full definition would include 0 and 1 (neutral elements), but in our considerations they will be literally 0 and 1
class FieldOps(ABC):
    @abstractmethod
    def add(self, a, b):
        pass

    @abstractmethod
    def add_inv(self, a):
        pass

    def subtract(self, a, b):
        return self.add(a, self.add_inv(b))
    
    @abstractmethod
    def mult(self, a, b):
        pass

    def pow(self, a, p):
        # could be implemented with an order of log(p) multiplications but who cares?
        if p == 0:
            return 1
        if p < 0:
            a = self.mult_inv(a)

        result = a
        for _ in range(abs(p) - 1):
            result = self.mult(result, a)
        return result
    
    @abstractmethod
    def mult_inv(self, a):
        pass


def ff_for_prime(p):
    class Ops(FieldOps):
        def add(self, a, b):
            return (a + b) % p

        def add_inv(self, a):
            return (-a) % p

        def mult(self, a, b):
            return (a * b) % p

        def mult_inv(self, a):
            _, _, inv = ext_euclidean(p, a)
            return inv

    return Ops()


# p: order of the base field
# field: operations on elements of the base field (represented as numbers 0, ..., p - 1)
# irr_p: list of irreducible polynomial coefficients, [p_0, ..., p_m]
def ff_for_irreducible_polynomial(p, field, irr_p):
    # number to polynomial
    def n2p(n):
        poly = base_repr(n, p)
        assert len(poly) < len(irr_p), f'{n} is too large for a correct representation of this field\'s element!'
        return poly


    # polynomial to number
    def p2n(poly):
        return base_val(poly, p)
        
    
    class Ops(FieldOps):
        def add(self, a, b):
            a_p, b_p = n2p(a), n2p(b)
            return p2n(add_poly(a_p, b_p, field))

        def add_inv(self, a):
            a_p = n2p(a)
            return p2n([field.add_inv(a_i) for a_i in a_p])

        def mult(self, a, b):
            a_p, b_p = n2p(a), n2p(b)
            ab_p = mult_poly(a_p, b_p, field)
            q, rem = div_poly(ab_p, irr_p, field)
            return p2n(rem)

        def mult_inv(self, a):
            _, _, inv = ext_euclidean_poly(irr_p, n2p(a), field)
            return p2n(inv)

    return Ops()


# q: order of the base field
# field: operations on elements of the base field (represented as numbers 0, ..., q - 1)
# prim_p: list of primitive polynomial coefficients [p_0, ..., p_m]
def ff_for_primitive_polynomial(q, field, prim_p):
    # number to polynomial
    def n2p(n):
        poly = base_repr(n, q)
        assert len(poly) < len(prim_p), f'{n} is too large for a correct representation of this field\'s element!'
        return poly


    # polynomial to number
    def p2n(poly):
        return base_val(poly, q)
        
    
    def remove_trailing_zeroes(poly):
        for i in range(len(poly) - 1, -1, -1):
            if poly[i] != 0:
                return poly[:i + 1]
        return [0]

    deg_prim_p = len(prim_p) - 1
    
    # find what polynomial of degree < deg_prim_p can replace alpha^{deg_prim_p} by using the primitive polynomial
    rhs_replacement = remove_trailing_zeroes([field.add_inv(p_i) for p_i in prim_p[:-1]])

    # let's build the lookup tables q-ary sequences (numbers) <-> primitive root powers
    num_to_power, power_to_num = {}, {}
    
    for power in range(q ** deg_prim_p - 1):
        poly = [0] * power + [1]
        
        while len(poly) > deg_prim_p:
            # use rhs_replacement to reduce degree of poly by rewriting the highest power
            hp = len(poly) - 1

            # multiply (p_{hp - deg_prim_p} * alpha^{hp - deg_prim_p}) by the RHS
            replacement = mult_poly([0] * (hp - deg_prim_p) + [poly[hp]], rhs_replacement, field)

            poly = add_poly(poly, replacement, field)
            poly[hp] = 0
            
            # ensure poly's getting shorter each iteration
            poly = remove_trailing_zeroes(poly)

        num = p2n(poly)
        num_to_power[num] = power
        power_to_num[power] = num

     # num_to_power maps all non-zero field elements
    mult_period = len(num_to_power)
    
    class Ops(FieldOps):
        def __init__(self):
            # let's add this class property so that we can verify with the table in Appendix C
            self.power_to_num = power_to_num

        def add(self, a, b):
            a_p, b_p = n2p(a), n2p(b)
            return p2n(add_poly(a_p, b_p, field))

        def add_inv(self, a):
            a_p = n2p(a)
            return p2n([field.add_inv(a_i) for a_i in a_p])

        def mult(self, a, b):
            if a == 0 or b == 0:
                return 0
            pow_a, pow_b = num_to_power[a], num_to_power[b]
            pow_ab = (pow_a + pow_b) % mult_period
            return power_to_num[pow_ab]

        def mult_inv(self, a):
            # no messed up Euclideans this time, yay
            # 0 not handled
            pow_a = num_to_power[a]
            return power_to_num[(-pow_a) % mult_period]

    return Ops()
