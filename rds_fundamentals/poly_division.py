#!/usr/env python3
# 
# Everywhere we assume coefficients go from c_0 to c_n, i.e. c_n * x^n + ... + c_0 = [c_0, c_1, ..., c_n]
# And everywhere we assume that c_n != 0
#
# I do it 1) for practice 2) because I'm not sure anymore if parity check matrix from RDS spec gives us proper division remainder


def divide_mod2(dividend, divisor):
    quotient = []
    remainder = dividend.copy()

    # don't assume dividend[-1] != 0
    start = len(dividend) - 1
    while dividend[start] == 0:
        start -= 1

    for i in range(start, len(divisor) - 2, -1):
        if remainder[i] == 0:
            quotient.append(0)  # must be reversed at the end
        else:
            quotient.append(1)
            for j, coefficient in enumerate(divisor[::-1]):
                remainder[i - j] = (remainder[i - j] + coefficient) % 2
                
    return quotient[::-1], remainder[:len(divisor) - 1]


def print_poly(coefficients):
    power = len(coefficients) - 1
    result = []
    for c in coefficients[::-1]:
        if c:
            coef = f'{c} * ' if c != 1 else ''
            var = 'x' if power == 1 else (f'x^{power}' if power != 0 else '1')
            result.append(coef + var)
        power -= 1
    print(' + '.join(result) if result else '0')


def main():
    # dividend = [1,0,1,1,0,1,1,0,0,0,1,1,0,1,1,0,0,1,1,1,0,1,1,1,0,1]
    # divisor = [0,1,0,1,1,1,0,1,1,0,1]
    dividend = [1] + [0] * 340 + [1]  # that's it, RDS generator polynomial is a divisor for x^341 - 1, certainly not for x^26 - 1
    dividend = [0] * 341
    dividend[-26] = 1
    divisor = [1,0,0,1,1,1,0,1,1,0,1]
    
    quotient, remainder = divide_mod2(dividend, divisor)

    for p in (dividend, divisor, quotient, remainder):
        print_poly(p)


if __name__ == '__main__':
    main()

