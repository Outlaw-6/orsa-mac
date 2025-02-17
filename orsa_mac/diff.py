#!/usr/bin/env python3

# Forward Mode Automatic Differentiation

class Dual:
    def __init__(self, real, dual):
        self.real = real
        self.dual = dual
        # real + dual epsilon

    def __add__(self, other):
        if (isinstance(other, Dual)):
            real = self.real + other.real
            dual = self.dual + other.dual
            return Dual(real,dual)
        return Dual(self.real + other, self.dual)
    __radd__ = __add__

    def __mul__(self, other):
        if (isinstance(other, Dual)):
            real = self.real * other.real
            dual = self.real * other.dual + self.dual * other.real
            return Dual(real,dual)
        return Dual(self.real * other, self.dual * other)
    __rmul__ = __mul__

def diff(f, x):
    return f(Dual(x,1)).dual
