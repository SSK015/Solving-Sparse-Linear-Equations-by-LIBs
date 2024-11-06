#!/usr/bin/python3

import sys, petsc4py
from petsc4py import PETSc

petsc4py.init(sys.argv)

m, n  = 32, 32
hx = 1.0/(m-1)
hy = 1.0/(n-1)

# create sparse matrix
A = PETSc.Mat()
partt = A.create(PETSc.COMM_WORLD)
A.setSizes([m*n, m*n])
A.setType('aij') # sparse
A.setFromOptions()

# precompute values for setting
# diagonal and non-diagonal entries
diagv = 2.0/hx**2 + 2.0/hy**2
offdx = -1.0/hx**2
offdy = -1.0/hy**2

Istart, Iend = A.getOwnershipRange()
for I in range(Istart, Iend) :
    A[I,I] = diagv
    i = I//n    # map row number to
    j = I - i*n # grid coordinates
    if i> 0  : J = I-n; A[I,J] = offdx
    if i< m-1: J = I+n; A[I,J] = offdx
    if j> 0  : J = I-1; A[I,J] = offdy
    if j< n-1: J = I+1; A[I,J] = offdy

# communicate off-processor values
# and setup internal data structures
# for performing parallel operations
A.assemblyBegin()
A.assemblyEnd()


# create linear solver,
ksp = PETSc.KSP()
ksp.create(PETSc.COMM_WORLD)

# use conjugate gradients method
ksp.setType('bcgs')
# and incomplete Cholesky
# ksp.getPC().setType('icc')

# obtain sol & rhs vectors
x, b = A.getVecs()
x.set(0)
b.set(1)

# m_start, m_end = partt.getOwnershipRange()
comm = PETSc.COMM_WORLD
rank = comm.Get_rank()
print(rank)
# print(m_start)
# and next solve
ksp.setOperators(A)
ksp.setFromOptions()
ksp.solve(b, x)

r = b.duplicate()
A.mult(x, r)  # r = Ax
r.scale(-1.0)  # r = -Ax
r.axpy(1.0, b)  # r = b - Ax

residual_norm = r.norm()

print(":", residual_norm)