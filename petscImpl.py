#!/usr/bin/python3
import os
import sys, petsc4py
from petsc4py import PETSc
import struct

num_files = 4
num_process = 0
path_prefix = "./data/small/"

A = PETSc.Mat()
global_rows = global_cols = 0

def read_binary_file_pre(file_name):
    with open(file_name, 'rb') as f:
        global_num_rows = struct.unpack('i', f.read(4))[0]  
        global_num_cols = struct.unpack('i', f.read(4))[0]  
        num_rows = struct.unpack('i', f.read(4))[0]         
        num_cols = struct.unpack('i', f.read(4))[0]         
        num_nonzeros = struct.unpack('i', f.read(4))[0] 
    return global_num_rows, global_num_cols, num_rows, num_cols, num_nonzeros

def read_binary_file(file_name):
    with open(file_name, 'rb') as f:
        global_num_rows = struct.unpack('i', f.read(4))[0]  
        global_num_cols = struct.unpack('i', f.read(4))[0]  
        num_rows = struct.unpack('i', f.read(4))[0]         
        num_cols = struct.unpack('i', f.read(4))[0]         
        num_nonzeros = struct.unpack('i', f.read(4))[0] 
        for i in range(num_nonzeros):
            II = struct.unpack('i', f.read(4))[0]
            JI = struct.unpack('i', f.read(4))[0]
            Val = struct.unpack('d', f.read(8))[0]
            II = II - 1
            JI = JI - 1
            A[II, JI] = Val
    return global_num_rows, global_num_cols, num_rows, num_cols, num_nonzeros


relative_path = './data/small/' + "A_coo_bin.0"
absolute_path = os.path.abspath(relative_path)
global_cols, global_rows, _, _, _ =  read_binary_file_pre(absolute_path)

A.setSizes([global_cols, global_rows])
A.setType('aij') # sparse
A.setFromOptions()

def distribute_tasks(num_files):
    comm = PETSc.COMM_WORLD
    size = comm.getSize()
    rank = comm.getRank()
    tasks_per_proc = num_files // size
    remainder = num_files % size 
    start = rank * tasks_per_proc + min(rank, remainder)
    end = start + tasks_per_proc + (1 if rank < remainder else 0)
    tasks = list(range(start, end))

    return tasks



petsc4py.init(sys.argv)

comm = PETSc.COMM_WORLD
rank = comm.Get_rank()
size = comm.getSize()
num_process = size


tasks = distribute_tasks(num_files)
for item in tasks:
    relative_path = path_prefix + "A_coo_bin." + str(item)
    absolute_path = os.path.abspath(relative_path)
    read_binary_file(absolute_path)
# print(tasks)

A.assemblyBegin()
A.assemblyEnd()



ksp = PETSc.KSP()
ksp.create(PETSc.COMM_WORLD)

ksp.setType('bcgs')
# ksp.getPC().setType('icc')

# obtain sol & rhs vectors
x, b = A.getVecs()
x.set(0)
b.set(1)

ksp.setOperators(A)
ksp.setFromOptions()
ksp.solve(b, x)



r = b.duplicate()
A.mult(x, r)  # r = Ax
r.scale(-1.0)  # r = -Ax
r.axpy(1.0, b)  # r = b - Ax

residual_norm = r.norm()

print(":", residual_norm)