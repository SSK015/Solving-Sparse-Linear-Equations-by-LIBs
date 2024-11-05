#!/bin/bash

export OMPI_MCA_btl_openib_allow_ib=1
export FI_PROVIDER=psm3

mpiexec -n 4 ./petscImpl.py
