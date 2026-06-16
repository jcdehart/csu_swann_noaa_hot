#!/bin/bash

unset LD_LIBRARY_PATH

julia --project=. objective_simplex.jl
