import time
T_START = time.time()          # before the heavy imports, to measure them

import numpy as np
from mpi4py import MPI
from gwlens.lens import nfw 


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    w_min, w_max = 0.001, 100
    y_min, y_max = 0.001, 5
    rs_min, rs_max = 0.00001,1
    num_w = 100
    num_y = 100
    num_rs = 100

    w_values = np.logspace(w_min, w_max, num_w)
    y_values = np.logspace(y_min, y_max, num_y)
    rs_values = np.logspace(rs_min,rs_max,num_rs)
    output_file = "nfw_amplification_grid.h5"

    if rank == 0:
        print(f"Grid: {num_w} x {num_y} = {num_w * num_y} points")
        print(f"Output: {output_file}")

    nfw.compute_nfw_lens_grid_mpi(w_values, y_values, rs, output_file, t_start=T_START)


if __name__ == "__main__":
    main()      # EVERY rank calls this — collectives require it
