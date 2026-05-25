#!/usr/bin/env bash

./table_nodes_mpc.py timings_mpc_*_1.log > table_nodes_mpc_1.tex
./table_nodes_mpc.py timings_mpc_*_2.log > table_nodes_mpc_2.tex
./table_nodes_mpc.py timings_mpc_*_3.log > table_nodes_mpc_3.tex
./table_nodes.py timings_mpt_*.log > table_mpt.tex
./table_nodes.py timings_basic_row_*.log > table_nodes.tex
./table.py timings_1_*.log timings_2_*.log timings_3_*.log timings_4_*.log timings_5_*.log timings_6_*.log > table_1.tex