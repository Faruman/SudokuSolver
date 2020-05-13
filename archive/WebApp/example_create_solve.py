# -*- coding: utf-8 -*-
"""
Created on Fri May  8 21:11:19 2020

@author: vinze
"""

import os
os.chdir("C:/Users/vinze/OneDrive/GitHub/Sudoku-solving-algorithm")

os.getcwd()
from packages.sudokusolver import sudokusolver
from packages.sudokucreator import sudokucreator


creator = sudokucreator()
solver = sudokusolver()


time_to_create, grid_input, output = creator.create_sudoku()
test = grid_input.copy()
validity, time, iterations, result = solver.solve(grid_input, 5)

if (result == output).all() :
    print('True')
