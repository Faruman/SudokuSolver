import numpy as np
import time

class sudokusolver:
    def __init__(self):
        self.full_set = np.arange(1, 10, 1)
        self.grid_possibilities = np.empty(shape=[9, 9], dtype=object)
        self.grid_possibilities.fill([])
        self.grid_result = np.empty(shape=[9, 9], dtype=object)
        self.grid_output = np.empty(shape=[9, 9], dtype=object)

    def reinit(self, input):
        self.full_set = np.arange(1, 10, 1)
        self.grid_possibilities = np.empty(shape=[9, 9], dtype=object)
        self.grid_possibilities.fill([])
        self.grid_result = input
        self.grid_output = input

    def create_possible_values(self):
        for row in range(0, 9):
            for column in range(0, 9):
                if self.grid_result[row][column] == 0:
                    # Take full Grid and substract values that are already in the row
                    poss_r = np.setdiff1d(self.full_set, self.grid_result[row])
                    # Take all values that are possible in a row and substract those already in the column
                    poss_c = np.setdiff1d(poss_r, self.grid_result[:, column])
    
                    # Check for values in subgrid
                    row_ind = (row // 3) * 3      # Use //3 to get whole numbers
                    col_ind = (column // 3) * 3
                    poss_overall = np.setdiff1d(
                        poss_c, self.grid_result[row_ind:row_ind + 3, col_ind:col_ind + 3].flatten())
                    self.grid_possibilities[row, column] = list(poss_overall)
        self.remove_pairs()
        return


    def check_single(self):
        for row in range(0, 9):
            for column in range(0, 9):
                # Check if only one value is possible for a cell! If so, assign it to the cell
                if np.size(self.grid_possibilities[row, column]) == 1:
                    result = self.grid_possibilities[row][column][0]
                    self.grid_result[row, column] = result
                    self.grid_possibilities[row, column] = []
                    self.create_possible_values()
                    self.check_single()
        return


    def check_unique(self):
        # Check for values that only appear once to be possible in each row
        for row in range(0, 9):
            row_poss = np.concatenate(
                self.grid_possibilities[row], axis=None).astype(int)
            poss_list = list(row_poss)
            unique_value = [x for x in poss_list if poss_list.count(x) == 1]
            for column in range(0, 9):
                if len(unique_value) >= 1:
                    for i in range(0, len(unique_value)):
                        if unique_value[i] in self.grid_possibilities[row, column]:
                            self.grid_possibilities[row, column] = []
                            self.grid_result[row, column] = unique_value[i]
                            self.create_possible_values()

        # Check for values that only appear once to be possible in each row
        for column in range(0, 9):
            column_poss = np.concatenate(
                self.grid_possibilities[0:, column], axis=None).astype(int)
            column_list = list(column_poss)
            unique_value = [x for x in column_list if column_list.count(x) == 1]
            for row in range(0, 9):
                if len(unique_value) >= 1:
                    for i in range(0, len(unique_value)):
                        if unique_value[i] in self.grid_possibilities[row, column]:
                            self.grid_possibilities[row, column] = []
                            self.grid_result[row, column] = unique_value[i]
                            self.create_possible_values()

        # Check for values that only appear once to be possible in each subgrid
        for sub_row in np.arange(0, 9, 3):
            for sub_col in np.arange(0, 9, 3):
                row_ind = (sub_row // 3) * 3
                col_ind = (sub_col // 3) * 3
                subgrid_possibilities = self.grid_possibilities[row_ind:row_ind +
                                                           3, col_ind:col_ind + 3].flatten()
                subgrid_poss = np.concatenate(
                    subgrid_possibilities, axis=None).astype(int)
                subgrid_list = list(subgrid_poss)
                unique_value = [
                    x for x in subgrid_list if subgrid_list.count(x) == 1]
                for subcol in range(col_ind, col_ind + 3):
                    for subrow in range(row_ind, row_ind + 3):
                        if len(unique_value) >= 1:
                            for i in range(0, len(unique_value)):
                                if unique_value[i] in self.grid_possibilities[subrow, subcol]:
                                    self.grid_possibilities[subrow, subcol] = []
                                    self.grid_result[subrow, subcol] = unique_value[i]
                                    self.create_possible_values()
        self.check_single()
        return


    def remove_pairs(self):
        # Remove Pairs from rows
        for row in range(0, 9):
            pair = self.find_pairs(self.grid_possibilities[row])
            if pair != []:
                value_1 = pair[0]
                value_2 = pair[1]
                for c in range(0, 9):
                    if (self.grid_possibilities[row, c] != pair and self.grid_possibilities[row, c] != []):
                        if value_1 in self.grid_possibilities[row, c]:
                            self.grid_possibilities[row, c].remove(value_1)
                        if value_2 in self.grid_possibilities[row, c]:
                            self.grid_possibilities[row, c].remove(value_2)

        # Remove Pairs from columns
        for column in range(0, 9):
            pair = self.find_pairs(self.grid_possibilities.T[column])
            if pair != []:
                value_1 = pair[0]
                value_2 = pair[1]
                for c in range(0, 9):
                    if (self.grid_possibilities.T[column, c] != pair and self.grid_possibilities.T[column, c] != []):
                        if value_1 in self.grid_possibilities.T[column, c]:
                            self.grid_possibilities.T[column, c].remove(value_1)
                        if value_2 in self.grid_possibilities.T[column, c]:
                            self.grid_possibilities.T[column, c].remove(value_2)

        return


    def solve(self, grid_input, tries):
        self.reinit(grid_input)
        iterration = 0
        start_time = time.time()
        while 0 in self.grid_result and tries > 0:
            self.create_possible_values()
            self.check_single()
            self.check_unique()
            iterration += 1
            tries -= 1
            self.grid_output = self.grid_result.copy()
        if self.solution_is_valid() == True:
            validity = True
        else:
            self.bf_solve()
            if self.solution_is_valid() == True:
                validity = True
            else:
                validity = False
        return validity, (time.time() - start_time), iterration, self.grid_output


    # ----- Helper Function
    def find_pairs(self, row):
        pair = []
        to_compare = []
        for col in row:
            if np.size(col) == 2:
                to_compare.append(col)
                if len(to_compare) > 1:
                    try:
                        pair = [x for x in to_compare if to_compare.count(x) == 2]
                        pair = pair[0]
                    except:
                        continue
        return pair


    def solution_is_valid(self):
        valid_rows = 0
        valid_columns = 0
        valid_subgrids = 0
        for row in self.grid_output:
            if np.array_equal(np.sort(row), self.full_set) == True:
                valid_rows += 1

        for column in self.grid_output.T:
            if np.array_equal(np.sort(column), self.full_set) == True:
                valid_columns += 1

        for subcol in np.arange(0, 9, 3):
            for subrow in np.arange(0, 9, 3):
                sorted_grid = np.sort(
                    self.grid_output[subcol:subcol + 3, subrow:subrow + 3].flatten())
                if np.array_equal(sorted_grid, self.full_set) == True:
                    valid_subgrids += 1
        if valid_rows == 9 and valid_subgrids == 9 and valid_subgrids == 9:
            return True
        else:
            return False


    def is_possible(self, row, column, value):
        # Check row
        for i in range(0, 9):
            if self.grid_result[row][i] == value:
                return False
        # Check column
        for i in range(0, 9):
            if self.grid_result[i][column] == value:
                return False
        # Check subgrid
        row0 = (row // 3) * 3
        column0 = (column // 3) * 3
        for i in range(0, 3):
            for j in range(0, 3):
                if self.grid_result[row0 + i][column0 + j] == value:
                    return False

        return True


    def bf_solve(self):
        for row in range(0, 9):
            for column in range(0, 9):
                if self.grid_result[row][column] == 0:
                    for value in self.grid_possibilities[row][column]:
                        if self.is_possible(row, column, value):
                            self.grid_result[row][column] = value
                            self.bf_solve()
                            self.grid_result[row][column] = 0
                    return
        self.grid_output = self.grid_result.copy()



# -------------------------- SUDOKUS ----------------------
# -------- Logically Solvable ---------------

#initilaize solver
SudokuSolver = sudokusolver()

# Creat inital sudoku - Later taken from input
grid_input = np.array([
    [2, 9, 0, 8, 7, 3, 0, 1, 0],
    [4, 0, 0, 0, 0, 5, 9, 2, 0],
    [0, 1, 0, 0, 2, 4, 0, 0, 0],
    [0, 0, 0, 0, 8, 9, 6, 0, 0],
    [0, 0, 4, 0, 0, 0, 8, 3, 0],
    [0, 8, 2, 3, 1, 0, 5, 0, 0],
    [0, 0, 9, 2, 3, 8, 0, 0, 7],
    [8, 0, 0, 0, 4, 7, 0, 0, 0],
    [3, 0, 5, 0, 9, 0, 2, 8, 4]])

print("Initial Sudoku")
print(SudokuSolver.solve(grid_input, 5))

# Test Sudoku 1
grid_input = np.array([
    [6, 0, 3, 0, 0, 0, 0, 0, 4],
    [9, 2, 0, 6, 1, 0, 7, 0, 3],
    [0, 0, 0, 9, 0, 0, 2, 5, 0],
    [5, 0, 0, 0, 0, 8, 0, 0, 2],
    [0, 8, 0, 4, 3, 1, 0, 7, 0],
    [7, 0, 0, 5, 0, 0, 0, 0, 8],
    [0, 5, 1, 0, 0, 9, 0, 0, 0],
    [3, 0, 7, 0, 5, 2, 0, 6, 1],
    [8, 0, 0, 0, 0, 0, 9, 0, 5]])

print("Test Sudoku 1")
print(SudokuSolver.solve(grid_input, 5))

# Test Sudoku 2
grid_input = np.array([
    [0, 0, 8, 6, 0, 0, 2, 5, 9],
    [3, 0, 2, 0, 1, 0, 0, 7, 4],
    [0, 7, 0, 2, 0, 0, 0, 0, 8],
    [0, 0, 0, 0, 3, 1, 5, 6, 0],
    [0, 0, 0, 9, 0, 7, 0, 0, 0],
    [0, 3, 1, 5, 6, 0, 0, 0, 0],
    [9, 0, 0, 0, 0, 6, 0, 4, 0],
    [7, 6, 0, 0, 8, 0, 9, 0, 5],
    [1, 8, 5, 0, 0, 4, 7, 0, 0]])

print("Test Sudoku 2")
print(SudokuSolver.solve(grid_input, 5))

# Test Sudoku 3
grid_input = np.array([
    [0, 0, 0, 2, 6, 0, 7, 0, 1],
    [6, 8, 0, 0, 7, 0, 0, 9, 0],
    [1, 9, 0, 0, 0, 4, 5, 0, 0],
    [8, 2, 0, 1, 0, 0, 0, 4, 0],
    [0, 0, 4, 6, 0, 2, 9, 0, 0],
    [0, 5, 0, 0, 0, 3, 0, 2, 8],
    [0, 0, 9, 3, 0, 0, 0, 7, 4],
    [0, 4, 0, 0, 5, 0, 0, 3, 6],
    [7, 0, 3, 0, 1, 8, 0, 0, 0]])

print("Test Sudoku 3")
print(SudokuSolver.solve(grid_input, 5))

# Test Sudoku 4
grid_input = np.array([
    [1, 0, 0, 4, 8, 9, 0, 0, 6],
    [7, 3, 0, 0, 0, 0, 0, 4, 0],
    [0, 0, 0, 0, 0, 1, 2, 9, 5],
    [0, 0, 7, 1, 2, 0, 6, 0, 0],
    [5, 0, 0, 7, 0, 3, 0, 0, 8],
    [0, 0, 6, 0, 9, 5, 7, 0, 0],
    [9, 1, 4, 6, 0, 0, 0, 0, 0],
    [0, 2, 0, 0, 0, 0, 0, 3, 7],
    [8, 0, 0, 5, 1, 2, 0, 0, 4]])

print("Test Sudoku 4")
print(SudokuSolver.solve(grid_input, 5))

# Test Sudoku 6
grid_input = np.array([
    [3, 5, 0, 0, 8, 6, 0, 0, 0],
    [0, 9, 7, 0, 0, 0, 0, 0, 0],
    [4, 0, 0, 0, 0, 7, 2, 0, 0],
    [2, 1, 8, 0, 0, 3, 0, 0, 9],
    [6, 0, 0, 0, 2, 0, 0, 0, 1],
    [9, 0, 0, 1, 0, 0, 8, 3, 2],
    [0, 0, 9, 8, 0, 0, 0, 0, 3],
    [0, 0, 0, 0, 0, 0, 9, 2, 0],
    [0, 0, 0, 2, 3, 0, 0, 7, 8]])

print("Test Sudoku 6")
print(SudokuSolver.solve(grid_input, 5))

# Sudoku.com(medium)
grid_input = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 7],
    [6, 4, 9, 2, 0, 0, 8, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 9, 0],
    [0, 6, 2, 0, 7, 0, 0, 5, 1],
    [0, 8, 5, 0, 0, 9, 0, 0, 6],
    [3, 0, 4, 0, 0, 0, 0, 0, 0],
    [4, 0, 0, 0, 0, 0, 0, 7, 5],
    [0, 1, 8, 0, 0, 0, 3, 0, 0],
    [0, 5, 0, 1, 0, 6, 2, 0, 8]])

print("Sudoku.com(medium) I")
print(SudokuSolver.solve(grid_input, 5))

# Sudoku.com(hard)
grid_input = np.array([
    [0, 8, 7, 0, 2, 0, 0, 5, 0],
    [0, 0, 0, 0, 0, 6, 0, 0, 4],
    [2, 5, 0, 0, 0, 9, 0, 0, 0],
    [3, 4, 5, 0, 0, 0, 0, 0, 0],
    [0, 0, 6, 0, 0, 0, 4, 0, 3],
    [8, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 3, 9, 8, 0],
    [0, 0, 9, 6, 5, 8, 3, 0, 0],
    [5, 0, 0, 2, 9, 0, 6, 0, 0]])

print("Sudoku.com(hard) I")
print(SudokuSolver.solve(grid_input, 5))

# Sudoku.com(hard)
grid_input = np.array([
    [8, 7, 0, 0, 4, 5, 0, 2, 0],
    [0, 0, 0, 7, 2, 0, 0, 0, 0],
    [0, 0, 6, 0, 0, 0, 0, 7, 0],
    [0, 0, 0, 0, 0, 0, 6, 0, 5],
    [0, 0, 0, 1, 0, 2, 0, 0, 0],
    [7, 6, 4, 0, 0, 8, 0, 0, 0],
    [0, 0, 0, 6, 0, 0, 3, 0, 0],
    [4, 2, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 3, 0, 5, 4, 1, 0, 0]])

print("Sudoku.com(hard) II")
print(SudokuSolver.solve(grid_input, 5))

# Sudoku.com(expert_2)
grid_input = np.array([
    [0, 0, 0, 0, 0, 2, 0, 0, 0],
    [7, 3, 0, 0, 5, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 0, 5, 3, 0],
    [5, 0, 0, 0, 4, 0, 0, 0, 0],
    [3, 4, 2, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 8, 6, 0, 0, 5, 0],
    [9, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 4, 3, 0, 0, 0, 6],
    [0, 0, 0, 0, 0, 0, 8, 0, 0]])

print("Sudoku.com(expert_2)")
print(SudokuSolver.solve(grid_input, 5))

# ------------ Brute Force --------------

# Test Sudoku 5
grid_input = np.array([
    [0, 2, 0, 6, 0, 8, 0, 0, 0],
    [5, 8, 0, 0, 0, 9, 7, 0, 0],
    [0, 0, 0, 0, 4, 0, 0, 0, 0],
    [3, 7, 0, 0, 0, 0, 5, 0, 0],
    [6, 0, 0, 0, 0, 0, 0, 0, 4],
    [0, 0, 8, 0, 0, 0, 0, 1, 3],
    [0, 0, 0, 0, 2, 0, 0, 0, 0],
    [0, 0, 9, 8, 0, 0, 0, 3, 6],
    [0, 0, 0, 3, 0, 6, 0, 9, 0]])

print("Test Sudoku 5")
print(SudokuSolver.solve(grid_input, 5))

# Hard Sudoku!
grid_input = np.array([
    [6, 0, 0, 0, 0, 0, 5, 3, 0],
    [0, 0, 0, 0, 0, 2, 7, 0, 0],
    [5, 0, 7, 0, 9, 6, 0, 1, 8],
    [0, 0, 6, 0, 0, 1, 0, 8, 0],
    [0, 9, 8, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 2, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 9, 0, 0],
    [0, 0, 0, 2, 0, 0, 0, 4, 3],
    [3, 1, 0, 0, 0, 9, 0, 6, 2]])

print("Hard Sudoku!")
print(SudokuSolver.solve(grid_input, 5))

# Sudoku.com(expert_Maybe_wrong!)
grid_input = np.array([
    [0, 0, 0, 0, 0, 0, 3, 9, 6],
    [7, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 2, 0, 8, 0, 0, 0],
    [0, 0, 0, 0, 0, 9, 6, 7, 0],
    [8, 0, 0, 3, 0, 6, 0, 0, 9],
    [0, 4, 0, 0, 0, 0, 0, 0, 0],
    [3, 0, 0, 0, 0, 5, 0, 0, 0],
    [0, 0, 5, 0, 2, 0, 7, 0, 0],
    [9, 0, 0, 0, 7, 0, 4, 1, 0]])

print("# Sudoku.com(expert_Maybe_wrong!)")
print(SudokuSolver.solve(grid_input, 5))

# Other hard one
grid_input = np.array([
    [0, 0, 2, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 5, 1, 0, 6, 9, 7, 4],
    [0, 7, 0, 0, 0, 5, 0, 0, 1],
    [0, 2, 6, 3, 9, 1, 8, 4, 0],
    [4, 0, 0, 8, 0, 0, 0, 5, 0],
    [1, 8, 4, 9, 0, 3, 5, 0, 0],
    [0, 9, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 4, 0, 0]])

print("Other hard one")
print(SudokuSolver.solve(grid_input, 5))

# Sudoku.com(expert_3)
grid_input = np.array([
    [0, 0, 0, 0, 0, 2, 0, 0, 0],
    [0, 0, 9, 1, 0, 0, 0, 0, 7],
    [0, 6, 1, 0, 0, 4, 0, 0, 0],
    [0, 0, 2, 0, 9, 0, 0, 0, 1],
    [7, 0, 5, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 3, 8, 4, 0],
    [0, 8, 0, 0, 3, 0, 0, 2, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 8],
    [0, 0, 0, 0, 0, 7, 4, 9, 0]])

print("Sudoku.com(expert_3)")
print(SudokuSolver.solve(grid_input, 5))

# Sudoku.com(expert_4)
grid_input = np.array([
    [0, 0, 0, 0, 0, 0, 5, 0, 0],
    [0, 7, 0, 9, 0, 0, 0, 0, 2],
    [0, 4, 0, 6, 8, 0, 1, 0, 0],
    [0, 0, 0, 0, 3, 2, 0, 0, 0],
    [0, 0, 6, 0, 9, 0, 0, 0, 4],
    [5, 0, 3, 0, 0, 4, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 6],
    [7, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 9, 0, 0, 0, 0, 7, 0]])

print("Sudoku.com(expert_4)")
print(SudokuSolver.solve(grid_input, 5))

# finnish guy claims to have created hardest sudoku:
grid_input = np.array([
    [8, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 3, 6, 0, 0, 0, 0, 0],
    [0, 7, 0, 0, 9, 0, 2, 0, 0],
    [0, 5, 0, 0, 0, 7, 0, 0, 0],
    [0, 0, 0, 0, 4, 5, 7, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 3, 0],
    [0, 0, 1, 0, 0, 0, 0, 6, 8],
    [0, 0, 8, 5, 0, 0, 0, 1, 0],
    [0, 9, 0, 0, 0, 0, 4, 0, 0]])

print("finnish guy claims to have created hardest sudoku")
print(SudokuSolver.solve(grid_input, 5))

