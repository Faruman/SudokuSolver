import numpy as np
import time
import random


class sudokucreator:
    def __init__(self):
        self.full_set = np.arange(1, 10, 1)
        self.grid_possibilities = np.empty(shape=[9, 9], dtype=object)
        self.grid_possibilities.fill([])
        self.grid_result = np.empty(shape=[9, 9], dtype=object)
        self.grid_output = np.empty(shape=[9, 9], dtype=object)
        self.iterration = 0
        self.grid_input = np.empty(shape=[9, 9], dtype=object)
        self.grid_input.fill(0)
        self.cell_poss = np.arange(0, 81, 1)

    def reinit(self):
        self.full_set = np.arange(1, 10, 1)
        self.grid_possibilities = np.empty(shape=[9, 9], dtype=object)
        self.grid_possibilities.fill([])
        self.grid_result = self.grid_input.copy()
        self.grid_output = self.grid_input.copy()

    def make_grid(self):
        self.iterration = 0
        self.grid_input = np.empty(shape=[9, 9], dtype=object)
        self.grid_input.fill(0)
        self.full_set = np.arange(1, 10, 1)

    def fill_grid(self):
        for row in range(0, 9):
            for column in range(0, 9):
                if self.grid_input[row][column] == 0:
                    random.shuffle(self.full_set)
                    for value in self.full_set:
                        if self.is_possible_2(row, column, value):
                            self.grid_input[row][column] = value
                            self.fill_grid()
                            if 0 not in self.grid_input:
                                return self.grid_input
                            self.fill_grid()
                            self.grid_input[row][column] = 0
                    return

    def is_possible_2(self, row, column, value):
        # Check row
        for i in range(0, 9):
            if self.grid_input[row][i] == value:
                return False
        # Check column
        for i in range(0, 9):
            if self.grid_input[i][column] == value:
                return False
        # Check subgrid
        row0 = (row // 3) * 3
        column0 = (column // 3) * 3
        for i in range(0, 3):
            for j in range(0, 3):
                if self.grid_input[row0 + i][column0 + j] == value:
                    return False
        return True

    def remove_values(self):
        while self.x <= self.number:
            c = random.choice(self.cell_poss)
            row, column = divmod(c, 9)
            value = self.grid_input[row][column]
            self.grid_input[row][column] = 0
            if self.solve_sudoku_logic():
                iterration = 0
                self.x += 1
                self.cell_poss = self.cell_poss[self.cell_poss != c]
            else:
                iterration = 0
                self.grid_input[row][column] = value
            self.remove_values()
        return

    def create_sudoku(self, input_level):
        start_time = time.time()
        self.x = 0
        self.iterration = 0
        self.cell_poss = np.arange(0, 81, 1)
        self.make_grid()
        self.fill_grid()
        if input_level == 'easy':
            self.number = 43
        elif input_level == 'medium':
            self.number = 49
        elif input_level == 'hard':
            self.number = 55
        else:
            self.number = 49
        self.remove_values()
        return (time.time() - start_time), self.grid_input, self.grid_output

    def create_possible_values(self):
        for row in range(0, 9):
            for column in range(0, 9):
                if self.grid_result[row][column] == 0:
                    # Take full Grid and substract values that are already in the row
                    poss_r = np.setdiff1d(self.full_set, self.grid_result[row])
                    # Take all values that are possible in a row and substract those already in the column
                    poss_c = np.setdiff1d(poss_r, self.grid_result[:, column])

                    # Check for values in subgrid
                    # Use //3 to get whole numbers
                    row_ind = (row // 3) * 3
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
            unique_value = [
                x for x in column_list if column_list.count(x) == 1]
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
                                    self.grid_possibilities[subrow, subcol] = [
                                    ]
                                    self.grid_result[subrow,
                                                     subcol] = unique_value[i]
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
                            self.grid_possibilities.T[column, c].remove(
                                value_1)
                        if value_2 in self.grid_possibilities.T[column, c]:
                            self.grid_possibilities.T[column, c].remove(
                                value_2)

        return

    def solve_sudoku_logic(self):
        iterration = 0
        tries = 10
        self.reinit()
        while 0 in self.grid_result and tries > 0:
            self.create_possible_values()
            self.check_single()
            self.check_unique()
            iterration += 1
            tries -= 1
            self.grid_output = self.grid_result.copy()
            if self.solution_is_valid() == True:
                return True
            else:
                return False
            return

    # ----- Helper Function

    def find_pairs(self, row):
        pair = []
        to_compare = []
        for col in row:
            if np.size(col) == 2:
                to_compare.append(col)
                if len(to_compare) > 1:
                    try:
                        pair = [
                            x for x in to_compare if to_compare.count(x) == 2]
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
