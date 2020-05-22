# Sudoku Solver

The [Sudoku Solver](https://mysudokusolver.ey.r.appspot.com/) is a web application that leverages deep learning and 
iterative processes to create the perfect environment to work on your sudokus. 

## Using the Sudoku Solver
**1. Creating a sudoku:**

To get started, there are three ways to create a sudoku.

* **"Upload"** converts images of physical sudokus into digital format.
* **"Generate"** automatically creates unique sudokus of three possible difficulty levels.
* **"Modify"** allows users to manually modify the loaded sudoku, or to create new ones from scratch.

**2. Working on the sudoku**

After creating a sudoku, you can enter your numbers directly in the white boxes.

**3. Evaluating the Sudoku**

* **"Check"** tests if the entered numbers are correct and marks any fields with wrong entries red.
* **"Solve"** solves the sudoku and fills in any missing numbers.

## Repository Structure 

**data:** *(sudoku images to test upload function)*

**deployment:** 
* .../Azure_CloudFunction
    * .../predict    
        * models.py  &nbsp; *(digit recognition model)*
        * predict.py  &nbsp; *(preprocessing & prediction)*
* .../GCloud_WebApp:
    * .../packages/
        * sudoku_creating_algorithm.py &nbsp; *(generate sudokus)*
        * sudoku_solving_algorithm.py &nbsp; *(solve sudoku)*
        
**mnistProcessor:**
* mnistClassification.py &nbsp; *(train model on augmented MNIST)*
* mnistGenerator.py &nbsp; *(augment MNIST dataset)*
* sudokuRecognition.py &nbsp; (xxxx)
* sudokuRecognitionClass.py &nbsp; (xxxx)






















