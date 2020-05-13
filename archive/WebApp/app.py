from flask import Flask, render_template, request
from packages.sudokusolver import sudokusolver
from packages.sudokudetector import sudokuProcessor
from packages.sudokucreator import sudokucreator
from datetime import timedelta
import json
import numpy as np
import cv2
import requests

app = Flask(__name__)
app.secret_key = "SZlMZmTBp2FvmoQGWPSq8n32UG8e02Lp"
app.permanent_session_lifetime = timedelta(minutes=20)

solver = sudokuProcessor("sudoku_net.sav")

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/generateSudoku", methods=['POST'])
def generateSudoku():
    creator = sudokucreator()
    difficulty = request.form.to_dict()['difficulty']

    _, puzzle, solution = creator.create_sudoku(difficulty)

    puzzle = puzzle.astype("float32").tolist()
    solution = solution.astype("float32").tolist()
    JSON_dict = {"puzzle": puzzle, "solution": solution}
    return json.dumps(JSON_dict)


@app.route("/readSudoku", methods=['POST'])
def readSudoku():
    # decode the array into an image
    puzzle = list()
    if len(request.data) != 0:
        img = cv2.imdecode(np.fromstring(request.data, dtype='uint8'), cv2.IMREAD_UNCHANGED)
        pred, prob = solver.process(img)
        _, pred_counts = np.unique(pred[pred != 0], return_counts=True)
        if np.mean(prob) > 0.5 and np.count_nonzero(pred) > 16 and max(pred_counts) <= 9:
            puzzle = pred.tolist()
    JSON_dict = {"puzzle": puzzle}
    return json.dumps(JSON_dict)


@app.route("/solveSudoku", methods=['POST'])
def solveSudoku():
    input = np.array(json.loads(request.form.to_dict()['puzzle']))
    solver = sudokusolver()
    validity, time, iterations, output = solver.solve(input, 5)
    output = output.tolist()
    JSON_dict = {"valid": validity, "time": time, "itertions": iterations, "solution": output}
    return json.dumps(JSON_dict)

if __name__ == '__main__':
    app.run(debug=True)
