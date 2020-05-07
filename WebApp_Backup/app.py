from flask import Flask, render_template, request
from packages.sudokusolver import sudokusolver
from packages.sudokudetector import sudokuProcessor
from datetime import timedelta
import json
import numpy as np
import cv2
import requests

app = Flask(__name__)
app.secret_key = "SZlMZmTBp2FvmoQGWPSq8n32UG8e02Lp"
app.permanent_session_lifetime = timedelta(minutes=20)

model_url = "https://mysudokusolver.s3.amazonaws.com/model_complex_v2_ext2.sav"
r = requests.get(model_url)
file = open("model.sav", "wb")
file.write(r.content)
file.close()
solver = sudokuProcessor("model.sav")

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/generateSudoku", methods=['POST'])
def generateSudoku():
    difficulty = request.form.to_dict()['difficulty']
    puzzle = np.array([
        [6, 0, 3, 0, 0, 0, 0, 0, 4],
        [9, 2, 0, 6, 1, 0, 7, 0, 3],
        [0, 0, 0, 9, 0, 0, 2, 5, 0],
        [5, 0, 0, 0, 0, 8, 0, 0, 2],
        [0, 8, 0, 4, 3, 1, 0, 7, 0],
        [7, 0, 0, 5, 0, 0, 0, 0, 8],
        [0, 5, 1, 0, 0, 9, 0, 0, 0],
        [3, 0, 7, 0, 5, 2, 0, 6, 1],
        [8, 0, 0, 0, 0, 0, 9, 0, 5]])
    solution = np.array([
        [6, 7, 3, 2, 8, 5, 1, 9, 4],
        [9, 2, 5, 6, 1, 4, 7, 8, 3],
        [1, 4, 8, 9, 7, 3, 2, 5, 6],
        [5, 3, 4, 7, 9, 8, 6, 1, 2],
        [2, 8, 6, 4, 3, 1, 5, 7, 9],
        [7, 1, 9, 5, 2, 6, 3, 4, 8],
        [4, 5, 1, 3, 6, 9, 8, 2, 7],
        [3, 9, 7, 8, 5, 2, 4, 6, 1],
        [8, 6, 2, 1, 4, 7, 9, 3, 5]
    ])
    puzzle = puzzle.tolist()
    solution = solution.tolist()
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
