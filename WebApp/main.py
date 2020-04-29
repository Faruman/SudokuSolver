from flask import Flask, render_template, request
from packages.sudokusolver import sudokusolver
from datetime import timedelta
import json
import numpy as np

app = Flask(__name__)
app.secret_key = "SZlMZmTBp2FvmoQGWPSq8n32UG8e02Lp"
app.permanent_session_lifetime = timedelta(minutes=20)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/generateSudoku", methods=['POST'])
def generateSudoku():
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
    puzzle = puzzle.tolist()
    JSON_dict = {"puzzle": puzzle}
    return json.dumps(JSON_dict)


@app.route("/readSudoku", methods=['POST'])
def readSudoku():
    return sudoku.asjson()


@app.route("/solveSudoku", methods=['POST'])
def solveSudoku():
    input = np.array(json.loads(request.form.to_dict()['puzzle']))
    solver = sudokusolver()
    validity, time, iterations, output = solver.solve(input, 5)
    output = output.tolist()
    JSON_dict = {"valid": validity, "time": time, "itertions": iterations, "solution": output}
    return json.dumps(JSON_dict)


if __name__ == '__main__':
    main.run(debug=True)
