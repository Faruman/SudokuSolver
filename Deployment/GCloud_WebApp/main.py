from flask import Flask, render_template, request, session
from packages.sudokusolver import sudokusolver
from packages.sudokucreator import sudokucreator
from datetime import timedelta
import json
import numpy as np
import random
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

app = Flask(__name__)
app.secret_key = "SZlMZmTBp2FvmoQGWPSq8n32UG8e02Lp"
app.permanent_session_lifetime = timedelta(minutes=20)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/generateSudoku", methods=['POST'])
def generateSudoku():
    creator = sudokucreator()
    difficulty = request.form.to_dict()['difficulty']

    if request.form.to_dict()['newload'] == "yes" and session.get("puzzle"):
        puzzle = session.get("puzzle")
        JSON_dict = {"puzzle": puzzle}
    elif request.form.to_dict()['newload'] == "yes":
        sudokus = np.load("data/sudokus.npy")
        rand = random.randint(0, 4)
        puzzle = sudokus[rand, 0, :, :].astype("float32").tolist()
        solution = sudokus[rand, 1, :, :].astype("float32").tolist()
        JSON_dict = {"puzzle": puzzle, "solution": solution}
    else:
        _, puzzle, solution = creator.create_sudoku(difficulty)
        puzzle = puzzle.astype("float32").tolist()
        solution = solution.astype("float32").tolist()
        session["puzzle"] = puzzle
        JSON_dict = {"puzzle": puzzle, "solution": solution}

    return json.dumps(JSON_dict)


@app.route("/readSudoku", methods=['POST'])
def readSudoku():
    #function outsourced to azure cloud functions
    data = {"data": request.data.decode("utf-8")}

    url = "https://sudokupredictor.azurewebsites.net/api/predict?code={{key}}"

    s = requests.Session()
    retries = Retry(total=3,
                    backoff_factor=0.1,
                    status_forcelist=[500, 502, 503, 504])

    s.mount('https://', HTTPAdapter(max_retries=retries))
    answer = s.post(url, json=data)
    if answer.status_code == 200:
        puzzle = json.loads(answer.content.decode("utf-8"))["puzzle"]
        error = "error: no sudoku identified"
        session["puzzle"] = puzzle
    else:
        puzzle = []
        error = "connection error: {}".format(answer.status_code)
    JSON_dict = {"puzzle": puzzle, "error": error}
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
