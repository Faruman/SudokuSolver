from flask import Flask, render_template, request, session
from packages.sudokusolver import sudokusolver
from packages.sudokucreator import sudokucreator
from datetime import timedelta
import json
import numpy as np
import requests

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
    url = "https://sudokupredictor.azurewebsites.net/api/predict?code=1rsJlv71SjSrMuXuDRvz9teD7U18pnIO9T9dJzaEVIkb4IeeweqH6g=="
    try:
        answer = requests.post(url, json=data, timeout=10)
        puzzle = json.loads(answer.content.decode("utf-8"))["puzzle"]
        session["puzzle"] = puzzle
    except:
        puzzle = []
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
