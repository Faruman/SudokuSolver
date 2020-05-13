import logging
import json
import azure.functions as func
from .predict import sudokuPrediction

def main(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    picture = req.params.get('data')
    logging.info('Python got picture.')
    logging.info(req.params)
    logging.info(req.get_body())

    if not picture:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            picture = req_body.get('data')

    if picture:
        solver = sudokuPrediction(context.function_directory + "/sudoku_net.sav")
        output = solver.predict(picture)
        return func.HttpResponse(json.dumps(output))
    else:
        return func.HttpResponse(
             "Please pass an image to the querry!",
             status_code=400
        )
