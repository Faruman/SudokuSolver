from setuptools import setup

setup(
    name="sudokudetector_package",
    version = "1.0.1",
    author = "Fabian Karst",
    author_email = "faruman.der.weise@gmail.com",
    description = ("Pytorch package containing a machine learning model for the prediction of the numbers in a sudoku grid."),
    include_package_data=True,
    scripts=["models.py", "gcloud_prediction.py"],
    url='http://mysudokusolver.ey.r.appspot.com/'
)