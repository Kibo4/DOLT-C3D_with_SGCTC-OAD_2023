import os

import wandb
from wandb.apis.public import Run
import shutil
API = wandb.Api()
RUNS = None

class NoFileWeightsFound(Exception):
    pass


class NoRunFound(Exception):
    pass

def download_weights(name, whereToDownload,projectName= "OLT-C3D_OAD_focus_on_earliness"):
    """ Telechargement des poids d'un modele dont le name est passe en parametre. 5 fichier a DL."""
    global RUNS
    RUNS = API.runs(projectName)

    foundRun = False
    foundFile = False
    for run in RUNS:
        if run.name == name:
            foundRun=True
            print("Downloading weights from wandb:" + run.name)
            for file in run.files():
                if "weights" in file.name:
                    file.download(whereToDownload+"/Weights/", replace=True)
                    foundFile = True
            break
    if not foundRun :
        raise NoRunFound("No run named " + name + " found" )
    if not foundFile :
        raise NoFileWeightsFound("no Weights file found")
    shutil.move(whereToDownload+"/Weights/weights/Weights/model",whereToDownload+"/Weights/",
                copy_function=shutil.copytree)
    shutil.move(whereToDownload + "/Weights/weights/config.txt",whereToDownload+"/Weights/",
                copy_function=shutil.copytree)


def getRun(name, projectName= "OLT-C3D_OAD_focus_on_earliness")->Run:
    """ Telechargement des poids d'un modele dont le name est passe en parametre. 5 fichier a DL."""
    global RUNS
    RUNS = API.runs(projectName)
    for run in RUNS:
        if not os.path.exists("Weigths/" + run.name):
            if run.name == name:
                return run
    raise Exception("No run named " + name + " found")
