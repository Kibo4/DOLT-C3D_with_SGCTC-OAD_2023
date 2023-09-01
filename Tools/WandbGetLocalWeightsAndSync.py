"""
File to get the models weights of runs which didnt finished correcty, and to sync them with wandb.
(To be able to use them simply in the evaluation)
"""
from shutil import copytree, copyfile

import wandb


listRuns = [("woven-grass-957","/gpfswork/rech/ckp/ukl91nl/DATA/PKUMMD/Log/20230513-001102/Weights/"),
            ("solar-wind-918","/gpfswork/rech/ckp/ukl91nl/DATA/PKUMMD/Log/20230512-114032/Weights/"),
            ("rose-tree-960","/gpfswork/rech/ckp/ukl91nl/DATA/PKUMMD/Log/20230513-094931/Weights/"),
            ("misty-fire-942","/gpfswork/rech/ckp/ukl91nl/DATA/PKUMMD/Log/20230512-141638/Weights/"),
            ("deft-spaceship-941","/gpfswork/rech/ckp/ukl91nl/DATA/PKUMMD/Log/20230512-141538/Weights/"),
            ("different-water-939","/gpfswork/rech/ckp/ukl91nl/DATA/PKUMMD/Log/20230512-141506/Weights/"),
            ("dauntless-meadow-938","/gpfswork/rech/ckp/ukl91nl/DATA/PKUMMD/Log/20230512-141238/Weights/"),
            ("pleasant-sun-937","/gpfswork/rech/ckp/ukl91nl/DATA/PKUMMD/Log/20230512-141237/Weights/"),
            ("mild-wood-933","/gpfswork/rech/ckp/ukl91nl/DATA/PKUMMD/Log/20230512-140459/Weights/"),
            ("whole-lake-932","/gpfswork/rech/ckp/ukl91nl/DATA/PKUMMD/Log/20230512-140459/Weights/"),
            ("still-energy-935","/gpfswork/rech/ckp/ukl91nl/DATA/PKUMMD/Log/20230512-140530/Weights/"),
            ("magic-snowball-948","/gpfswork/rech/ckp/ukl91nl/DATA/PKUMMD/Log/20230512-154902/Weights/"),
            ("dainty-glade-943","/gpfswork/rech/ckp/ukl91nl/DATA/PKUMMD/Log/20230512-153306/Weights/"),
            ("smart-elevator-946","/gpfswork/rech/ckp/ukl91nl/DATA/PKUMMD/Log/20230512-153723/Weights/"),
            ("colorful-water-964","/gpfswork/rech/ckp/ukl91nl/DATA/PKUMMD/Log/20230513-095302/Weights/"),
            ("sparkling-moon-963","/gpfswork/rech/ckp/ukl91nl/DATA/PKUMMD/Log/20230513-095244/Weights/"),
            ("absurd-sea-962","/gpfswork/rech/ckp/ukl91nl/DATA/PKUMMD/Log/20230513-095244/Weights/"),
]
listRuns = [("bumbling-spaceship-936","/gpfswork/rech/ckp/ukl91nl/DATA/PKUMMD/Log/20230512-140629/Weights/")]
listOfRunHere = [("polar-forest-731","C:\workspace2\Datasets\PKUMMD\oldexpOutInterm\cs1\polar-forest-731\Weights\\"),
                 ("wise-cherry-912","C:\workspace2\Datasets\PKUMMD\expOut\CS\wise-cherry-912\Weights\\"),
                 ("classic-shape-911","C:\workspace2\Datasets\PKUMMD\expOut\CV\classic-shape-911\Weights\\"),
                 ("misty-fire-894","C:\workspace2\Datasets\PKUMMD\expOut\CV\misty-fire-894\Weights\\"),
]
API = wandb.Api()

def findIdForName_Wandb(name):
    runs = API.runs("intuidoc-gesture-reco/OLT-C3D_OAD_focus_on_earliness")
    for run in runs:
        if run.name == name:
            return run.id
    raise Exception("run not found")

# send the model to the existing run on  wandb
for run in listRuns:
    print("start for ",run)
    runname = run[0]
    path = run[1]
    wandb.init(id=findIdForName_Wandb(runname), resume="must", project="OLT-C3D_OAD_focus_on_earliness", entity="intuidoc-gesture-reco")
    wandbRunDir = wandb.run.dir
    # wandb.save(path+"*", base_path=path)
    copytree(path, wandbRunDir + "/" + "weights/Weights/")
    copyfile(path+"../config.txt", wandbRunDir + "/" + "weights/Weights/config.txt")
    wandb.finish()

    print("done for ", runname)
