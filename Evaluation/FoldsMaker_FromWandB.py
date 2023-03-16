import wandb
import os
from wandb.apis.public import Run
import shutil
import pandas as pd

db = "Chalearn"
pathDB, separator = "C:\workspace2\\Datasets\\" + db + "/", "/"
pathDB, separator = "/srv/tempdd/wmocaer/data/" + db + "/", "/"
pathOutModels = pathDB + "modelRunsGrouped"+separator

API = wandb.Api()
#state finished and splitsize != 70000
RUNS = API.runs("OLT-C3D_OAD_focus_on_earliness",
                filters={"state": "finished", "config.splitSize": {"$ne": 70000}})
                         # "config.useSegmentationGuidedCTC":False})
# group by these keys in config :
groups = ["modeVoxelisation", "thresholdCuDi", "weightCTCLoss", "weightSmoothing", "doSSG", "weightPrior",
          "useSegmentationGuidedCTC","splitSize"]
forNameGroup = ["vox", "cuDi", "wCTC", "smooth", "SSG", "wprior","useSG","split"]
# create a pandas dataframe with the config of each run
# for run in RUNS:
#     print(run.name)
#     for group in groups:
#         print(group,run.config[group])

# build df with the name of the run and the selected config (groups)
df = pd.DataFrame(columns=["name"] + groups)
for run in RUNS:
    df = pd.concat([df, pd.DataFrame({"name": run.name, **{group: run.config[group] for group in groups}}, index=[0])],
                   ignore_index=True)
    df["doSSG"] = df["doSSG"].astype(bool)
    df["useSegmentationGuidedCTC"] = df["useSegmentationGuidedCTC"].astype(bool)

# create new groups with subgroups (6 hierarchies)
# df["group"] = df["modeVoxelisation"] + "_" + df["thresholdCuDi"].astype(str) + "_" + df["weightCTCLoss"].astype(str) + "_" + df["weightSmoothing"].astype(str) + "_" + df["doSSG"].astype(str) + "_" + df["weightPrior"].astype(str)
# do it with a generic values of group
df["group"] = df[groups].apply(
    lambda x: "_".join([forNameGroup[idGroup] + str(x[group]) for idGroup, group in enumerate(groups)]), axis=1)

# groupy by the new groups and aggregate the name of the runs and count them
df = df.groupby("group").agg({"name": list, "group": "count"})
# df = df.groupby("group").agg({"name":list})

# print warnings if there is no 5 runs in a group
for group in df.index:
    if df.loc[group, "group"] != 5:
        print("WARNING : group " + group + " has " + str(df.loc[group, "group"]) + " runs")
    else:
        print("GOOD : group " + group +" has 5 runs")
# visualize the groups
print(df)

# create the folder for the groups
if not os.path.exists(pathOutModels):
    os.makedirs(pathOutModels)

# create one file for each group, the name is the groupe name, the content is the list of the runs in the group
for group in df.index:
    with open(pathOutModels + group, "w") as f:
        f.write(",".join(df.loc[group, "name"]))

print("Done")
