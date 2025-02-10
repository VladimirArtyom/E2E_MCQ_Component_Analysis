import pandas as pd
import pickle
import os
import datasets

generated_path = "./generated"
dg_all_path: str = os.path.join(generated_path, "dg_all_strict/dg_all.pickle")

qg_1_path: str = os.path.join(generated_path, "qg_strict/qg_1.pickle")
qg_2_path: str = os.path.join(generated_path, "qg_strict/qg_2.pickle")

qag_path: str = os.path.join(generated_path, "qag_strict/qag_strict.pickle")

qg_dataset = "VosLannack/squad_id_512"

de = datasets.load_dataset(qg_dataset)
vsa = pd.DataFrame(de["validation"])
print(vsa[-2:])


with open(dg_all_path, "rb") as f:
    dg = pickle.load(f)

with open(qg_1_path, "rb") as f:
    qg_1 = pickle.load(f)

with open(qg_1_path, "rb") as f:
    qg_2 = pickle.load(f)
    qg_1 += (qg_2)

with open(qag_path, "rb") as f:
    qag = pickle.load(f)

print(qg_2[-2:])
print(qg_1[-2:])