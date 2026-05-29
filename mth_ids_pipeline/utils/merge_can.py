import pandas as pd
import glob
import os
import re

path = os.getcwd() + "/data/CAN_*.txt"
can_dataset_files = glob.glob(pathname=path)
can_dataset_files_names = {file: os.path.splitext(os.path.basename(file))[0] for file in can_dataset_files}
can_labels = {file_name: file_name.split("_")[1] for file_name in can_dataset_files_names.values()}
can_labels["CAN_Attack_free_dataset"] = "Normal"

can_columns = ["timestamp", "can_id", "flag", "dlc", "payload", "label"]

for file in can_dataset_files:
    with open(file, "r") as infile, open(f"{os.getcwd()}/data/{can_dataset_files_names[file]}.csv", "w") as outfile:
        outfile.write(",".join(can_columns) + "\n")

        for line in infile:
            match = re.match(r"Timestamp:\s+([\d.]+)\s+ID:\s+(\w+)\s+(\d+)\s+DLC:\s+(\d+)\s+(.*)", line)
            
            if match:
                timestamp, can_id, flag, dlc, payload = match.groups()
                outfile.write(f"{timestamp},{can_id},{flag},{dlc},\"{payload}\",{can_labels[can_dataset_files_names[file]]}\n")

new_path = os.getcwd() + "/data/CAN_*.csv"
can_dataset_files_csv = glob.glob(pathname=new_path)

df = pd.concat(
    [pd.read_csv(file, encoding="latin1") for file in can_dataset_files_csv],
    ignore_index=True
)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.columns = df.columns.str.strip()
print(df["label"].value_counts())

df.to_csv("./data/CAN_Intrusion_Dataset.csv", index=False)