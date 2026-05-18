#Read dataset
import pandas as pd
import glob

arquivos_csv = glob.glob("./data/MachineLearningCSV/*.csv")

df = pd.concat(
    [pd.read_csv(arquivo, encoding="latin1") for arquivo in arquivos_csv],
    ignore_index=True
)

df.columns = df.columns.str.strip()
print(df["Label"].value_counts())

# Agrupar labels semelhantes em categorias principais
df["Label"] = df["Label"].replace({
    # Ataques DoS
    "DoS Hulk": "DoS",
    "DDoS": "DoS",
    "DoS GoldenEye": "DoS",
    "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",
    "Heartbleed": "DoS",

    # Ataques de força bruta
    "FTP-Patator": "BruteForce",
    "SSH-Patator": "BruteForce",

    # Ataques Web
    "Web Attack ï¿½ Brute Force": "WebAttack",
    "Web Attack ï¿½ XSS": "WebAttack",
    "Web Attack ï¿½ Sql Injection": "WebAttack",
})

# Conferir a nova distribuição das classes
print(df["Label"].value_counts())
df.to_csv("./data/CICIDS2017.csv", index=False)


