import math
import pandas as pd
from collections import Counter


def entropy(data):
    n = len(data)
    nGrup = Counter(data)
    ent = 0.0
    for c in nGrup.values():
        p = c / n
        ent -= p * math.log2(p)
    return ent



def gain(data_input, data_output, fitur):
    base_ent = entropy(data_output)
    base_n = len(data_output)

    # kelompok berdasarkan nilai atribut
    subsets = {}
    for x, label in zip(data_input, data_output):
        key = x[fitur]
        subsets.setdefault(key, []).append(label)

    subset_ent = 0.0
    for labels in subsets.values():
        subset_ent += (len(labels) / base_n) * entropy(labels)

    return base_ent - subset_ent


# Tree
def buat_tree(data_input, data_output, feature_indices):
    # semua output (Label) sama -> Leaf
    if len(set(data_output)) == 1:
        return data_output[0]

    # fitur kosong -> habis
    if not feature_indices:
        return Counter(data_output).most_common(1)[0][0]

    # Pilih fitur terbaik berdasarkan gain
    gains = [(gain(data_input, data_output, idx), idx) for idx in feature_indices]
    gains.sort(reverse=True)
    best_gain, best_fitur = gains[0]

    # Jika gain = 0 -> tidak ada peningkatan
    if best_gain == 0:
        return Counter(data_output).most_common(1)[0][0]

    tree = {best_fitur: {}}

    # Nilai-nilai unik fitur pada data saat ini
    values = set(x[best_fitur] for x in data_input)
    for val in values:
        # buat subset data untuk nilai fitur = val
        sub_input = [x for x, label in zip(data_input, data_output) if x[best_fitur] == val]
        sub_output = [label for x, label in zip(data_input, data_output) if x[best_fitur] == val]

        # jika subset kosong
        if not sub_input:
            tree[best_fitur][val] = Counter(data_output).most_common(1)[0][0]
        else:
            sisa_fitur = [i for i in feature_indices if i != best_fitur]
            tree[best_fitur][val] = buat_tree(sub_input, sub_output, sisa_fitur)

    return tree


def load_data_with_pandas(filename):
    # Membaca CSV ke DataFrame
    df = pd.read_csv(filename)
    
    print("--- Head Dataframe ---")
    print(df.head())
    print("-" * 30)

    if 'No' in df.columns:
        df = df.drop(columns=['No'])
        

    # Asumsi: Kolom terakhir adalah Target ('Play')
    target_col = df.columns[-1]
    feature_cols = df.columns[:-1].tolist()
    
    print(f"Fitur : {feature_cols}")
    print(f"Target: {target_col}\n")

    # Fungsi manual kita butuh List of Dictionaries, bukan DataFrame.
    # orient='records' mengubah df menjadi [{'Outlook':'Sunny', 'Temp':'Hot'}, {...}]
    data_input = df[feature_cols].to_dict(orient='records')
    
    # Target diubah menjadi list biasa ['No', 'No', 'Yes', ...]
    data_output = df[target_col].tolist()
    
    return data_input, data_output, feature_cols


if __name__ == "__main__":
    file_csv = './dataset/weather_data.csv'
    
    try:
        # Load data pakai Pandas
        X, y, features = load_data_with_pandas(file_csv)
        
        # Jalankan Algoritma
        print("--- Membangun Tree... ---")
        pohon_keputusan = buat_tree(X, y, features)
        
        print("\n--- Hasil Akhir (Tree Structure) ---")
        print(pohon_keputusan)
        
    except FileNotFoundError:
        print(f"File {file_csv} tidak ditemukan!")