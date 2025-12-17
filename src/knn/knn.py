# import numpy as np

# def calculate_distance(x1, y1, x2, y2) -> float:
#     return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

# result = calculate_distance(32, 4, 25, 3)
# print(result)


import numpy as np
from collections import Counter
from typing import List, Any, Tuple, Dict

# --- Tipe Data Kustom ---
# DistanceTuple: Tuple[float, Any] -> (jarak, label_kelas)

## 1. Fungsi Jarak (Tidak Berubah)
def euclidean_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    """
    Menghitung Jarak Euclidean antara dua vektor (titik) dengan dimensi berapapun.
    """
    return np.linalg.norm(point1 - point2)

## 2. Fungsi Pencari Tetangga Terdekat (DIUBAH)
def get_neighbors(training_data: List[List[Any]], query_point: List[float], k: int) -> List[Tuple[float, Any]]:
    """
    Menghitung jarak dan mengembalikan list tuple (jarak, label) dari K tetangga terdekat.
    """
    distances: List[Tuple[float, Any]] = []
    
    # Konversi data ke NumPy array
    train_features = np.array([row[:-1] for row in training_data], dtype=float)
    train_labels = [row[-1] for row in training_data]
    query_vector = np.array(query_point, dtype=float)
    
    # Hitung jarak
    for i in range(len(train_features)):
        dist = euclidean_distance(query_vector, train_features[i])
        distances.append((dist, train_labels[i]))
        
    # Urutkan berdasarkan jarak dan ambil K teratas
    sorted_distances: List[Tuple[float, Any]] = sorted(distances, key=lambda x: x[0])
    k_nearest_neighbors: List[Tuple[float, Any]] = sorted_distances[:k]
    
    # K-nearest neighbors sekarang berisi list (jarak, label)
    return k_nearest_neighbors

## 3. Fungsi Prediksi (Voting) (DIUBAH untuk menerima rincian)
def predict_knn(k_nearest_neighbors: List[Tuple[float, Any]]) -> Any:
    """
    Melakukan voting mayoritas dari label tetangga terdekat.
    """
    # Ekstrak hanya label/kelas dari tetangga terdekat
    neighbor_labels: List[Any] = [item[1] for item in k_nearest_neighbors]
    
    # Gunakan Counter untuk menghitung frekuensi setiap label
    label_counts = Counter(neighbor_labels)
    
    # Ambil label yang paling sering muncul
    most_common_label = label_counts.most_common(1)[0][0]
    
    return most_common_label

## 4. Fungsi Utama (Main Pipeline) (DIUBAH untuk mengembalikan rincian)
def knn_classifier_pipeline(training_data: List[List[Any]], query_point: List[float], k: int = 3) -> Tuple[Any, List[Tuple[float, Any]], Dict[Any, int]]:
    """
    Pipeline lengkap untuk klasifikasi KNN.
    
    Mengembalikan: (prediksi, rincian_tetangga, hasil_voting)
    """
    if k <= 0:
        raise ValueError("Nilai 'k' harus lebih besar dari 0.")

    # 1. Cari K tetangga terdekat (jarak, label)
    k_nearest_neighbors = get_neighbors(training_data, query_point, k)
    
    # 2. Prediksi kelas berdasarkan voting
    prediction = predict_knn(k_nearest_neighbors)
    
    # 3. Hitung hasil voting untuk ditampilkan
    neighbor_labels: List[Any] = [item[1] for item in k_nearest_neighbors]
    result_voting: Dict[Any, int] = dict(Counter(neighbor_labels))

    return prediction, k_nearest_neighbors, result_voting

# --- Contoh Penggunaan ---

# Data Pelatihan (Sama seperti sebelumnya)
iris_data: List[List[Any]] = [
    [5.3, 3.7, 'Setosa'],
    [5.1, 3.8, 'Setosa'],
    [7.2, 3.0, 'Virginica'],
    [5.4, 3.4, 'Setosa'],
    [5.1, 3.3, 'Setosa'],
    [5.4, 3.9, 'Setosa'],
    [7.4, 2.8, 'Virginica'],
    [6.1, 2.8, 'Versicolor'],
    [7.3, 2.9, 'Virginica'],
    [6.0, 2.7, 'Versicolor'],
    [5.8, 2.8, 'Versicolor'],
    [6.3, 2.3, 'Versicolor'],
    [5.1, 2.5, 'Versicolor'],
    [5.5, 2.5, 'Versicolor'],
    [6.3, 2.4, 'Versicolor'],
]

# Titik Data Baru yang Akan Diprediksi
new_data_point: List[float] = [5.2, 3.1]

# Nilai K
K_VALUE = 5 

print(f"### 🌸 Mulai Prediksi KNN (K={K_VALUE}) ###")
print(f"Titik Data Baru (Fitur): {new_data_point}")
print("---")

# Panggil pipeline utama yang sekarang mengembalikan 3 nilai
predicted_species, neighbors_details, voting_results = knn_classifier_pipeline(iris_data, new_data_point, K_VALUE)

## --- Tampilan Rincian ---

print(f"## 🔎 Detail {K_VALUE} Tetangga Terdekat")
print("Rincian Jarak & Kelas dari data pelatihan:")
print("-" * 35)
print(f"{'No.':<4}{'Jarak':<10}{'Kelas':<20}")
print("-" * 35)
for i, (distance, label) in enumerate(neighbors_details, 1):
    # Untuk melihat data mana yang terpilih, kita cari jaraknya di iris_data.
    # Namun, karena kita tidak menyimpan data fitur asli saat membuat tuple jarak,
    # kita hanya akan menampilkan jarak dan label:
    print(f"{i:<4}{distance:.6f} {label:<20}")
print("-" * 35)

print("\n## 🗳️ Hasil Voting")
print("Kelas | Suara")
print(":--- | :---")
for label, count in voting_results.items():
    print(f"{label} | {count}")

print("\n## ✨ Hasil Prediksi Akhir")
print(f"Titik data {new_data_point} diklasifikasikan sebagai: **{predicted_species}**")