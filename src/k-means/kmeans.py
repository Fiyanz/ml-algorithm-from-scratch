import numpy as np
from typing import List, Tuple, Dict, Any

# Tipe data: 
# DataPoint: List[float]
# Centroid: np.ndarray

## 1. Fungsi Jarak Euclidean
def euclidean_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    """
    Menghitung Jarak Euclidean antara dua vektor.
    """
    return np.linalg.norm(point1 - point2)

# ----------------------------------------------------------------------

## 2. Fungsi Penugasan Cluster (Assignment Step)
def assign_cluster(data: np.ndarray, centroids: Dict[str, np.ndarray]) -> Dict[str, List[np.ndarray]]:
    """
    Menetapkan setiap titik data ke centroid terdekatnya (berdasarkan jarak).
    
    Mengembalikan: Dictionary di mana kunci adalah nama centroid 
    dan nilai adalah list dari data point yang termasuk dalam cluster tersebut.
    """
    clusters: Dict[str, List[np.ndarray]] = {key: [] for key in centroids.keys()}
    
    for point in data:
        min_distance = float('inf')
        closest_centroid_key = ""
        
        for key, centroid in centroids.items():
            dist = euclidean_distance(point, centroid)
            
            if dist < min_distance:
                min_distance = dist
                closest_centroid_key = key
        
        clusters[closest_centroid_key].append(point)
        
    return clusters

# ----------------------------------------------------------------------

## 3. Fungsi Pembaruan Centroid (Update Step)
def update_centroids(clusters: Dict[str, List[np.ndarray]]) -> Dict[str, np.ndarray]:
    """
    Menghitung centroid baru sebagai rata-rata (mean) dari semua titik dalam setiap cluster.
    
    Mengembalikan: Dictionary dengan centroid yang baru (np.ndarray).
    """
    new_centroids: Dict[str, np.ndarray] = {}
    
    for key, points in clusters.items():
        if points:
            new_centroid = np.mean(np.stack(points), axis=0)
            new_centroids[key] = new_centroid
        else:
            print(f"Peringatan: Cluster {key} kosong dan tidak diperbarui.")
    
    return new_centroids

# ----------------------------------------------------------------------

## 4. Fungsi Utama (Pipeline K-Means)
def kmeans_clustering_pipeline(data: List[List[float]], initial_centroids: Dict[str, List[float]], max_iter: int = 100, tolerance: float = 1e-4) -> Tuple[Dict[str, np.ndarray], Dict[str, List[np.ndarray]]]:
    """
    Pipeline K-Means Clustering.
    
    Args:
        data: List of data points [[x1, y1], [x2, y2], ...].
        initial_centroids: Dictionary dengan kunci (nama cluster) dan nilai (koordinat awal).
        max_iter: Jumlah iterasi maksimum.
        tolerance: Kriteria berhenti jika perpindahan centroid sangat kecil.
        
    Mengembalikan: (Centroid akhir, Penugasan cluster akhir)
    """
    
    data_np = np.array(data, dtype=float)
    
    current_centroids: Dict[str, np.ndarray] = {
        key: np.array(value, dtype=float) for key, value in initial_centroids.items()
    }
    
    print("--- Proses Iterasi K-Means Dimulai ---")
    
    for i in range(max_iter):
        print(f"\nITERASI {i + 1}")
        
        clusters = assign_cluster(data_np, current_centroids)
        
        print("  [ASSIGNMENT]")
        for key, points in clusters.items():
            print(f"    Cluster {key}: {len(points)} titik")
        
        new_centroids = update_centroids(clusters)
        
        movement = 0.0
        for key in current_centroids.keys():
            if key in new_centroids:
                movement += euclidean_distance(current_centroids[key], new_centroids[key])
        
        print(f"  [UPDATE] Total perpindahan centroid: {movement:.4f}")
        
        if movement < tolerance:
            print(f"K-Means konvergen. Berhenti pada iterasi {i + 1}.")
            break
        
        current_centroids = new_centroids

    print("\n--- Proses Iterasi K-Means Selesai ---")
    
    return current_centroids, clusters


if __name__ == "__main__":

    data_points: List[List[float]] = [
        [2, 10], # A1
        [2, 5],  # A2
        [8, 4],  # A3
        [5, 8],  # B1
        [7, 5],  # B2
        [6, 4],  # B3
        [1, 2],  # C1
        [4, 9],  # C2
    ]

    initial_centroids_dict: Dict[str, List[float]] = {
        'Centroid A': [2, 10],
        'Centroid B': [5, 8],
        'Centroid C': [1, 2],
    }

    final_centroids, final_clusters = kmeans_clustering_pipeline(
        data=data_points,
        initial_centroids=initial_centroids_dict,
        max_iter=10
    )

    # --- Tampilkan Hasil Akhir ---

    print("\n====================================")
    print("## ✨ HASIL K-MEANS CLUSTERING AKHIR")
    print("====================================")

    print("\n### Lokasi Centroid Akhir:")
    for key, centroid in final_centroids.items():
        print(f"  {key}: {np.round(centroid, 4)}")

    print("\n### Penugasan Kluster Akhir:")
    data_point_labels = []
    for key, points in final_clusters.items():
        print(f"\nCluster {key} ({len(points)} Titik):")
        for point in points:
            data_point_labels.append((point, key))
            print(f"  {point}")

    # Verifikasi perhitungan jarak (Contoh dari Gambar)
    p1 = np.array([4, 9])
    p2 = np.array([5, 8])
    dist_c2_b = euclidean_distance(p1, p2)

    print("\n--- Verifikasi Jarak (Sesuai Gambar) ---")
    print(f"Jarak C2 (4, 9) ke Centroid B (5, 8): {dist_c2_b:.4f}")

    p1_c1 = np.array([1, 2])
    p2_b = np.array([5, 8])
    dist_c1_b = euclidean_distance(p1_c1, p2_b)
    print(f"Jarak C1 (1, 2) ke Centroid B (5, 8): {dist_c1_b:.4f}")