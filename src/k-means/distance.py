import numpy as np

# --- 1. Fungsi Helper ---
def hitung_jarak(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# --- 2. Data Input ---
data_points = {
    'A1': [2, 10], 'A2': [2, 5], 'A3': [8, 4],
    'B1': [5, 8],  'B2': [7, 5], 'B3': [6, 4],
    'C1': [1, 2],  'C2': [4, 9],
}

# Inisialisasi Centroid Awal (Disimpan dalam list agar mudah di-indeks)
# Index 0 = Centroid 1, Index 1 = Centroid 2, Index 2 = Centroid 3
centroids = [
    [2, 10], # C1
    [5, 8],  # C2
    [1, 2]   # C3
]

iteration = 0

print("MULAI PROSES ITERASI K-MEANS")
print("=" * 50)

# --- 3. LOOPING SAMPAI KONVERGEN (DATA SAMA) ---
while True:
    iteration += 1
    print(f"\n>>> ITERASI KE-{iteration}")
    print(f"Posisi Centroid Saat Ini: {centroids}")
    
    # Siapkan wadah untuk pengelompokan titik (Reset setiap iterasi)
    # Kita buat list of lists: [ [titik cluster 1], [titik cluster 2], [titik cluster 3] ]
    clusters_points = [[], [], []]
    
    # Header Tabel
    print("-" * 65)
    print(f"{'Point':<6} | {'Jarak C1':<10} | {'Jarak C2':<10} | {'Jarak C3':<10} | {'Min':<10} | {'Label':<5}")
    print("-" * 65)
    
    # --- A. Assignment Step (Hitung Jarak & Label) ---
    for nama, titik in data_points.items():
        # Hitung jarak ke setiap centroid saat ini
        dists = [hitung_jarak(titik, c) for c in centroids]
        
        # Cari jarak terpendek
        min_dist = min(dists)
        
        # Tentukan label (0, 1, 2 komputer -> 1, 2, 3 manusia)
        cluster_index = dists.index(min_dist)
        label_manusia = cluster_index + 1
        
        # Tampilkan baris tabel
        print(f"{nama:<6} | {dists[0]:<10.4f} | {dists[1]:<10.4f} | {dists[2]:<10.4f} | {min_dist:<10.4f} | {label_manusia:<5}")
        
        # Masukkan titik ke wadah cluster yang sesuai
        clusters_points[cluster_index].append(titik)

    print("-" * 65)

    # --- B. Update Step (Hitung Centroid Baru) ---
    new_centroids = []
    
    print("\nPERHITUNGAN CENTROID BARU:")
    for i in range(3): # Loop untuk Cluster 1, 2, 3
        points_in_cluster = clusters_points[i]
        
        if len(points_in_cluster) > 0:
            # Hitung rata-rata (Tanpa Rounding agar akurat)
            rata_rata = np.mean(points_in_cluster, axis=0)
            centroid_baru = rata_rata.tolist()
        else:
            # Jika cluster kosong, centroid tetap di posisi lama
            centroid_baru = centroids[i]
            
        new_centroids.append(centroid_baru)
        print(f"  -> Cluster {i+1} ({len(points_in_cluster)} data): Rata-rata {centroid_baru}")

    # --- C. Comparison Step (Bandingkan Lama vs Baru) ---
    print(f"\nCEK KONSISTENSI ITERASI {iteration}:")
    print(f"  Lama: {centroids}")
    print(f"  Baru: {new_centroids}")
    
    if new_centroids == centroids:
        print("\n[HASIL] Centroid TIDAK BERUBAH. Loop Dihentikan.")
        print("KONVERGENSI TERCAPAI! ✅")
        break
    else:
        print("\n[HASIL] Centroid BERUBAH. Lanjutkan ke Iterasi berikutnya... 🔄")
        # Update centroid untuk iterasi selanjutnya
        centroids = new_centroids