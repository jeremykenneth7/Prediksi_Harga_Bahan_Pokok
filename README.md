# Prediksi Harga Bahan Pokok

Proyek ini bertujuan untuk memprediksi harga berbagai bahan makanan pokok menggunakan model SARIMA (Seasonal AutoRegressive Integrated Moving Average). Prediksi harga mencakup periode dari Juni 2024 hingga Desember 2026. Data yang digunakan untuk prediksi ini berasal dari `DataBahanPokok.csv`.

## Daftar Isi
- [Deskripsi Proyek](#deskripsi-proyek)
- [Instalasi](#instalasi)
- [Penggunaan](#penggunaan)
- [Hasil](#hasil)

## Deskripsi Proyek

Tujuan utama dari proyek ini adalah untuk memprediksi harga masa depan dari bahan makanan pokok berikut:
- Daging Ayam Kampung
- Kacang Kedelai Lokal
- Telur Ayam Kampung
- Garam Beryodium Halus

Dengan menggunakan model SARIMA, kami berusaha memberikan prediksi harga yang akurat untuk bahan-bahan ini, membantu para pemangku kepentingan dalam membuat keputusan yang tepat.

## Instalasi

1. Clone repositori:
    ```sh
    git clone https://github.com/jeremykenneth7/Prediksi_Harga_Bahan_Pokok.git
    cd prediksi-harga-bahan-pokok
    ```

2. Buat dan aktifkan virtual environment (opsional tapi disarankan):
    ```sh
    python -m venv env
    source env/bin/activate # Di Windows, gunakan `env\Scripts\activate`
    ```

## Penggunaan

1. Letakkan file data Anda (`DataBahanPokok.csv`) di direktori proyek.

2. Jalankan skrip `data_processing.py` untuk memfit model SARIMA dan menghasilkan prediksi:
    ```sh
    python data_processing.py
    ```

3. Skrip akan mengeluarkan Root Mean Squared Error (RMSE) untuk setiap item dan menampilkan plot yang menunjukkan harga historis dan harga yang diprediksi.

## Hasil

Setelah menjalankan skrip, Anda akan melihat plot untuk setiap item yang tercantum di atas. Plot tersebut akan menunjukkan harga historis serta harga yang diprediksi dari Juni 2024 hingga Desember 2026. Selain itu, nilai RMSE untuk setiap item akan dicetak, menunjukkan akurasi model.
