# preprocessing/automate_adinda_putri.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

def preprocess_and_save(input_path, output_dir):
    # Load dataset
    df = pd.read_csv(input_path)

    # Hapus kolom Id jika ada
    if "Id" in df.columns:
        df = df.drop(columns=["Id"])

    # Pisahkan fitur dan target
    X = df.drop("Species", axis=1)
    y = df["Species"]

    # Encode target
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42
    )

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Buat folder output
    os.makedirs(output_dir, exist_ok=True)

    # Simpan scaler dan label encoder
    joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))
    joblib.dump(le, os.path.join(output_dir, "label_encoder.pkl"))

    # Simpan data latih & uji dalam CSV
    train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    train_df["Species"] = y_train
    train_df.to_csv(os.path.join(output_dir, "iris_train_preprocessed.csv"), index=False)

    test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
    test_df["Species"] = y_test
    test_df.to_csv(os.path.join(output_dir, "iris_test_preprocessed.csv"), index=False)

    print(f"Preprocessing selesai. Hasil disimpan di: {output_dir}")

if __name__ == "__main__":
    preprocess_and_save("../Iris.csv", "iris_preprocessing")