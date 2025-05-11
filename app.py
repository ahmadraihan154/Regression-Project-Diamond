import pandas as pd
import numpy as np
import joblib
import streamlit as st
from datetime import datetime

# Fungsi untuk prediksi harga
def predict_price(models, input_dict):
    try:
        # Convert input_dict ke DataFrame
        input_df = pd.DataFrame([input_dict])

        # === 1. Transformasi fitur numerik ===
        num_features = ['carat', 'table', 'x', 'y', 'z']
        for col in num_features:
            transformer = models['power_transformers'][col]
            input_df[f'transform_{col}'] = transformer.transform(input_df[[col]])
        transformed_num = input_df[[f'transform_{col}' for col in num_features]]

        # === 2. Encoding fitur kategorikal ===
        cat_features = ['cut', 'color', 'clarity']
        encoded = models['encoder'].transform(input_df[cat_features])
        encoded_df = pd.DataFrame(
            encoded,
            columns=models['encoder'].get_feature_names_out(cat_features),
            index=input_df.index
        )

        # === 3. Gabungkan fitur hasil transformasi numerik dan kategorikal ===
        input_processed = pd.concat([encoded_df, transformed_num], axis=1)

        # === 4. Lengkapi kolom agar sesuai urutan training ===
        for col in models['final_columns']:
            if col not in input_processed:
                input_processed[col] = 0  # tambahkan kolom yang hilang dengan nol

        input_processed = input_processed[models['final_columns']]  # urutkan kolom sesuai model

        # === 5. Prediksi dan inverse transform harga ===
        pred = models['lgbm_model'].predict(input_processed)
        price_transformer = models['power_transformers']['price']
        predicted_price = price_transformer.inverse_transform(np.array(pred).reshape(-1, 1))[0][0]

        return predicted_price

    except Exception as e:
        st.error(f"Error saat prediksi: {str(e)}")
        return None

# Load resources dengan caching 
@st.cache_resource
def load_resources():
    return {
        'models': {
            'lgbm_model': joblib.load('lgbm_model.joblib'),
            'power_transformers': joblib.load('power_transformers.joblib'),
            'encoder': joblib.load('onehot_encoder.joblib'),
            'final_columns': joblib.load('final_feature_columns.joblib')
        }
    }

def main():
    st.set_page_config(
        page_title="Diamond Price Prediction",
        page_icon="💎",
        layout="centered"  # Diubah dari "wide" ke "centered"
    )
    
    # Inisialisasi session state (disederhanakan)
    if 'resources' not in st.session_state:
        st.session_state.resources = load_resources()
    if 'history' not in st.session_state:
        st.session_state.history = pd.DataFrame(columns=[
            'Timestamp', 'Carat', 'Cut', 'Color', 'Clarity', 
            'Table', 'Length', 'Width', 'Depth', 'Predicted Price'
        ])
    
    # Navigasi sederhana
    page = st.sidebar.radio("Menu", ["Prediction", "History"])
    
    if page == "Prediction":
        render_prediction_page()
    else:
        render_history_page()

def render_prediction_page():
    st.title("💎 Diamond Price Prediction")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            carat = st.number_input(
                "Carat Weight", 
                min_value=0.1, 
                max_value=10.0, 
                value=None,
                placeholder="Masukkan nilai 0.1 sampai 10.0",
                step=0.1
            )
            cut = st.selectbox(
                "Cut Quality (fair terendah - ideal tertinggi)", 
                ['', 'Fair', 'Good', 'Very Good', 'Premium', 'Ideal'], 
                index=0
            )
            color = st.selectbox(
                "Color Grade (J terendah - D tertinggi)", 
                ['', 'J', 'I', 'H', 'G', 'F', 'E', 'D'], 
                index=0
            )
            clarity = st.selectbox(
                "Clarity Grade (I1 terendah - IF tertinggi)", 
                ['', 'I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'], 
                index=0
            )
        
        with col2:
            table = st.number_input(
                "Table", 
                min_value=50.0, 
                max_value=80.0, 
                value=None,
                placeholder="Masukkan nilai 50.0 sampai 80.0",
                step=0.1
            )
            x = st.number_input(
                "Length", 
                min_value=0.0, 
                max_value=20.0, 
                value=None,
                placeholder="Masukkan nilai 0.0 sampai 20.0",
                step=0.1
            )
            y = st.number_input(
                "Width", 
                min_value=0.0, 
                max_value=20.0, 
                value=None,
                placeholder="Masukkan nilai 0.0 sampai 20.0",
                step=0.1
            )
            z = st.number_input(
                "Depth", 
                min_value=0.0, 
                max_value=20.0, 
                value=None,
                placeholder="Masukkan nilai 0.0 sampai 20.0",
                step=0.1
            )
        
        submitted = st.form_submit_button("Predict Price", type="primary")
        
        if submitted:
            if not all([carat, cut, color, clarity, table, x, y, z]):
                st.error("Harap isi semua field!")
            else:
                input_dict = {
                    'carat': carat, 'cut': cut, 'color': color, 'clarity': clarity,
                    'table': table, 'x': x, 'y': y, 'z': z
                }
                predicted_price = predict_price(st.session_state.resources['models'], input_dict)
                
                if predicted_price:
                    # Tambahkan ke history
                    new_entry = {
                        'Timestamp': datetime.now(),
                        'Carat': carat,
                        'Cut': cut,
                        'Color': color,
                        'Clarity': clarity,
                        'Table': table,
                        'Length': x,
                        'Width': y,
                        'Depth': z,
                        'Predicted Price': predicted_price
                    }
                    
                    st.session_state.history = pd.concat([
                        st.session_state.history,
                        pd.DataFrame([new_entry])
                    ], ignore_index=True)
                    
                    st.success(f"💎 Predicted Price: ${predicted_price:,.2f}")

def render_history_page():
    st.title("📊 Prediction History")
    
    if st.session_state.history.empty:
        st.warning("No prediction history yet!")
    else:
        # Format harga dan tampilkan tabel
        history_display = st.session_state.history.copy()
        history_display['Predicted Price'] = history_display['Predicted Price'].apply(lambda x: f"${x:,.2f}")
        
        st.dataframe(
            history_display.sort_values('Timestamp', ascending=False),
            hide_index=True,
            use_container_width=True
        )
        
        # Opsi download history
        csv = st.session_state.history.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download History as CSV",
            data=csv,
            file_name='diamond_price_predictions.csv',
            mime='text/csv'
        )

if __name__ == "__main__":
    main()
