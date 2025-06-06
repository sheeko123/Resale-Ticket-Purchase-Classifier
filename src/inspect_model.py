import pickle
import streamlit as st

def inspect_model():
    try:
        with open("models/trained_models/model.pkl", 'rb') as f:
            model_data = pickle.load(f)
            st.write("Model data type:", type(model_data))
            st.write("Model data contents:", model_data)
    except Exception as e:
        st.error(f"Error inspecting model: {str(e)}")

if __name__ == "__main__":
    inspect_model() 