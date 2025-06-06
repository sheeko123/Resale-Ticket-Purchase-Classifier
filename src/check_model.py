import pickle
import sys

try:
    with open('models/trained_models/model.pkl', 'rb') as f:
        model_dict = pickle.load(f)
        print("Model type:", type(model_dict))
        if isinstance(model_dict, dict):
            print("\nModel keys:", model_dict.keys())
            model = model_dict['model']
            print("\nModel features:", model.feature_names_)
            print("\nModel parameters:", model.get_params())
except Exception as e:
    print("Error:", str(e)) 