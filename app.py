import json
import pickle
import numpy as np
from sensor import SensorArray
from smell_model import SmellPredictor

def main():
    print("==================================================")
    print("  ELECTRONIC NOSE & OPENPOM INTEGRATED PIPELINE   ")
    print("==================================================\n")

    # 1. Load trained Random Forest nose classifier
    print("Loading Electronic Nose Classifier ('nose_model.pkl')...")
    with open("nose_model.pkl", "rb") as f:
        nose_clf = pickle.load(f)
        
    # 2. Load samples dataset
    with open("samples.json", "r") as f:
        samples = json.load(f)
        
    # 3. Initialize components
    sensor_array = SensorArray()
    print("Loading OpenPOM Ensemble GNN Model (10 Checkpoints)...")
    smell_predictor = SmellPredictor()
    print("All components loaded successfully!\n")

    # Display available test aromas
    print("Available aroma samples to sniff:")
    sample_keys = list(samples.keys())
    for idx, key in enumerate(sample_keys, 1):
        print(f"  [{idx}] {samples[key]['name']}")
        
    choice = input("\nSelect a sample number to sniff (1-6) [Default: 5 - Banana]: ").strip()
    if not choice or not choice.isdigit() or int(choice) < 1 or int(choice) > len(sample_keys):
        selected_key = "banana"
    else:
        selected_key = sample_keys[int(choice) - 1]
        
    target_sample = samples[selected_key]
    print(f"\n---> Sniffing '{target_sample['name']}' with 8-channel MOX sensor array...")
    
    # 4. Generate a noisy sensor reading
    raw_signal = sensor_array.sniff(target_sample["smiles"], noise_level=0.05, humidity_factor=1.0)
    print("\n--- 8-Channel Sensor Signal Output ---")
    for i, val in enumerate(raw_signal, 1):
        print(f"  Sensor {i}: {val:.3f}")

    # 5. Electronic Nose Classification (Random Forest)
    predicted_label = nose_clf.predict([raw_signal])[0]
    confidence = np.max(nose_clf.predict_proba([raw_signal])) * 100
    predicted_name = samples[predicted_label]["name"]
    
    print(f"\n--- Electronic Nose Classifier Output ---")
    print(f"  Predicted Identity : {predicted_name}")
    print(f"  Confidence Score   : {confidence:.1f}%")

    # 6. Retrieve constituent SMILES & run OpenPOM GNN model
    constituent_smiles = samples[predicted_label]["smiles"]
    print(f"\n--- Running OpenPOM GNN Model on Constituent Molecules ---")
    for s in constituent_smiles:
        print(f"  - {s}")

    odors = smell_predictor.predict_sample_odors(constituent_smiles, top_n=10)
    
    print(f"\n--- Top Predicted Smell Descriptors for '{predicted_name}' ---")
    for o in odors:
        print(f"  {o['descriptor']:<15}: {o['probability']:.4f}")
    print("\n==================================================")

if __name__ == "__main__":
    main()