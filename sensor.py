import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

class SensorArray:
    """
    Simulates an 8-channel MOX gas sensor array reacting to molecular features.
    """
    def __init__(self, num_sensors=8):
        self.num_sensors = num_sensors

    def extract_features(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(4)
        
        mw = Descriptors.ExactMolWt(mol)
        logp = Descriptors.MolLogP(mol)
        o_count = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 8)
        s_count = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 16)
        
        return np.array([mw, logp, o_count, s_count])

    def get_raw_response(self, smiles_list):
        features = np.array([self.extract_features(s) for s in smiles_list])
        avg_features = np.mean(features, axis=0)
        mw, logp, o_count, s_count = avg_features
        
        # 8 distinct response equations simulating sensor selectivities
        responses = np.zeros(self.num_sensors)
        responses[0] = 0.05 * mw + 0.8 * logp
        responses[1] = 0.10 * mw + 1.2 * o_count
        responses[2] = 1.5 * logp + 2.0 * o_count
        responses[3] = 3.5 * s_count + 0.4 * logp
        responses[4] = 0.02 * mw + 2.5 * s_count
        responses[5] = 0.8 * o_count + 0.5 * logp
        responses[6] = 0.08 * mw + 0.3 * logp
        responses[7] = 1.0 * o_count + 1.8 * s_count
        
        return np.maximum(0, responses)

    def sniff(self, smiles_list, noise_level=0.05, humidity_factor=1.0):
        base_signal = self.get_raw_response(smiles_list)
        # Apply environmental humidity shift and Gaussian sensor noise
        humidity_effect = base_signal * (0.95 + 0.1 * humidity_factor)
        noise = np.random.normal(0, noise_level, size=self.num_sensors)
        noisy_signal = humidity_effect + noise
        return np.maximum(0, noisy_signal)

if __name__ == "__main__":
    # Test run on Banana aroma
    banana_smiles = ["CC(C)CCOC(=O)C", "CC(=O)OCCCCC", "CCCC(=O)OCC"]
    array = SensorArray()
    signal = array.sniff(banana_smiles)
    print("Simulated 8-Channel Sensor Signal (Banana):")
    print(np.round(signal, 3))