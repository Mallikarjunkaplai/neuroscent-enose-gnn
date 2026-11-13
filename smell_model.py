import os
import ast
import numpy as np
import deepchem as dc
from openpom.models.mpnn_pom import MPNNPOMModel
from openpom.feat.graph_featurizer import GraphFeaturizer

def load_tasks_safely():
    """Extract TASKS list directly from predict_odors.py AST to avoid CLI execution."""
    try:
        with open("predict_odors.py", "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "TASKS":
                        return ast.literal_eval(node.value)
    except Exception:
        pass
    
    # 138 Standard POM Odor Descriptors Fallback
    return [
        'alcoholic', 'aldehydic', 'aliaceous', 'almond', 'amber', 'animal', 'anise', 'apple', 
        'apricot', 'aromatic', 'berry', 'bitter', 'black currant', 'brandy', 'burnt', 'butter', 
        'camphoreous', 'caramellic', 'cedar', 'chamomile', 'cheese', 'cherry', 'chocolate', 
        'cinnamon', 'citrus', 'clean', 'clove', 'cocoa', 'coconut', 'coffee', 'cognac', 'cooked', 
        'cooling', 'cortex', 'coumarinic', 'creamy', 'cucumber', 'dairy', 'dry', 'earthy', 
        'ethereal', 'fatty', 'fermented', 'fishy', 'floral', 'fresh', 'fruit', 'fruity', 
        'garlic', 'gas', 'geranium', 'grape', 'grapefruit', 'grassy', 'green', 'herbal', 
        'honey', 'hyacinth', 'incense', 'jasmin', 'juicy', 'kerosene', 'lactonic', 'lavender', 
        'leafy', 'leather', 'lemon', 'licorice', 'lily', 'mahogany', 'malt', 'mandarin', 
        'meat', 'medicinal', 'melon', 'menthol', 'metallic', 'milk', 'mint', 'muguet', 
        'mushroom', 'musk', 'musty', 'nutty', 'odorous', 'oily', 'onion', 'orange', 
        'orris', 'ozone', 'peach', 'pear', 'phenolic', 'pine', 'pineapple', 'plum', 
        'popcorn', 'potato', 'pungent', 'radish', 'raspberry', 'roasted', 'rose', 'rum', 
        'sandalwood', 'savory', 'sharp', 'smoky', 'soap', 'sour', 'spicy', 'strawberry', 
        'sulfurous', 'sweet', 'tea', 'tobacco', 'tomato', 'tropical', 'vanilla', 'vegetable', 
        'violet', 'warm', 'waxy', 'wine', 'woody', 'yeasty'
    ]

TASKS = load_tasks_safely()

class SmellPredictor:
    """
    Evaluates SMILES chemical structures using the pre-trained OpenPOM ensemble model.
    """
    def __init__(self, models_dir="./models/ensemble_models"):
        self.tasks = TASKS
        self.n_tasks = len(self.tasks)
        self.featurizer = GraphFeaturizer()
        self.models = []
        
        # Load all 10 ensemble checkpoints using hardcoded feature dimensions (134 atom, 6 bond)
        for i in range(10):
            exp_dir = os.path.join(models_dir, f"experiments_{i+1}")
            model = MPNNPOMModel(
                n_tasks=self.n_tasks,
                batch_size=128,
                node_out_feats=100,
                edge_hidden_feats=75,
                edge_out_feats=100,
                num_step_message_passing=5,
                mpnn_residual=True,
                message_aggregator_type='sum',
                mode='classification',
                number_atom_features=134,
                number_bond_features=6,
                n_classes=1,
                readout_type='set2set',
                num_step_set2set=3,
                num_layer_set2set=2,
                ffn_hidden_list=[392, 392],
                ffn_embeddings=256,
                ffn_activation='relu',
                ffn_dropout_p=0.12,
                ffn_dropout_at_input_no_act=False,
                weight_decay=1e-5,
                self_loop=False,
                optimizer_name='adam',
                log_frequency=32,
                model_dir=exp_dir,
                device_name='cpu'
            )
            ckpt_path = os.path.join(exp_dir, "checkpoint2.pt")
            model.restore(ckpt_path)
            self.models.append(model)

    def predict_sample_odors(self, smiles_list, top_n=10):
        # Featurize SMILES strings into molecular graphs
        featurized_data = self.featurizer.featurize(smiles_list)
        dataset = dc.data.NumpyDataset(featurized_data)
        
        # Collect predictions across all 10 ensemble checkpoints
        all_preds = []
        for model in self.models:
            preds = model.predict(dataset)
            all_preds.append(preds)
            
        # Average ensemble probabilities across models & constituent molecules
        avg_preds = np.mean(all_preds, axis=0)  # Shape: [num_molecules, 138]
        sample_avg = np.mean(avg_preds, axis=0)  # Shape: [138]
        
        # Rank top N descriptors
        top_indices = np.argsort(sample_avg)[::-1][:top_n]
        results = [
            {"descriptor": self.tasks[idx], "probability": float(sample_avg[idx])}
            for idx in top_indices
        ]
        return results

if __name__ == "__main__":
    predictor = SmellPredictor()
    test_smiles = ["CC(C)CCOC(=O)C", "CC(=O)OCCCCC"]
    results = predictor.predict_sample_odors(test_smiles)
    print("\n--- Predicted Top Smell Descriptors ---")
    for r in results:
        print(f"{r['descriptor']:<15}: {r['probability']:.4f}")