import json

docs = json.load(open("pipelines/pubmed_cache/documents.json", encoding="utf-8"))

CONDITIONS = {
    "Atelectasis": ["atelectasis", "lung collapse", "volume loss"],
    "Cardiomegaly": ["cardiomegaly", "enlarged heart", "cardiothoracic ratio"],
    "Consolidation": ["consolidation", "lobar consolidation", "air bronchogram"],
    "Edema": ["pulmonary edema", "fluid overload", "kerley lines"],
    "Effusion": ["pleural effusion", "effusion", "thoracentesis"],
    "Emphysema": ["emphysema", "copd", "hyperinflation", "bullae"],
    "Fibrosis": ["fibrosis", "interstitial lung disease", "honeycombing"],
    "Hernia": ["hernia", "diaphragmatic hernia", "hiatal"],
    "Infiltration": ["infiltrate", "infiltration", "opacity"],
    "Mass": ["mass", "tumor", "malignancy", "lung cancer", "carcinoma"],
    "Nodule": ["nodule", "pulmonary nodule", "solitary nodule"],
    "Pleural Thickening": ["pleural thickening", "asbestos", "mesothelioma"],
    "Pneumonia": ["pneumonia", "community-acquired", "bacterial pneumonia"],
    "Pneumothorax": ["pneumothorax", "chest tube", "tension pneumothorax"],
    "Normal": ["normal chest", "normal radiograph", "no acute"],
}

print(f"Corpus: {len(docs)} documents\n")
print(f"{'Condition':<22} {'Docs':>6}")
print("-" * 32)

for condition, keywords in CONDITIONS.items():
    count = sum(1 for doc in docs if any(kw in (doc.get("title","") + " " + doc.get("abstract","")).lower() for kw in keywords))
    status = "OK" if count >= 10 else "LOW" if count >= 3 else "GAP"
    print(f"{condition:<22} {count:>6}  {status}")