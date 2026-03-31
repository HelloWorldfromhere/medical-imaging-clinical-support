"""Fetch PubMed articles for weak conditions identified in spot-check."""

import json, time, os
from dotenv import load_dotenv
from Bio import Entrez

load_dotenv()
Entrez.email = os.getenv("ENTREZ_EMAIL")
print("Using email from .env")

# Targeted terms for conditions with poor retrieval
new_terms = [
    # TB - need AFB testing, isolation, treatment protocols
    "tuberculosis AFB smear culture diagnosis sensitivity specificity",
    "tuberculosis airborne isolation precautions infection control",
    "tuberculosis treatment isoniazid rifampin pyrazinamide ethambutol",
    "cavitary tuberculosis chest radiograph upper lobe",
    # PE - completely missing from corpus
    "pulmonary embolism diagnosis D-dimer CT angiography",
    "pulmonary embolism Wells score clinical prediction",
    "pulmonary embolism anticoagulation heparin treatment",
    "deep vein thrombosis pulmonary embolism risk factors",
    # Atelectasis - low coverage (15 mentions)
    "atelectasis post operative lung expansion management",
    "obesity hypoventilation atelectasis CPAP respiratory",
    # Cardiomegaly - very low (6 mentions)
    "cardiomegaly chest radiograph cardiothoracic ratio heart failure",
    # Bronchoscopy - low (18 mentions)
    "bronchoscopy non resolving pneumonia endobronchial obstruction",
    # Asthma exacerbation specifics
    "asthma exacerbation emergency management oxygen bronchodilator",
    # Lupus pleuritis specifics
    "lupus pleuritis pleural effusion ANA complement rheumatology treatment",
]

existing = json.load(open("pipelines/pubmed_cache/documents.json", encoding="utf-8"))
existing_ids = {d["id"] for d in existing}
print(f"Starting with {len(existing)} docs")

new_count = 0
for term in new_terms:
    print(f"  Searching: {term[:55]}...")
    try:
        sh = Entrez.esearch(db="pubmed", term=term, retmax=20, sort="relevance")
        pmids = Entrez.read(sh)["IdList"]
        sh.close()
        if not pmids:
            continue
        fh = Entrez.efetch(db="pubmed", id=pmids, rettype="abstract", retmode="xml")
        records = Entrez.read(fh).get("PubmedArticle", [])
        fh.close()
        for r in records:
            try:
                c = r["MedlineCitation"]
                a = c["Article"]
                pmid = str(c["PMID"])
                pid = f"PMID:{pmid}"
                if pid in existing_ids:
                    continue
                title = str(a.get("ArticleTitle", "")).strip()
                ap = a.get("Abstract", {}).get("AbstractText", [])
                abstract = (
                    " ".join(str(p) for p in ap).strip()
                    if isinstance(ap, list)
                    else str(ap).strip()
                )
                if not abstract or len(abstract) < 50:
                    continue
                existing.append(
                    {"id": pid, "title": title, "abstract": abstract, "full_text": "", "keywords": []}
                )
                existing_ids.add(pid)
                new_count += 1
            except:
                continue
        time.sleep(0.5)
    except Exception as e:
        print(f"    Error: {e}")
        time.sleep(2)

with open("pipelines/pubmed_cache/documents.json", "w", encoding="utf-8") as f:
    json.dump(existing, f, indent=2, ensure_ascii=False)
print(f"\nAdded {new_count} new docs. Total: {len(existing)}")

# Coverage check
text = " ".join(d["title"].lower() + " " + d["abstract"].lower() for d in existing)
print("\nCondition coverage after expansion:")
for condition in [
    "tuberculosis", "AFB", "isolation", "pulmonary embolism",
    "D-dimer", "angiography", "atelectasis", "cardiomegaly",
    "bronchoscopy", "asthma", "lupus",
]:
    count = text.count(condition.lower())
    print(f"  {condition:<22} mentions: {count:>4}")
