"""Pydantic models for API request and response schemas."""

from pydantic import BaseModel, Field

CONDITIONS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia",
    "Infiltration", "Mass", "Nodule", "Pleural Thickening",
    "Pneumonia", "Pneumothorax", "Normal",
]

CONDITION_CONTEXT = {
    "Atelectasis": "atelectasis lung collapse volume loss chest radiograph management",
    "Cardiomegaly": "cardiomegaly enlarged heart cardiothoracic ratio heart failure echocardiography",
    "Consolidation": "pulmonary consolidation lobar pneumonia air bronchograms chest X-ray",
    "Edema": "pulmonary edema fluid overload heart failure diuretics chest radiograph",
    "Effusion": "pleural effusion thoracentesis Light criteria transudative exudative",
    "Emphysema": "emphysema COPD hyperinflation bullae pulmonary function",
    "Fibrosis": "pulmonary fibrosis interstitial lung disease honeycombing HRCT restrictive",
    "Hernia": "diaphragmatic hernia hiatal hernia chest radiograph surgical repair",
    "Infiltration": "pulmonary infiltrate differential diagnosis infection inflammation chest imaging",
    "Mass": "pulmonary mass lung cancer solitary nodule malignancy biopsy staging",
    "Nodule": "pulmonary nodule lung cancer screening calcification malignancy risk Fleischner",
    "Pleural Thickening": "pleural thickening asbestos mesothelioma chronic pleuritis imaging",
    "Pneumonia": "pneumonia community-acquired empiric antibiotics CURB-65 chest radiograph treatment",
    "Pneumothorax": "pneumothorax chest tube tension pneumothorax management visceral pleural line",
    "Normal": "normal chest radiograph anatomy cardiac silhouette costophrenic angles",
}


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=5)
    condition: str = Field(default="")

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=5)
    cnn_prediction: str = Field(default="unknown")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    condition: str = Field(default="")

class RetrievedChunk(BaseModel):
    chunk_text: str
    similarity: float
    doc_id: str
    chunk_index: int

class RetrieveResponse(BaseModel):
    query: str
    condition: str
    chunks: list[RetrievedChunk]
    retrieval_latency_ms: float
    total_latency_ms: float

class QueryResponse(BaseModel):
    query: str
    cnn_prediction: str
    confidence: float
    condition: str
    chunks: list[RetrievedChunk]
    generated_response: str
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float

class ConditionPrediction(BaseModel):
    condition: str
    probability: float
    detected: bool

class PredictResponse(BaseModel):
    needs_manual_selection: bool = False
    predictions: list[ConditionPrediction]
    model_loaded: bool
    inference_latency_ms: float

class AnalyzeResponse(BaseModel):
    predictions: list[ConditionPrediction]
    detected_conditions: list[str]
    query: str
    chunks: list[RetrievedChunk]
    generated_response: str
    model_loaded: bool
    inference_latency_ms: float
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float

class HealthResponse(BaseModel):
    status: str
    corpus_loaded: bool
    corpus_size: int
    chunk_count: int
    cnn_model_loaded: bool

class StatsResponse(BaseModel):
    corpus_size: int
    chunk_count: int
    embedding_model: str
    chunking_strategy: str
    retrieval_k: int
    conditions: list[str]
    cnn_model_loaded: bool
