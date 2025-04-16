from cog import BasePredictor, Input
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import json
import base64

class Predictor(BasePredictor):
    def setup(self):
        """Load the Hugging Face model into memory."""
        self.model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    def _decode_data_uri(self, data_uri: str) -> str:
        """Decode a data URI and return the decoded JSON string."""
        prefix = "data:application/json;base64,"
        if data_uri.startswith(prefix):
            base64_str = data_uri[len(prefix):]
            decoded_bytes = base64.b64decode(base64_str)
            return decoded_bytes.decode("utf-8")
        return data_uri

    def predict(self, texts: str = Input(description="A JSON-formatted list of texts for sentiment analysis")) -> dict:
        """
        Run sentiment analysis in batch mode.
        The input 'texts' is expected to be a JSON-formatted string representing a list.
        Returns a dictionary with predicted_labels and confidences.
        """
        # Decode data URI if necessary.
        if isinstance(texts, str) and texts.startswith("data:"):
            texts = self._decode_data_uri(texts)

        # Attempt to parse the input JSON.
        try:
            parsed_texts = json.loads(texts)
        except Exception:
            parsed_texts = [texts]
        
        if not isinstance(parsed_texts, list):
            parsed_texts = [parsed_texts]

        batch_size = 32
        predicted_labels = []
        confidences = []
        
        # Process texts in batches.
        for i in range(0, len(parsed_texts), batch_size):
            batch_texts = parsed_texts[i:i + batch_size]
            cleaned_texts = [text if isinstance(text, str) and text.strip() else "[EMPTY]" 
                              for text in batch_texts]

            inputs = self.tokenizer(
                cleaned_texts, 
                padding=True, 
                truncation=True, 
                return_tensors="pt", 
                max_length=128
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                batch_probs = F.softmax(logits, dim=-1)

            batch_probs_cpu = batch_probs.cpu()
            predicted_labels_batch = torch.argmax(batch_probs_cpu, dim=1)
            confidences_batch = torch.max(batch_probs_cpu, dim=1).values
            confidences_batch = torch.round(confidences_batch * 10000) / 10000.0

            predicted_labels.extend(predicted_labels_batch.tolist())
            confidences.extend(confidences_batch.tolist())

        return {
            "predicted_labels": predicted_labels,
            "confidences": confidences
        }
