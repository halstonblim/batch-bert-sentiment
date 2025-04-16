# Batch BERT Sentiment Analysis

This repository hosts a [DistilBERT sentiment analysis](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english) model wrapped with [Cog](https://github.com/replicate/cog) for easy deployment on [Replicate](https://replicate.com/). The model handles **batched** input texts in a single API call to improve performance and reduce costs. This is especially important for using Public cold start models, as passing all inputs in a single API means we only need to boot up once. 

For more details, see the Replicate model page: [halstonblim/distilbert-base-uncased-finetuned-sst-2-english](https://replicate.com/halstonblim/distilbert-base-uncased-finetuned-sst-2-english)

---

## Contents

- **predict.py**: Contains the `Predictor` class, which loads the model and performs inference (including batched logic).
- **cog.yaml**: Specifies the build and runtime environment for Cog.
- **requirements.txt**: Lists Python dependencies (for local development).
- **Dockerfile** *(optional)*: Defines a custom build if needed.
- **test.py** *(optional)*: Minimal local testing script for direct Python usage.

---

## Quick Start

### 1. Install Cog (optional)
If you're testing locally, install [Cog](https://replicate.com/docs/guides/push-a-model#install-cog):
```bash
pip install cog
```

### 2. Run Locally with Cog

```bash
cog predict -i texts="[\"I hate apple\", \"I love apple\"]"
```

**Or**, provide a JSON file (e.g., `input.json`):
```json
["I hate apple", "I love apple"]
```
Then call:
```bash
cog predict -i texts=@input.json
```
Cog will invoke the `predict()` method and return a JSON response with predicted labels (`0` for negative, `1` for positive) and confidence values.

> **Note**: On Windows/WSL or PowerShell, you may need special quoting or a data URI format. The predictor code includes extra logic for decoding base64 data URIs.

---

## Usage on Replicate

1. **Push to Replicate**  
   After customizing your code, push the model:
   ```bash
   cog login
   cog push r8.im/<username>/<repo-name>
   ```
2. **Run Predictions**  
   On Replicate, you can call your model’s endpoint with a JSON list of texts. For example:
   ```bash
   replicate run <username>/<repo-name>:latest --texts '["I hate apple", "I love apple"]'
   ```
   (If you see issues with quotes or data URIs, consult the [Replicate docs](https://replicate.com/docs) for usage details.)

**Why Batch?**  
By passing multiple texts in a single API call, you minimize repeated overhead for container startup and model loading. This can improve throughput and potentially lower your Replicate usage costs, especially if you have many inputs.

---

## Local Testing (Optional)

You can also test the cog build importing the Predictor in a Python shell (`cog run python`) or sript (`cog run python script.py`)

```python
from predict import Predictor

predictor = Predictor()
predictor.setup()

# Provide a JSON-formatted string representing a list
test_input = '["I hate apple", "I love apple"]'
result = predictor.predict(test_input)
print(result)
```

You should see an output dictionary with `predicted_labels` and `confidences` for both inputs.
