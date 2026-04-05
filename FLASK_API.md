# 🚀 Flask API & Web Interface Guide

## Overview

The Flask API provides a web-based interface for the Nepali Honorifics Translator. It includes:
- **REST API endpoint** for programmatic translation
- **Interactive HTML webpage** with a modern UI for manual translation

---

## 🔧 Installation

### 1. Install Flask
```bash
pip install flask
```

Or update from requirements:
```bash
pip install -r requirements.txt
```

### 2. Ensure Model is Trained
Make sure you've trained and saved the model:
```bash
python scripts/train.py
```

The model should be saved at: `outputs/models/best_honorifics_model/`

---

## ▶️ Running the Flask Server

### Start the Server
```bash
python app.py
```

You should see:
```
🚀 Starting Flask API server...
📖 Visit http://localhost:5000 to access the translator
```

### Access the Web Interface
Open your browser and go to: **http://localhost:5000**

---

## 🌐 Web Interface Features

### User Interface
- **English Input**: Text area for entering English text
- **Nepali Output**: Read-only area displaying the translation
- **Translate Button**: Submits text for translation
- **Clear Button**: Clears both input and output
- **Copy Button**: Copies output to clipboard
- **Character Counter**: Shows input length (max 500 chars)
- **Status Messages**: Success/error feedback

### Usage
1. Enter English text in the input area
2. Click **"Translate"** or press **Ctrl+Enter**
3. View the Nepali translation in the output area
4. Click **"Copy"** to copy the translation to clipboard
5. Click **"Clear"** to start a new translation

---

## 📡 REST API Endpoints

### 1. **Translate Endpoint**
- **URL**: `http://localhost:5000/api/translate`
- **Method**: POST
- **Content-Type**: application/json

#### Request Format
```json
{
  "text": "Please sit down, sir."
}
```

#### Response (Success)
```json
{
  "success": true,
  "input": "Please sit down, sir.",
  "translation": "कृपया बस्नुहोस्, सर ।"
}
```

#### Response (Error)
```json
{
  "success": false,
  "error": "Text cannot be empty"
}
```

### 2. **Health Check Endpoint**
- **URL**: `http://localhost:5000/api/health`
- **Method**: GET

#### Response
```json
{
  "status": "ok",
  "model_loaded": true
}
```

---

## 🧪 Testing the API

### Using cURL (Windows PowerShell)
```powershell
$body = @{
    text = "I am going home"
} | ConvertTo-Json

Invoke-WebRequest -Uri "http://localhost:5000/api/translate" `
    -Method POST `
    -ContentType "application/json" `
    -Body $body
```

### Using Python
```python
import requests

url = "http://localhost:5000/api/translate"
data = {"text": "Please sit down, sir."}

response = requests.post(url, json=data)
print(response.json())
```

### Using JavaScript (Fetch API)
```javascript
fetch('http://localhost:5000/api/translate', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json'
    },
    body: JSON.stringify({ text: 'I am going home' })
})
.then(response => response.json())
.then(data => console.log(data))
.catch(error => console.error('Error:', error));
```

---

## 📁 File Structure

```
nllb-honorifics-nepali/
├── app.py                          # Flask application
├── templates/
│   └── index.html                  # Web interface
├── src/
│   ├── translator.py               # Translation logic
│   ├── config.py                   # Configuration
│   └── ...
└── outputs/models/
    └── best_honorifics_model/      # Trained model
```

---

## ⚙️ Configuration

### API Settings (in app.py)
- **Host**: `0.0.0.0` (accessible from all interfaces)
- **Port**: `5000`
- **Debug Mode**: `True` (auto-reload on code changes)

### To change settings:
Edit `app.py` line:
```python
app.run(debug=True, host='0.0.0.0', port=5000)
```

---

## 🔒 Security Notes

### For Production Use
1. **Set debug mode to False**:
   ```python
   app.run(debug=False, host='127.0.0.1', port=5000)
   ```

2. **Use a production server** (e.g., Gunicorn):
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

3. **Add rate limiting** to prevent abuse

4. **Enable HTTPS** with SSL certificates

5. **Add authentication** if needed

---

## 🐛 Troubleshooting

### Model not found error
```
❌ Model not found. Please train the model first.
```
**Solution**: Train the model first using `python scripts/train.py`

### Port already in use
```
OSError: [Errno 48] Address already in use
```
**Solution**: Change the port in `app.py` or kill the process using the port:
```bash
# Windows PowerShell
Get-Process -Name python | Stop-Process

# Or specify different port in app.py
app.run(debug=True, host='0.0.0.0', port=5001)
```

### CORS issues (if calling from different domain)
Add CORS support to `app.py`:
```python
from flask_cors import CORS
CORS(app)
```
Then install: `pip install flask-cors`

### Slow translation
- Model loading takes time on first request
- Subsequent requests are faster (model cached in memory)
- Use GPU for faster inference

### Connection refused
- Ensure Flask server is running
- Check if firewall is blocking port 5000
- Verify correct URL: `http://localhost:5000`

---

## 📊 Example API Usage Scenarios

### Scenario 1: Translate multiple texts
```python
texts = [
    "I am going home",
    "Please sit down, sir.",
    "Good morning, madam"
]

for text in texts:
    response = requests.post(
        "http://localhost:5000/api/translate",
        json={"text": text}
    )
    print(f"Input: {text}")
    print(f"Output: {response.json()['translation']}")
    print()
```

### Scenario 2: Batch translation with error handling
```python
def batch_translate(texts, api_url="http://localhost:5000/api/translate"):
    results = []
    for text in texts:
        try:
            response = requests.post(api_url, json={"text": text}, timeout=10)
            if response.status_code == 200:
                results.append(response.json())
            else:
                results.append({"error": f"Status {response.status_code}"})
        except Exception as e:
            results.append({"error": str(e)})
    return results
```

---

## 📝 Notes

- Maximum input length: 500 characters
- Supports Devanagari script in output
- Model trained on honorific domain data
- Automatic text cleaning and normalization

---

## 🎯 Next Steps

1. Try translating different sentences
2. Integrate the API into your applications
3. Customize the HTML interface for your needs
4. Deploy to production server
5. Monitor API performance and user feedback

---

**Enjoy translation! 🚀**
