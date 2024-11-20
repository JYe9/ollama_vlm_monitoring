# ollama_vlm_monitoring

## ✨ Features
- Upload and analyze videos through an intuitive web interface
- Real-time frame-by-frame analysis using multimodal AI
- Natural language object description support
- Visual results display with confidence scores
- Image preprocessing for better detection accuracy
- Streaming response for real-time analysis feedback

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Ollama with Llama Vision model installed
- OpenCV

### Installation

1. Clone the repository
```bash
git clone https://github.com/JYe9/ollama_vlm_monitoring.git
cd ollama_vlm_monitoring
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Make sure Ollama is running with Llama Vision model
```bash
ollama run llama3.2-vision
```

4. Start the application
```bash
python main.py
```

5. Access the web interface at `http://localhost:8000`

## 🛠️ Usage
1. Open the web interface
2. Upload a video file
3. Enter a description of the object/person you want to find
4. Click "Start Analysis"
5. View results as they appear in real-time

## 📦 Dependencies
- FastAPI
- OpenCV
- Ollama
- Jinja2
- uvicorn
