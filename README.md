# ML Model Predictor (Node.js + Python) in Docker

This minimal app serves a static frontend and a Node.js API that calls a Python script to load `model.pkl` and return predictions.

Files added:
- `server.js` - Node.js Express server that forwards POST /predict to `predict.py`.
- `predict.py` - Python script that loads `model.pkl` and returns a JSON prediction.
- `public/index.html` - Simple frontend to submit feature values and show predictions.
- `Dockerfile`, `docker-compose.yml` - for building and running the app.

How to run (Docker required):

1. Place your trained model file named `model.pkl` in the project root (next to this README).
2. Build and run:

```bash
docker-compose up --build
```

3. Open http://localhost:3000 in your browser and use the form.

Notes:
- The backend executes `predict.py` with `python3` and expects a `model.pkl` loadable by `joblib`.
- Input is a JSON array of numbers for one sample (e.g., `[5.1,3.5,1.4,0.2]`).
# MLOps
Tugas Mata Kuliah MLOps Universitas Brawijaya Fakultas Ilmu Komputer Tahun 2025
