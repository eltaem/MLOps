const express = require('express');
const cors = require('cors');
const { execFile } = require('child_process');
const path = require('path');

const app = express();
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

const PYTHON_CMD = 'python3';
const PREDICT_SCRIPT = path.join(__dirname, 'predict.py');

// Contract: POST /predict { features: [num, ...] }
// Response: { prediction, probabilities, classes }
app.post('/predict', (req, res) => {
  const body = req.body || {};
  const features = body.features;
  if (!Array.isArray(features)) {
    return res.status(400).json({ error: 'features must be an array' });
  }

  // Basic size check for iris-like models (4 features) — not enforced strictly
  if (features.length === 0) {
    return res.status(400).json({ error: 'features array is empty' });
  }

  const arg = JSON.stringify({ features });
  execFile(PYTHON_CMD, [PREDICT_SCRIPT, arg], { maxBuffer: 10 * 1024 * 1024 }, (err, stdout, stderr) => {
    // If python process errored, check if it still printed a JSON payload on stdout
    if (err) {
      // Try to parse stdout first (predict.py prints JSON even on error)
      if (stdout) {
        try {
          const out = JSON.parse(stdout);
          return res.status(out.error ? 500 : 200).json(out);
        } catch (parseErr) {
          console.error('Prediction error, invalid JSON from python stdout:', stdout, parseErr, 'stderr:', stderr);
          return res.status(500).json({ error: 'prediction failed', details: stderr || String(err) });
        }
      }
      console.error('Prediction error:', err, stderr);
      return res.status(500).json({ error: 'prediction failed', details: stderr || String(err) });
    }
    try {
      const out = JSON.parse(stdout);
      return res.json(out);
    } catch (e) {
      console.error('Invalid JSON from python:', stdout, e);
      return res.status(500).json({ error: 'invalid response from predictor' });
    }
  });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server listening on port 3000 web: http://localhost:${PORT}`);
});
