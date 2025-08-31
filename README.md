# ScenePhys: Physics Video Understanding Benchmark

A comprehensive benchmark for evaluating Vision-Language Models (VLMs) on physics video understanding tasks using PhET Interactive Simulations.

## 📊 Dataset Overview

- **Total Videos**: 382 physics simulation clips
- **Total Q/A Pairs**: 1,146 (3 questions per video)
- **Categories**: 17 physics topics across 4 major fields
- **Models Evaluated**: GPT-4o-mini, Gemini-2.5-Flash-Lite, Qwen-VL-Plus

## 🏗️ Dataset Structure

| Field | Topics | Clips | Q/A Pairs |
|-------|--------|-------|-----------|
| **Mechanics & Fluids** | Buoyancy, Collision, Mass-spring, Pendulum, Projectile | 79 | 237 |
| **Optics** | Concave/Convex Lenses & Mirrors, Plane Mirror | 50 | 150 |
| **Electromagnetism & Circuits** | Capacitors, Coulomb's Law, Induction, RC Circuits | 130 | 390 |
| **Quantum Mechanics** | Hydrogen Atom, Quantum Tunneling, Photon | 123 | 369 |

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ScenePhys.git
cd ScenePhys

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset

The dataset is available on Hugging Face:
```bash
# Download from Hugging Face
# Visit: https://huggingface.co/datasets/ScenePhys/ScenePhys/tree/main
# Download DataSet_lastversion.zip and extract to data/videos/
```

### 3. Configuration

1. Edit `config.py` and add your AvalAI API key:
```python
AVALAI_API_KEY = "your_api_key_here"
```

2. Ensure your dataset structure:
```
data/
├── metadata/          # CSV files with questions and answers
└── videos/           # MP4 files (download from Hugging Face)
```

### 4. Run Evaluation

```bash
# Run the complete evaluation pipeline
python src/batch_runner.py \
    --dataset_path data/metadata \
    --video_path data/videos \
    --max_workers 8 \
    --output results/batch_results.json

# Generate visualizations
python scripts/plot_standard_results.py
```

## 📁 Repository Structure

```
ScenePhys/
├── src/                    # Core evaluation modules
│   ├── robust_evaluation.py      # Evaluation metrics
│   ├── working_video_analysis.py # Video processing
│   ├── batch_runner.py           # Main pipeline
│   └── rerun_standard_judge.py   # LLM-as-a-judge
├── scripts/                # Utility scripts
│   └── plot_standard_results.py  # Visualization
├── data/                   # Dataset files
│   └── metadata/           # CSV files with Q/A pairs
├── results/                # Output files
│   └── batch_result_new.json
├── docs/                   # Documentation
├── config.py              # Configuration settings
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## 🔬 Evaluation Metrics

### Objective Metrics (Non-LLM)
- **Keypoint-F1**: Coverage of physics concepts for conceptual questions
- **Checklist-F1**: Identification of simulation limitations for error-detection
- **Mathematical Accuracy**: Numerical precision and unit handling

### Subjective Metrics (LLM-as-a-Judge)
- **Judge Score**: 1-5 scale rating with confidence
- **Dual Evaluation**: Two independent judge passes for reliability
- **Structured Output**: JSON format with detailed reasoning

## 🎯 Question Types

Each video has 3 question types:

1. **Conceptual**: Understanding physics principles and relationships
2. **Numerical**: Mathematical calculations and quantitative analysis
3. **Error-Detection**: Identifying limitations and assumptions in simulations

## 🔧 Configuration

Key settings in `config.py`:

```python
# Video Processing
DEFAULT_FPS = 3.0              # Frame extraction rate
DEFAULT_MAX_FRAMES = 40        # Maximum frames per video
DEFAULT_JPG_QUALITY = 95       # Image quality for VLM analysis

# Evaluation
DEFAULT_MAX_WORKERS = 8        # Parallel processing
JUDGE_MODEL = "gpt-4o-mini"    # LLM-as-a-judge model
```

## 📈 Results

The evaluation produces comprehensive results including:
- Per-model performance across all categories
- Detailed metrics for each question type
- Comparative analysis between VLMs
- Visualization plots and charts

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **PhET Interactive Simulations** for providing the physics simulation videos
- **AvalAI** for the unified VLM API access
- **OpenAI, Google, Alibaba** for the VLM models

## 📞 Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.

---

**Citation**: If you use this dataset in your research, please cite our paper (citation details to be added).
