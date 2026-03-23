# LLM Evaluation Framework

A comprehensive Python framework for evaluating multiple LLMs from OpenRouter and Cerebras APIs for RAG-based tutoring systems.

## 🎯 Purpose

This framework helps you compare model performance across different providers and models, measuring:
- **Factual Accuracy**: How correct is the information? (0-100 scale)
- **Clarity**: How easy to understand is the explanation? (0-100 scale)
- **Relevance**: How well does it address the prompt? (0-100 scale)
- **Latency**: How fast are the responses?

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r evaluation_requirements.txt
```

### 2. Set Up API Keys

Create a `.env` file with your API keys:

```bash
# OpenRouter API Key
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Cerebras API Key  
CEREBRAS_API_KEY=your_cerebras_api_key_here
```

### 3. Run Evaluation

```bash
python evaluate_models.py
```

## 📊 Models Tested

### Cerebras Models
- `qwen-3-235b-a22b-instruct-2507`
- `llama-3.3-70b`
- `llama-4-maverick-17b-128e-instruct`
- `llama-4-scout-17b-16e-instruct`

## 🔄 Evaluation Process

The framework now follows a more realistic workflow:

1. **Generate Lesson**: Create a lesson for each topic
2. **Generate Quiz**: Immediately create a quiz based on the lesson content
3. **Evaluate Both**: Score both lesson and quiz quality
4. **Repeat**: Continue for all models and topics

This ensures quizzes are directly tied to the lesson content, making the evaluation more realistic.

## 📁 Output Files

- `model_eval_raw.csv` - Raw responses and timing data
- `model_eval_scores.csv` - Scored results with evaluation metrics (0-100 scale)
- `model_eval_summary.csv` - Summary statistics per model
- `model_evaluation_results.png` - Performance visualizations
- `evaluation.log` - Detailed execution log

## 📈 Analysis

Run the analysis script to get detailed insights:

```bash
python analyze_results.py
```

This will generate:
- Model rankings and performance metrics
- Best models by category (overall, fastest, most accurate, etc.)
- Performance by prompt type (lessons vs quizzes)
- Consistency analysis
- Custom visualizations
- Recommendations

## 🔧 Key Features

- **Integrated Environment Check**: Automatically checks prerequisites
- **Realistic Quiz Generation**: Quizzes are based on actual lesson content
- **Granular Scoring**: 0-100 scale for more precise evaluation
- **Comprehensive Analysis**: Detailed performance breakdowns
- **Visual Reports**: Charts and graphs for easy interpretation

## 📊 Score Interpretation

- **90-100**: Excellent
- **80-89**: Good
- **70-79**: Average
- **60-69**: Below Average
- **0-59**: Poor

The framework uses a more critical evaluation approach to avoid inflated scores.