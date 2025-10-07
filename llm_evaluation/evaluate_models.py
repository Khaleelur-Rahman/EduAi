#!/usr/bin/env python3
"""
LLM Evaluation Framework for RAG-based Tutoring System

This script evaluates multiple LLMs from OpenRouter and Cerebras APIs
for lesson and quiz generation, measuring performance across various metrics.
"""

import os
import json
import csv
import time
import logging
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import asyncio
import aiohttp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Try to import dotenv, but don't fail if it's missing
try:
    from dotenv import load_dotenv
    load_dotenv()
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    print("⚠️ python-dotenv not available, will check environment variables directly")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for each model to test"""
    name: str
    provider: str
    api_key_env: str
    base_url: str
    model_id: str
    max_tokens: int = 1000
    temperature: float = 0.7

@dataclass
class EvaluationResult:
    """Result of a single model evaluation"""
    model_name: str
    prompt_type: str
    prompt_text: str
    response_text: str
    latency_seconds: float
    timestamp: str
    factual_accuracy: int = None
    clarity: int = None
    relevance: int = None
    overall_score: float = None

class LLMEvaluator:
    """Main evaluation class for testing multiple LLMs"""
    
    def __init__(self):
        self.results: List[EvaluationResult] = []
        self.models = self._setup_models()
        self.lesson_prompts = [
            "Explain photosynthesis to a 12-year-old.",
            "Describe the process of mitosis.",
            "What are alkaline elements?",
            "Explain how acids and bases react together.",
            "Give a summary of chemical bonding in simple terms."
        ]
        
    def check_environment(self) -> bool:
        """Check if the environment is properly set up"""
        print("🔍 Checking Environment Setup")
        print("=" * 30)
        
        # Check API keys
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        cerebras_key = os.getenv("CEREBRAS_API_KEY")
        
        print(f"OpenRouter API Key: {'✅ Found' if openrouter_key else '❌ Missing'}")
        print(f"Cerebras API Key: {'✅ Found' if cerebras_key else '❌ Missing'}")
        
        if not openrouter_key and not cerebras_key:
            print("\n❌ No API keys found!")
            print("Please set environment variables:")
            print("export OPENROUTER_API_KEY=your_key_here")
            print("export CEREBRAS_API_KEY=your_key_here")
            return False
        
        # Check required dependencies
        required_packages = ["aiohttp", "pandas", "matplotlib", "seaborn"]
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
            print("Install them with: pip install -r evaluation_requirements.txt")
            return False
        
        print("✅ Environment check passed!")
        return True
        
    def _setup_models(self) -> List[ModelConfig]:
        """Setup model configurations"""
        evaluator_model_deepseek = ModelConfig(
            name="deepseek-chat-v3",
            provider="openrouter",
            api_key_env="OPENRPUTER_API_KEY",
            base_url="https://openrouter.ai/api/v1",
            model_id="deepseek/deepseek-chat-v3.1:free",
            max_tokens=1000
        )
        models = [
            # Cerebras Models
            ModelConfig(
                name="qwen-3-235b-a22b-instruct",
                provider="cerebras",
                api_key_env="CEREBRAS_API_KEY",
                base_url="https://api.cerebras.ai/v1",
                model_id="qwen-3-235b-a22b-instruct-2507",
                max_tokens=1000
            ),
            ModelConfig(
                name="llama-3.3-70b",
                provider="cerebras",
                api_key_env="CEREBRAS_API_KEY",
                base_url="https://api.cerebras.ai/v1",
                model_id="llama-3.3-70b",
                max_tokens=1000
            ),
            ModelConfig(
                name="llama-4-maverick-17b",
                provider="cerebras",
                api_key_env="CEREBRAS_API_KEY",
                base_url="https://api.cerebras.ai/v1",
                model_id="llama-4-maverick-17b-128e-instruct",
                max_tokens=1000
            ),
            ModelConfig(
                name="llama-4-scout-17b",
                provider="cerebras",
                api_key_env="CEREBRAS_API_KEY",
                base_url="https://api.cerebras.ai/v1",
                model_id="llama-4-scout-17b-16e-instruct",
                max_tokens=1000
            )
        ]
        
        # Filter out models without API keys
        available_models = []
        for model in models:
            api_key = os.getenv(model.api_key_env)
            if api_key:
                available_models.append(model)
                logger.info(f"✅ {model.name} ({model.provider}) - API key found")
            else:
                logger.warning(f"❌ {model.name} ({model.provider}) - No API key found")
        
        return available_models
    
    async def generate_response(self, model: ModelConfig, prompt: str) -> Tuple[str, float]:
        """
        Generate response from a model with timing
        
        Args:
            model: Model configuration
            prompt: Input prompt
            
        Returns:
            Tuple of (response_text, latency_seconds)
        """
        api_key = os.getenv(model.api_key_env)
        if not api_key:
            raise ValueError(f"No API key found for {model.name}")
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Prepare request payload
        payload = {
            "model": model.model_id,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": model.max_tokens,
            "temperature": model.temperature
        }
        
        # Add streaming for Cerebras
        if model.provider == "cerebras":
            payload["stream"] = True
        
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{model.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"API error {response.status}: {error_text}")
                    
                    if model.provider == "cerebras" and payload.get("stream"):
                        # Handle streaming response for Cerebras
                        response_text = ""
                        async for line in response.content:
                            line = line.decode('utf-8').strip()
                            if line.startswith('data: '):
                                data = line[6:]  # Remove 'data: ' prefix
                                if data == '[DONE]':
                                    break
                                try:
                                    chunk = json.loads(data)
                                    if 'choices' in chunk and len(chunk['choices']) > 0:
                                        delta = chunk['choices'][0].get('delta', {})
                                        if 'content' in delta:
                                            response_text += delta['content']
                                except json.JSONDecodeError:
                                    continue
                    else:
                        # Handle non-streaming response
                        response_data = await response.json()
                        response_text = response_data['choices'][0]['message']['content']
                    
                    latency = time.time() - start_time
                    return response_text.strip(), latency
                    
        except Exception as e:
            logger.error(f"Error generating response from {model.name}: {str(e)}")
            return f"Error: {str(e)}", time.time() - start_time
    
    async def generate_quiz_from_lesson(self, model: ModelConfig, lesson_content: str, topic: str) -> Tuple[str, float]:
        """
        Generate a quiz based on the lesson content
        
        Args:
            model: Model configuration
            lesson_content: The lesson content to create quiz from
            topic: The topic of the lesson
            
        Returns:
            Tuple of (quiz_text, latency_seconds)
        """
        quiz_prompt = f"""
Based on the following lesson content about {topic}, create a 3-question multiple-choice quiz.

Lesson Content:
{lesson_content}

Please create exactly 3 multiple-choice questions with 4 options each (A, B, C, D).
Format your response as:

Q1: [Question text]
A) [Option A]
B) [Option B] 
C) [Option C]
D) [Option D]
Answer: [Correct letter]

Q2: [Question text]
A) [Option A]
B) [Option B]
C) [Option C] 
D) [Option D]
Answer: [Correct letter]

Q3: [Question text]
A) [Option A]
B) [Option B]
C) [Option C]
D) [Option D]
Answer: [Correct letter]

Make sure the questions test understanding of the key concepts from the lesson content.
"""
        
        return await self.generate_response(model, quiz_prompt)
    
    async def evaluate_response(self, response: str, prompt: str, evaluator_model: ModelConfig) -> Dict[str, int]:
        """
        Evaluate response quality using another LLM with 0-100 scale
        
        Args:
            response: Model response to evaluate
            prompt: Original prompt
            evaluator_model: Model to use for evaluation
            
        Returns:
            Dictionary with evaluation scores (0-100)
        """
        evaluation_prompt = f"""
You are an expert educational content evaluator. Please evaluate the following model response for a tutoring system.

Original Prompt: "{prompt}"
Model Response: "{response}"

Evaluate the following criteria (each scored 0–100):

1. Factual Accuracy — Are the statements scientifically correct and appropriate for the educational level?
2. Clarity — Are the explanations simple, coherent, and easy for a young learner to follow?
3. Relevance — Does the response stay focused on the topic and fulfill the learning intent?

Be critical and use the full range of the scale. Avoid inflated scores — average content should score around 60–75.

Return ONLY a JSON object with these exact keys:
{{
    "factual_accuracy": <integer>,
    "clarity": <integer>,
    "relevance": <integer>
}}
"""
        
        try:
            eval_response, _ = await self.generate_response(evaluator_model, evaluation_prompt)
            
            # Extract JSON from response
            eval_response = eval_response.strip()
            if eval_response.startswith('```json'):
                eval_response = eval_response[7:-3]
            elif eval_response.startswith('```'):
                eval_response = eval_response[3:-3]
            
            scores = json.loads(eval_response)
            
            # Ensure scores are within 0-100 range
            for key in scores:
                scores[key] = max(0, min(100, int(scores[key])))
            
            return scores
            
        except Exception as e:
            logger.error(f"Error evaluating response: {str(e)}")
            return {"factual_accuracy": 50, "clarity": 50, "relevance": 50}
    
    async def run_evaluation(self):
        """Run the complete evaluation process"""
        logger.info("🚀 Starting LLM Evaluation Framework")
        logger.info(f"Testing {len(self.models)} models")
        
        evaluator_model = self.evaluator_model_deepseek
        if not evaluator_model:
            logger.error("No models available for evaluation!")
            return
        
        logger.info(f"Using {evaluator_model.name} as evaluator model")
        
        total_tests = len(self.models) * len(self.lesson_prompts) * 2  # lesson + quiz for each
        current_test = 0
        
        for model in self.models:
            logger.info(f"\n📊 Testing {model.name} ({model.provider})")
            
            # Test each lesson prompt
            for i, prompt in enumerate(self.lesson_prompts):
                current_test += 1
                logger.info(f"  [{current_test}/{total_tests}] Lesson {i+1}: {prompt[:50]}...")
                
                try:
                    # Generate lesson
                    lesson_response, lesson_latency = await self.generate_response(model, prompt)
                    
                    # Evaluate lesson quality
                    lesson_scores = await self.evaluate_response(lesson_response, prompt, evaluator_model)
                    
                    lesson_result = EvaluationResult(
                        model_name=model.name,
                        prompt_type="lesson",
                        prompt_text=prompt,
                        response_text=lesson_response,
                        latency_seconds=lesson_latency,
                        timestamp=datetime.now().isoformat(),
                        factual_accuracy=lesson_scores.get("factual_accuracy", 0),
                        clarity=lesson_scores.get("clarity", 0),
                        relevance=lesson_scores.get("relevance", 0),
                        overall_score=(lesson_scores.get("factual_accuracy", 0) + 
                                     lesson_scores.get("clarity", 0) + 
                                     lesson_scores.get("relevance", 0)) / 3
                    )
                    
                    self.results.append(lesson_result)
                    logger.info(f"    ✅ Lesson completed (latency: {lesson_latency:.2f}s, score: {lesson_result.overall_score:.1f})")
                    
                    # Generate quiz based on the lesson content
                    current_test += 1
                    logger.info(f"  [{current_test}/{total_tests}] Quiz {i+1}: Based on lesson content...")
                    
                    quiz_response, quiz_latency = await self.generate_quiz_from_lesson(model, lesson_response, prompt)
                    
                    # Evaluate quiz quality
                    quiz_scores = await self.evaluate_response(quiz_response, f"Generate a quiz about: {prompt}", evaluator_model)
                    
                    quiz_result = EvaluationResult(
                        model_name=model.name,
                        prompt_type="quiz",
                        prompt_text=f"Generate a quiz about: {prompt}",
                        response_text=quiz_response,
                        latency_seconds=quiz_latency,
                        timestamp=datetime.now().isoformat(),
                        factual_accuracy=quiz_scores.get("factual_accuracy", 0),
                        clarity=quiz_scores.get("clarity", 0),
                        relevance=quiz_scores.get("relevance", 0),
                        overall_score=(quiz_scores.get("factual_accuracy", 0) + 
                                     quiz_scores.get("clarity", 0) + 
                                     quiz_scores.get("relevance", 0)) / 3
                    )
                    
                    self.results.append(quiz_result)
                    logger.info(f"    ✅ Quiz completed (latency: {quiz_latency:.2f}s, score: {quiz_result.overall_score:.1f})")
                    
                except Exception as e:
                    logger.error(f"    ❌ Failed: {str(e)}")
        
        logger.info(f"\n🎉 Evaluation completed! {len(self.results)} tests completed")
    
    def save_results(self):
        """Save results to CSV files"""
        logger.info("💾 Saving results to CSV files...")
        
        # Save raw results
        raw_data = []
        for result in self.results:
            raw_data.append({
                'model_name': result.model_name,
                'prompt_type': result.prompt_type,
                'prompt_text': result.prompt_text,
                'response_text': result.response_text,
                'latency_seconds': result.latency_seconds,
                'timestamp': result.timestamp
            })
        
        df_raw = pd.DataFrame(raw_data)
        df_raw.to_csv('model_eval_raw.csv', index=False)
        logger.info("✅ Raw results saved to model_eval_raw.csv")
        
        # Save scored results
        scored_data = []
        for result in self.results:
            scored_data.append({
                'model_name': result.model_name,
                'prompt_type': result.prompt_type,
                'prompt_text': result.prompt_text,
                'response_text': result.response_text,
                'latency_seconds': result.latency_seconds,
                'factual_accuracy': result.factual_accuracy,
                'clarity': result.clarity,
                'relevance': result.relevance,
                'overall_score': result.overall_score,
                'timestamp': result.timestamp
            })
        
        df_scored = pd.DataFrame(scored_data)
        df_scored.to_csv('model_eval_scores.csv', index=False)
        logger.info("✅ Scored results saved to model_eval_scores.csv")
        
        # Generate summary statistics
        self.generate_summary_report(df_scored)
    
    def generate_summary_report(self, df: pd.DataFrame):
        """Generate summary statistics and visualizations"""
        logger.info("📊 Generating summary report...")
        
        # Calculate summary statistics
        summary = df.groupby('model_name').agg({
            'overall_score': ['mean', 'std', 'min', 'max'],
            'latency_seconds': ['mean', 'std', 'min', 'max'],
            'factual_accuracy': 'mean',
            'clarity': 'mean',
            'relevance': 'mean'
        }).round(1)
        
        # Flatten column names
        summary.columns = ['_'.join(col).strip() for col in summary.columns]
        summary = summary.reset_index()
        
        # Save summary
        summary.to_csv('model_eval_summary.csv', index=False)
        logger.info("✅ Summary statistics saved to model_eval_summary.csv")
        
        # Create visualizations
        self.create_visualizations(df)
    
    def create_visualizations(self, df: pd.DataFrame):
        """Create performance visualizations"""
        logger.info("📈 Creating visualizations...")
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('LLM Performance Evaluation Results (0-100 Scale)', fontsize=16, fontweight='bold')
        
        # 1. Overall Score by Model
        ax1 = axes[0, 0]
        score_by_model = df.groupby('model_name')['overall_score'].mean().sort_values(ascending=True)
        score_by_model.plot(kind='barh', ax=ax1, color='skyblue')
        ax1.set_title('Average Overall Score by Model')
        ax1.set_xlabel('Overall Score (0-100)')
        ax1.grid(True, alpha=0.3)
        
        # 2. Latency by Model
        ax2 = axes[0, 1]
        latency_by_model = df.groupby('model_name')['latency_seconds'].mean().sort_values(ascending=True)
        latency_by_model.plot(kind='barh', ax=ax2, color='lightcoral')
        ax2.set_title('Average Response Latency by Model')
        ax2.set_xlabel('Latency (seconds)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Score Components Comparison
        ax3 = axes[1, 0]
        score_components = df.groupby('model_name')[['factual_accuracy', 'clarity', 'relevance']].mean()
        score_components.plot(kind='bar', ax=ax3, width=0.8)
        ax3.set_title('Score Components by Model')
        ax3.set_ylabel('Score (0-100)')
        ax3.set_xlabel('Model')
        ax3.legend(title='Component')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # 4. Score vs Latency Scatter Plot
        ax4 = axes[1, 1]
        scatter_data = df.groupby('model_name').agg({
            'overall_score': 'mean',
            'latency_seconds': 'mean'
        })
        ax4.scatter(scatter_data['latency_seconds'], scatter_data['overall_score'], 
                   s=100, alpha=0.7, c='green')
        
        # Add model labels
        for i, model in enumerate(scatter_data.index):
            ax4.annotate(model, 
                        (scatter_data.iloc[i]['latency_seconds'], 
                         scatter_data.iloc[i]['overall_score']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.8)
        
        ax4.set_title('Performance vs Speed Trade-off')
        ax4.set_xlabel('Average Latency (seconds)')
        ax4.set_ylabel('Average Overall Score (0-100)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('model_evaluation_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("✅ Visualizations saved to model_evaluation_results.png")

async def main():
    """Main execution function"""
    evaluator = LLMEvaluator()
    
    # Check environment first
    if not evaluator.check_environment():
        logger.error("❌ Environment check failed!")
        return
    
    if not evaluator.models:
        logger.error("❌ No models available! Please check your API keys.")
        logger.info("Required environment variables:")
        logger.info("  - OPENROUTER_API_KEY (for OpenRouter models)")
        logger.info("  - CEREBRAS_API_KEY (for Cerebras models)")
        return
    
    try:
        await evaluator.run_evaluation()
        evaluator.save_results()
        
        logger.info("\n🎯 Evaluation Summary:")
        logger.info(f"  - Models tested: {len(evaluator.models)}")
        logger.info(f"  - Total tests: {len(evaluator.results)}")
        logger.info(f"  - Files generated:")
        logger.info(f"    • model_eval_raw.csv")
        logger.info(f"    • model_eval_scores.csv")
        logger.info(f"    • model_eval_summary.csv")
        logger.info(f"    • model_evaluation_results.png")
        logger.info(f"    • evaluation.log")
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Run the evaluation
    asyncio.run(main())