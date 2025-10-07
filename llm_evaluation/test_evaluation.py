#!/usr/bin/env python3
"""
Test script for the LLM evaluation framework

This script tests the evaluation framework with a minimal setup to ensure it works correctly.
"""

import os
import asyncio
import json

def check_environment():
    """Check if the environment is properly set up"""
    print("Checking Environment Setup")
    print("=" * 30)
    
    # Check API keys
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    cerebras_key = os.getenv("CEREBRAS_API_KEY")
    
    print(f"OpenRouter API Key: {'Found' if openrouter_key else 'Missing'}")
    print(f"Cerebras API Key: {'Found' if cerebras_key else 'Missing'}")
    
    if not openrouter_key and not cerebras_key:
        print("\nNo API keys found!")
        print("Please create a .env file with:")
        print("OPENROUTER_API_KEY=your_key_here")
        print("CEREBRAS_API_KEY=your_key_here")
        return False
    
    # Check required files
    required_files = [
        "evaluate_models.py",
        "evaluation_config.json",
        "evaluation_requirements.txt"
    ]
    
    print(f"\nChecking Required Files:")
    all_files_exist = True
    for file in required_files:
        exists = os.path.exists(file)
        print(f"  {file}: {'Found' if exists else 'Missing'}")
        if not exists:
            all_files_exist = False
    
    return all_files_exist

def check_dependencies():
    """Test if all required dependencies are installed"""
    print("\nTesting Dependencies")
    print("=" * 25)
    
    required_packages = [
        "aiohttp",
        "pandas", 
        "matplotlib",
        "seaborn"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  {package}: Installed")
        except ImportError:
            print(f"  {package}: Missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Install them with: pip install -r evaluation_requirements.txt")
        return False
    
    return True

def check_config():
    """Check if the configuration file is valid"""
    print("\nChecking Configuration")
    print("=" * 25)
    
    try:
        with open("evaluation_config.json", "r") as f:
            config = json.load(f)
        
        # Check structure
        if "evaluation_config" not in config:
            print("Invalid config structure: missing 'evaluation_config'")
            return False
        
        eval_config = config["evaluation_config"]
        
        # Check required sections
        required_sections = ["lesson_prompts", "models"]
        for section in required_sections:
            if section not in eval_config:
                print(f"Missing config section: {section}")
                return False
        
        # Check models
        models = eval_config["models"]
        if "openrouter" not in models and "cerebras" not in models:
            print("No model providers configured")
            return False
        
        # Count available models
        total_models = 0
        if "openrouter" in models:
            total_models += len(models["openrouter"])
            print(f"  OpenRouter models: {len(models['openrouter'])}")
        
        if "cerebras" in models:
            total_models += len(models["cerebras"])
            print(f"  Cerebras models: {len(models['cerebras'])}")
        
        print(f"  Total models configured: {total_models}")
        
        # Check lesson prompts
        lesson_prompts = eval_config["lesson_prompts"]
        print(f"  Lesson prompts: {len(lesson_prompts)}")
        
        print("✅ Configuration is valid")
        return True
        
    except FileNotFoundError:
        print("evaluation_config.json not found")
        return False
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in config file: {e}")
        return False
    except Exception as e:
        print(f"Error checking config: {e}")
        return False

def check_available_models():
    """Check which models are actually available based on API keys"""
    print("\n🤖 Checking Available Models")
    print("=" * 30)
    
    try:
        with open("evaluation_config.json", "r") as f:
            config = json.load(f)
        
        eval_config = config["evaluation_config"]
        models = eval_config["models"]
        
        available_count = 0
        
        # Check OpenRouter models
        if "openrouter" in models and os.getenv("OPENROUTER_API_KEY"):
            openrouter_models = models["openrouter"]
            print(f"OpenRouter models ({len(openrouter_models)}):")
            for model in openrouter_models:
                print(f"{model['name']} - {model['model_id']}")
                available_count += 1
        
        # Check Cerebras models
        if "cerebras" in models and os.getenv("CEREBRAS_API_KEY"):
            cerebras_models = models["cerebras"]
            print(f"\nCerebras models ({len(cerebras_models)}):")
            for model in cerebras_models:
                print(f"{model['name']} - {model['model_id']}")
                available_count += 1
        
        print(f"\nTotal available models: {available_count}")
        
        if available_count == 0:
            print("No models available! Check your API keys.")
            return False
        
        return True
        
    except Exception as e:
        print(f"Error checking models: {e}")
        return False

async def test_single_model():
    """Test a single model to ensure the framework works"""
    print("\nTesting Single Model")
    print("=" * 25)
    
    try:
        # Import the evaluator
        from evaluate_models import LLMEvaluator
        
        # Create evaluator
        evaluator = LLMEvaluator()
        
        if not evaluator.models:
            print("No models available for testing")
            return False
        
        # Test with the first available model
        test_model = evaluator.models[0]
        test_prompt = "Explain photosynthesis to a 12-year-old."
        
        print(f"Testing {test_model.name} ({test_model.provider})")
        print(f"Prompt: {test_prompt[:50]}...")
        
        # Generate response
        response, latency = await evaluator.generate_response(test_model, test_prompt)
        
        print(f"   Response generated successfully!")
        print(f"   Latency: {latency:.2f} seconds")
        print(f"   Response length: {len(response)} characters")
        print(f"   Response preview: {response[:100]}...")
        
        # Test quiz generation
        print(f"\n Testing Quiz Generation")
        quiz_response, quiz_latency = await evaluator.generate_quiz_from_lesson(
            test_model, response, "photosynthesis"
        )
        
        print(f"   Quiz generated successfully!")
        print(f"   Latency: {quiz_latency:.2f} seconds")
        print(f"   Quiz length: {len(quiz_response)} characters")
        print(f"   Quiz preview: {quiz_response[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_evaluation_scoring():
    """Test the evaluation scoring system"""
    print("\nTesting Evaluation Scoring")
    print("=" * 30)
    
    try:
        from evaluate_models import LLMEvaluator
        
        evaluator = LLMEvaluator()
        
        if not evaluator.models:
            print("No models available for testing")
            return False
        
        # Use first model as evaluator
        evaluator_model = evaluator.models[0]
        test_response = "Photosynthesis is the process by which plants convert sunlight into energy."
        test_prompt = "Explain photosynthesis to a 12-year-old."
        
        print(f"Testing evaluation with {evaluator_model.name}")
        print(f"Response to evaluate: {test_response}")
        
        # Test evaluation
        scores = await evaluator.evaluate_response(test_response, test_prompt, evaluator_model)
        
        print(f"   Evaluation completed!")
        print(f"   Factual Accuracy: {scores.get('factual_accuracy', 'N/A')}/100")
        print(f"   Clarity: {scores.get('clarity', 'N/A')}/100")
        print(f"   Relevance: {scores.get('relevance', 'N/A')}/100")
        
        # Check if scores are in valid range
        for key, score in scores.items():
            if not (0 <= score <= 100):
                print(f"Invalid score for {key}: {score} (should be 0-100)")
                return False
        
        print("All scores are in valid range (0-100)")
        return True
        
    except Exception as e:
        print(f"Evaluation test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function"""
    print("LLM Evaluation Framework Test")
    print("=" * 35)
    
    # Check environment
    if not check_environment():
        print("\nEnvironment check failed!")
        return
    
    # Check dependencies
    if not check_dependencies():
        print("\nDependency check failed!")
        return
    
    # Check configuration
    if not check_config():
        print("\nConfiguration check failed!")
        return
    
    # Check available models
    if not check_available_models():
        print("\nModel availability check failed!")
        return
    
    # Test single model
    if not await test_single_model():
        print("\nModel test failed!")
        return
    
    # Test evaluation scoring
    if not await test_evaluation_scoring():
        print("\nEvaluation scoring test failed!")
        return
    
    print("\n✅ All tests passed!")
    print("\nThe evaluation framework is ready to use!")
    print("\nNext steps:")
    print("1. Run: python evaluate_models.py")
    print("2. Analyze results: python analyze_results.py")
    print("3. Check generated CSV files and visualizations")
    print("\nExpected output files:")
    print("  • model_eval_raw.csv")
    print("  • model_eval_scores.csv")
    print("  • model_eval_summary.csv")
    print("  • model_evaluation_results.png")
    print("  • evaluation.log")

if __name__ == "__main__":
    asyncio.run(main())