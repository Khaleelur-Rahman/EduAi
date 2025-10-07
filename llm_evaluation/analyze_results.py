#!/usr/bin/env python3
"""
Example analysis script for LLM evaluation results

This script demonstrates how to analyze the results from the evaluation framework.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List

def load_evaluation_results() -> pd.DataFrame:
    """Load the evaluation results from CSV"""
    try:
        df = pd.read_csv('model_eval_scores.csv')
        print(f"✅ Loaded {len(df)} evaluation results")
        return df
    except FileNotFoundError:
        print("❌ model_eval_scores.csv not found. Please run evaluate_models.py first.")
        return None

def analyze_model_performance(df: pd.DataFrame) -> Dict:
    """Analyze overall model performance"""
    print("\n📊 Model Performance Analysis")
    print("=" * 50)
    
    # Calculate summary statistics
    summary = df.groupby('model_name').agg({
        'overall_score': ['mean', 'std', 'count'],
        'latency_seconds': ['mean', 'std'],
        'factual_accuracy': 'mean',
        'clarity': 'mean',
        'relevance': 'mean'
    }).round(1)
    
    # Flatten column names
    summary.columns = ['_'.join(col).strip() for col in summary.columns]
    summary = summary.reset_index()
    
    # Sort by overall score
    summary = summary.sort_values('overall_score_mean', ascending=False)
    
    print("\n🏆 Model Rankings (by Overall Score - 0-100 scale):")
    for i, row in summary.iterrows():
        print(f"{i+1:2d}. {row['model_name']:30s} | Score: {row['overall_score_mean']:.1f} | Latency: {row['latency_seconds_mean']:.2f}s")
    
    return summary

def find_best_models(df: pd.DataFrame) -> Dict[str, str]:
    """Find the best model for different criteria"""
    print("\n🎯 Best Models by Category")
    print("=" * 30)
    
    best_models = {}
    
    # Best overall score
    best_overall = df.groupby('model_name')['overall_score'].mean().idxmax()
    best_models['overall'] = best_overall
    print(f"🥇 Best Overall: {best_overall}")
    
    # Fastest model
    fastest = df.groupby('model_name')['latency_seconds'].mean().idxmin()
    best_models['fastest'] = fastest
    print(f"⚡ Fastest: {fastest}")
    
    # Most accurate
    most_accurate = df.groupby('model_name')['factual_accuracy'].mean().idxmax()
    best_models['most_accurate'] = most_accurate
    print(f"🎯 Most Accurate: {most_accurate}")
    
    # Clearest explanations
    clearest = df.groupby('model_name')['clarity'].mean().idxmax()
    best_models['clearest'] = clearest
    print(f"💡 Clearest: {clearest}")
    
    # Most relevant
    most_relevant = df.groupby('model_name')['relevance'].mean().idxmax()
    best_models['most_relevant'] = most_relevant
    print(f"🎪 Most Relevant: {most_relevant}")
    
    # Most efficient (score per second)
    df['efficiency'] = df['overall_score'] / df['latency_seconds']
    most_efficient = df.groupby('model_name')['efficiency'].mean().idxmax()
    best_models['most_efficient'] = most_efficient
    print(f"⚖️ Most Efficient: {most_efficient}")
    
    return best_models

def analyze_by_prompt_type(df: pd.DataFrame):
    """Analyze performance by prompt type (lesson vs quiz)"""
    print("\n📚 Performance by Prompt Type")
    print("=" * 35)
    
    by_type = df.groupby(['model_name', 'prompt_type']).agg({
        'overall_score': 'mean',
        'latency_seconds': 'mean'
    }).round(1)
    
    # Pivot for easier comparison
    score_pivot = by_type['overall_score'].unstack()
    latency_pivot = by_type['latency_seconds'].unstack()
    
    print("\n📖 Lesson Generation Scores (0-100):")
    lesson_scores = score_pivot['lesson'].sort_values(ascending=False)
    for model, score in lesson_scores.items():
        print(f"  {model:30s} | {score:.1f}")
    
    print("\n🧩 Quiz Generation Scores (0-100):")
    quiz_scores = score_pivot['quiz'].sort_values(ascending=False)
    for model, score in quiz_scores.items():
        print(f"  {model:30s} | {score:.1f}")
    
    # Find best for each type
    best_lesson = lesson_scores.idxmax()
    best_quiz = quiz_scores.idxmax()
    
    print(f"\n🏆 Best for Lessons: {best_lesson}")
    print(f"🏆 Best for Quizzes: {best_quiz}")

def analyze_consistency(df: pd.DataFrame):
    """Analyze model consistency (lower std = more consistent)"""
    print("\n📈 Model Consistency Analysis")
    print("=" * 35)
    
    consistency = df.groupby('model_name')['overall_score'].agg(['mean', 'std']).round(1)
    consistency['cv'] = (consistency['std'] / consistency['mean'] * 100).round(1)  # Coefficient of variation
    
    # Sort by consistency (lower CV = more consistent)
    consistency = consistency.sort_values('cv')
    
    print("Model Consistency (lower CV = more consistent):")
    for model, row in consistency.iterrows():
        print(f"  {model:30s} | Mean: {row['mean']:.1f} | Std: {row['std']:.1f} | CV: {row['cv']:.1f}%")

def create_custom_visualizations(df: pd.DataFrame):
    """Create custom visualizations for analysis"""
    print("\n📊 Creating custom visualizations...")
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Custom LLM Analysis (0-100 Scale)', fontsize=16, fontweight='bold')
    
    # 1. Score distribution by model
    ax1 = axes[0, 0]
    df.boxplot(column='overall_score', by='model_name', ax=ax1)
    ax1.set_title('Score Distribution by Model')
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Overall Score (0-100)')
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Latency distribution by model
    ax2 = axes[0, 1]
    df.boxplot(column='latency_seconds', by='model_name', ax=ax2)
    ax2.set_title('Latency Distribution by Model')
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Latency (seconds)')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Score components heatmap
    ax3 = axes[1, 0]
    score_components = df.groupby('model_name')[['factual_accuracy', 'clarity', 'relevance']].mean()
    sns.heatmap(score_components.T, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax3)
    ax3.set_title('Score Components Heatmap (0-100)')
    
    # 4. Efficiency scatter plot
    ax4 = axes[1, 1]
    efficiency_data = df.groupby('model_name').agg({
        'overall_score': 'mean',
        'latency_seconds': 'mean'
    })
    efficiency_data['efficiency'] = efficiency_data['overall_score'] / efficiency_data['latency_seconds']
    
    scatter = ax4.scatter(efficiency_data['latency_seconds'], 
                         efficiency_data['overall_score'],
                         s=efficiency_data['efficiency'] * 2,
                         alpha=0.7, c=efficiency_data['efficiency'], 
                         cmap='viridis')
    
    # Add model labels
    for model in efficiency_data.index:
        ax4.annotate(model, 
                    (efficiency_data.loc[model, 'latency_seconds'], 
                     efficiency_data.loc[model, 'overall_score']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, alpha=0.8)
    
    ax4.set_title('Performance vs Speed (bubble size = efficiency)')
    ax4.set_xlabel('Average Latency (seconds)')
    ax4.set_ylabel('Average Overall Score (0-100)')
    ax4.grid(True, alpha=0.3)
    
    # Add colorbar
    plt.colorbar(scatter, ax=ax4, label='Efficiency Score')
    
    plt.tight_layout()
    plt.savefig('custom_analysis_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Custom visualizations saved to custom_analysis_results.png")

def generate_recommendations(df: pd.DataFrame, best_models: Dict[str, str]) -> List[str]:
    """Generate recommendations based on analysis"""
    print("\n💡 Recommendations")
    print("=" * 20)
    
    recommendations = []
    
    # Overall recommendation
    best_overall = best_models['overall']
    fastest = best_models['fastest']
    
    recommendations.append(f"🏆 For best overall quality: Use {best_overall}")
    recommendations.append(f"⚡ For fastest responses: Use {fastest}")
    
    # Efficiency recommendation
    most_efficient = best_models['most_efficient']
    recommendations.append(f"⚖️ For best efficiency (quality/speed): Use {most_efficient}")
    
    # Task-specific recommendations
    lesson_performance = df[df['prompt_type'] == 'lesson'].groupby('model_name')['overall_score'].mean()
    quiz_performance = df[df['prompt_type'] == 'quiz'].groupby('model_name')['overall_score'].mean()
    
    best_lesson = lesson_performance.idxmax()
    best_quiz = quiz_performance.idxmax()
    
    recommendations.append(f"📚 For lesson generation: Use {best_lesson}")
    recommendations.append(f"🧩 For quiz generation: Use {best_quiz}")
    
    # Consistency recommendation
    consistency = df.groupby('model_name')['overall_score'].std().sort_values()
    most_consistent = consistency.index[0]
    recommendations.append(f"📈 For most consistent results: Use {most_consistent}")
    
    # Print recommendations
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
    
    return recommendations

def main():
    """Main analysis function"""
    print("🔍 LLM Evaluation Results Analysis")
    print("=" * 40)
    
    # Load data
    df = load_evaluation_results()
    if df is None:
        return
    
    # Run analyses
    summary = analyze_model_performance(df)
    best_models = find_best_models(df)
    analyze_by_prompt_type(df)
    analyze_consistency(df)
    
    # Create visualizations
    create_custom_visualizations(df)
    
    # Generate recommendations
    recommendations = generate_recommendations(df, best_models)
    
    print(f"\n✅ Analysis complete! Check 'custom_analysis_results.png' for visualizations.")
    print(f"📊 Analyzed {len(df)} test results across {df['model_name'].nunique()} models.")

if __name__ == "__main__":
    main()