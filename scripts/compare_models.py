import os
import json
import argparse
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
from tabulate import tabulate


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Compare trained cryptocurrency price prediction models')

    parser.add_argument('--models_dir', type=str, default='models',
                        help='Directory containing model folders')

    parser.add_argument('--output_dir', type=str, default='models/comparison',
                        help='Directory to save comparison results')

    parser.add_argument('--metric', type=str, default='f1_score',
                        choices=['accuracy', 'precision', 'recall', 'f1_score'],
                        help='Metric to use for comparing models')

    parser.add_argument('--sort_by', type=str, default='test_score',
                        choices=['name', 'test_score', 'train_score', 'train_test_gap', 'timestamp'],
                        help='How to sort models in the comparison')

    parser.add_argument('--max_models', type=int, default=20,
                        help='Maximum number of models to include in comparison')

    return parser.parse_args()


def find_model_dirs(models_dir):
    """Find all model directories in the base directory."""
    # Find all results.json files
    results_files = glob.glob(os.path.join(models_dir, '**/results.json'), recursive=True)
    
    # Extract the directories containing these files
    model_dirs = [os.path.dirname(file) for file in results_files]
    
    return model_dirs


def load_model_results(model_dir):
    """Load results data from a model directory."""
    results_path = os.path.join(model_dir, 'results.json')
    
    try:
        with open(results_path, 'r') as f:
            results = json.load(f)
            
        # Add model directory name
        results['model_dir'] = os.path.basename(model_dir)
        
        return results
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading results from {model_dir}: {e}")
        return None


def extract_model_metadata(results):
    """Extract key metadata from model results for comparison."""
    if not results:
        return None
    
    try:
        args = results['args']
        
        metadata = {
            'model_dir': results['model_dir'],
            'timestamp': results.get('timestamp', 'Unknown'),
            'model_type': args.get('model_type', 'Unknown'),
            'days_ahead': args.get('days_ahead', 0),
            'threshold': args.get('threshold', 0),
            'feature_set': args.get('feature_set', 'Unknown'),
            'selection_method': args.get('selection', 'Unknown'),
            'train_size': len(results.get('train_coins', [])),
            'test_size': len(results.get('test_coins', [])),
            'runtime_seconds': results.get('runtime_seconds', 0),
            'train_metrics': results.get('train_metrics', {}),
            'test_metrics': results.get('test_metrics', {})
        }
        
        return metadata
    except KeyError as e:
        print(f"Error extracting metadata: {e}")
        return None


def create_comparison_dataframe(model_metadata_list, metric):
    """Create a pandas DataFrame for model comparison."""
    rows = []
    
    for metadata in model_metadata_list:
        if not metadata:
            continue
            
        # Get train and test scores for the specified metric
        train_score = metadata['train_metrics'].get(metric, 0)
        test_score = metadata['test_metrics'].get(metric, 0)
        
        # Calculate the gap between train and test scores
        train_test_gap = train_score - test_score
        
        # Parse timestamp
        try:
            timestamp = datetime.strptime(metadata['timestamp'], '%Y-%m-%d %H:%M:%S')
        except:
            timestamp = datetime.now()  # Fallback if parsing fails
        
        row = {
            'Name': metadata['model_dir'],
            'Type': metadata['model_type'],
            'Days Ahead': metadata['days_ahead'],
            'Threshold': f"{metadata['threshold']:.2f}",
            'Features': metadata['feature_set'],
            'Selection': metadata['selection_method'],
            'Train Size': metadata['train_size'],
            'Test Size': metadata['test_size'],
            'Runtime (s)': int(metadata['runtime_seconds']),
            f'Train {metric.capitalize()}': train_score,
            f'Test {metric.capitalize()}': test_score,
            'Train-Test Gap': train_test_gap,
            'Timestamp': timestamp
        }
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def plot_comparison(df, metric, output_dir, sort_by):
    """Create visualizations for model comparison."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine sort column
    if sort_by == 'name':
        df = df.sort_values('Name')
    elif sort_by == 'test_score':
        df = df.sort_values(f'Test {metric.capitalize()}', ascending=False)
    elif sort_by == 'train_score':
        df = df.sort_values(f'Train {metric.capitalize()}', ascending=False)
    elif sort_by == 'train_test_gap':
        df = df.sort_values('Train-Test Gap')
    elif sort_by == 'timestamp':
        df = df.sort_values('Timestamp', ascending=False)
        
    # Create a horizontal bar chart comparing models
    plt.figure(figsize=(12, max(8, len(df)*0.4)))
    
    # Plot test scores
    test_bars = plt.barh(df['Name'], df[f'Test {metric.capitalize()}'], color='seagreen', alpha=0.7, label='Test')
    
    # Plot train scores
    train_bars = plt.barh(df['Name'], df[f'Train {metric.capitalize()}'], 
                          color='navy', alpha=0.4, label='Train')
    
    plt.xlabel(f'{metric.capitalize()}')
    plt.title(f'Model Comparison by {metric.capitalize()}')
    plt.legend()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, f'model_comparison_{metric}.png'))
    plt.close()
    
    # Create a scatter plot of train vs test scores
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(df[f'Train {metric.capitalize()}'], 
                         df[f'Test {metric.capitalize()}'],
                         c=df['Train-Test Gap'],
                         cmap='coolwarm_r',  # Red = high gap (potential overfitting)
                         s=100,
                         alpha=0.7)
    
    # Add model names as annotations
    for i, row in df.iterrows():
        plt.annotate(row['Name'], 
                    (row[f'Train {metric.capitalize()}'], row[f'Test {metric.capitalize()}']),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8)
    
    # Add a colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Train-Test Gap')
    
    # Add identity line (where train = test)
    max_val = max(df[f'Train {metric.capitalize()}'].max(), df[f'Test {metric.capitalize()}'].max())
    min_val = min(df[f'Train {metric.capitalize()}'].min(), df[f'Test {metric.capitalize()}'].min())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3)
    
    plt.xlabel(f'Train {metric.capitalize()}')
    plt.ylabel(f'Test {metric.capitalize()}')
    plt.title(f'Train vs Test {metric.capitalize()}')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, f'train_test_scatter_{metric}.png'))
    plt.close()
    
    # Create a feature importance comparison if multiple random forest models exist
    rf_df = df[df['Type'] == 'random_forest']
    if len(rf_df) >= 2:
        pass  # Could add feature importance comparison in future enhancement
    
    # Save the summary to a CSV file
    df.to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)
    
    # Create a summary text file
    with open(os.path.join(output_dir, 'model_comparison.txt'), 'w') as f:
        f.write(f"Model Comparison ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n")
        f.write(f"Sorted by: {sort_by}\n")
        f.write(f"Metric: {metric}\n\n")
        f.write(tabulate(df[[
            'Name', 'Type', 'Days Ahead', 'Threshold', 'Features', 
            f'Train {metric.capitalize()}', f'Test {metric.capitalize()}', 'Train-Test Gap'
        ]], headers='keys', tablefmt='grid'))
        
    return df


def compare_models(args):
    """Main function to compare trained models."""
    # Find all model directories
    model_dirs = find_model_dirs(args.models_dir)
    print(f"Found {len(model_dirs)} model directories")
    
    if not model_dirs:
        print(f"No model directories found in {args.models_dir}")
        return
    
    # Load results from each model
    all_results = []
    for model_dir in model_dirs:
        results = load_model_results(model_dir)
        if results:
            all_results.append(results)
    
    print(f"Successfully loaded {len(all_results)} model results")
    
    # Extract metadata
    model_metadata = [extract_model_metadata(results) for results in all_results]
    model_metadata = [m for m in model_metadata if m is not None]
    
    # Create comparison dataframe
    comparison_df = create_comparison_dataframe(model_metadata, args.metric)
    
    # If we have too many models, limit to the top N by test metric
    if len(comparison_df) > args.max_models:
        print(f"Limiting comparison to top {args.max_models} models by test {args.metric}")
        comparison_df = comparison_df.sort_values(f'Test {args.metric.capitalize()}', ascending=False).head(args.max_models)
    
    # Create visualizations
    df = plot_comparison(comparison_df, args.metric, args.output_dir, args.sort_by)
    
    print(f"Model comparison completed. Results saved to {args.output_dir}")
    return df


def main():
    args = parse_args()
    compare_models(args)


if __name__ == "__main__":
    main() 