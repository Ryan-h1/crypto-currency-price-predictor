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
                        choices=['accuracy', 'precision', 'recall', 'f1_score', 'roi', 'profit_factor', 'win_rate'],
                        help='Metric to use for comparing models')

    parser.add_argument('--sort_by', type=str, default='test_score',
                        choices=['name', 'test_score', 'train_score', 'train_test_gap', 'timestamp', 'roi', 'profit_factor'],
                        help='How to sort models in the comparison')

    parser.add_argument('--max_models', type=int, default=20,
                        help='Maximum number of models to include in comparison')
                        
    parser.add_argument('--financial_focus', action='store_true',
                        help='Focus on financial metrics in visualizations and reports')
                        
    parser.add_argument('--threshold_analysis', action='store_true',
                        help='Include probability threshold analysis if available')

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
        
        # Extract basic metadata
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
        
        # Extract financial metrics if available
        financial_metrics = results.get('financial_metrics', {})
        if financial_metrics:
            metadata['roi'] = financial_metrics.get('roi', 0)
            metadata['profit_factor'] = financial_metrics.get('profit_factor', 0)
            metadata['win_rate'] = financial_metrics.get('win_rate', 0)
            metadata['avg_win'] = financial_metrics.get('avg_win', 0)
            metadata['avg_loss'] = financial_metrics.get('avg_loss', 0)
            metadata['max_drawdown'] = financial_metrics.get('max_drawdown', 0)
            metadata['volatility'] = financial_metrics.get('volatility', 0)
            metadata['sharpe_ratio'] = financial_metrics.get('sharpe_ratio', 0)
        
        # Calculate estimated financial performance if not available
        if 'roi' not in metadata and 'test_metrics' in results:
            # Estimate ROI based on precision and expected returns
            # Assuming a 15% price increase threshold means ~15% average return on true positives
            # and an average loss of ~5% on false positives
            precision = results['test_metrics'].get('precision', 0)
            avg_expected_gain = args.get('threshold', 0.15) * 100  # Convert to percentage
            avg_expected_loss = -5  # Assume 5% average loss on false positives
            
            if precision > 0:
                estimated_roi = (precision * avg_expected_gain) + ((1-precision) * avg_expected_loss)
                metadata['roi'] = estimated_roi
                metadata['estimated_financials'] = True  # Flag that these are estimated
                metadata['profit_factor'] = (precision * avg_expected_gain) / max(0.001, abs((1-precision) * avg_expected_loss))
                metadata['win_rate'] = precision
            else:
                metadata['roi'] = 0
                metadata['profit_factor'] = 0
                metadata['win_rate'] = 0
                metadata['estimated_financials'] = True
        
        # Extract per-coin performance if available
        if 'coin_results' in results:
            top_coins = sorted(results['coin_results'].items(), 
                             key=lambda x: x[1].get('f1_score', 0), 
                             reverse=True)[:5]
            
            metadata['top_performing_coins'] = [coin for coin, _ in top_coins]
            
            # Calculate average returns by market cap segment
            if len(results['coin_results']) > 0:
                # Group coins by market cap
                coins_with_mcap = [(coin, data.get('market_cap', 0)) 
                                  for coin, data in results['coin_results'].items()]
                
                if coins_with_mcap:
                    # Sort by market cap
                    coins_with_mcap.sort(key=lambda x: x[1], reverse=True)
                    
                    # Split into segments
                    n_segments = 3  # Top, middle, bottom
                    segment_size = max(1, len(coins_with_mcap) // n_segments)
                    
                    segments = []
                    for i in range(n_segments):
                        start_idx = i * segment_size
                        end_idx = start_idx + segment_size
                        if i == n_segments - 1:  # Last segment takes all remaining
                            end_idx = len(coins_with_mcap)
                        segments.append(coins_with_mcap[start_idx:end_idx])
                    
                    # Calculate average performance by segment
                    for i, segment in enumerate(segments):
                        segment_coins = [coin for coin, _ in segment]
                        segment_results = [results['coin_results'][coin] for coin in segment_coins 
                                          if coin in results['coin_results']]
                        
                        if segment_results:
                            avg_f1 = sum(r.get('f1_score', 0) for r in segment_results) / len(segment_results)
                            metadata[f'mcap_segment_{i+1}_f1'] = avg_f1
        
        # Extract threshold analysis if available
        if 'threshold_analysis' in results:
            metadata['threshold_analysis'] = results['threshold_analysis']
        
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
        
        # Check if we have financial metrics
        has_financials = any(key in metadata for key in ['roi', 'profit_factor', 'win_rate'])
        
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
        
        # Add financial metrics if available
        if has_financials:
            row['ROI (%)'] = metadata.get('roi', 0)
            row['Profit Factor'] = metadata.get('profit_factor', 0)
            row['Win Rate'] = metadata.get('win_rate', 0)
            row['Avg Win (%)'] = metadata.get('avg_win', 0)
            row['Avg Loss (%)'] = metadata.get('avg_loss', 0)
            row['Volatility'] = metadata.get('volatility', 0)
            row['Sharpe Ratio'] = metadata.get('sharpe_ratio', 0)
            row['Estimated Financials'] = metadata.get('estimated_financials', False)
        
        # Add market cap segment performance if available
        for i in range(1, 4):
            key = f'mcap_segment_{i}_f1'
            if key in metadata:
                row[f'MCap Segment {i} F1'] = metadata[key]
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def plot_comparison(df, metric, output_dir, sort_by, financial_focus=False):
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
    elif sort_by == 'roi' and 'ROI (%)' in df.columns:
        df = df.sort_values('ROI (%)', ascending=False)
    elif sort_by == 'profit_factor' and 'Profit Factor' in df.columns:
        df = df.sort_values('Profit Factor', ascending=False)
        
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
    
    # If we have financial metrics, create financial performance visualizations
    if financial_focus and 'ROI (%)' in df.columns:
        # ROI vs F1 Score plot
        plt.figure(figsize=(10, 8))
        if 'Test f1_score' in df.columns:
            financial_scatter = plt.scatter(df['Test f1_score'], 
                                         df['ROI (%)'],
                                         c=df['Profit Factor'],
                                         cmap='viridis',
                                         s=100,
                                         alpha=0.7)
            
            # Add model names as annotations
            for i, row in df.iterrows():
                plt.annotate(row['Name'], 
                            (row['Test f1_score'], row['ROI (%)']),
                            xytext=(5, 5),
                            textcoords='offset points',
                            fontsize=8)
            
            plt.xlabel('F1 Score')
            plt.ylabel('ROI (%)')
            plt.title('F1 Score vs ROI')
            
            # Add a colorbar
            cbar = plt.colorbar(financial_scatter)
            cbar.set_label('Profit Factor')
            
            plt.grid(alpha=0.3)
            plt.tight_layout()
            
            # Save the plot
            plt.savefig(os.path.join(output_dir, 'f1_vs_roi.png'))
            plt.close()
        
        # ROI Comparison Bar Chart
        plt.figure(figsize=(12, max(8, len(df)*0.4)))
        roi_bars = plt.barh(df['Name'], df['ROI (%)'], color='goldenrod', alpha=0.7)
        
        # Add win rate as color intensity
        if 'Win Rate' in df.columns:
            bar_colors = plt.cm.Blues(df['Win Rate'] / df['Win Rate'].max())
            for i, bar in enumerate(roi_bars):
                bar.set_color(bar_colors[i])
                
        plt.xlabel('ROI (%)')
        plt.title('Model ROI Comparison')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(output_dir, 'roi_comparison.png'))
        plt.close()
        
        # Market Cap Segment Performance if available
        if 'MCap Segment 1 F1' in df.columns:
            # Select a representative model for the plot
            top_model = df.sort_values('ROI (%)', ascending=False).iloc[0]
            
            seg_columns = [col for col in df.columns if col.startswith('MCap Segment')]
            if seg_columns and len(seg_columns) >= 2:
                # Get segment data for the top model
                segments = ['High MCap', 'Medium MCap', 'Low MCap'][:len(seg_columns)]
                values = [top_model[col] for col in seg_columns]
                
                plt.figure(figsize=(10, 6))
                plt.bar(segments, values, color='skyblue')
                plt.ylabel('F1 Score')
                plt.title(f'Performance by Market Cap Segment ({top_model["Name"]})')
                plt.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                
                # Save the plot
                plt.savefig(os.path.join(output_dir, 'market_cap_performance.png'))
                plt.close()
    
    # Save the summary to a CSV file
    df.to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)
    
    # Columns for text report
    report_columns = ['Name', 'Type', 'Days Ahead', 'Threshold', 'Features']
    
    # Add classification metrics
    report_columns.extend([f'Train {metric.capitalize()}', f'Test {metric.capitalize()}', 'Train-Test Gap'])
    
    # Add financial metrics if available
    if 'ROI (%)' in df.columns and financial_focus:
        report_columns.extend(['ROI (%)', 'Profit Factor', 'Win Rate'])
    
    # Create a summary text file
    with open(os.path.join(output_dir, 'model_comparison.txt'), 'w') as f:
        f.write(f"Model Comparison ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n")
        f.write(f"Sorted by: {sort_by}\n")
        f.write(f"Metric: {metric}\n\n")
        f.write(tabulate(df[report_columns], headers='keys', tablefmt='grid'))
        
        # Add financial performance summary if available
        if 'ROI (%)' in df.columns:
            f.write("\n\nFINANCIAL PERFORMANCE SUMMARY\n")
            
            # Sort by ROI
            df_sorted = df.sort_values('ROI (%)', ascending=False)
            
            f.write("\nTop Models by ROI:\n")
            top_financial = df_sorted.head(3)
            f.write(tabulate(top_financial[['Name', 'Type', 'ROI (%)', 'Profit Factor', 'Win Rate']], 
                            headers='keys', tablefmt='simple'))
            
            # Add note about estimated financials
            if 'Estimated Financials' in df.columns and df['Estimated Financials'].any():
                f.write("\n\nNote: Some financial metrics are estimated based on classification performance.\n")
                f.write("For more accurate financial analysis, run backtest simulations on the models.\n")
    
    return df


def plot_threshold_analysis(model_metadata_list, output_dir):
    """Create visualizations for threshold analysis if available."""
    models_with_threshold_analysis = [m for m in model_metadata_list 
                                    if m and 'threshold_analysis' in m]
    
    if not models_with_threshold_analysis:
        print("No threshold analysis data found in any model.")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    for metadata in models_with_threshold_analysis:
        # Extract threshold analysis data
        threshold_data = metadata['threshold_analysis']
        
        # Convert to DataFrame
        threshold_df = pd.DataFrame(threshold_data)
        
        model_name = metadata['model_dir']
        
        # Create visualization of threshold vs. performance metrics
        plt.figure(figsize=(12, 8))
        
        # Plot ROI
        if 'roi' in threshold_df.columns:
            plt.plot(threshold_df['threshold'], threshold_df['roi'], 
                    'o-', color='darkgreen', label='ROI (%)')
        
        # Plot F1 Score on secondary axis
        if 'f1_score' in threshold_df.columns:
            ax2 = plt.gca().twinx()
            ax2.plot(threshold_df['threshold'], threshold_df['f1_score'], 
                    'o--', color='navy', label='F1 Score')
            ax2.set_ylabel('F1 Score', color='navy')
            ax2.tick_params(axis='y', colors='navy')
        
        # Find optimal threshold(s)
        if 'roi' in threshold_df.columns:
            optimal_roi_threshold = threshold_df.loc[threshold_df['roi'].idxmax(), 'threshold']
            plt.axvline(optimal_roi_threshold, color='darkgreen', linestyle='--', alpha=0.5, 
                       label=f'Optimal ROI Threshold: {optimal_roi_threshold:.2f}')
        
        if 'f1_score' in threshold_df.columns:
            optimal_f1_threshold = threshold_df.loc[threshold_df['f1_score'].idxmax(), 'threshold']
            plt.axvline(optimal_f1_threshold, color='navy', linestyle=':', alpha=0.5,
                       label=f'Optimal F1 Threshold: {optimal_f1_threshold:.2f}')
        
        plt.xlabel('Probability Threshold')
        plt.ylabel('ROI (%)')
        plt.title(f'Threshold Analysis for {model_name}')
        plt.grid(alpha=0.3)
        
        # Combine both legends
        lines1, labels1 = plt.gca().get_legend_handles_labels()
        if 'f1_score' in threshold_df.columns:
            lines2, labels2 = ax2.get_legend_handles_labels()
            plt.legend(lines1 + lines2, labels1 + labels2, loc='best')
        else:
            plt.legend(loc='best')
            
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(output_dir, f'threshold_analysis_{model_name}.png'))
        plt.close()
        
        # Save threshold data
        threshold_df.to_csv(os.path.join(output_dir, f'threshold_data_{model_name}.csv'), index=False)


def create_roi_summary_report(df, output_dir):
    """Create a detailed summary of financial performance."""
    if 'ROI (%)' not in df.columns:
        return
    
    with open(os.path.join(output_dir, 'financial_summary.txt'), 'w') as f:
        f.write(f"CRYPTOCURRENCY PREDICTION MODEL FINANCIAL SUMMARY\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Sort by ROI
        df_sorted = df.sort_values('ROI (%)', ascending=False)
        
        # Overall statistics
        f.write("OVERALL PERFORMANCE STATISTICS\n")
        f.write(f"Number of models: {len(df)}\n")
        f.write(f"Average ROI: {df['ROI (%)'].mean():.2f}%\n")
        f.write(f"Median ROI: {df['ROI (%)'].median():.2f}%\n")
        f.write(f"ROI Range: {df['ROI (%)'].min():.2f}% to {df['ROI (%)'].max():.2f}%\n\n")
        
        # Top 3 models by ROI
        f.write("TOP 3 MODELS BY ROI\n")
        top_3 = df_sorted.head(3)
        for i, row in enumerate(top_3.iterrows(), 1):
            idx, data = row
            f.write(f"{i}. {data['Name']} ({data['Type']})\n")
            f.write(f"   ROI: {data['ROI (%)']:.2f}%\n")
            if 'Profit Factor' in data:
                f.write(f"   Profit Factor: {data['Profit Factor']:.2f}\n")
            if 'Win Rate' in data:
                f.write(f"   Win Rate: {data['Win Rate']:.2f}\n")
            if 'Volatility' in data:
                f.write(f"   Volatility: {data['Volatility']:.2f}\n")
            
            # Check for capitalized metric name (fix for KeyError)
            f1_column = next((col for col in df.columns if col.lower() == 'test f1_score'), None)
            if f1_column and f1_column in data:
                f.write(f"   F1 Score: {data[f1_column]:.4f}\n")
            else:
                f.write(f"   F1 Score: N/A\n")
            f.write("\n")
        
        # Model comparison by type
        f.write("ROI BY MODEL TYPE\n")
        model_types = df['Type'].unique()
        for model_type in model_types:
            type_df = df[df['Type'] == model_type]
            f.write(f"{model_type}:\n")
            f.write(f"   Average ROI: {type_df['ROI (%)'].mean():.2f}%\n")
            f.write(f"   Best Model: {type_df.sort_values('ROI (%)', ascending=False).iloc[0]['Name']}\n")
            f.write(f"   Best ROI: {type_df['ROI (%)'].max():.2f}%\n\n")
        
        # ROI vs classification metrics correlation
        f.write("CORRELATION BETWEEN FINANCIAL AND CLASSIFICATION METRICS\n")
        # Find f1_score column using case-insensitive matching
        f1_column = next((col for col in df.columns if col.lower() == 'test f1_score'), None)
        if f1_column and not df[f1_column].isna().all():
            corr = df['ROI (%)'].corr(df[f1_column])
            f.write(f"ROI vs Test F1 Score correlation: {corr:.4f}\n")
        
        # Find precision column using case-insensitive matching
        precision_column = next((col for col in df.columns if col.lower() == 'test precision'), None)
        if precision_column and not df[precision_column].isna().all():
            corr = df['ROI (%)'].corr(df[precision_column])
            f.write(f"ROI vs Test Precision correlation: {corr:.4f}\n")
        
        # Find recall column using case-insensitive matching
        recall_column = next((col for col in df.columns if col.lower() == 'test recall'), None)
        if recall_column and not df[recall_column].isna().all():
            corr = df['ROI (%)'].corr(df[recall_column])
            f.write(f"ROI vs Test Recall correlation: {corr:.4f}\n")
        
        f.write("\n\nINVESTMENT IMPLICATIONS\n")
        f.write("Based on the model comparison, the following investment strategies are recommended:\n\n")
        
        best_model = df_sorted.iloc[0]
        f.write(f"1. Primary Strategy: Use {best_model['Name']} ({best_model['Type']}) model\n")
        f.write(f"   - Expected ROI: {best_model['ROI (%)']:.2f}%\n")
        if 'Win Rate' in best_model:
            f.write(f"   - Win Rate: {best_model['Win Rate']:.2f}\n")
        f.write(f"   - Prediction Threshold: {best_model['Threshold']}\n")
        f.write(f"   - Features: {best_model['Features']}\n\n")
        
        # Risk analysis
        f.write("2. Risk Management:\n")
        if 'Volatility' in best_model and best_model['Volatility'] > 0:
            f.write(f"   - Expected Volatility: {best_model['Volatility']:.2f}%\n")
        if 'Avg Loss (%)' in best_model:
            f.write(f"   - Average Loss Per Trade: {best_model['Avg Loss (%)']:.2f}%\n")
        else:
            f.write("   - Consider setting stop-losses at 10% below entry price\n")
        
        f.write("\n3. Market Cap Considerations:\n")
        if any(col.startswith('MCap Segment') for col in df.columns):
            top_model_seg_cols = [col for col in df.columns if col.startswith('MCap Segment') and pd.notna(best_model.get(col))]
            if top_model_seg_cols:
                # Find which segment performs best
                seg_performances = [(col, best_model[col]) for col in top_model_seg_cols]
                best_seg = max(seg_performances, key=lambda x: x[1])
                worst_seg = min(seg_performances, key=lambda x: x[1])
                
                f.write(f"   - Best performing market cap segment: {best_seg[0].replace('MCap Segment ', '').replace(' F1', '')}\n")
                f.write(f"   - Worst performing market cap segment: {worst_seg[0].replace('MCap Segment ', '').replace(' F1', '')}\n")
                f.write(f"   - Consider focusing investments in {best_seg[0].replace('MCap Segment ', '').replace(' F1', '')} market cap range\n")
        else:
            f.write("   - Consider allocating 70% to large cap, 30% to mid cap cryptocurrencies\n")


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
    df = plot_comparison(comparison_df, args.metric, args.output_dir, args.sort_by, args.financial_focus)
    
    # Create threshold analysis plots if requested
    if args.threshold_analysis:
        plot_threshold_analysis(model_metadata, args.output_dir)
    
    # Create detailed ROI report if we have financial metrics
    if args.financial_focus and 'ROI (%)' in df.columns:
        create_roi_summary_report(df, args.output_dir)
    
    print(f"Model comparison completed. Results saved to {args.output_dir}")
    return df


def main():
    args = parse_args()
    compare_models(args)


if __name__ == "__main__":
    main() 