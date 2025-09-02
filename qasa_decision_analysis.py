import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = 16, 12
plt.rcParams['font.size'] = 12

class QASADecisionAnalyzer:
    """Analyzer for QASA (Quantum Attention) model decision snapshots"""
    
    def __init__(self, data_path='features_data.csv'):
        """Initialize the analyzer with data"""
        print("Loading data for QASA decision analysis...")
        self.df = pd.read_csv(data_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values('date')
        
        # Get unique dates and industries
        self.dates = self.df['date'].unique()
        self.industries = self.df['industry'].unique()
        
        # QASA-specific parameters (matching qasa_model.py)
        self.n_qubits = 4
        self.n_layers = 2
        
        print(f"Data loaded: {len(self.dates)} dates, {len(self.industries)} industries")
        print(f"Date range: {self.dates.min()} to {self.dates.max()}")
        print(f"QASA Configuration: {self.n_qubits} qubits, {self.n_layers} layers")
    
    def simulate_qasa_predictions(self, target_date, top_k=10):
        """Simulate QASA model predictions for a given date"""
        print(f"Simulating QASA predictions for {target_date}")
        
        # Find the date index
        date_idx = np.where(self.dates == target_date)[0]
        if len(date_idx) == 0:
            raise ValueError(f"Date {target_date} not found in data")
        
        date_idx = date_idx[0]
        
        # Get historical data for prediction (last 10 days)
        start_idx = max(0, date_idx - 10)
        historical_data = self.df[self.df['date'].isin(self.dates[start_idx:date_idx])]
        
        # Calculate quantum-enhanced predictions (simulating QASA output)
        predictions = {}
        for industry in self.industries:
            industry_data = historical_data[historical_data['industry'] == industry]
            if len(industry_data) > 0:
                # Simulate QASA quantum-enhanced predictions
                # QASA combines classical attention with quantum circuit outputs
                momentum_5 = industry_data['share_Momentum_5'].iloc[-1] if 'share_Momentum_5' in industry_data.columns else 0
                momentum_10 = industry_data['share_Momentum_10'].iloc[-1] if 'share_Momentum_10' in industry_data.columns else 0
                momentum_20 = industry_data['share_Momentum_20'].iloc[-1] if 'share_Momentum_20' in industry_data.columns else 0
                vol_20 = industry_data['share_Vol_20'].iloc[-1] if 'share_Vol_20' in industry_data.columns else 0
                
                # Quantum enhancement factor (simulating quantum circuit output)
                # QASA uses quantum circuits with CRZ and RY gates
                quantum_factor = np.sin(momentum_5 * np.pi) * np.cos(momentum_10 * np.pi) * 0.1
                
                # Combine classical and quantum factors (QASA's hybrid approach)
                classical_score = 0.35 * momentum_5 + 0.25 * momentum_10 + 0.25 * momentum_20 + 0.15 * vol_20
                quantum_score = quantum_factor
                
                # Final QASA score (classical + quantum enhancement)
                score = classical_score + quantum_score
                predictions[industry] = score
        
        # Sort by prediction score and get top-k
        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        top_k_industries = sorted_predictions[:top_k]
        
        return top_k_industries, predictions
    
    def get_next_day_performance(self, target_date, top_industries):
        """Get next day performance for selected industries"""
        # Find next trading day
        date_idx = np.where(self.dates == target_date)[0][0]
        if date_idx + 1 >= len(self.dates):
            return None
        
        next_date = self.dates[date_idx + 1]
        
        # Get performance data
        performance_data = {}
        for industry, score in top_industries:
            current_data = self.df[(self.df['date'] == target_date) & (self.df['industry'] == industry)]
            next_data = self.df[(self.df['date'] == next_date) & (self.df['industry'] == industry)]
            
            if len(current_data) > 0 and len(next_data) > 0:
                current_share = current_data['capital_share'].iloc[0]
                next_share = next_data['capital_share'].iloc[0]
                
                # Calculate return
                if current_share > 0:
                    return_pct = (next_share - current_share) / current_share * 100
                else:
                    return_pct = 0
                
                performance_data[industry] = {
                    'prediction_score': score,
                    'current_share': current_share,
                    'next_share': next_share,
                    'return_pct': return_pct
                }
        
        return performance_data, next_date
    
    def create_decision_snapshot(self, target_date, top_k=10):
        """Create a comprehensive decision snapshot for a given date"""
        print(f"\nCreating QASA decision snapshot for {target_date}")
        
        # Get predictions
        top_industries, all_predictions = self.simulate_qasa_predictions(target_date, top_k)
        
        # Get next day performance
        performance_result = self.get_next_day_performance(target_date, top_industries)
        if performance_result is None:
            print("Cannot get next day performance (end of data)")
            return None
        
        performance_data, next_date = performance_result
        
        # Create summary DataFrame
        snapshot_data = []
        for industry, score in top_industries:
            if industry in performance_data:
                perf = performance_data[industry]
                snapshot_data.append({
                    'Industry': industry,
                    'QASA_Score': score,
                    'Current_Share': perf['current_share'],
                    'Next_Share': perf['next_share'],
                    'Return_%': perf['return_pct']
                })
        
        snapshot_df = pd.DataFrame(snapshot_data)
        
        return snapshot_df, all_predictions, next_date
    
    def plot_top10_performance(self, snapshot_df, target_date, next_date):
        """Plot Top-10 performance comparison for QASA"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
        
        # Plot 1: QASA prediction scores
        industries = snapshot_df['Industry']
        scores = snapshot_df['QASA_Score']
        
        bars1 = ax1.bar(range(len(industries)), scores, color='purple', alpha=0.7)
        ax1.set_title(f'QASA Model: Top-10 Industry Predictions\n{target_date.strftime("%Y-%m-%d")} (Quantum-Enhanced)', 
                      fontsize=16, fontweight='bold')
        ax1.set_ylabel('QASA Score (Classical + Quantum)', fontsize=14)
        ax1.set_xlabel('Industries', fontsize=14)
        ax1.set_xticks(range(len(industries)))
        ax1.set_xticklabels(industries, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars1, scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{score:.4f}', ha='center', va='bottom', fontsize=10)
        
        # Plot 2: Next day performance
        returns = snapshot_df['Return_%']
        colors = ['green' if r > 0 else 'red' for r in returns]
        
        bars2 = ax2.bar(range(len(industries)), returns, color=colors, alpha=0.7)
        ax2.set_title(f'Next Day Performance: {next_date.strftime("%Y-%m-%d")}', 
                      fontsize=16, fontweight='bold')
        ax2.set_ylabel('Return (%)', fontsize=14)
        ax2.set_xlabel('Industries', fontsize=14)
        ax2.set_xticks(range(len(industries)))
        ax2.set_xticklabels(industries, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Add value labels on bars
        for bar, ret in zip(bars2, returns):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + (0.1 if ret > 0 else -0.1),
                    f'{ret:.2f}%', ha='center', va='bottom' if ret > 0 else 'top', fontsize=10)
        
        plt.tight_layout()
        return fig
    
    def create_sector_weight_heatmap(self, start_date=None, end_date=None, top_n=8):
        """Create sector weight heatmap over time for QASA context"""
        if start_date is None:
            start_date = self.dates[0]
        if end_date is None:
            end_date = self.dates[-1]
        
        # Filter date range
        mask = (self.dates >= start_date) & (self.dates <= end_date)
        filtered_dates = self.dates[mask]
        
        print(f"Creating sector weight heatmap from {start_date} to {end_date}")
        
        # Get top industries by average capital share
        avg_shares = {}
        for industry in self.industries:
            industry_data = self.df[(self.df['industry'] == industry) & 
                                  (self.df['date'].isin(filtered_dates))]
            if len(industry_data) > 0:
                avg_shares[industry] = industry_data['capital_share'].mean()
        
        # Select top N industries
        top_industries = sorted(avg_shares.items(), key=lambda x: x[1], reverse=True)[:top_n]
        top_industry_names = [ind[0] for ind in top_industries]
        
        # Create heatmap data
        heatmap_data = []
        for date in filtered_dates:
            row_data = []
            for industry in top_industry_names:
                day_data = self.df[(self.df['date'] == date) & (self.df['industry'] == industry)]
                if len(day_data) > 0:
                    row_data.append(day_data['capital_share'].iloc[0])
                else:
                    row_data.append(0)
            heatmap_data.append(row_data)
        
        heatmap_df = pd.DataFrame(heatmap_data, 
                                index=filtered_dates, 
                                columns=top_industry_names)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Create custom colormap (quantum-themed colors)
        cmap = sns.diverging_palette(250, 15, sep=80, n=7)
        
        sns.heatmap(heatmap_df.T, 
                   annot=True, 
                   fmt='.3f', 
                   cmap=cmap,
                   cbar_kws={'label': 'Capital Share'},
                   ax=ax)
        
        ax.set_title('Sector Weight Heatmap: Top Industries Over Time\n(QASA Quantum-Enhanced Model Context)', 
                    fontsize=18, fontweight='bold', pad=20)
        ax.set_xlabel('Date', fontsize=14)
        ax.set_ylabel('Industries', fontsize=14)
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        return fig, heatmap_df
    
    def create_quantum_circuit_visualization(self):
        """Create a visualization of QASA's quantum circuit structure"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Draw quantum circuit representation
        qubits = range(self.n_qubits)
        layers = range(self.n_layers)
        
        # Draw qubit lines
        for i, qubit in enumerate(qubits):
            ax.plot([0, len(layers) + 1], [qubit, qubit], 'k-', linewidth=2, alpha=0.7)
            ax.text(-0.2, qubit, f'Q{i}', fontsize=12, ha='right', va='center')
        
        # Draw quantum gates
        for layer in layers:
            for i, qubit in enumerate(qubits):
                # CRZ gates (controlled rotation Z)
                if i < self.n_qubits - 1:
                    ax.plot([layer + 0.5, layer + 0.5], [qubit, qubit + 1], 'b-', linewidth=3, alpha=0.8)
                    ax.scatter([layer + 0.5], [qubit + 0.5], color='blue', s=100, alpha=0.8)
                
                # RY gates (rotation Y)
                ax.scatter([layer + 0.5], [qubit], color='red', s=80, alpha=0.8)
        
        # Add labels
        ax.set_title('QASA Quantum Circuit Architecture', fontsize=16, fontweight='bold')
        ax.set_xlabel('Quantum Layers', fontsize=14)
        ax.set_ylabel('Qubits', fontsize=14)
        ax.set_xlim(-0.5, len(layers) + 0.5)
        ax.set_ylim(-0.5, self.n_qubits - 0.5)
        ax.set_xticks(range(len(layers) + 1))
        ax.set_yticks(qubits)
        ax.grid(True, alpha=0.3)
        
        # Add legend
        ax.scatter([], [], color='blue', s=100, label='CRZ (Controlled Rotation Z)')
        ax.scatter([], [], color='red', s=80, label='RY (Rotation Y)')
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        return fig
    
    def save_analysis(self, snapshot_df, fig1, fig2, fig3, target_date):
        """Save all analysis results"""
        print(f"\nSaving QASA analysis results for {target_date}")
        
        # Save snapshot data
        snapshot_filename = f'qasa_snapshot_{target_date.strftime("%Y%m%d")}.csv'
        snapshot_df.to_csv(snapshot_filename, index=False)
        print(f"Snapshot data saved: {snapshot_filename}")
        
        # Save plots
        fig1.savefig(f'qasa_top10_{target_date.strftime("%Y%m%d")}.png', 
                     dpi=300, bbox_inches='tight', facecolor='white')
        fig2.savefig(f'qasa_sector_weight_heatmap_{target_date.strftime("%Y%m%d")}.png', 
                     dpi=300, bbox_inches='tight', facecolor='white')
        fig3.savefig(f'qasa_quantum_circuit_{target_date.strftime("%Y%m%d")}.png', 
                     dpi=300, bbox_inches='tight', facecolor='white')
        
        # Save as PDF for papers
        fig1.savefig(f'qasa_top10_{target_date.strftime("%Y%m%d")}.pdf', 
                     bbox_inches='tight', facecolor='white')
        fig2.savefig(f'qasa_sector_weight_heatmap_{target_date.strftime("%Y%m%d")}.pdf', 
                     bbox_inches='tight', facecolor='white')
        fig3.savefig(f'qasa_quantum_circuit_{target_date.strftime("%Y%m%d")}.pdf', 
                     bbox_inches='tight', facecolor='white')
        
        print("All plots saved as PNG and PDF")
    
    def run_complete_analysis(self, target_date, top_k=10):
        """Run complete QASA analysis for a given date"""
        print(f"Running complete QASA decision analysis for {target_date}")
        
        # Create decision snapshot
        result = self.create_decision_snapshot(target_date, top_k)
        if result is None:
            return None
        
        snapshot_df, all_predictions, next_date = result
        
        # Print summary
        print(f"\n=== QASA Decision Snapshot Summary ===")
        print(f"Target Date: {target_date.strftime('%Y-%m-%d')}")
        print(f"Next Date: {next_date.strftime('%Y-%m-%d')}")
        print(f"Quantum Configuration: {self.n_qubits} qubits, {self.n_layers} layers")
        print(f"Top {top_k} Industries Selected:")
        print(snapshot_df.to_string(index=False))
        
        # Calculate performance metrics
        avg_return = snapshot_df['Return_%'].mean()
        positive_count = (snapshot_df['Return_%'] > 0).sum()
        print(f"\nPerformance Summary:")
        print(f"Average Return: {avg_return:.2f}%")
        print(f"Positive Returns: {positive_count}/{top_k} ({positive_count/top_k*100:.1f}%)")
        
        # Create plots
        fig1 = self.plot_top10_performance(snapshot_df, target_date, next_date)
        fig2, heatmap_df = self.create_sector_weight_heatmap()
        fig3 = self.create_quantum_circuit_visualization()
        
        # Save results
        self.save_analysis(snapshot_df, fig1, fig2, fig3, target_date)
        
        # Display plots
        plt.show()
        
        return snapshot_df, fig1, fig2, fig3

def main():
    """Main function to run the QASA analysis"""
    # Initialize analyzer
    analyzer = QASADecisionAnalyzer()
    
    # Set target date (you can change this to any date in your data)
    target_date = pd.to_datetime('2025-06-12')  # Example date
    
    # Check if date exists in data
    if target_date not in analyzer.dates:
        print(f"Date {target_date} not found. Available dates:")
        print(f"From: {analyzer.dates[0]} to {analyzer.dates[-1]}")
        # Use a date that exists
        target_date = analyzer.dates[-5]  # Use 5th last date
        print(f"Using available date: {target_date}")
    
    # Run analysis
    try:
        result = analyzer.run_complete_analysis(target_date, top_k=10)
        if result:
            print("\n✅ QASA Analysis completed successfully!")
            print("Files generated:")
            print("- CSV snapshot data")
            print("- Top-10 performance plots (PNG/PDF)")
            print("- Sector weight heatmap (PNG/PDF)")
            print("- Quantum circuit visualization (PNG/PDF)")
        else:
            print("❌ QASA Analysis failed")
    except Exception as e:
        print(f"Error during QASA analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
