#!/usr/bin/env python3
"""
Comprehensive Multi-Model Visualizer for Reverse Attribution Analysis
Supports IMDb BERT, Yelp RoBERTa, and CIFAR-10 ResNet with extensive comparisons

Usage:
    python visualizer.py --auto-discover --outdir comprehensive_analysis/
    python visualizer.py --input evaluation_results.json --outdir multi_model_dashboard/
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple

# Set UTF-8 encoding for Windows compatibility
os.environ['PYTHONUTF8'] = '1'

# Handle stdout/stderr encoding for Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

# Configure matplotlib for Windows Unicode compatibility
matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['savefig.dpi'] = 300
matplotlib.rcParams['savefig.bbox'] = 'tight'

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('comprehensive_visualizer.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ComprehensiveModelVisualizer:
    """
    Advanced multi-model visualizer for comprehensive Reverse Attribution analysis.
    Handles IMDb BERT, Yelp RoBERTa, and CIFAR-10 ResNet with extensive comparisons.
    """
    
    def __init__(self, output_dir: str = "comprehensive_analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create comprehensive directory structure
        self.subdirs = {
            'performance': self.output_dir / "performance_analysis",
            'attribution': self.output_dir / "attribution_analysis", 
            'comparison': self.output_dir / "model_comparison",
            'interactive': self.output_dir / "interactive_dashboards",
            'summary': self.output_dir / "summary_reports",
            'cross_domain': self.output_dir / "cross_domain_analysis"
        }
        
        for subdir in self.subdirs.values():
            subdir.mkdir(exist_ok=True, parents=True)
        
        self.data = {}
        self.models = {}
        self.formats = ['png', 'pdf', 'html']
        
        # Model configuration
        self.model_configs = {
            'imdb': {
                'name': 'IMDb BERT',
                'type': 'text',
                'color': '#2E86AB',
                'architecture': 'BERT-base',
                'domain': 'Natural Language Processing',
                'task': 'Sentiment Classification'
            },
            'yelp': {
                'name': 'Yelp RoBERTa', 
                'type': 'text',
                'color': '#A23B72',
                'architecture': 'RoBERTa-base',
                'domain': 'Natural Language Processing',
                'task': 'Review Classification'
            },
            'cifar10': {
                'name': 'CIFAR-10 ResNet',
                'type': 'vision',
                'color': '#F18F01',
                'architecture': 'ResNet-56',
                'domain': 'Computer Vision',
                'task': 'Image Classification'
            }
        }
        
        logger.info(f"ComprehensiveModelVisualizer initialized")
        logger.info(f"Output directory: {self.output_dir}")
        
    def discover_all_results(self, search_dirs: List[str] = None) -> Dict[str, Any]:
        """
        Comprehensive discovery of all model results across multiple locations.
        """
        if search_dirs is None:
            search_dirs = [".", "reproduction_results", "../reproduction_results", 
                          "checkpoints", "results"]
        
        discovered_data = {}
        
        # File patterns for comprehensive search
        file_patterns = {
            'evaluation_results': ['evaluation_results.json'],
            'jmlr_metrics': ['jmlr_metrics.json'],
            'training_summary': ['training_summary.json', 'training_results_summary.json'],
            'comprehensive_results': ['comprehensive_evaluation_results', 'comprehensive_evaluation_results.json'],
            'imdb_results': ['imdb_results.json', 'bert_imdb_results.json'],
            'yelp_results': ['yelp_results.json', 'roberta_yelp_results.json'],
            'cifar_results': ['cifar10_results.json', 'resnet_cifar_results.json']
        }
        
        for search_dir in search_dirs:
            search_path = Path(search_dir)
            if not search_path.exists():
                continue
                
            logger.info(f"Searching for results in: {search_path.absolute()}")
            
            for file_type, patterns in file_patterns.items():
                if file_type in discovered_data:
                    continue
                    
                for pattern in patterns:
                    file_path = search_path / pattern
                    if file_path.exists():
                        try:
                            if file_path.suffix == '.json':
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    discovered_data[file_type] = json.load(f)
                            else:
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    content = f.read()
                                    if content.strip().startswith('{'):
                                        discovered_data[file_type] = json.loads(content)
                                    else:
                                        discovered_data[file_type] = {'raw_content': content}
                            
                            logger.info(f"✅ Found {file_type}: {file_path}")
                            break
                        except Exception as e:
                            logger.error(f"❌ Failed to load {file_path}: {e}")
        
        # Extract and standardize model data
        self.data = discovered_data
        self.models = self._extract_all_models(discovered_data)
        
        logger.info(f"📊 Discovered {len(self.models)} model datasets: {list(self.models.keys())}")
        return discovered_data
    
    def _extract_all_models(self, data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Extract and standardize data for all models from various result files.
        """
        models = {}
        
        # Process evaluation results
        if 'evaluation_results' in data:
            eval_data = data['evaluation_results']
            if isinstance(eval_data, dict):
                for key, value in eval_data.items():
                    if isinstance(value, dict) and 'standard_metrics' in value:
                        model_name = self._identify_model_name(key, value)
                        if model_name:
                            models[model_name] = self._standardize_model_data(value, model_name)
        
        # Process JMLR metrics
        if 'jmlr_metrics' in data:
            jmlr_data = data['jmlr_metrics']
            if isinstance(jmlr_data, dict):
                for key, value in jmlr_data.items():
                    if isinstance(value, dict):
                        model_name = self._identify_model_name(key, value)
                        if model_name and model_name not in models:
                            models[model_name] = self._standardize_model_data(value, model_name)
                        elif model_name and model_name in models:
                            # Merge additional metrics
                            models[model_name].update(self._standardize_model_data(value, model_name))
        
        # Process individual model result files
        for file_key in ['imdb_results', 'yelp_results', 'cifar_results']:
            if file_key in data:
                model_name = file_key.split('_')[0]
                if model_name in ['imdb', 'yelp', 'cifar']:
                    if model_name == 'cifar':
                        model_name = 'cifar10'
                    
                    if model_name not in models:
                        models[model_name] = self._standardize_model_data(data[file_key], model_name)
        
        return models
    
    def _identify_model_name(self, key: str, data: Dict[str, Any]) -> Optional[str]:
        """
        Identify model name from key and data content.
        """
        key_lower = key.lower()
        
        # Direct key matching
        if 'imdb' in key_lower or 'bert' in key_lower:
            return 'imdb'
        elif 'yelp' in key_lower or 'roberta' in key_lower:
            return 'yelp'
        elif 'cifar' in key_lower or 'resnet' in key_lower:
            return 'cifar10'
        
        # Data content analysis
        if isinstance(data, dict):
            model_type = data.get('model_type', '').lower()
            model_class = data.get('model_class', '').lower()
            
            if 'bert' in model_type or 'bert' in model_class:
                if 'imdb' in str(data).lower():
                    return 'imdb'
                elif 'yelp' in str(data).lower():
                    return 'yelp'
            elif 'resnet' in model_type or 'resnet' in model_class:
                return 'cifar10'
        
        return None
    
    def _standardize_model_data(self, data: Dict[str, Any], model_name: str) -> Dict[str, Any]:
        """
        Standardize model data format for consistent visualization.
        """
        standardized = {
            'model_name': model_name,
            'config': self.model_configs.get(model_name, {}),
            'performance_metrics': {},
            'ra_metrics': {},
            'training_info': {},
            'raw_data': data
        }
        
        # Extract performance metrics
        if 'standard_metrics' in data:
            standardized['performance_metrics'] = data['standard_metrics']
        
        # Extract common metrics from top level
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'loss', 'ece', 'brier_score']:
            if metric in data:
                standardized['performance_metrics'][metric] = data[metric]
        
        # Extract RA metrics
        if 'ra_analysis' in data:
            ra_data = data['ra_analysis']
            if 'summary' in ra_data:
                standardized['ra_metrics'] = ra_data['summary']
            if 'detailed_results' in ra_data:
                standardized['detailed_ra_results'] = ra_data['detailed_results']
        
        # Extract RA metrics from top level
        for metric in ['avg_a_flip', 'std_a_flip', 'avg_counter_evidence_count', 
                      'avg_counter_evidence_strength', 'samples_analyzed']:
            if metric in data:
                standardized['ra_metrics'][metric] = data[metric]
        
        # Extract training info
        for key in ['model_type', 'model_class', 'architecture', 'total_parameters', 
                   'training_time', 'epochs', 'batch_size']:
            if key in data:
                standardized['training_info'][key] = data[key]
        
        return standardized
    
    def create_comprehensive_dashboard(self) -> str:
        """
        Create comprehensive multi-model performance dashboard.
        """
        if not self.models:
            logger.warning("⚠️ No model data available for comprehensive dashboard")
            return ""
        
        # Create large figure with multiple subplots
        fig = plt.figure(figsize=(24, 18))
        gs = GridSpec(4, 6, figure=fig, hspace=0.4, wspace=0.3)
        
        models_list = list(self.models.keys())
        colors = [self.model_configs.get(model, {}).get('color', '#666666') for model in models_list]
        
        # 1. Overall Performance Comparison (top row, spans 3 columns)
        ax1 = fig.add_subplot(gs[0, :3])
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        x = np.arange(len(metrics))
        width = 0.25
        
        for i, model in enumerate(models_list):
            model_data = self.models[model]
            perf_metrics = model_data['performance_metrics']
            values = [perf_metrics.get(metric, 0) for metric in metrics]
            
            ax1.bar(x + i * width, values, width, 
                   label=model_data['config'].get('name', model.upper()),
                   color=colors[i], alpha=0.8)
        
        ax1.set_xlabel('Performance Metrics')
        ax1.set_ylabel('Score')
        ax1.set_title('Multi-Model Performance Comparison', fontsize=16, fontweight='bold')
        ax1.set_xticks(x + width)
        ax1.set_xticklabels([m.capitalize() for m in metrics])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for i, model in enumerate(models_list):
            perf_metrics = self.models[model]['performance_metrics']
            for j, metric in enumerate(metrics):
                value = perf_metrics.get(metric, 0)
                if value > 0:
                    ax1.text(j + i * width, value + 0.01, f'{value:.3f}', 
                            ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # 2. Attribution Analysis Comparison (top row, right)
        ax2 = fig.add_subplot(gs[0, 3:])
        ra_metrics_names = ['avg_a_flip', 'avg_counter_evidence_count']
        ra_labels = ['A-Flip Score', 'Counter-Evidence Count']
        
        # Normalize A-Flip scores for visualization
        aflip_values = []
        ce_values = []
        
        for model in models_list:
            ra_data = self.models[model]['ra_metrics']
            aflip_values.append(ra_data.get('avg_a_flip', 0))
            ce_values.append(ra_data.get('avg_counter_evidence_count', 0))
        
        # Create dual-axis plot
        ax2_twin = ax2.twinx()
        
        x_pos = np.arange(len(models_list))
        bars1 = ax2.bar(x_pos - 0.2, aflip_values, 0.4, 
                       color=[c + '80' for c in colors], label='A-Flip Score')
        bars2 = ax2_twin.bar(x_pos + 0.2, ce_values, 0.4,
                            color=[c + 'CC' for c in colors], label='Counter-Evidence Count')
        
        ax2.set_xlabel('Models')
        ax2.set_ylabel('A-Flip Score', color='blue')
        ax2_twin.set_ylabel('Counter-Evidence Count', color='red')
        ax2.set_title('Attribution Analysis Comparison', fontsize=16, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([self.model_configs.get(m, {}).get('name', m.upper()) for m in models_list])
        
        # Add value labels
        for i, (aflip, ce) in enumerate(zip(aflip_values, ce_values)):
            if aflip > 0:
                ax2.text(i - 0.2, aflip + max(aflip_values) * 0.02, f'{aflip:.1f}', 
                        ha='center', va='bottom', fontweight='bold')
            if ce > 0:
                ax2_twin.text(i + 0.2, ce + max(ce_values) * 0.02, f'{ce:.1f}', 
                             ha='center', va='bottom', fontweight='bold')
        
        # 3. Model Architecture Comparison (second row, left)
        ax3 = fig.add_subplot(gs[1, :2])
        ax3.axis('off')
        
        # Create architecture comparison table
        arch_data = []
        for model in models_list:
            model_data = self.models[model]
            config = model_data['config']
            training_info = model_data['training_info']
            
            arch_data.append([
                config.get('name', model.upper()),
                config.get('architecture', 'Unknown'),
                config.get('domain', 'Unknown'),
                f"{training_info.get('total_parameters', 0):,}",
                f"{model_data['performance_metrics'].get('accuracy', 0):.3f}"
            ])
        
        table = ax3.table(cellText=arch_data,
                         colLabels=['Model', 'Architecture', 'Domain', 'Parameters', 'Accuracy'],
                         cellLoc='center', loc='center', bbox=[0.1, 0.3, 0.8, 0.6])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)
        ax3.set_title('Model Architecture Comparison', fontsize=14, fontweight='bold', y=0.9)
        
        # Style table
        for (i, j), cell in table.get_celld().items():
            if i == 0:
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#E6E6E6')
            else:
                model_idx = i - 1
                if model_idx < len(colors):
                    cell.set_facecolor(colors[model_idx] + '20')
        
        # 4. Cross-Domain Performance Analysis (second row, middle-right)
        ax4 = fig.add_subplot(gs[1, 2:4])
        
        # Group models by domain
        text_models = [m for m in models_list if self.model_configs.get(m, {}).get('type') == 'text']
        vision_models = [m for m in models_list if self.model_configs.get(m, {}).get('type') == 'vision']
        
        domain_comparison = []
        if text_models:
            text_acc = np.mean([self.models[m]['performance_metrics'].get('accuracy', 0) for m in text_models])
            text_aflip = np.mean([self.models[m]['ra_metrics'].get('avg_a_flip', 0) for m in text_models])
            domain_comparison.append(['Text Models', text_acc, text_aflip])
        
        if vision_models:
            vision_acc = np.mean([self.models[m]['performance_metrics'].get('accuracy', 0) for m in vision_models])
            vision_aflip = np.mean([self.models[m]['ra_metrics'].get('avg_a_flip', 0) for m in vision_models])
            domain_comparison.append(['Vision Models', vision_acc, vision_aflip])
        
        if domain_comparison:
            domains = [d[0] for d in domain_comparison]
            accuracies = [d[1] for d in domain_comparison]
            aflips = [d[2] for d in domain_comparison]
            
            x_pos = np.arange(len(domains))
            ax4_twin = ax4.twinx()
            
            bars1 = ax4.bar(x_pos - 0.2, accuracies, 0.4, color='#2E86AB', alpha=0.7, label='Accuracy')
            bars2 = ax4_twin.bar(x_pos + 0.2, aflips, 0.4, color='#F18F01', alpha=0.7, label='A-Flip Score')
            
            ax4.set_xlabel('Model Domains')
            ax4.set_ylabel('Accuracy', color='blue')
            ax4_twin.set_ylabel('A-Flip Score', color='orange')
            ax4.set_title('Cross-Domain Performance Analysis', fontsize=14, fontweight='bold')
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(domains)
        
        # 5. Detailed Performance Radar Chart (second row, right)
        ax5 = plt.subplot(gs[1, 4:], projection='polar')
        
        # Radar chart for comprehensive comparison
        categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for i, model in enumerate(models_list[:3]):  # Limit to 3 models for clarity
            model_data = self.models[model]
            perf_metrics = model_data['performance_metrics']
            
            values = [
                perf_metrics.get('accuracy', 0),
                perf_metrics.get('precision', 0),
                perf_metrics.get('recall', 0),
                perf_metrics.get('f1', 0)
            ]
            values += values[:1]  # Complete the circle
            
            ax5.plot(angles, values, 'o-', linewidth=2, 
                    label=model_data['config'].get('name', model.upper()),
                    color=colors[i])
            ax5.fill(angles, values, alpha=0.25, color=colors[i])
        
        ax5.set_xticks(angles[:-1])
        ax5.set_xticklabels(categories)
        ax5.set_ylim(0, 1)
        ax5.set_title('Multi-Model Performance Radar', y=1.08, fontsize=14, fontweight='bold')
        ax5.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        
        # 6. Attribution Stability Analysis (third row)
        ax6 = fig.add_subplot(gs[2, :3])
        
        # A-Flip distribution comparison
        for i, model in enumerate(models_list):
            model_data = self.models[model]
            if 'detailed_ra_results' in model_data:
                detailed_results = model_data['detailed_ra_results']
                if detailed_results:
                    aflip_scores = [r.get('a_flip', 0) for r in detailed_results if 'a_flip' in r]
                    if aflip_scores:
                        ax6.hist(aflip_scores, bins=20, alpha=0.6, 
                               label=model_data['config'].get('name', model.upper()),
                               color=colors[i])
        
        ax6.set_xlabel('A-Flip Score')
        ax6.set_ylabel('Frequency')
        ax6.set_title('Attribution Stability Distribution Comparison', fontsize=14, fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Model Efficiency Analysis (third row, right)
        ax7 = fig.add_subplot(gs[2, 3:])
        
        # Create efficiency scatter plot (accuracy vs model size)
        model_names = []
        accuracies = []
        param_counts = []
        
        for model in models_list:
            model_data = self.models[model]
            model_names.append(model_data['config'].get('name', model.upper()))
            accuracies.append(model_data['performance_metrics'].get('accuracy', 0))
            param_counts.append(model_data['training_info'].get('total_parameters', 0))
        
        scatter = ax7.scatter(param_counts, accuracies, 
                             c=[colors[i] for i in range(len(models_list))],
                             s=200, alpha=0.7)
        
        for i, name in enumerate(model_names):
            ax7.annotate(name, (param_counts[i], accuracies[i]), 
                        xytext=(5, 5), textcoords='offset points', fontweight='bold')
        
        ax7.set_xlabel('Model Parameters')
        ax7.set_ylabel('Accuracy')
        ax7.set_title('Model Efficiency Analysis\n(Accuracy vs Model Size)', fontsize=14, fontweight='bold')
        ax7.grid(True, alpha=0.3)
        
        # Format x-axis for readability
        ax7.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
        
        # 8. Summary Statistics Table (bottom row)
        ax8 = fig.add_subplot(gs[3, :])
        ax8.axis('off')
        
        # Create comprehensive summary table
        summary_data = []
        for model in models_list:
            model_data = self.models[model]
            config = model_data['config']
            perf = model_data['performance_metrics']
            ra = model_data['ra_metrics']
            
            summary_data.append([
                config.get('name', model.upper()),
                f"{perf.get('accuracy', 0):.3f}",
                f"{perf.get('f1', 0):.3f}",
                f"{perf.get('ece', 0):.4f}" if perf.get('ece', 0) > 0 else "N/A",
                f"{ra.get('avg_a_flip', 0):.1f}" if ra.get('avg_a_flip', 0) > 0 else "N/A",
                f"{ra.get('avg_counter_evidence_count', 0):.1f}" if ra.get('avg_counter_evidence_count', 0) > 0 else "N/A",
                f"{ra.get('samples_analyzed', 0)}" if ra.get('samples_analyzed', 0) > 0 else "N/A",
                config.get('task', 'Unknown')
            ])
        
        summary_table = ax8.table(
            cellText=summary_data,
            colLabels=['Model', 'Accuracy', 'F1-Score', 'ECE', 'Avg A-Flip', 'Avg Counter-Ev', 'Samples', 'Task'],
            cellLoc='center',
            loc='center',
            bbox=[0.05, 0.2, 0.9, 0.6]
        )
        
        summary_table.auto_set_font_size(False)
        summary_table.set_fontsize(11)
        summary_table.scale(1.2, 2.0)
        
        # Style summary table
        for (i, j), cell in summary_table.get_celld().items():
            if i == 0:
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#E6E6E6')
            else:
                model_idx = i - 1
                if model_idx < len(colors):
                    cell.set_facecolor(colors[model_idx] + '15')
        
        ax8.set_title('Comprehensive Model Analysis Summary', 
                     fontsize=16, fontweight='bold', y=0.9)
        
        plt.suptitle('Comprehensive Multi-Model Analysis Dashboard\nReverse Attribution Framework', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        # Save the comprehensive dashboard
        output_path = self.subdirs['summary'] / "comprehensive_dashboard"
        for fmt in ['png', 'pdf']:
            plt.savefig(f"{output_path}.{fmt}", format=fmt, bbox_inches='tight', dpi=300)
        
        plt.close()
        logger.info(f"✅ Comprehensive dashboard saved to: {output_path}")
        return str(output_path)
    
    def create_interactive_plotly_dashboard(self) -> str:
        """
        Create interactive Plotly dashboard for comprehensive model analysis.
        """
        if not self.models:
            logger.warning("⚠️ No model data available for interactive dashboard")
            return ""
        
        # Prepare data for interactive visualizations
        models_list = list(self.models.keys())
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=['Performance Comparison', 'Attribution Analysis', 
                          'Cross-Domain Analysis', 'Model Efficiency',
                          'Detailed Metrics Heatmap', 'Training Information'],
            specs=[[{"secondary_y": False}, {"secondary_y": True}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"colspan": 2}, None]],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # 1. Performance Comparison Bar Chart
        for i, model in enumerate(models_list):
            model_data = self.models[model]
            perf = model_data['performance_metrics']
            config = model_data['config']
            
            fig.add_trace(
                go.Bar(
                    name=config.get('name', model.upper()),
                    x=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                    y=[perf.get('accuracy', 0), perf.get('precision', 0), 
                       perf.get('recall', 0), perf.get('f1', 0)],
                    marker_color=config.get('color', '#666666'),
                    text=[f"{perf.get('accuracy', 0):.3f}", f"{perf.get('precision', 0):.3f}",
                          f"{perf.get('recall', 0):.3f}", f"{perf.get('f1', 0):.3f}"],
                    textposition='auto'
                ),
                row=1, col=1
            )
        
        # 2. Attribution Analysis (A-Flip vs Counter-Evidence)
        for i, model in enumerate(models_list):
            model_data = self.models[model]
            ra = model_data['ra_metrics']
            config = model_data['config']
            
            fig.add_trace(
                go.Scatter(
                    name=config.get('name', model.upper()),
                    x=[ra.get('avg_a_flip', 0)],
                    y=[ra.get('avg_counter_evidence_count', 0)],
                    mode='markers+text',
                    marker=dict(size=15, color=config.get('color', '#666666')),
                    text=[config.get('name', model.upper())],
                    textposition="top center"
                ),
                row=1, col=2
            )
        
        # 3. Cross-Domain Comparison
        text_models = [m for m in models_list if self.model_configs.get(m, {}).get('type') == 'text']
        vision_models = [m for m in models_list if self.model_configs.get(m, {}).get('type') == 'vision']
        
        if text_models and vision_models:
            text_acc = np.mean([self.models[m]['performance_metrics'].get('accuracy', 0) for m in text_models])
            vision_acc = np.mean([self.models[m]['performance_metrics'].get('accuracy', 0) for m in vision_models])
            
            fig.add_trace(
                go.Bar(
                    name='Domain Comparison',
                    x=['Text Models', 'Vision Models'],
                    y=[text_acc, vision_acc],
                    marker_color=['#2E86AB', '#F18F01'],
                    text=[f"{text_acc:.3f}", f"{vision_acc:.3f}"],
                    textposition='auto',
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # 4. Model Efficiency Scatter Plot
        param_counts = []
        accuracies = []
        model_names = []
        
        for model in models_list:
            model_data = self.models[model]
            param_counts.append(model_data['training_info'].get('total_parameters', 0))
            accuracies.append(model_data['performance_metrics'].get('accuracy', 0))
            model_names.append(model_data['config'].get('name', model.upper()))
        
        fig.add_trace(
            go.Scatter(
                name='Model Efficiency',
                x=param_counts,
                y=accuracies,
                mode='markers+text',
                marker=dict(
                    size=[15] * len(models_list),
                    color=[self.model_configs.get(m, {}).get('color', '#666666') for m in models_list]
                ),
                text=model_names,
                textposition="top center",
                showlegend=False
            ),
            row=2, col=2
        )
        
        # 5. Comprehensive Metrics Heatmap
        metrics_matrix = []
        metric_names = ['Accuracy', 'F1-Score', 'A-Flip Score', 'Counter-Evidence']
        
        for model in models_list:
            model_data = self.models[model]
            perf = model_data['performance_metrics']
            ra = model_data['ra_metrics']
            
            row_data = [
                perf.get('accuracy', 0),
                perf.get('f1', 0),
                ra.get('avg_a_flip', 0) / 1000 if ra.get('avg_a_flip', 0) > 0 else 0,  # Normalize
                ra.get('avg_counter_evidence_count', 0) / 10 if ra.get('avg_counter_evidence_count', 0) > 0 else 0
            ]
            metrics_matrix.append(row_data)
        
        fig.add_trace(
            go.Heatmap(
                z=metrics_matrix,
                x=metric_names,
                y=[self.model_configs.get(m, {}).get('name', m.upper()) for m in models_list],
                colorscale='RdYlBu_r',
                showscale=True
            ),
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text="Interactive Multi-Model Analysis Dashboard<br><sub>Reverse Attribution Framework</sub>",
                x=0.5,
                font=dict(size=20)
            ),
            height=1200,
            showlegend=True,
            font=dict(size=12)
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Metrics", row=1, col=1)
        fig.update_yaxes(title_text="Score", row=1, col=1)
        
        fig.update_xaxes(title_text="A-Flip Score", row=1, col=2)
        fig.update_yaxes(title_text="Counter-Evidence Count", row=1, col=2)
        
        fig.update_xaxes(title_text="Model Domains", row=2, col=1)
        fig.update_yaxes(title_text="Average Accuracy", row=2, col=1)
        
        fig.update_xaxes(title_text="Model Parameters", row=2, col=2)
        fig.update_yaxes(title_text="Accuracy", row=2, col=2)
        
        # Save interactive dashboard
        output_path = self.subdirs['interactive'] / "interactive_dashboard.html"
        fig.write_html(str(output_path))
        
        logger.info(f"✅ Interactive dashboard saved to: {output_path}")
        return str(output_path)
    
    def create_detailed_model_reports(self) -> Dict[str, str]:
        """
        Create detailed individual reports for each model.
        """
        reports = {}
        
        for model_name, model_data in self.models.items():
            config = model_data['config']
            perf = model_data['performance_metrics']
            ra = model_data['ra_metrics']
            training = model_data['training_info']
            
            # Create detailed report for each model
            report_content = f"""# {config.get('name', model_name.upper())} - Detailed Analysis Report

## Model Information
- **Architecture**: {config.get('architecture', 'Unknown')}
- **Domain**: {config.get('domain', 'Unknown')}
- **Task**: {config.get('task', 'Unknown')}
- **Parameters**: {training.get('total_parameters', 0):,}

## Performance Metrics
| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Accuracy** | {perf.get('accuracy', 0):.4f} | {self._interpret_accuracy(perf.get('accuracy', 0))} |
| **Precision** | {perf.get('precision', 0):.4f} | Precision of positive predictions |
| **Recall** | {perf.get('recall', 0):.4f} | Coverage of actual positive cases |
| **F1-Score** | {perf.get('f1', 0):.4f} | Harmonic mean of precision and recall |
| **ECE** | {perf.get('ece', 0):.4f} | {self._interpret_ece(perf.get('ece', 0))} |
| **Brier Score** | {perf.get('brier_score', 0):.4f} | {self._interpret_brier(perf.get('brier_score', 0))} |

## Reverse Attribution Analysis
| RA Metric | Value | Interpretation |
|-----------|-------|----------------|
| **Average A-Flip Score** | {ra.get('avg_a_flip', 0):.2f} | {self._interpret_aflip(ra.get('avg_a_flip', 0))} |
| **A-Flip Std Deviation** | {ra.get('std_a_flip', 0):.2f} | Variability in attribution stability |
| **Samples Analyzed** | {ra.get('samples_analyzed', 0)} | Number of test samples processed |
| **Avg Counter-Evidence Count** | {ra.get('avg_counter_evidence_count', 0):.2f} | Features contradicting prediction |
| **Avg Counter-Evidence Strength** | {ra.get('avg_counter_evidence_strength', 0):.4f} | Average strength of contradictory evidence |

## Model Training Information
- **Model Type**: {training.get('model_type', 'Unknown')}
- **Model Class**: {training.get('model_class', 'Unknown')}
- **Epochs**: {training.get('epochs', 'Unknown')}
- **Batch Size**: {training.get('batch_size', 'Unknown')}

## Key Insights
{self._generate_model_insights(model_data)}

---
*Report generated by Comprehensive Model Visualizer on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
            
            # Save individual report
            report_path = self.subdirs['summary'] / f"{model_name}_detailed_report.md"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            reports[model_name] = str(report_path)
            logger.info(f"✅ Detailed report for {model_name} saved to: {report_path}")
        
        return reports
    
    def _interpret_accuracy(self, accuracy: float) -> str:
        if accuracy > 0.95:
            return "Excellent performance"
        elif accuracy > 0.9:
            return "Very good performance"
        elif accuracy > 0.8:
            return "Good performance"
        else:
            return "Performance could be improved"
    
    def _interpret_ece(self, ece: float) -> str:
        if ece < 0.05:
            return "Well-calibrated predictions"
        elif ece < 0.1:
            return "Reasonably calibrated"
        else:
            return "Poorly calibrated predictions"
    
    def _interpret_brier(self, brier: float) -> str:
        if brier < 0.02:
            return "Low prediction uncertainty"
        elif brier < 0.1:
            return "Moderate prediction uncertainty"
        else:
            return "High prediction uncertainty"
    
    def _interpret_aflip(self, aflip: float) -> str:
        if aflip < 100:
            return "Very stable attributions"
        elif aflip < 500:
            return "Stable attributions"
        elif aflip < 1000:
            return "Moderate attribution instability"
        else:
            return "High attribution instability"
    
    def _generate_model_insights(self, model_data: Dict[str, Any]) -> str:
        insights = []
        
        config = model_data['config']
        perf = model_data['performance_metrics']
        ra = model_data['ra_metrics']
        
        # Performance insights
        accuracy = perf.get('accuracy', 0)
        if accuracy > 0.93:
            insights.append(f"• The model achieves excellent accuracy ({accuracy:.3f}), demonstrating strong learning capability")
        
        # Domain-specific insights
        if config.get('type') == 'text':
            insights.append("• Text-based model with natural language understanding capabilities")
        elif config.get('type') == 'vision':
            insights.append("• Computer vision model with image recognition capabilities")
        
        # Attribution insights
        aflip = ra.get('avg_a_flip', 0)
        if aflip > 0:
            if aflip > 800:
                insights.append("• Attribution maps show moderate instability, indicating sensitivity to input perturbations")
            else:
                insights.append("• Attribution maps demonstrate good stability")
        
        # Counter-evidence insights
        ce_count = ra.get('avg_counter_evidence_count', 0)
        if ce_count > 3:
            insights.append(f"• High counter-evidence detection ({ce_count:.1f} per sample) shows effective contradictory feature identification")
        
        return '\n'.join(insights) if insights else "• Model shows standard performance characteristics"
    
    def create_cross_domain_analysis(self) -> str:
        """
        Create comprehensive cross-domain analysis comparing text vs vision models.
        """
        text_models = [m for m in self.models.keys() if self.model_configs.get(m, {}).get('type') == 'text']
        vision_models = [m for m in self.models.keys() if self.model_configs.get(m, {}).get('type') == 'vision']
        
        if not text_models or not vision_models:
            logger.warning("⚠️ Need both text and vision models for cross-domain analysis")
            return ""
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Cross-Domain Analysis: Text vs Vision Models', fontsize=18, fontweight='bold')
        
        # 1. Performance Comparison by Domain
        ax1 = axes[0, 0]
        domains = ['Text Models', 'Vision Models']
        
        text_metrics = {
            'accuracy': np.mean([self.models[m]['performance_metrics'].get('accuracy', 0) for m in text_models]),
            'f1': np.mean([self.models[m]['performance_metrics'].get('f1', 0) for m in text_models])
        }
        
        vision_metrics = {
            'accuracy': np.mean([self.models[m]['performance_metrics'].get('accuracy', 0) for m in vision_models]),
            'f1': np.mean([self.models[m]['performance_metrics'].get('f1', 0) for m in vision_models])
        }
        
        x = np.arange(len(domains))
        width = 0.35
        
        ax1.bar(x - width/2, [text_metrics['accuracy'], vision_metrics['accuracy']], 
               width, label='Accuracy', color='#2E86AB', alpha=0.8)
        ax1.bar(x + width/2, [text_metrics['f1'], vision_metrics['f1']], 
               width, label='F1-Score', color='#F18F01', alpha=0.8)
        
        ax1.set_xlabel('Model Domains')
        ax1.set_ylabel('Performance Score')
        ax1.set_title('Performance Comparison by Domain')
        ax1.set_xticks(x)
        ax1.set_xticklabels(domains)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Attribution Stability Comparison
        ax2 = axes[0, 1]
        
        text_aflip = [self.models[m]['ra_metrics'].get('avg_a_flip', 0) for m in text_models if self.models[m]['ra_metrics'].get('avg_a_flip', 0) > 0]
        vision_aflip = [self.models[m]['ra_metrics'].get('avg_a_flip', 0) for m in vision_models if self.models[m]['ra_metrics'].get('avg_a_flip', 0) > 0]
        
        if text_aflip:
            ax2.hist(text_aflip, bins=15, alpha=0.7, label='Text Models', color='#2E86AB')
        if vision_aflip:
            ax2.hist(vision_aflip, bins=15, alpha=0.7, label='Vision Models', color='#F18F01')
        
        ax2.set_xlabel('A-Flip Score')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Attribution Stability Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Model Complexity Comparison
        ax3 = axes[0, 2]
        
        text_params = [self.models[m]['training_info'].get('total_parameters', 0) for m in text_models]
        vision_params = [self.models[m]['training_info'].get('total_parameters', 0) for m in vision_models]
        
        model_names = [self.model_configs.get(m, {}).get('name', m.upper()) for m in text_models + vision_models]
        param_counts = text_params + vision_params
        colors = ['#2E86AB'] * len(text_models) + ['#F18F01'] * len(vision_models)
        
        bars = ax3.bar(model_names, param_counts, color=colors, alpha=0.8)
        ax3.set_xlabel('Models')
        ax3.set_ylabel('Parameter Count')
        ax3.set_title('Model Complexity Comparison')
        ax3.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, count in zip(bars, param_counts):
            if count > 0:
                ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(param_counts) * 0.01,
                        f'{count/1e6:.1f}M', ha='center', va='bottom', fontweight='bold')
        
        # 4. Counter-Evidence Analysis
        ax4 = axes[1, 0]
        
        text_ce = [self.models[m]['ra_metrics'].get('avg_counter_evidence_count', 0) for m in text_models]
        vision_ce = [self.models[m]['ra_metrics'].get('avg_counter_evidence_count', 0) for m in vision_models]
        
        domain_ce = []
        if text_ce and any(ce > 0 for ce in text_ce):
            domain_ce.append(['Text', np.mean([ce for ce in text_ce if ce > 0])])
        if vision_ce and any(ce > 0 for ce in vision_ce):
            domain_ce.append(['Vision', np.mean([ce for ce in vision_ce if ce > 0])])
        
        if domain_ce:
            domains = [d[0] for d in domain_ce]
            ce_values = [d[1] for d in domain_ce]
            
            ax4.bar(domains, ce_values, color=['#2E86AB', '#F18F01'][:len(domains)], alpha=0.8)
            ax4.set_xlabel('Model Domains')
            ax4.set_ylabel('Avg Counter-Evidence Count')
            ax4.set_title('Counter-Evidence Detection by Domain')
            
            for i, (domain, value) in enumerate(domain_ce):
                ax4.text(i, value + max(ce_values) * 0.02, f'{value:.1f}', 
                        ha='center', va='bottom', fontweight='bold')
        
        # 5. Task Complexity Analysis
        ax5 = axes[1, 1]
        
        # Create a summary comparison
        comparison_data = {
            'Text Models': {
                'avg_accuracy': text_metrics['accuracy'],
                'avg_f1': text_metrics['f1'],
                'avg_params': np.mean(text_params) if text_params else 0,
                'model_count': len(text_models)
            },
            'Vision Models': {
                'avg_accuracy': vision_metrics['accuracy'],
                'avg_f1': vision_metrics['f1'],
                'avg_params': np.mean(vision_params) if vision_params else 0,
                'model_count': len(vision_models)
            }
        }
        
        # Create radar chart comparison
        categories = ['Accuracy', 'F1-Score', 'Complexity\n(Normalized)']
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        ax5 = plt.subplot(2, 3, 5, projection='polar')
        
        max_params = max([comparison_data[d]['avg_params'] for d in comparison_data.keys()])
        
        for i, (domain, data) in enumerate(comparison_data.items()):
            values = [
                data['avg_accuracy'],
                data['avg_f1'],
                data['avg_params'] / max_params if max_params > 0 else 0
            ]
            values += values[:1]
            
            color = '#2E86AB' if 'Text' in domain else '#F18F01'
            ax5.plot(angles, values, 'o-', linewidth=2, label=domain, color=color)
            ax5.fill(angles, values, alpha=0.25, color=color)
        
        ax5.set_xticks(angles[:-1])
        ax5.set_xticklabels(categories)
        ax5.set_ylim(0, 1)
        ax5.set_title('Domain Comparison Radar', y=1.08, fontweight='bold')
        ax5.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        
        # 6. Summary Statistics Table
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        summary_stats = []
        for domain, data in comparison_data.items():
            summary_stats.append([
                domain,
                f"{data['avg_accuracy']:.3f}",
                f"{data['avg_f1']:.3f}",
                f"{data['avg_params']/1e6:.1f}M" if data['avg_params'] > 0 else "N/A",
                str(data['model_count'])
            ])
        
        table = ax6.table(
            cellText=summary_stats,
            colLabels=['Domain', 'Avg Accuracy', 'Avg F1', 'Avg Parameters', 'Model Count'],
            cellLoc='center',
            loc='center',
            bbox=[0.1, 0.3, 0.8, 0.4]
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)
        ax6.set_title('Cross-Domain Summary Statistics', fontweight='bold', y=0.8)
        
        # Style table
        for (i, j), cell in table.get_celld().items():
            if i == 0:
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#E6E6E6')
            else:
                color = '#2E86AB20' if 'Text' in summary_stats[i-1][0] else '#F18F0120'
                cell.set_facecolor(color)
        
        plt.tight_layout()
        
        # Save cross-domain analysis
        output_path = self.subdirs['cross_domain'] / "cross_domain_analysis"
        for fmt in ['png', 'pdf']:
            plt.savefig(f"{output_path}.{fmt}", format=fmt, bbox_inches='tight', dpi=300)
        
        plt.close()
        logger.info(f"✅ Cross-domain analysis saved to: {output_path}")
        return str(output_path)
    
    def generate_comprehensive_summary(self) -> str:
        """
        Generate comprehensive summary report with all findings.
        """
        summary_path = self.subdirs['summary'] / "comprehensive_analysis_summary.md"
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("# Comprehensive Multi-Model Analysis Summary\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Executive Summary\n\n")
            f.write(f"This report presents a comprehensive analysis of {len(self.models)} trained models ")
            f.write("across text and vision domains using the Reverse Attribution framework.\n\n")
            
            f.write("## Models Analyzed\n\n")
            for model_name, model_data in self.models.items():
                config = model_data['config']
                f.write(f"### {config.get('name', model_name.upper())}\n")
                f.write(f"- **Architecture**: {config.get('architecture', 'Unknown')}\n")
                f.write(f"- **Domain**: {config.get('domain', 'Unknown')}\n")
                f.write(f"- **Task**: {config.get('task', 'Unknown')}\n")
                f.write(f"- **Parameters**: {model_data['training_info'].get('total_parameters', 0):,}\n")
                f.write(f"- **Accuracy**: {model_data['performance_metrics'].get('accuracy', 0):.3f}\n\n")
            
            f.write("## Key Performance Findings\n\n")
            
            # Find best performing model
            best_model = max(self.models.keys(), 
                           key=lambda x: self.models[x]['performance_metrics'].get('accuracy', 0))
            best_accuracy = self.models[best_model]['performance_metrics'].get('accuracy', 0)
            
            f.write(f"- **Best Overall Performance**: {self.model_configs.get(best_model, {}).get('name', best_model.upper())} ({best_accuracy:.3f} accuracy)\n")
            
            # Attribution analysis summary
            models_with_ra = [m for m in self.models.keys() 
                            if self.models[m]['ra_metrics'].get('avg_a_flip', 0) > 0]
            
            if models_with_ra:
                most_stable = min(models_with_ra, 
                                key=lambda x: self.models[x]['ra_metrics'].get('avg_a_flip', float('inf')))
                stability_score = self.models[most_stable]['ra_metrics'].get('avg_a_flip', 0)
                
                f.write(f"- **Most Stable Attributions**: {self.model_configs.get(most_stable, {}).get('name', most_stable.upper())} (A-Flip: {stability_score:.1f})\n")
            
            # Cross-domain insights
            text_models = [m for m in self.models.keys() if self.model_configs.get(m, {}).get('type') == 'text']
            vision_models = [m for m in self.models.keys() if self.model_configs.get(m, {}).get('type') == 'vision']
            
            if text_models and vision_models:
                text_avg_acc = np.mean([self.models[m]['performance_metrics'].get('accuracy', 0) for m in text_models])
                vision_avg_acc = np.mean([self.models[m]['performance_metrics'].get('accuracy', 0) for m in vision_models])
                
                f.write(f"- **Text Models Average Accuracy**: {text_avg_acc:.3f}\n")
                f.write(f"- **Vision Models Average Accuracy**: {vision_avg_acc:.3f}\n")
            
            f.write("\n## Attribution Analysis Summary\n\n")
            f.write("The Reverse Attribution analysis reveals important insights about model interpretability:\n\n")
            
            for model_name, model_data in self.models.items():
                ra = model_data['ra_metrics']
                if ra.get('avg_a_flip', 0) > 0:
                    config = model_data['config']
                    f.write(f"### {config.get('name', model_name.upper())}\n")
                    f.write(f"- **Attribution Stability**: {self._interpret_aflip(ra.get('avg_a_flip', 0))}\n")
                    f.write(f"- **Counter-Evidence Detection**: {ra.get('avg_counter_evidence_count', 0):.1f} features per sample\n")
                    f.write(f"- **Samples Analyzed**: {ra.get('samples_analyzed', 0)}\n\n")
            
            f.write("## Generated Visualizations\n\n")
            f.write("This analysis generated the following comprehensive visualizations:\n\n")
            f.write("- **Comprehensive Dashboard**: Multi-panel performance and attribution comparison\n")
            f.write("- **Interactive Dashboard**: Plotly-based interactive analysis interface\n")
            f.write("- **Cross-Domain Analysis**: Text vs Vision model comparison\n")
            f.write("- **Individual Model Reports**: Detailed analysis for each model\n")
            f.write("- **Attribution Stability Analysis**: Comprehensive RA framework evaluation\n\n")
            
            f.write("## Recommendations\n\n")
            recommendations = self._generate_comprehensive_recommendations()
            for rec in recommendations:
                f.write(f"- {rec}\n")
            
            f.write(f"\n## File Structure\n\n")
            f.write("```")
            f.write(f"{self.output_dir}/\n")
            for subdir_name, subdir_path in self.subdirs.items():
                f.write(f"├── {subdir_name}/\n")
            f.write("```\n\n")
            
            f.write("---\n")
            f.write("*Report generated by Comprehensive Multi-Model Visualizer*\n")
        
        logger.info(f"✅ Comprehensive summary saved to: {summary_path}")
        return str(summary_path)
    
    def _generate_comprehensive_recommendations(self) -> List[str]:
        recommendations = []
        
        # Performance-based recommendations
        best_model = max(self.models.keys(), 
                        key=lambda x: self.models[x]['performance_metrics'].get('accuracy', 0))
        best_accuracy = self.models[best_model]['performance_metrics'].get('accuracy', 0)
        
        recommendations.append(f"Continue development with {self.model_configs.get(best_model, {}).get('name', best_model.upper())} as the primary model due to superior accuracy ({best_accuracy:.3f})")
        
        # Attribution-based recommendations
        models_with_ra = [m for m in self.models.keys() 
                         if self.models[m]['ra_metrics'].get('avg_a_flip', 0) > 0]
        
        if models_with_ra:
            most_stable = min(models_with_ra, 
                             key=lambda x: self.models[x]['ra_metrics'].get('avg_a_flip', float('inf')))
            recommendations.append(f"Consider {self.model_configs.get(most_stable, {}).get('name', most_stable.upper())} for applications requiring stable and interpretable attributions")
        
        # Cross-domain recommendations
        text_models = [m for m in self.models.keys() if self.model_configs.get(m, {}).get('type') == 'text']
        vision_models = [m for m in self.models.keys() if self.model_configs.get(m, {}).get('type') == 'vision']
        
        if len(text_models) > 1:
            recommendations.append("Investigate ensemble methods combining text models for improved robustness")
        
        if text_models and vision_models:
            recommendations.append("Explore multimodal approaches combining text and vision models for comprehensive analysis")
        
        recommendations.append("Implement regular monitoring of attribution stability for production models")
        recommendations.append("Use counter-evidence analysis for model validation and debugging")
        
        return recommendations
    
    def visualize_all(self) -> Dict[str, str]:
        """
        Generate comprehensive multi-model analysis with all visualizations.
        """
        logger.info("🚀 Starting comprehensive multi-model visualization pipeline...")
        
        results = {}
        
        try:
            # Step 1: Discover all model results
            discovered_data = self.discover_all_results()
            
            if not self.models:
                logger.error("❌ No model data found. Please check your result files.")
                return {}
            
            logger.info(f"✅ Found {len(self.models)} models: {list(self.models.keys())}")
            
            # Step 2: Generate comprehensive dashboard
            dashboard_path = self.create_comprehensive_dashboard()
            if dashboard_path:
                results['comprehensive_dashboard'] = dashboard_path
            
            # Step 3: Generate interactive dashboard
            interactive_path = self.create_interactive_plotly_dashboard()
            if interactive_path:
                results['interactive_dashboard'] = interactive_path
            
            # Step 4: Generate cross-domain analysis
            if len([m for m in self.models.keys() if self.model_configs.get(m, {}).get('type') == 'text']) > 0 and \
               len([m for m in self.models.keys() if self.model_configs.get(m, {}).get('type') == 'vision']) > 0:
                cross_domain_path = self.create_cross_domain_analysis()
                if cross_domain_path:
                    results['cross_domain_analysis'] = cross_domain_path
            
            # Step 5: Generate detailed individual reports
            individual_reports = self.create_detailed_model_reports()
            results.update(individual_reports)
            
            # Step 6: Generate comprehensive summary
            summary_path = self.generate_comprehensive_summary()
            if summary_path:
                results['comprehensive_summary'] = summary_path
            
            logger.info("🎉 All comprehensive visualizations generated successfully!")
            logger.info(f"📁 Output directory: {self.output_dir}")
            
        except Exception as e:
            logger.error(f"❌ Error during comprehensive visualization: {e}")
            raise
        
        return results


def main():
    """Main CLI interface for comprehensive multi-model visualizer."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Multi-Model Visualizer for Reverse Attribution Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Comprehensive analysis with auto-discovery
  python visualizer.py --auto-discover --outdir comprehensive_analysis/
  
  # Analysis with specific input file
  python visualizer.py --input evaluation_results.json --outdir multi_model_dashboard/
  
  # Generate specific visualization types
  python visualizer.py --auto-discover --types dashboard interactive cross-domain --outdir custom_analysis/
        """
    )
    
    parser.add_argument('--input', '-i', type=str,
                       help='Path to specific results JSON file')
    parser.add_argument('--outdir', '-o', type=str, default='comprehensive_analysis',
                       help='Output directory for comprehensive analysis (default: comprehensive_analysis)')
    parser.add_argument('--auto-discover', action='store_true', default=True,
                       help='Automatically discover all model results (default: enabled)')
    parser.add_argument('--types', nargs='+',
                       choices=['dashboard', 'interactive', 'cross-domain', 'reports', 'summary', 'all'],
                       default=['all'],
                       help='Types of analyses to generate (default: all)')
    parser.add_argument('--formats', nargs='+', choices=['png', 'pdf', 'html', 'svg'],
                       default=['png', 'pdf', 'html'],
                       help='Output formats (default: png pdf html)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("🎨 Comprehensive Multi-Model Visualizer")
    print("📊 Advanced Analysis for Reverse Attribution Framework")
    print("🔬 Supporting IMDb BERT, Yelp RoBERTa, and CIFAR-10 ResNet")
    print("=" * 80)
    
    try:
        # Initialize comprehensive visualizer
        visualizer = ComprehensiveModelVisualizer(output_dir=args.outdir)
        visualizer.formats = args.formats
        
        # Load data
        if args.input:
            # Load from specific file
            specific_data = visualizer.discover_all_results([str(Path(args.input).parent)])
        else:
            # Auto-discover all results
            specific_data = visualizer.discover_all_results()
        
        if not visualizer.models:
            print("❌ No model data found!")
            print("   Please ensure your evaluation results contain model performance data.")
            print("   Expected files: evaluation_results.json, jmlr_metrics.json, training_summary.json")
            return 1
        
        # Generate requested analyses
        analysis_types = args.types if 'all' not in args.types else ['dashboard', 'interactive', 'cross-domain', 'reports', 'summary']
        
        print(f"🔍 Discovered {len(visualizer.models)} models: {', '.join(visualizer.models.keys()).upper()}")
        print(f"📈 Generating {len(analysis_types)} analysis types...")
        
        # Generate comprehensive analysis
        results = visualizer.visualize_all()
        
        # Print comprehensive summary
        print(f"\n🎉 Comprehensive Multi-Model Analysis Complete!")
        print("=" * 80)
        print(f"📁 Output directory: {args.outdir}")
        print(f"📊 Generated {len(results)} visualization sets")
        print(f"🎯 Models analyzed: {len(visualizer.models)}")
        print(f"📄 Formats: {', '.join(args.formats).upper()}")
        
        print("\n📋 Generated Analysis Components:")
        for analysis_type, path in results.items():
            if path:
                analysis_name = analysis_type.replace('_', ' ').title()
                print(f"  ✅ {analysis_name}: {path}")
        
        print(f"\n📖 Access your comprehensive analysis:")
        print(f"  📊 Main Dashboard: {args.outdir}/performance_analysis/")
        print(f"  🎛️ Interactive Dashboard: {args.outdir}/interactive_dashboards/")
        print(f"  🔄 Cross-Domain Analysis: {args.outdir}/cross_domain_analysis/")
        print(f"  📝 Summary Reports: {args.outdir}/summary_reports/")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⏹️ Analysis interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"💥 Comprehensive analysis failed: {e}")
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
