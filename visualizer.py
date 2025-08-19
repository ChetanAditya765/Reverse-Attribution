#!/usr/bin/env python3
"""
Perfect Visualizer for Reverse Attribution Framework
Includes ExplanationVisualizer class for reproduce_results.py compatibility
and comprehensive multi-model analysis capabilities

Usage:
    python visualizer.py --auto-discover --outdir figs/
    python reproduce_results.py  # Uses ExplanationVisualizer class
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

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

# Configure matplotlib for Windows Unicode compatibility
matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['savefig.dpi'] = 300
matplotlib.rcParams['savefig.bbox'] = 'tight'
matplotlib.rcParams['savefig.pad_inches'] = 0.2

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('visualizer.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ExplanationVisualizer:
    """
    Perfect ExplanationVisualizer class for reproduce_results.py compatibility.
    This class provides all the visualization functionality needed by the
    Reverse Attribution framework while maintaining backward compatibility.
    """
    def __init__(self, save_dir: str | Path = "visuals", color_scheme: str = "RdYlBu"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.color_scheme = color_scheme

    def visualize_ra_explanation(
        self,
        ra_result: Dict[str, Any],
        input_data: Any,
        input_type: str = "text",          # "text" | "image"
        tokens: Optional[List[str]] = None,
        show_details: bool = True,
        interactive: bool = True
    ) -> Any:
        """
        Unified entrypoint to visualize Reverse Attribution results.

        Args:
            ra_result: dict returned by ReverseAttribution.explain(...)
                       expected keys: 'phi', 'counter_evidence', 'a_flip', and optionally 'model_type'
            input_data: the original input (string for text, CHW or NCHW tensor/array for image)
            input_type: "text" or "image"
            tokens: optional pre-tokenized text
            show_details: include auxiliary panels/metrics
            interactive: Plotly (True) or Matplotlib (False)

        Returns:
            Plotly/Matplotlib figure OR a dict of artifact paths.
        """
        if input_type == "text":
            return self._visualize_text(ra_result, input_data, tokens, show_details, interactive)
        elif input_type == "image":
            return self._visualize_image(ra_result, input_data, show_details, interactive)
        else:
            raise ValueError(f"Unsupported input_type: {input_type}")

    # ----------------------------- TEXT ---------------------------------
    def _visualize_text(
        self,
        ra_result: Dict[str, Any],
        text: str,
        tokens: Optional[List[str]],
        show_details: bool,
        interactive: bool
    ) -> Any:
        phi = np.array(ra_result.get("phi", []), dtype=float)
        ce  = ra_result.get("counter_evidence", [])
        a_flip = float(ra_result.get("a_flip", 0.0))

        if tokens is None:
            # fallback tokenization
            tokens = text.split()

        n = min(len(tokens), len(phi))
        tokens = tokens[:n]
        phi    = phi[:n]

        title = f"Reverse Attribution (A-Flip: {a_flip:.3f})"

        if interactive and go is not None:
            fig = make_subplots(
                rows=2 if show_details else 1, cols=1,
                subplot_titles=("Token Attributions", "Counter-Evidence (top-k)") if show_details else ("Token Attributions",),
                vertical_spacing=0.12
            )
            # bar for token attributions
            fig.add_bar(x=list(range(n)), y=phi.tolist(), name="φ", row=1, col=1)
            fig.update_xaxes(title_text="token index", row=1, col=1)
            fig.update_yaxes(title_text="attribution (φ)", row=1, col=1)
            if show_details and ce:
                # ce entries can be tuples like (idx, attr, delta)
                idxs = [int(t[0]) for t in ce[: min(len(ce), 20)] if len(t) >= 1]
                deltas = [float(t[2]) if len(t) >= 3 else 0.0 for t in ce[: min(len(ce), 20)]]
                fig.add_scatter(x=idxs, y=deltas, mode="markers", name="counter-evidence Δ", row=2, col=1)
                fig.update_xaxes(title_text="token index", row=2, col=1)
                fig.update_yaxes(title_text="Δ (flip pressure)", row=2, col=1)
            fig.update_layout(title=title, showlegend=False)
            return fig

        # Matplotlib fallback
        if plt is not None:
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.bar(np.arange(n), phi, width=0.8)
            ax.set_title(title)
            ax.set_xlabel("token index")
            ax.set_ylabel("attribution (φ)")
            fig.tight_layout()
            return fig

        # If no plotting backend, just return a simple artifact dict
        out = (self.save_dir / "text_attrib.npy")
        np.save(out, phi)
        return {"heatmap": str(out)}

    # ----------------------------- IMAGE --------------------------------
    def _visualize_image(
        self,
        ra_result: Dict[str, Any],
        x: Any,
        show_details: bool,
        interactive: bool
    ) -> Any:
        phi = np.array(ra_result.get("phi", []), dtype=float)
        a_flip = float(ra_result.get("a_flip", 0.0))
        title = f"Reverse Attribution (A-Flip: {a_flip:.3f})"

        # flatten handling (supports HxW or CxHxW)
        if phi.ndim == 1 and hasattr(x, "shape"):
            # try to infer H, W from input
            if hasattr(x, "shape"):
                if len(x.shape) == 4:  # (N,C,H,W) -> take first
                    _, c, h, w = x.shape
                elif len(x.shape) == 3:  # (C,H,W)
                    c, h, w = x.shape
                else:
                    h = w = int(np.sqrt(len(phi)))
                    c = 1
            else:
                h = w = int(np.sqrt(len(phi)))
                c = 1
            if h * w == len(phi):
                phi = phi.reshape(h, w)

        if interactive and go is not None and phi.ndim == 2:
            fig = go.Figure()
            fig.add_heatmap(z=phi, colorscale=self.color_scheme, zmid=0)
            fig.update_layout(title=title, xaxis_title="x", yaxis_title="y")
            return fig

        if plt is not None and phi.ndim == 2:
            fig, ax = plt.subplots(figsize=(4, 4))
            im = ax.imshow(phi, cmap=self.color_scheme)
            ax.set_title(title)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            return fig

        out = (self.save_dir / "image_attrib.npy")
        np.save(out, phi)
        return {"overlay": str(out)}    
    def __init__(self, output_dir: str = "figs", **kwargs):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create organized subdirectories
        self.subdirs = {
            'performance': self.output_dir / "performance",
            'attribution': self.output_dir / "attribution", 
            'comparison': self.output_dir / "comparison",
            'summary': self.output_dir / "summary",
            'individual': self.output_dir / "individual_models"
        }
        
        for subdir in self.subdirs.values():
            subdir.mkdir(exist_ok=True, parents=True)
        
        self.data = {}
        self.models = {}
        self.formats = ['png', 'pdf']
        
        # Model configuration
        self.model_configs = {
            'imdb': {
                'name': 'IMDb BERT',
                'type': 'text',
                'color': '#2E86AB',
                'architecture': 'BERT-base',
                'domain': 'Natural Language Processing'
            },
            'yelp': {
                'name': 'Yelp RoBERTa', 
                'type': 'text',
                'color': '#A23B72',
                'architecture': 'RoBERTa-base',
                'domain': 'Natural Language Processing'
            },
            'cifar10': {
                'name': 'CIFAR-10 ResNet',
                'type': 'vision',
                'color': '#F18F01',
                'architecture': 'ResNet-56',
                'domain': 'Computer Vision'
            }
        }
        
        logger.info(f"ExplanationVisualizer initialized with output directory: {self.output_dir}")
    
    def load_results(self, file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load evaluation results from files. If no file_path specified,
        auto-discover results in standard locations.
        """
        if file_path:
            return self._load_specific_file(file_path)
        else:
            return self._auto_discover_results()
    
    def _auto_discover_results(self) -> Dict[str, Any]:
        """Auto-discover all result files in standard locations."""
        search_dirs = [".", "reproduction_results", "../reproduction_results", "results"]
        discovered_data = {}
        
        file_patterns = {
            'evaluation_results': ['evaluation_results.json'],
            'jmlr_metrics': ['jmlr_metrics.json'],
            'training_summary': ['training_summary.json'],
            'comprehensive_results': ['comprehensive_evaluation_results']
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
                            
                            logger.info(f"✅ Found {file_type}: {file_path}")
                            break
                        except Exception as e:
                            logger.error(f"❌ Failed to load {file_path}: {e}")
        
        # Extract model data
        self.data = discovered_data
        self.models = self._extract_all_models(discovered_data)
        
        logger.info(f"📊 Discovered {len(self.models)} models: {list(self.models.keys())}")
        return discovered_data
    
    def _load_specific_file(self, file_path: str) -> Dict[str, Any]:
        """Load results from a specific file."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Results file not found: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.data = {'evaluation_results': data}
            self.models = self._extract_all_models(self.data)
            
            logger.info(f"✅ Loaded results from: {file_path}")
            return data
        except Exception as e:
            logger.error(f"❌ Failed to load {file_path}: {e}")
            raise
    
    def _extract_all_models(self, data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Extract and standardize data for all models."""
        models = {}
        
        # Process evaluation results
        if 'evaluation_results' in data:
            eval_data = data['evaluation_results']
            if isinstance(eval_data, dict):
                for key, value in eval_data.items():
                    if isinstance(value, dict):
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
                        if model_name:
                            if model_name not in models:
                                models[model_name] = self._standardize_model_data(value, model_name)
                            else:
                                # Merge additional metrics
                                models[model_name].update(self._standardize_model_data(value, model_name))
        
        return models
    
    def _identify_model_name(self, key: str, data: Dict[str, Any]) -> Optional[str]:
        """Identify model name from key and data content."""
        key_lower = key.lower()
        
        if 'imdb' in key_lower or ('bert' in key_lower and 'imdb' in str(data).lower()):
            return 'imdb'
        elif 'yelp' in key_lower or ('roberta' in key_lower and 'yelp' in str(data).lower()):
            return 'yelp'
        elif 'cifar' in key_lower or 'resnet' in key_lower:
            return 'cifar10'
        
        return None
    
    def _standardize_model_data(self, data: Dict[str, Any], model_name: str) -> Dict[str, Any]:
        """Standardize model data format for consistent visualization."""
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
        
        for metric in ['accuracy', 'loss', 'ece', 'brier_score']:
            if metric in data:
                standardized['performance_metrics'][metric] = data[metric]
        
        # Extract RA metrics
        if 'ra_analysis' in data:
            ra_data = data['ra_analysis']
            if 'summary' in ra_data:
                standardized['ra_metrics'] = ra_data['summary']
            if 'detailed_results' in ra_data:
                standardized['detailed_ra_results'] = ra_data['detailed_results']
        
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
    
    def create_performance_comparison(self) -> str:
        """Create comprehensive performance comparison visualization."""
        if not self.models:
            logger.warning("⚠️ No model data available for performance comparison")
            return ""
        
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.3)
        
        models_list = list(self.models.keys())
        colors = [self.model_configs.get(model, {}).get('color', '#666666') for model in models_list]
        
        # 1. Overall Performance Comparison
        ax1 = fig.add_subplot(gs[0, :2])
        metrics = ['accuracy']
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
        
        # Add value labels
        for i, model in enumerate(models_list):
            perf_metrics = self.models[model]['performance_metrics']
            for j, metric in enumerate(metrics):
                value = perf_metrics.get(metric, 0)
                if value > 0:
                    ax1.text(j + i * width, value + 0.01, f'{value:.3f}', 
                            ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # 2. Attribution Analysis Comparison
        ax2 = fig.add_subplot(gs[0, 2:])
        aflip_values = []
        ce_values = []
        model_names = []
        
        for model in models_list:
            ra_data = self.models[model]['ra_metrics']
            aflip_values.append(ra_data.get('avg_a_flip', 0))
            ce_values.append(ra_data.get('avg_counter_evidence_count', 0))
            model_names.append(self.models[model]['config'].get('name', model.upper()))
        
        x_pos = np.arange(len(models_list))
        ax2_twin = ax2.twinx()
        
        bars1 = ax2.bar(x_pos - 0.2, aflip_values, 0.4, 
                       color=[c + '80' for c in colors], label='A-Flip Score')
        bars2 = ax2_twin.bar(x_pos + 0.2, ce_values, 0.4,
                            color=[c + 'CC' for c in colors], label='Counter-Evidence')
        
        ax2.set_xlabel('Models')
        ax2.set_ylabel('A-Flip Score', color='blue')
        ax2_twin.set_ylabel('Counter-Evidence Count', color='red')
        ax2.set_title('Attribution Analysis Comparison', fontsize=16, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([name.split()[0] for name in model_names])  # Shortened labels
        
        # 3. Model Architecture Summary
        ax3 = fig.add_subplot(gs[1, :])
        ax3.axis('off')
        
        table_data = []
        for model in models_list:
            model_data = self.models[model]
            config = model_data['config']
            perf = model_data['performance_metrics']
            ra = model_data['ra_metrics']
            training = model_data['training_info']
            
            table_data.append([
                config.get('name', model.upper()),
                config.get('architecture', 'Unknown'),
                f"{perf.get('accuracy', 0):.3f}",
                f"{ra.get('avg_a_flip', 0):.1f}" if ra.get('avg_a_flip', 0) > 0 else "N/A",
            ])
        
        table = ax3.table(
            cellText=table_data,
            colLabels=['Model', 'Architecture', 'Accuracy', 'A-Flip'],
            cellLoc='center',
            loc='center',
            bbox=[0.1, 0.3, 0.8, 0.4]
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 2.0)
        ax3.set_title('Model Summary Comparison', fontsize=16, fontweight='bold', y=0.8)
        
        # Style table
        for (i, j), cell in table.get_celld().items():
            if i == 0:
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#E6E6E6')
            else:
                model_idx = i - 1
                if model_idx < len(colors):
                    cell.set_facecolor(colors[model_idx] + '20')
        
        # 4. Cross-Domain Analysis
        ax4 = fig.add_subplot(gs[2, :2])
        text_models = [m for m in models_list if self.model_configs.get(m, {}).get('type') == 'text']
        vision_models = [m for m in models_list if self.model_configs.get(m, {}).get('type') == 'vision']
        
        if text_models and vision_models:
            text_acc = np.mean([self.models[m]['performance_metrics'].get('accuracy', 0) for m in text_models])
            vision_acc = np.mean([self.models[m]['performance_metrics'].get('accuracy', 0) for m in vision_models])
            
            domains = ['Text Models', 'Vision Models']
            accuracies = [text_acc, vision_acc]
            
            bars = ax4.bar(domains, accuracies, color=['#2E86AB', '#F18F01'], alpha=0.7)
            ax4.set_title('Cross-Domain Performance', fontsize=14, fontweight='bold')
            ax4.set_ylabel('Average Accuracy')
            
            for bar, acc in zip(bars, accuracies):
                ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                        f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 5. Attribution Stability Analysis
        ax5 = fig.add_subplot(gs[2, 2:])
        if any(self.models[m]['ra_metrics'].get('avg_a_flip', 0) > 0 for m in models_list):
            stability_data = []
            model_labels = []
            
            for model in models_list:
                aflip = self.models[model]['ra_metrics'].get('avg_a_flip', 0)
                if aflip > 0:
                    stability_data.append(1 / (1 + aflip / 100))  # Convert to stability score
                    model_labels.append(self.models[model]['config'].get('name', model.upper()))
            
            if stability_data:
                bars = ax5.bar(range(len(model_labels)), stability_data, 
                              color=[colors[i] for i in range(len(model_labels))], alpha=0.7)
                ax5.set_title('Attribution Stability Scores', fontsize=14, fontweight='bold')
                ax5.set_ylabel('Stability Score (higher = more stable)')
                ax5.set_xticks(range(len(model_labels)))
                ax5.set_xticklabels([label.split()[0] for label in model_labels])
                
                for bar, score in zip(bars, stability_data):
                    ax5.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                            f'{score:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.suptitle('Comprehensive Multi-Model Analysis Dashboard\nReverse Attribution Framework', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        # Save the visualization
        output_path = self.subdirs['summary'] / "performance_comparison"
        for fmt in self.formats:
            plt.savefig(f"{output_path}.{fmt}", format=fmt, bbox_inches='tight', dpi=300)
        
        plt.close()
        logger.info(f"✅ Performance comparison saved to: {output_path}")
        return str(output_path)
    
    def create_individual_model_reports(self) -> Dict[str, str]:
        """Create detailed visualizations for each individual model."""
        reports = {}
        
        for model_name, model_data in self.models.items():
            config = model_data['config']
            perf = model_data['performance_metrics']
            ra = model_data['ra_metrics']
            
            # Create individual model visualization
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f"{config.get('name', model_name.upper())} - Detailed Analysis", 
                        fontsize=16, fontweight='bold')
            
            # 1. Performance Metrics
            ax1 = axes[0, 0]
            metrics = ['accuracy']
            values = [perf.get(metric, 0) for metric in metrics]
            
            bars = ax1.bar(metrics, values, color=config.get('color', '#666666'), alpha=0.7)
            ax1.set_title('Performance Metrics')
            ax1.set_ylabel('Score')
            ax1.set_ylim(0, 1)
            
            for bar, value in zip(bars, values):
                if value > 0:
                    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                            f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # 2. RA Metrics
            ax2 = axes[0, 1]
            if any(ra.get(metric, 0) > 0 for metric in ['avg_a_flip', 'avg_counter_evidence_count']):
                ra_metrics = ['avg_a_flip', 'avg_counter_evidence_count', 'samples_analyzed']
                ra_values = [ra.get(metric, 0) for metric in ra_metrics]
                ra_labels = ['A-Flip Score', 'Counter-Evidence', 'Samples']
                
                # Normalize for visualization
                normalized_values = []
                for i, value in enumerate(ra_values):
                    if i == 0 and value > 0:  # A-Flip
                        normalized_values.append(value / max(ra_values[0], 1000) * 100)
                    elif i == 1:  # Counter-Evidence
                        normalized_values.append(value * 10)
                    else:  # Samples
                        normalized_values.append(value / max(ra_values[2], 100) * 100)
                
                bars = ax2.bar(ra_labels, normalized_values, color=config.get('color', '#666666'), alpha=0.7)
                ax2.set_title('RA Analysis (Normalized)')
                ax2.set_ylabel('Normalized Score')
                
                for bar, orig_val, norm_val in zip(bars, ra_values, normalized_values):
                    if orig_val > 0:
                        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(normalized_values) * 0.02,
                                f'{orig_val:.1f}', ha='center', va='bottom', fontweight='bold')
            else:
                ax2.text(0.5, 0.5, 'No RA Data Available', ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('RA Analysis')
            
            # 3. Attribution Distribution (if available)
            ax3 = axes[1, 0]
            if 'detailed_ra_results' in model_data:
                detailed_results = model_data['detailed_ra_results']
                aflip_scores = [r.get('a_flip', 0) for r in detailed_results if 'a_flip' in r]
                
                if aflip_scores:
                    ax3.hist(aflip_scores, bins=20, color=config.get('color', '#666666'), alpha=0.7, edgecolor='black')
                    ax3.set_title('A-Flip Distribution')
                    ax3.set_xlabel('A-Flip Score')
                    ax3.set_ylabel('Frequency')
                    
                    mean_aflip = np.mean(aflip_scores)
                    ax3.axvline(mean_aflip, color='red', linestyle='--', 
                               label=f'Mean: {mean_aflip:.1f}')
                    ax3.legend()
                else:
                    ax3.text(0.5, 0.5, 'No A-Flip Data', ha='center', va='center', transform=ax3.transAxes)
                    ax3.set_title('A-Flip Distribution')
            else:
                ax3.text(0.5, 0.5, 'No Detailed RA Data', ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('A-Flip Distribution')
            
            # 4. Model Information
            ax4 = axes[1, 1]
            ax4.axis('off')
            
            info_text = f"""Model Information:

Architecture: {config.get('architecture', 'Unknown')}
Domain: {config.get('domain', 'Unknown')}
Type: {config.get('type', 'Unknown').title()}

Performance:
• Accuracy: {perf.get('accuracy', 0):.3f}

Attribution Analysis:
• A-Flip Score: {ra.get('avg_a_flip', 0):.1f}
• Counter-Evidence: {ra.get('avg_counter_evidence_count', 0):.1f}
• Samples Analyzed: {ra.get('samples_analyzed', 0)}
"""
            
            ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes, fontsize=11,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor=config.get('color', '#666666') + '20'))
            
            plt.tight_layout()
            
            # Save individual model report
            output_path = self.subdirs['individual'] / f"{model_name}_analysis"
            for fmt in self.formats:
                plt.savefig(f"{output_path}.{fmt}", format=fmt, bbox_inches='tight', dpi=300)
            
            plt.close()
            reports[model_name] = str(output_path)
            logger.info(f"✅ Individual analysis for {model_name} saved to: {output_path}")
        
        return reports
    
    def create_attribution_analysis(self) -> str:
        """Create comprehensive attribution analysis visualization."""
        if not self.models:
            return ""
        
        # Filter models with RA data
        models_with_ra = {k: v for k, v in self.models.items() 
                         if v['ra_metrics'].get('avg_a_flip', 0) > 0}
        
        if not models_with_ra:
            logger.warning("⚠️ No models with RA data available")
            return ""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Comprehensive Attribution Analysis', fontsize=16, fontweight='bold')
        
        models_list = list(models_with_ra.keys())
        colors = [self.model_configs.get(model, {}).get('color', '#666666') for model in models_list]
        
        # 1. A-Flip Comparison
        ax1 = axes[0, 0]
        aflip_scores = [models_with_ra[m]['ra_metrics'].get('avg_a_flip', 0) for m in models_list]
        model_names = [models_with_ra[m]['config'].get('name', m.upper()) for m in models_list]
        
        bars = ax1.bar(range(len(models_list)), aflip_scores, color=colors, alpha=0.7)
        ax1.set_title('A-Flip Score Comparison')
        ax1.set_ylabel('A-Flip Score')
        ax1.set_xticks(range(len(models_list)))
        ax1.set_xticklabels([name.split()[0] for name in model_names])
        
        for bar, score in zip(bars, aflip_scores):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(aflip_scores) * 0.02,
                    f'{score:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Counter-Evidence Analysis
        ax2 = axes[0, 1]
        ce_counts = [models_with_ra[m]['ra_metrics'].get('avg_counter_evidence_count', 0) for m in models_list]
        
        bars = ax2.bar(range(len(models_list)), ce_counts, color=colors, alpha=0.7)
        ax2.set_title('Counter-Evidence Count')
        ax2.set_ylabel('Average Count')
        ax2.set_xticks(range(len(models_list)))
        ax2.set_xticklabels([name.split()[0] for name in model_names])
        
        for bar, count in zip(bars, ce_counts):
            if count > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(ce_counts) * 0.02,
                        f'{count:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Attribution Stability Radar
        ax3 = plt.subplot(2, 3, 3, projection='polar')
        
        categories = ['Stability', 'Coverage', 'Reliability']
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        for i, model in enumerate(models_list):
            ra_data = models_with_ra[model]['ra_metrics']
            
            # Normalize metrics for radar chart
            stability = 1 / (1 + ra_data.get('avg_a_flip', 1000) / 1000)
            coverage = min(ra_data.get('samples_analyzed', 0) / 1000, 1.0)
            reliability = min(ra_data.get('avg_counter_evidence_count', 0) / 10, 1.0)
            
            values = [stability, coverage, reliability]
            values += values[:1]
            
            ax3.plot(angles, values, 'o-', linewidth=2, 
                    label=model_names[i].split()[0], color=colors[i])
            ax3.fill(angles, values, alpha=0.25, color=colors[i])
        
        ax3.set_xticks(angles[:-1])
        ax3.set_xticklabels(categories)
        ax3.set_ylim(0, 1)
        ax3.set_title('Attribution Quality Radar', y=1.08)
        ax3.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        
        # 4. Sample Analysis Coverage
        ax4 = axes[1, 0]
        samples_analyzed = [models_with_ra[m]['ra_metrics'].get('samples_analyzed', 0) for m in models_list]
        
        bars = ax4.bar(range(len(models_list)), samples_analyzed, color=colors, alpha=0.7)
        ax4.set_title('Sample Analysis Coverage')
        ax4.set_ylabel('Samples Analyzed')
        ax4.set_xticks(range(len(models_list)))
        ax4.set_xticklabels([name.split()[0] for name in model_names])
        
        for bar, count in zip(bars, samples_analyzed):
            if count > 0:
                ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(samples_analyzed) * 0.02,
                        f'{count}', ha='center', va='bottom', fontweight='bold')
        
        # 5. Attribution Distribution
        ax5 = axes[1, 1]
        for i, model in enumerate(models_list):
            if 'detailed_ra_results' in models_with_ra[model]:
                detailed_results = models_with_ra[model]['detailed_ra_results']
                aflip_scores = [r.get('a_flip', 0) for r in detailed_results if 'a_flip' in r]
                
                if aflip_scores:
                    ax5.hist(aflip_scores, bins=20, alpha=0.6, 
                           label=model_names[i].split()[0], color=colors[i])
        
        ax5.set_title('A-Flip Distribution Overlay')
        ax5.set_xlabel('A-Flip Score')
        ax5.set_ylabel('Frequency')
        ax5.legend()
        
        # 6. Summary Statistics
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        summary_text = "Attribution Analysis Summary:\n\n"
        
        for i, model in enumerate(models_list):
            ra_data = models_with_ra[model]['ra_metrics']
            model_name = model_names[i].split()[0]
            
            summary_text += f"{model_name}:\n"
            summary_text += f"  A-Flip: {ra_data.get('avg_a_flip', 0):.1f}\n"
            summary_text += f"  Counter-Ev: {ra_data.get('avg_counter_evidence_count', 0):.1f}\n"
            summary_text += f"  Samples: {ra_data.get('samples_analyzed', 0)}\n\n"
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        
        # Save attribution analysis
        output_path = self.subdirs['attribution'] / "attribution_analysis"
        for fmt in self.formats:
            plt.savefig(f"{output_path}.{fmt}", format=fmt, bbox_inches='tight', dpi=300)
        
        plt.close()
        logger.info(f"✅ Attribution analysis saved to: {output_path}")
        return str(output_path)
    
    def generate_summary_report(self) -> str:
        """Generate comprehensive markdown summary report."""
        report_path = self.output_dir / "comprehensive_analysis_report.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Comprehensive Reverse Attribution Analysis Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Executive Summary\n\n")
            f.write(f"This report presents a comprehensive analysis of {len(self.models)} trained models ")
            f.write("using the Reverse Attribution framework for explainable AI analysis.\n\n")
            
            if self.models:
                f.write("## Models Analyzed\n\n")
                for model_name, model_data in self.models.items():
                    config = model_data['config']
                    perf = model_data['performance_metrics']
                    
                    f.write(f"### {config.get('name', model_name.upper())}\n")
                    f.write(f"- **Architecture**: {config.get('architecture', 'Unknown')}\n")
                    f.write(f"- **Domain**: {config.get('domain', 'Unknown')}\n")
                    f.write(f"- **Accuracy**: {perf.get('accuracy', 0):.3f}\n")
                
                # Performance insights
                best_model = max(self.models.keys(), 
                               key=lambda x: self.models[x]['performance_metrics'].get('accuracy', 0))
                best_accuracy = self.models[best_model]['performance_metrics'].get('accuracy', 0)
                
                f.write("## Key Findings\n\n")
                f.write(f"- **Best Performance**: {self.model_configs.get(best_model, {}).get('name', best_model.upper())} ({best_accuracy:.3f} accuracy)\n")
                
                # Attribution insights
                models_with_ra = [m for m in self.models.keys() 
                                if self.models[m]['ra_metrics'].get('avg_a_flip', 0) > 0]
                
                if models_with_ra:
                    most_stable = min(models_with_ra, 
                                    key=lambda x: self.models[x]['ra_metrics'].get('avg_a_flip', float('inf')))
                    stability_score = self.models[most_stable]['ra_metrics'].get('avg_a_flip', 0)
                    
                    f.write(f"- **Most Stable Attributions**: {self.model_configs.get(most_stable, {}).get('name', most_stable.upper())} (A-Flip: {stability_score:.1f})\n")
                
                f.write(f"- **Total Models Analyzed**: {len(self.models)}\n")
                f.write(f"- **Models with RA Data**: {len(models_with_ra)}\n")
            
            f.write("\n## Generated Visualizations\n\n")
            f.write("- Performance Comparison Dashboard\n")
            f.write("- Individual Model Analysis Reports\n")
            f.write("- Attribution Analysis Visualization\n")
            f.write("- Comprehensive Summary Report (this document)\n\n")
            
            f.write("---\n")
            f.write("*Generated by ExplanationVisualizer - Reverse Attribution Framework*\n")
        
        logger.info(f"✅ Summary report saved to: {report_path}")
        return str(report_path)
    
    def visualize_all(self, auto_discover: bool = True) -> Dict[str, str]:
        """
        Generate all visualizations. This is the main method called by reproduce_results.py
        and also supports the CLI interface.
        """
        logger.info("🚀 Starting comprehensive visualization pipeline...")
        
        results = {}
        
        try:
            # Load data
            if auto_discover:
                self._auto_discover_results()
            
            if not self.models:
                logger.error("❌ No model data found for visualization")
                return {}
            
            logger.info(f"✅ Found {len(self.models)} models: {list(self.models.keys())}")
            
            # Generate all visualizations
            
            # 1. Performance comparison
            perf_path = self.create_performance_comparison()
            if perf_path:
                results['performance_comparison'] = perf_path
            
            # 2. Individual model reports
            individual_reports = self.create_individual_model_reports()
            results.update(individual_reports)
            
            # 3. Attribution analysis
            attr_path = self.create_attribution_analysis()
            if attr_path:
                results['attribution_analysis'] = attr_path
            
            # 4. Summary report
            summary_path = self.generate_summary_report()
            if summary_path:
                results['summary_report'] = summary_path
            
            logger.info("🎉 All visualizations generated successfully!")
            logger.info(f"📁 Output directory: {self.output_dir}")
            
        except Exception as e:
            logger.error(f"❌ Error during visualization: {e}")
            raise
        
        return results
    
    # Legacy method names for backward compatibility
    def create_explanation_plots(self, *args, **kwargs):
        """Legacy method for backward compatibility."""
        return self.visualize_all(*args, **kwargs)
    
    def plot_results(self, *args, **kwargs):
        """Legacy method for backward compatibility."""
        return self.visualize_all(*args, **kwargs)
    
    def save_plots(self, output_dir: str = None):
        """Legacy method for backward compatibility."""
        if output_dir:
            self.output_dir = Path(output_dir)
        return self.visualize_all()


def main():
    """Main CLI interface for the visualizer."""
    parser = argparse.ArgumentParser(
        description="Perfect Visualizer for Reverse Attribution Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-discover and visualize all results
  python visualizer.py --auto-discover --outdir figs/
  
  # Visualize specific results file
  python visualizer.py --input evaluation_results.json --outdir analysis/
  
  # Generate with verbose logging
  python visualizer.py --auto-discover --verbose --outdir comprehensive/
        """
    )
    
    parser.add_argument('--input', '-i', type=str,
                       help='Path to specific results JSON file')
    parser.add_argument('--outdir', '-o', type=str, default='figs',
                       help='Output directory for visualizations (default: figs)')
    parser.add_argument('--auto-discover', action='store_true', default=True,
                       help='Automatically discover result files (default: enabled)')
    parser.add_argument('--formats', nargs='+', choices=['png', 'pdf', 'svg'],
                       default=['png', 'pdf'],
                       help='Output formats (default: png pdf)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("🎨 Perfect Visualizer for Reverse Attribution Framework")
    print("📊 Compatible with reproduce_results.py and comprehensive analysis")
    print("=" * 80)
    
    try:
        # Initialize visualizer
        visualizer = ExplanationVisualizer(output_dir=args.outdir)
        visualizer.formats = args.formats
        
        # Load data
        if args.input:
            visualizer.load_results(args.input)
        else:
            visualizer.load_results()  # Auto-discover
        
        if not visualizer.models:
            print("❌ No model data found!")
            print("   Please ensure your evaluation results contain model performance data.")
            print("   Expected files: evaluation_results.json, jmlr_metrics.json")
            return 1
        
        # Generate visualizations
        results = visualizer.visualize_all(auto_discover=False)  # Data already loaded
        
        # Print summary
        print(f"\n🎉 Visualization Generation Complete!")
        print("=" * 80)
        print(f"📁 Output directory: {args.outdir}")
        print(f"📊 Generated {len(results)} visualization sets")
        print(f"🎯 Models analyzed: {len(visualizer.models)}")
        print(f"📄 Formats: {', '.join(args.formats).upper()}")
        
        print(f"\n🔍 Discovered models: {', '.join(visualizer.models.keys()).upper()}")
        
        print("\n📋 Generated Visualizations:")
        for viz_type, path in results.items():
            if path:
                viz_name = viz_type.replace('_', ' ').title()
                print(f"  ✅ {viz_name}: {path}")
        
        print(f"\n📖 Access your analysis:")
        print(f"  📊 Performance Dashboard: {args.outdir}/summary/")
        print(f"  📝 Individual Reports: {args.outdir}/individual_models/")
        print(f"  🎯 Attribution Analysis: {args.outdir}/attribution/")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⏹️ Visualization interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"💥 Visualization failed: {e}")
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
