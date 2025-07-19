"""
Comprehensive visualization utilities for Reverse Attribution.
Handles rendering of explanations, saliency maps, interactive plots, and comparison views.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
import torch
import cv2
from typing import Dict, List, Any, Optional, Tuple, Union
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from PIL import Image
import base64
from io import BytesIO
import colorcet as cc
from wordcloud import WordCloud


class ExplanationVisualizer:
    """
    Main visualization class for Reverse Attribution explanations.
    Handles both text and image visualization with interactive components.
    """
    
    def __init__(self, color_scheme: str = "RdYlBu_r"):
        self.color_scheme = color_scheme
        self.colors = {
            'positive': '#2E7D32',  # Green
            'negative': '#C62828',  # Red  
            'neutral': '#757575',   # Gray
            'highlight': '#FF9800'  # Orange
        }
        
    def visualize_ra_explanation(
        self,
        ra_result: Dict[str, Any],
        input_data: Any,
        input_type: str = "text",
        tokens: List[str] = None,
        show_details: bool = True,
        interactive: bool = True
    ) -> Any:
        """
        Main visualization function for RA explanations.
        
        Args:
            ra_result: Result from ReverseAttribution.explain()
            input_data: Original input (text or image)
            input_type: Type of input ("text" or "image")
            tokens: Tokenized input for text visualization
            show_details: Whether to show detailed metrics
            interactive: Whether to use interactive plots
            
        Returns:
            Matplotlib figure or Plotly figure based on interactive setting
        """
        if input_type == "text":
            return self._visualize_text_explanation(
                ra_result, input_data, tokens, show_details, interactive
            )
        elif input_type == "image":
            return self._visualize_image_explanation(
                ra_result, input_data, show_details, interactive
            )
        else:
            raise ValueError(f"Unsupported input type: {input_type}")
    
    def _visualize_text_explanation(
        self,
        ra_result: Dict[str, Any],
        text: str,
        tokens: List[str] = None,
        show_details: bool = True,
        interactive: bool = True
    ) -> Any:
        """Visualize RA explanation for text input."""
        
        phi = ra_result['phi']
        counter_evidence = ra_result['counter_evidence']
        a_flip = ra_result['a_flip']
        
        if tokens is None:
            tokens = text.split()  # Simple tokenization
        
        # Ensure phi and tokens have same length
        min_len = min(len(phi), len(tokens))
        phi = phi[:min_len]
        tokens = tokens[:min_len]
        
        if interactive:
            return self._create_interactive_text_plot(
                tokens, phi, counter_evidence, a_flip, show_details
            )
        else:
            return self._create_static_text_plot(
                tokens, phi, counter_evidence, a_flip, show_details
            )
    
    def _create_interactive_text_plot(
        self,
        tokens: List[str],
        attributions: np.ndarray,
        counter_evidence: List[Tuple],
        a_flip: float,
        show_details: bool
    ) -> go.Figure:
        """Create interactive text visualization using Plotly."""
        
        # Create subplot layout
        if show_details:
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=("Attribution Scores", "Counter-Evidence Analysis", "Text Highlights"),
                row_heights=[0.4, 0.3, 0.3],
                vertical_spacing=0.1
            )
        else:
            fig = go.Figure()
        
        # Main attribution plot
        colors = ['red' if attr < 0 else 'green' for attr in attributions]
        intensities = np.abs(attributions)
        
        fig.add_trace(
            go.Bar(
                x=list(range(len(tokens))),
                y=attributions,
                text=tokens,
                textposition='outside',
                marker=dict(
                    color=attributions,
                    colorscale='RdYlGn',
                    cmid=0,
                    colorbar=dict(title="Attribution Score")
                ),
                hovertemplate="<b>%{text}</b><br>Attribution: %{y:.3f}<extra></extra>",
                name="Attributions"
            ),
            row=1 if show_details else None,
            col=1 if show_details else None
        )
        
        if show_details:
            # Counter-evidence highlighting
            ce_indices = [ce[0] for ce in counter_evidence]
            ce_values = [ce[1] for ce in counter_evidence]
            ce_tokens = [tokens[i] if i < len(tokens) else f"Token_{i}" for i in ce_indices]
            
            fig.add_trace(
                go.Scatter(
                    x=ce_indices,
                    y=ce_values,
                    mode='markers+text',
                    marker=dict(
                        size=15,
                        color='red',
                        symbol='x'
                    ),
                    text=ce_tokens,
                    textposition="top center",
                    name="Counter-Evidence",
                    hovertemplate="<b>Counter-Evidence</b><br>Token: %{text}<br>Attribution: %{y:.3f}<extra></extra>"
                ),
                row=2, col=1
            )
            
            # Summary metrics
            fig.add_annotation(
                text=f"A-Flip Score: {a_flip:.3f}<br>Counter-Evidence Found: {len(counter_evidence)}",
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                showarrow=False,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="black",
                borderwidth=1
            )
        
        # Layout updates
        fig.update_layout(
            title="Reverse Attribution Analysis",
            showlegend=show_details,
            height=800 if show_details else 400
        )
        
        fig.update_xaxes(title="Token Index")
        fig.update_yaxes(title="Attribution Score")
        
        return fig
    
    def _create_static_text_plot(
        self,
        tokens: List[str],
        attributions: np.ndarray,
        counter_evidence: List[Tuple],
        a_flip: float,
        show_details: bool
    ) -> plt.Figure:
        """Create static text visualization using Matplotlib."""
        
        if show_details:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(15, 6))
        
        # Main attribution bar plot
        colors = ['red' if attr < 0 else 'green' for attr in attributions]
        bars = ax1.bar(range(len(tokens)), attributions, color=colors, alpha=0.7)
        
        # Add token labels
        for i, (bar, token) in enumerate(zip(bars, tokens)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01 * np.sign(height),
                    token, ha='center', va='bottom' if height > 0 else 'top',
                    rotation=45, fontsize=8)
        
        ax1.set_title(f"Reverse Attribution Analysis (A-Flip: {a_flip:.3f})")
        ax1.set_xlabel("Token Index")
        ax1.set_ylabel("Attribution Score")
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Highlight counter-evidence
        ce_indices = [ce[0] for ce in counter_evidence]
        for idx in ce_indices:
            if idx < len(attributions):
                ax1.axvline(x=idx, color='red', linestyle='--', alpha=0.8)
        
        if show_details:
            # Counter-evidence analysis
            if counter_evidence:
                ce_indices, ce_attrs, ce_deltas = zip(*counter_evidence)
                
                ax2.scatter(ce_indices, ce_deltas, c='red', s=100, alpha=0.7)
                for i, (idx, delta) in enumerate(zip(ce_indices, ce_deltas)):
                    ax2.annotate(f'Token {idx}', (idx, delta), 
                               textcoords="offset points", xytext=(0,10), ha='center')
                
                ax2.set_title("Counter-Evidence Suppression Strength")
                ax2.set_xlabel("Token Index")
                ax2.set_ylabel("Suppression Delta")
                ax2.grid(True, alpha=0.3)
            
            # Text highlighting visualization
            self._create_text_highlight_plot(ax3, tokens, attributions, counter_evidence)
        
        plt.tight_layout()
        return fig
    
    def _create_text_highlight_plot(
        self,
        ax: plt.Axes,
        tokens: List[str],
        attributions: np.ndarray,
        counter_evidence: List[Tuple]
    ):
        """Create text highlighting visualization."""
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 2)
        ax.axis('off')
        
        # Create highlighted text
        x_pos = 0.1
        y_pos = 1.5
        
        ce_indices = set([ce[0] for ce in counter_evidence])
        
        for i, (token, attr) in enumerate(zip(tokens, attributions)):
            # Color based on attribution
            if i in ce_indices:
                color = 'red'
                weight = 'bold'
            elif attr > 0:
                color = 'green'
                weight = 'normal'
            else:
                color = 'gray'
                weight = 'normal'
            
            # Add token with appropriate styling
            ax.text(x_pos, y_pos, token, fontsize=12, color=color, weight=weight)
            
            # Update position
            x_pos += len(token) * 0.15 + 0.1
            if x_pos > 9:
                x_pos = 0.1
                y_pos -= 0.3
        
        ax.set_title("Highlighted Text (Red=Counter-Evidence, Green=Supportive)")
    
    def _visualize_image_explanation(
        self,
        ra_result: Dict[str, Any],
        image: np.ndarray,
        show_details: bool = True,
        interactive: bool = True
    ) -> Any:
        """Visualize RA explanation for image input."""
        
        phi = ra_result['phi'].reshape(image.shape[:2])  # Reshape to image dimensions
        counter_evidence = ra_result['counter_evidence']
        a_flip = ra_result['a_flip']
        
        if interactive:
            return self._create_interactive_image_plot(
                image, phi, counter_evidence, a_flip, show_details
            )
        else:
            return self._create_static_image_plot(
                image, phi, counter_evidence, a_flip, show_details
            )
    
    def _create_static_image_plot(
        self,
        image: np.ndarray,
        attributions: np.ndarray,
        counter_evidence: List[Tuple],
        a_flip: float,
        show_details: bool
    ) -> plt.Figure:
        """Create static image visualization."""
        
        if show_details:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Original image
        ax1.imshow(image)
        ax1.set_title("Original Image")
        ax1.axis('off')
        
        # Attribution heatmap
        im = ax2.imshow(attributions, cmap='RdYlGn', alpha=0.8)
        ax2.imshow(image, alpha=0.3)  # Overlay original image
        ax2.set_title(f"Attribution Heatmap (A-Flip: {a_flip:.3f})")
        ax2.axis('off')
        plt.colorbar(im, ax=ax2, label="Attribution Score")
        
        if show_details:
            # Counter-evidence regions
            ax3.imshow(image)
            
            # Highlight counter-evidence pixels
            ce_mask = np.zeros_like(attributions, dtype=bool)
            for ce_idx, _, _ in counter_evidence[:10]:  # Show top 10
                # Convert flat index to 2D coordinates
                y, x = np.unravel_index(ce_idx, attributions.shape)
                if 0 <= y < attributions.shape[0] and 0 <= x < attributions.shape[1]:
                    ce_mask[max(0, y-2):min(attributions.shape[0], y+3), 
                            max(0, x-2):min(attributions.shape[1], x+3)] = True
            
            ax3.contour(ce_mask, levels=[0.5], colors=['red'], linewidths=2)
            ax3.set_title("Counter-Evidence Regions")
            ax3.axis('off')
            
            # Attribution distribution
            ax4.hist(attributions.flatten(), bins=50, alpha=0.7, edgecolor='black')
            ax4.axvline(x=0, color='red', linestyle='--', label='Zero Attribution')
            ax4.set_title("Attribution Distribution")
            ax4.set_xlabel("Attribution Score")
            ax4.set_ylabel("Frequency")
            ax4.legend()
        
        plt.tight_layout()
        return fig
    
    def _create_interactive_image_plot(
        self,
        image: np.ndarray,
        attributions: np.ndarray,
        counter_evidence: List[Tuple],
        a_flip: float,
        show_details: bool
    ) -> go.Figure:
        """Create interactive image visualization using Plotly."""
        
        if show_details:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("Original Image", "Attribution Heatmap", 
                              "Counter-Evidence", "Attribution Distribution"),
                specs=[[{"type": "xy"}, {"type": "xy"}],
                       [{"type": "xy"}, {"type": "xy"}]]
            )
        else:
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Original Image", "Attribution Heatmap")
            )
        
        # Original image
        fig.add_trace(
            go.Image(z=image),
            row=1, col=1
        )
        
        # Attribution heatmap
        fig.add_trace(
            go.Heatmap(
                z=attributions,
                colorscale='RdYlGn',
                zmid=0,
                name="Attributions",
                hovertemplate="Attribution: %{z:.3f}<extra></extra>"
            ),
            row=1, col=2
        )
        
        if show_details:
            # Counter-evidence scatter plot
            if counter_evidence:
                ce_indices, ce_attrs, ce_deltas = zip(*counter_evidence[:20])  # Top 20
                ce_coords = [np.unravel_index(idx, attributions.shape) for idx in ce_indices]
                ce_y, ce_x = zip(*ce_coords)
                
                fig.add_trace(
                    go.Scatter(
                        x=ce_x, y=ce_y,
                        mode='markers',
                        marker=dict(size=8, color='red', symbol='x'),
                        name="Counter-Evidence",
                        hovertemplate="Position: (%{x}, %{y})<br>Delta: %{customdata:.3f}<extra></extra>",
                        customdata=ce_deltas
                    ),
                    row=2, col=1
                )
            
            # Attribution distribution histogram
            fig.add_trace(
                go.Histogram(
                    x=attributions.flatten(),
                    nbinsx=50,
                    name="Attribution Distribution",
                    opacity=0.7
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title=f"Reverse Attribution Analysis (A-Flip: {a_flip:.3f})",
            showlegend=False,
            height=800 if show_details else 400
        )
        
        return fig
    
    def create_comparison_visualization(
        self,
        ra_result: Dict[str, Any],
        baseline_results: Dict[str, Any],
        input_data: Any,
        input_type: str = "text",
        tokens: List[str] = None
    ) -> Any:
        """
        Create side-by-side comparison of RA vs baseline explanations.
        
        Args:
            ra_result: RA explanation result
            baseline_results: Dictionary of baseline explanation results
            input_data: Original input
            input_type: Type of input
            tokens: Tokenized input for text
            
        Returns:
            Plotly figure with comparison
        """
        n_methods = len(baseline_results) + 1  # +1 for RA
        
        fig = make_subplots(
            rows=1, cols=n_methods,
            subplot_titles=['Reverse Attribution'] + list(baseline_results.keys()),
            horizontal_spacing=0.05
        )
        
        if input_type == "text":
            # RA visualization
            phi = ra_result['phi']
            if tokens is None:
                tokens = input_data.split()
            
            min_len = min(len(phi), len(tokens))
            phi = phi[:min_len]
            
            fig.add_trace(
                go.Bar(
                    x=list(range(len(phi))),
                    y=phi,
                    text=tokens[:len(phi)],
                    name="RA",
                    marker=dict(color=phi, colorscale='RdYlGn', cmid=0)
                ),
                row=1, col=1
            )
            
            # Baseline visualizations
            for i, (method, result) in enumerate(baseline_results.items(), 2):
                if 'attributions' in result:
                    attr = result['attributions'][:min_len]
                    fig.add_trace(
                        go.Bar(
                            x=list(range(len(attr))),
                            y=attr,
                            text=tokens[:len(attr)],
                            name=method,
                            marker=dict(color=attr, colorscale='RdYlGn', cmid=0)
                        ),
                        row=1, col=i
                    )
        
        elif input_type == "image":
            # Image comparison visualization
            phi = ra_result['phi'].reshape(input_data.shape[:2])
            
            fig.add_trace(
                go.Heatmap(z=phi, colorscale='RdYlGn', zmid=0, name="RA"),
                row=1, col=1
            )
            
            for i, (method, result) in enumerate(baseline_results.items(), 2):
                if 'attributions' in result:
                    attr = result['attributions'].reshape(input_data.shape[:2])
                    fig.add_trace(
                        go.Heatmap(z=attr, colorscale='RdYlGn', zmid=0, name=method),
                        row=1, col=i
                    )
        
        fig.update_layout(
            title="Method Comparison",
            showlegend=False,
            height=400
        )
        
        return fig
    
    def create_streamlit_text_visualization(
        self,
        ra_result: Dict[str, Any],
        text: str,
        tokens: List[str] = None
    ):
        """
        Create Streamlit-compatible text visualization with highlighting.
        """
        phi = ra_result['phi']
        counter_evidence = ra_result['counter_evidence']
        
        if tokens is None:
            tokens = text.split()
        
        min_len = min(len(phi), len(tokens))
        phi = phi[:min_len]
        tokens = tokens[:min_len]
        
        # Create HTML with color-coded tokens
        html_parts = []
        ce_indices = set([ce[0] for ce in counter_evidence])
        
        for i, (token, attr) in enumerate(zip(tokens, phi)):
            if i in ce_indices:
                # Counter-evidence: red background
                color = f"background-color: rgba(255, 0, 0, 0.3);"
            elif attr > 0:
                # Positive attribution: green intensity
                intensity = min(abs(attr) * 2, 0.8)
                color = f"background-color: rgba(0, 255, 0, {intensity});"
            elif attr < 0:
                # Negative attribution: red intensity
                intensity = min(abs(attr) * 2, 0.8)
                color = f"background-color: rgba(255, 0, 0, {intensity});"
            else:
                color = ""
            
            html_parts.append(f'<span style="{color}">{token}</span>')
        
        html_text = ' '.join(html_parts)
        
        # Display with legend
        st.markdown("### Text with Attribution Highlighting")
        st.markdown(
            '<div style="padding: 10px; border: 1px solid #ddd; border-radius: 5px;">' +
            html_text + '</div>',
            unsafe_allow_html=True
        )
        
        # Legend
        st.markdown("""
        **Legend:**
        - <span style="background-color: rgba(255, 0, 0, 0.3);">Red highlight</span>: Counter-evidence (suppresses true class)
        - <span style="background-color: rgba(0, 255, 0, 0.5);">Green intensity</span>: Positive attribution
        - <span style="background-color: rgba(255, 0, 0, 0.5);">Red intensity</span>: Negative attribution
        """, unsafe_allow_html=True)
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("A-Flip Score", f"{ra_result['a_flip']:.3f}")
        with col2:
            st.metric("Counter-Evidence Found", len(counter_evidence))
        with col3:
            avg_strength = np.mean([abs(ce[2]) for ce in counter_evidence]) if counter_evidence else 0
            st.metric("Avg Suppression", f"{avg_strength:.3f}")
    
    def export_visualization(
        self,
        fig: Any,
        filename: str,
        format: str = "png",
        width: int = 1200,
        height: int = 800
    ):
        """
        Export visualization to file.
        
        Args:
            fig: Matplotlib or Plotly figure
            filename: Output filename
            format: Export format ('png', 'pdf', 'svg', 'html')
            width: Figure width in pixels
            height: Figure height in pixels
        """
        if hasattr(fig, 'write_image'):  # Plotly figure
            if format == 'html':
                fig.write_html(filename)
            else:
                fig.write_image(filename, format=format, width=width, height=height)
        else:  # Matplotlib figure
            fig.savefig(filename, format=format, dpi=300, bbox_inches='tight')
    
    def create_summary_dashboard(
        self,
        multiple_results: List[Dict[str, Any]],
        labels: List[str] = None
    ) -> go.Figure:
        """
        Create summary dashboard for multiple RA analyses.
        
        Args:
            multiple_results: List of RA results
            labels: Labels for each result
            
        Returns:
            Plotly dashboard figure
        """
        if labels is None:
            labels = [f"Sample {i+1}" for i in range(len(multiple_results))]
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("A-Flip Scores", "Counter-Evidence Counts", 
                          "Attribution Distributions", "Instability Analysis"),
            specs=[[{"type": "xy"}, {"type": "xy"}],
                   [{"type": "xy"}, {"type": "xy"}]]
        )
        
        # A-Flip scores
        a_flip_scores = [result['a_flip'] for result in multiple_results]
        fig.add_trace(
            go.Bar(x=labels, y=a_flip_scores, name="A-Flip Scores"),
            row=1, col=1
        )
        
        # Counter-evidence counts
        ce_counts = [len(result['counter_evidence']) for result in multiple_results]
        fig.add_trace(
            go.Bar(x=labels, y=ce_counts, name="Counter-Evidence"),
            row=1, col=2
        )
        
        # Attribution distributions (violin plot)
        all_attributions = []
        sample_labels = []
        
        for i, (result, label) in enumerate(zip(multiple_results, labels)):
            phi = result['phi']
            all_attributions.extend(phi)
            sample_labels.extend([label] * len(phi))
        
        fig.add_trace(
            go.Violin(x=sample_labels, y=all_attributions, name="Attribution Distribution"),
            row=2, col=1
        )
        
        # Instability vs Performance scatter
        # This would require additional performance metrics
        fig.add_trace(
            go.Scatter(
                x=a_flip_scores,
                y=ce_counts,
                mode='markers+text',
                text=labels,
                textposition="top center",
                name="Instability Analysis",
                hovertemplate="Sample: %{text}<br>A-Flip: %{x:.3f}<br>Counter-Evidence: %{y}<extra></extra>"
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Reverse Attribution Summary Dashboard",
            showlegend=False,
            height=800
        )
        
        return fig


# Utility functions for visualization
def create_word_cloud_from_attributions(
    tokens: List[str],
    attributions: np.ndarray,
    focus_on: str = "negative"
) -> WordCloud:
    """
    Create word cloud highlighting important tokens.
    
    Args:
        tokens: List of tokens
        attributions: Attribution scores
        focus_on: 'positive', 'negative', or 'all'
        
    Returns:
        WordCloud object
    """
    # Filter based on focus
    if focus_on == "negative":
        mask = attributions < 0
    elif focus_on == "positive":
        mask = attributions > 0
    else:
        mask = np.ones(len(attributions), dtype=bool)
    
    # Create frequency dictionary
    frequencies = {}
    for token, attr in zip(tokens, attributions):
        if mask[list(tokens).index(token)]:
            frequencies[token] = abs(attr)
    
    if not frequencies:
        return None
    
    wordcloud = WordCloud(
        width=800, height=400,
        background_color='white',
        colormap='RdYlBu_r' if focus_on == "negative" else 'YlGn'
    ).generate_from_frequencies(frequencies)
    
    return wordcloud


def save_explanation_report(
    ra_result: Dict[str, Any],
    filename: str,
    include_details: bool = True
):
    """
    Save detailed explanation report to file.
    
    Args:
        ra_result: RA explanation result
        filename: Output filename
        include_details: Whether to include detailed analysis
    """
    report_lines = [
        "# Reverse Attribution Analysis Report\n",
        f"Generated on: {pd.Timestamp.now()}\n\n",
        "## Summary Metrics\n",
        f"- **A-Flip Score:** {ra_result['a_flip']:.4f}\n",
        f"- **Predicted Class:** {ra_result['y_hat']}\n",
        f"- **Runner-up Class:** {ra_result['runner_up']}\n",
        f"- **Counter-Evidence Features:** {len(ra_result['counter_evidence'])}\n\n"
    ]
    
    if include_details and ra_result['counter_evidence']:
        report_lines.extend([
            "## Counter-Evidence Analysis\n",
            "| Feature Index | Attribution | Suppression Delta |\n",
            "|---------------|-------------|-------------------|\n"
        ])
        
        for idx, attr, delta in ra_result['counter_evidence'][:10]:
            report_lines.append(f"| {idx} | {attr:.4f} | {delta:.4f} |\n")
        
        report_lines.append("\n")
    
    # Attribution statistics
    phi = ra_result['phi']
    report_lines.extend([
        "## Attribution Statistics\n",
        f"- **Mean Attribution:** {np.mean(phi):.4f}\n",
        f"- **Attribution Std:** {np.std(phi):.4f}\n",
        f"- **Positive Features:** {np.sum(phi > 0)}\n",
        f"- **Negative Features:** {np.sum(phi < 0)}\n",
        f"- **Strong Negative Features (< -0.1):** {np.sum(phi < -0.1)}\n"
    ])
    
    with open(filename, 'w') as f:
        f.writelines(report_lines)


if __name__ == "__main__":
    # Example usage and testing
    
    # Create dummy RA result for testing
    dummy_ra_result = {
        'phi': np.random.randn(20),
        'counter_evidence': [(2, -0.3, -0.15), (7, -0.2, -0.12), (15, -0.25, -0.18)],
        'a_flip': 0.42,
        'y_hat': 1,
        'runner_up': 0
    }
    
    dummy_tokens = ["The", "movie", "was", "not", "very", "good", "and", "the", 
                   "acting", "was", "terrible", "but", "the", "music", "was", 
                   "decent", "overall", "disappointing", "experience", "unfortunately"]
    
    # Test visualizer
    visualizer = ExplanationVisualizer()
    
    # Test static text visualization
    fig = visualizer._create_static_text_plot(
        dummy_tokens, dummy_ra_result['phi'], 
        dummy_ra_result['counter_evidence'], 
        dummy_ra_result['a_flip'], True
    )
    
    print("✅ Visualizer testing completed successfully!")
    
    # Test report generation
    save_explanation_report(dummy_ra_result, "test_report.md")
    print("📄 Test report saved to test_report.md")
