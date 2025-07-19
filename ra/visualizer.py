"""
Visualization helpers for the Reverse-Attribution framework.

Highlights
----------
1.  Works with **BERTSentimentClassifier** and **ResNetCIFAR** out-of-the-box.  
2.  Produces token-level heat-maps for text and pixel/segment overlays for images.  
3.  Accepts raw RA outputs, Captum/SHAP/LIME attributions, or blended scores.  
4.  Saves interactive HTML (Plotly) and static PNG/PDF (Matplotlib/Seaborn).  
5.  Designed to be imported, but can be executed as a CLI tool:
    `python visualizer.py --input ra_output.json --outdir figs/`.
"""

from __future__ import annotations
import json
import os
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.io as pio
import torch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Text utilities
# --------------------------------------------------------------------------- #
def _draw_text_heatmap(tokens: List[str],
                       scores: np.ndarray,
                       title: str,
                       save_path: Path | None = None) -> None:
    """Colour each token by attribution score."""
    norm = (scores - scores.min()) / (scores.ptp() + 1e-9)
    cmap = sns.color_palette("coolwarm", as_cmap=True)

    fig, ax = plt.subplots(figsize=(max(6, len(tokens) * .4), 1.8))
    ax.axis("off")

    x = 0.0
    for tok, s in zip(tokens, norm):
        width = .015 * len(tok) + .03
        rect = plt.Rectangle((x, 0), width, 1, color=cmap(s))
        ax.add_patch(rect)
        ax.text(x + width / 2, .5, tok,
                ha="center", va="center", fontsize=10, rotation=0,
                color="white" if s > .5 else "black")
        x += width

    ax.set_xlim(0, x)
    ax.set_ylim(0, 1)
    plt.title(title, fontsize=12)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# --------------------------------------------------------------------------- #
# Vision utilities
# --------------------------------------------------------------------------- #
def _overlay_heatmap(img: np.ndarray,
                     heat: np.ndarray,
                     alpha: float = .5) -> np.ndarray:
    """Return RGB array with heat-map overlay."""
    from matplotlib import cm
    heat_norm = (heat - heat.min()) / (heat.ptp() + 1e-9)
    heat_rgb = cm.jet(heat_norm)[..., :3]          # drop alpha
    overlay = (1 - alpha) * img + alpha * heat_rgb
    return np.clip(overlay, 0, 1)


def plot_image_explanation(img_tensor: torch.Tensor,
                           attribution: np.ndarray,
                           title: str,
                           save_path: Path | None = None,
                           show_original: bool = True) -> None:
    """
    Show original image and heat-map overlay.

    Parameters
    ----------
    img_tensor : (3, H, W) float tensor in [0,1] range **after de-norm**.
    attribution: flattened or (H,W) array of scores.
    """
    img = img_tensor.cpu().permute(1, 2, 0).numpy()
    h, w = img.shape[:2]
    heat = attribution.reshape(h, w)

    overlay = _overlay_heatmap(img, heat)

    fig, axs = plt.subplots(1, 2 if show_original else 1,
                            figsize=(6 if show_original else 4.5, 4.5))

    if show_original:
        axs[0].imshow(img); axs[0].set_title("Original"); axs[0].axis("off")
        ax = axs[1]
    else:
        ax = axs

    ax.imshow(overlay); ax.set_title(title); ax.axis("off")
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# --------------------------------------------------------------------------- #
# Interactive Plotly utilities
# --------------------------------------------------------------------------- #
def tokens_plotly(tokens: List[str],
                  scores: np.ndarray,
                  html_path: Path) -> None:
    """Interactive bar chart of token attributions."""
    norm = (scores - scores.min()) / (scores.ptp() + 1e-9)
    fig = go.Figure(go.Bar(
        x=list(range(len(tokens))),
        y=scores,
        text=tokens,
        marker=dict(color=norm, colorscale="RdBu"),
        hovertemplate="<b>%{text}</b><br>score=%{y:.3f}<extra></extra>"
    ))
    fig.update_layout(xaxis_title="Token index", yaxis_title="Attribution")
    html_path.parent.mkdir(parents=True, exist_ok=True)
    pio.write_html(fig, file=str(html_path), auto_open=False)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def visualize_ra_output(ra_json: Dict[str, Any],
                        tokenizer=None,
                        save_dir: str | Path = "figs") -> Dict[str, str]:
    """
    Create visual artefacts for one RA explanation result.

    Returns dict mapping figure type → file path.
    """
    save_dir = Path(save_dir)
    artefacts: Dict[str, str] = {}

    if ra_json["model_type"] in {"bert_sentiment", "custom_text"}:
        idx_to_tok = (tokenizer.convert_ids_to_tokens
                      if tokenizer else (lambda ids: [f"id_{i}" for i in ids]))
        tokens = idx_to_tok(ra_json["input_ids"])
        scores = np.asarray(ra_json["phi"])
        fname = save_dir / f"text_heat_{ra_json['sample_id']}.png"
        _draw_text_heatmap(tokens, scores,
                           title=f"A-Flip: {ra_json['a_flip']:.3f}",
                           save_path=fname)
        artefacts["text_heatmap"] = str(fname)

        # interactive
        html = save_dir / f"text_heat_{ra_json['sample_id']}.html"
        tokens_plotly(tokens, scores, html)
        artefacts["text_html"] = str(html)

    else:  # vision
        img = torch.tensor(ra_json["input"]).float()  # (3,H,W) already de-norm
        phi = np.asarray(ra_json["phi"])
        fname = save_dir / f"img_overlay_{ra_json['sample_id']}.png"
        plot_image_explanation(img, phi,
                               title=f"A-Flip: {ra_json['a_flip']:.3f}",
                               save_path=fname)
        artefacts["image_explanation"] = str(fname)

    return artefacts


# --------------------------------------------------------------------------- #
# Additional visualization functions
# --------------------------------------------------------------------------- #
def create_word_cloud_from_attributions(tokens: List[str], 
                                       attributions: np.ndarray,
                                       save_path: Path | None = None) -> None:
    """Create word cloud from attribution scores."""
    try:
        from wordcloud import WordCloud
        
        # Create frequency dict from attributions
        word_freq = {}
        for token, attr in zip(tokens, attributions):
            # Use absolute attribution as frequency, minimum of 1
            word_freq[token] = max(1, int(abs(attr) * 100))
        
        wordcloud = WordCloud(
            width=800, 
            height=400,
            background_color='white',
            colormap='coolwarm',
            max_words=100
        ).generate_from_frequencies(word_freq)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Attribution Word Cloud')
        
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
        
    except ImportError:
        logger.warning("WordCloud not available. Install with: pip install wordcloud")


def save_explanation_report(explanations: Dict[str, Any],
                          save_path: Path | None = None) -> str:
    """Generate comprehensive explanation report."""
    
    if save_path is None:
        save_path = Path("explanation_report.html")
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Reverse Attribution Explanation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background-color: #f0f0f0; padding: 10px; }}
            .section {{ margin: 20px 0; }}
            .metric {{ display: inline-block; margin: 10px; padding: 10px; 
                      background-color: #e8f4fd; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Reverse Attribution Explanation Report</h1>
            <p>Generated for model: {explanations.get('model_type', 'Unknown')}</p>
        </div>
        
        <div class="section">
            <h2>Key Metrics</h2>
            <div class="metric">
                <strong>A-Flip Score:</strong> {explanations.get('a_flip', 'N/A'):.4f}
            </div>
            <div class="metric">
                <strong>Predicted Class:</strong> {explanations.get('y_hat', 'N/A')}
            </div>
            <div class="metric">
                <strong>Runner-up Class:</strong> {explanations.get('runner_up', 'N/A')}
            </div>
            <div class="metric">
                <strong>Counter-Evidence Features:</strong> {len(explanations.get('counter_evidence', []))}
            </div>
        </div>
        
        <div class="section">
            <h2>Counter-Evidence Analysis</h2>
            <table border="1" style="border-collapse: collapse; width: 100%;">
                <tr>
                    <th>Rank</th>
                    <th>Feature Index</th>
                    <th>Attribution</th>
                    <th>Delta (Suppression)</th>
                </tr>
    """
    
    # Add counter-evidence table rows
    for i, (feat_idx, attr, delta) in enumerate(explanations.get('counter_evidence', [])):
        html_content += f"""
                <tr>
                    <td>{i+1}</td>
                    <td>{feat_idx}</td>
                    <td>{attr:.4f}</td>
                    <td>{delta:.4f}</td>
                </tr>
        """
    
    html_content += """
            </table>
        </div>
    </body>
    </html>
    """
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"Explanation report saved to: {save_path}")
    return str(save_path)


class ExplanationVisualizer:
    """
    Main visualization class that integrates with your model implementations.
    """
    
    def __init__(self, save_dir: str = "visualizations"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        logger.info(f"ExplanationVisualizer initialized with save_dir: {self.save_dir}")
    
    def visualize_text_explanation(self,
                                 tokens: List[str],
                                 attributions: np.ndarray,
                                 title: str = "Text Attribution",
                                 save_name: str = None) -> Dict[str, str]:
        """Visualize text explanation with multiple formats."""
        
        if save_name is None:
            save_name = "text_explanation"
        
        results = {}
        
        # Static heatmap
        heatmap_path = self.save_dir / f"{save_name}_heatmap.png"
        _draw_text_heatmap(tokens, attributions, title, heatmap_path)
        results['heatmap'] = str(heatmap_path)
        
        # Interactive plot
        html_path = self.save_dir / f"{save_name}_interactive.html"
        tokens_plotly(tokens, attributions, html_path)
        results['interactive'] = str(html_path)
        
        # Word cloud
        wordcloud_path = self.save_dir / f"{save_name}_wordcloud.png"
        create_word_cloud_from_attributions(tokens, attributions, wordcloud_path)
        results['wordcloud'] = str(wordcloud_path)
        
        logger.info(f"Text visualization saved: {len(results)} files")
        return results
    
    def visualize_image_explanation(self,
                                  image: torch.Tensor,
                                  attributions: np.ndarray,
                                  title: str = "Image Attribution",
                                  save_name: str = None) -> Dict[str, str]:
        """Visualize image explanation."""
        
        if save_name is None:
            save_name = "image_explanation"
        
        results = {}
        
        # Overlay visualization
        overlay_path = self.save_dir / f"{save_name}_overlay.png"
        plot_image_explanation(image, attributions, title, overlay_path)
        results['overlay'] = str(overlay_path)
        
        logger.info(f"Image visualization saved: {len(results)} files")
        return results
    
    def visualize_ra_explanation(self,
                               ra_results: Dict[str, Any],
                               input_data: Any = None,
                               tokenizer: Any = None) -> Dict[str, str]:
        """Visualize complete RA explanation results."""
        
        model_type = ra_results.get('model_type', 'unknown')
        sample_id = ra_results.get('sample_id', 'unknown')
        
        all_results = {}
        
        if model_type in ['bert_sentiment', 'custom_text'] and tokenizer is not None:
            # Text visualization
            if 'input_ids' in ra_results:
                tokens = tokenizer.convert_ids_to_tokens(ra_results['input_ids'])
            else:
                tokens = [f"token_{i}" for i in range(len(ra_results['phi']))]
                
            attributions = np.array(ra_results['phi'])
            title = f"RA Explanation - A-Flip: {ra_results['a_flip']:.3f}"
            
            text_results = self.visualize_text_explanation(
                tokens, attributions, title, f"ra_{sample_id}_text"
            )
            all_results.update(text_results)
            
        elif model_type in ['resnet_cifar', 'custom_vision'] and input_data is not None:
            # Image visualization
            attributions = np.array(ra_results['phi'])
            title = f"RA Explanation - A-Flip: {ra_results['a_flip']:.3f}"
            
            image_results = self.visualize_image_explanation(
                input_data, attributions, title, f"ra_{sample_id}_image"
            )
            all_results.update(image_results)
        
        # Generate HTML report
        report_path = self.save_dir / f"ra_{sample_id}_report.html"
        save_explanation_report(ra_results, report_path)
        all_results['report'] = str(report_path)
        
        logger.info(f"Complete RA visualization saved: {len(all_results)} files")
        return all_results


# CLI entry-point ----------------------------------------------------------- #
def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="RA JSON file or dir with *.json")
    p.add_argument("--outdir", default="figs", help="Output directory")
    args = p.parse_args()

    paths = [Path(args.input)]
    if paths[0].is_dir():
        paths = list(paths[0].glob("*.json"))

    for jpath in paths:
        ra_json = json.loads(Path(jpath).read_text())
        visualize_ra_output(ra_json, save_dir=args.outdir)
        logger.info(f"Visualized {jpath}")


if __name__ == "__main__":
    _cli()
