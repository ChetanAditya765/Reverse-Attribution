"""
User Study Framework for Reverse Attribution Research
Implements the user study methodology from the JMLR paper including:
- Trust calibration measurement
- Debugging time assessment  
- Interactive explanation interfaces
- Data collection and export utilities
"""

import streamlit as st
import time
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import os
from datetime import datetime
import uuid
import pickle

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ra.ra import ReverseAttribution
from ra.model_factory import ModelFactory
from ra.dataset_utils import DatasetLoader
from explainer_utils import ExplainerHub
from visualizer import ExplanationVisualizer
from metrics import trust_change, debug_time_improvement


class UserStudySession:
    """
    Manages individual user study sessions including data collection,
    timing measurements, and response recording.
    """
    
    def __init__(self, participant_id: str = None, study_type: str = "trust_calibration"):
        self.participant_id = participant_id or str(uuid.uuid4())
        self.study_type = study_type
        self.session_start = datetime.now()
        self.responses = []
        self.timing_data = []
        self.explanation_views = []
        
    def start_task_timer(self) -> str:
        """Start timing a debugging/explanation task."""
        task_id = str(uuid.uuid4())
        self.timing_data.append({
            'task_id': task_id,
            'start_time': time.time(),
            'end_time': None,
            'duration': None
        })
        return task_id
    
    def end_task_timer(self, task_id: str):
        """End timing for a specific task."""
        current_time = time.time()
        for task in self.timing_data:
            if task['task_id'] == task_id:
                task['end_time'] = current_time
                task['duration'] = current_time - task['start_time']
                break
    
    def record_trust_rating(
        self, 
        sample_id: str, 
        pre_explanation_trust: float, 
        post_explanation_trust: float,
        explanation_type: str,
        additional_data: Dict[str, Any] = None
    ):
        """Record trust ratings before and after seeing explanation."""
        response = {
            'sample_id': sample_id,
            'pre_explanation_trust': pre_explanation_trust,
            'post_explanation_trust': post_explanation_trust,
            'trust_change': post_explanation_trust - pre_explanation_trust,
            'explanation_type': explanation_type,
            'timestamp': datetime.now().isoformat(),
            'additional_data': additional_data or {}
        }
        self.responses.append(response)
    
    def record_explanation_view(
        self,
        sample_id: str,
        explanation_method: str,
        view_duration: float,
        user_feedback: Dict[str, Any] = None
    ):
        """Record when user views an explanation."""
        view_record = {
            'sample_id': sample_id,
            'explanation_method': explanation_method,
            'view_duration': view_duration,
            'timestamp': datetime.now().isoformat(),
            'user_feedback': user_feedback or {}
        }
        self.explanation_views.append(view_record)
    
    def export_session_data(self, output_dir: str = "./user_study_data") -> str:
        """Export all session data to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        session_data = {
            'participant_id': self.participant_id,
            'study_type': self.study_type,
            'session_start': self.session_start.isoformat(),
            'session_end': datetime.now().isoformat(),
            'responses': self.responses,
            'timing_data': self.timing_data,
            'explanation_views': self.explanation_views,
            'summary_stats': self._compute_session_summary()
        }
        
        # Save as JSON
        json_path = os.path.join(output_dir, f"session_{self.participant_id}.json")
        with open(json_path, 'w') as f:
            json.dump(session_data, f, indent=2, default=str)
        
        # Save as CSV for easy analysis
        if self.responses:
            df_responses = pd.DataFrame(self.responses)
            csv_path = os.path.join(output_dir, f"responses_{self.participant_id}.csv")
            df_responses.to_csv(csv_path, index=False)
        
        return json_path
    
    def _compute_session_summary(self) -> Dict[str, Any]:
        """Compute summary statistics for the session."""
        summary = {
            'total_responses': len(self.responses),
            'total_tasks': len(self.timing_data),
            'total_explanation_views': len(self.explanation_views)
        }
        
        if self.responses:
            trust_changes = [r['trust_change'] for r in self.responses]
            summary.update({
                'avg_trust_change': np.mean(trust_changes),
                'std_trust_change': np.std(trust_changes),
                'positive_trust_changes': sum(1 for tc in trust_changes if tc > 0),
                'negative_trust_changes': sum(1 for tc in trust_changes if tc < 0)
            })
        
        if self.timing_data:
            completed_tasks = [t for t in self.timing_data if t['duration'] is not None]
            if completed_tasks:
                durations = [t['duration'] for t in completed_tasks]
                summary.update({
                    'avg_task_duration': np.mean(durations),
                    'total_time_spent': sum(durations),
                    'completed_tasks': len(completed_tasks)
                })
        
        return summary


class TrustCalibrationStudy:
    """
    Implements the trust calibration study from the JMLR paper.
    Measures how RA explanations affect human trust in model predictions.
    """
    
    def __init__(
        self,
        model: Any,
        ra_explainer: ReverseAttribution,
        baseline_explainers: ExplainerHub,
        visualizer: Any
    ):
        self.model = model
        self.ra_explainer = ra_explainer
        self.baseline_explainers = baseline_explainers
        self.visualizer = visualizer
        self.current_session = None
        
    def create_study_interface(self):
        """Create Streamlit interface for trust calibration study."""
        st.title("🔬 Trust Calibration Study")
        st.markdown("Help us understand how explanations affect trust in AI predictions")
        
        # Participant setup
        if 'participant_id' not in st.session_state:
            st.session_state.participant_id = st.text_input(
                "Enter Participant ID:", 
                value=str(uuid.uuid4())[:8]
            )
            
            if st.button("Start Study"):
                self.current_session = UserStudySession(
                    st.session_state.participant_id,
                    "trust_calibration"
                )
                st.session_state.study_started = True
                st.rerun()
        
        if st.session_state.get('study_started', False):
            self._run_trust_study()
    
    def _run_trust_study(self):
        """Main trust calibration study workflow."""
        # Study progress tracking
        if 'current_sample' not in st.session_state:
            st.session_state.current_sample = 0
            st.session_state.study_samples = self._prepare_study_samples()
        
        total_samples = len(st.session_state.study_samples)
        current_idx = st.session_state.current_sample
        
        if current_idx >= total_samples:
            self._show_study_completion()
            return
        
        # Show progress
        progress = current_idx / total_samples
        st.progress(progress, text=f"Sample {current_idx + 1} of {total_samples}")
        
        # Get current sample
        sample = st.session_state.study_samples[current_idx]
        
        st.subheader(f"Sample {current_idx + 1}")
        
        # Display the sample (text or image)
        self._display_sample(sample)
        
        # Show model prediction
        prediction = self._get_model_prediction(sample)
        st.info(f"**Model Prediction:** {prediction['predicted_class']} (Confidence: {prediction['confidence']:.2%})")
        
        # Pre-explanation trust rating
        if f'pre_trust_{current_idx}' not in st.session_state:
            st.markdown("### Initial Trust Rating")
            pre_trust = st.slider(
                "How much do you trust this prediction?",
                min_value=1.0, max_value=5.0, value=3.0, step=0.1,
                key=f"pre_trust_slider_{current_idx}",
                help="1 = Very Low Trust, 5 = Very High Trust"
            )
            
            if st.button("Record Initial Trust", key=f"record_pre_{current_idx}"):
                st.session_state[f'pre_trust_{current_idx}'] = pre_trust
                st.rerun()
        
        else:
            pre_trust = st.session_state[f'pre_trust_{current_idx}']
            st.success(f"✅ Initial trust recorded: {pre_trust:.1f}/5.0")
            
            # Show explanation
            self._show_explanations(sample, current_idx)
            
            # Post-explanation trust rating
            if f'post_trust_{current_idx}' not in st.session_state:
                st.markdown("### Updated Trust Rating")
                post_trust = st.slider(
                    "After seeing the explanation, how much do you trust this prediction?",
                    min_value=1.0, max_value=5.0, value=pre_trust, step=0.1,
                    key=f"post_trust_slider_{current_idx}",
                    help="1 = Very Low Trust, 5 = Very High Trust"
                )
                
                # Additional feedback
                feedback = st.text_area(
                    "Optional: Any comments about the explanation?",
                    key=f"feedback_{current_idx}"
                )
                
                if st.button("Submit Response", key=f"submit_{current_idx}"):
                    # Record the response
                    self.current_session.record_trust_rating(
                        sample_id=str(current_idx),
                        pre_explanation_trust=pre_trust,
                        post_explanation_trust=post_trust,
                        explanation_type="reverse_attribution",
                        additional_data={
                            'feedback': feedback,
                            'model_prediction': prediction,
                            'sample_data': sample
                        }
                    )
                    
                    st.session_state[f'post_trust_{current_idx}'] = post_trust
                    st.session_state.current_sample += 1
                    st.rerun()
            
            else:
                # Show next button
                if st.button("Next Sample", key=f"next_{current_idx}"):
                    st.session_state.current_sample += 1
                    st.rerun()
    
    def _prepare_study_samples(self, n_samples: int = 20) -> List[Dict[str, Any]]:
        """Prepare samples for the trust study."""
        # This should be customized based on your specific datasets
        # For now, return dummy samples
        samples = []
        for i in range(n_samples):
            samples.append({
                'id': i,
                'text': f"Sample text {i} for trust calibration study...",
                'true_label': np.random.choice([0, 1]),
                'domain': 'text'
            })
        return samples
    
    def _display_sample(self, sample: Dict[str, Any]):
        """Display a sample to the user."""
        if sample['domain'] == 'text':
            st.markdown("**Text Sample:**")
            st.write(sample['text'])
        elif sample['domain'] == 'image':
            st.markdown("**Image Sample:**")
            # st.image(sample['image'])  # Would display actual image
            st.write("Image display would go here")
    
    def _get_model_prediction(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Get model prediction for a sample."""
        # This should interface with your actual model
        return {
            'predicted_class': f"Class {np.random.choice(['A', 'B'])}",
            'confidence': np.random.beta(2, 2)
        }
    
    def _show_explanations(self, sample: Dict[str, Any], sample_idx: int):
        """Display RA and baseline explanations."""
        st.markdown("### Explanation")
        
        explanation_tab1, explanation_tab2 = st.tabs(["Reverse Attribution", "Baseline Method"])
        
        with explanation_tab1:
            st.markdown("**Reverse Attribution Explanation:**")
            # This would show actual RA visualization
            st.info("🎯 Counter-evidence features identified:")
            st.write("• Feature 1: Strong negative impact (-0.23)")
            st.write("• Feature 2: Moderate negative impact (-0.15)")
            st.write("• Feature 3: Weak negative impact (-0.08)")
            
            st.warning(f"⚠️ Attribution instability score: 0.42 (moderate)")
        
        with explanation_tab2:
            st.markdown("**SHAP Explanation:**")
            # This would show actual SHAP visualization  
            st.info("📊 Top important features:")
            st.write("• Feature A: +0.31")
            st.write("• Feature B: +0.18") 
            st.write("• Feature C: -0.12")
    
    def _show_study_completion(self):
        """Show study completion screen and export data."""
        st.success("🎉 Study Complete!")
        st.markdown("Thank you for participating in our trust calibration study.")
        
        if st.button("Export My Data"):
            if self.current_session:
                export_path = self.current_session.export_session_data()
                st.success(f"Data exported to: {export_path}")
                
                # Show summary statistics
                summary = self.current_session._compute_session_summary()
                st.json(summary)


class DebuggingTimeStudy:
    """
    Measures debugging time with and without RA explanations.
    Implements the debugging time methodology from the JMLR paper.
    """
    
    def __init__(self, model, ra_explainer, baseline_explainers):
        self.model = model
        self.ra_explainer = ra_explainer
        self.baseline_explainers = baseline_explainers
        self.current_session = None
    
    def create_debugging_interface(self):
        """Create interface for debugging time study."""
        st.title("⏱️ Debugging Time Study")
        st.markdown("Help us measure how explanations affect debugging efficiency")
        
        # Participant setup
        if 'debug_participant_id' not in st.session_state:
            st.session_state.debug_participant_id = st.text_input(
                "Enter Participant ID:",
                value=str(uuid.uuid4())[:8],
                key="debug_participant"
            )
            
            if st.button("Start Debugging Study"):
                self.current_session = UserStudySession(
                    st.session_state.debug_participant_id,
                    "debugging_time"
                )
                st.session_state.debug_study_started = True
                st.rerun()
        
        if st.session_state.get('debug_study_started', False):
            self._run_debugging_study()
    
    def _run_debugging_study(self):
        """Main debugging study workflow."""
        # Initialize study state
        if 'debug_phase' not in st.session_state:
            st.session_state.debug_phase = 'without_ra'  # Start without RA
            st.session_state.debug_sample_idx = 0
            st.session_state.current_task_id = None
            st.session_state.debugging_samples = self._prepare_debugging_samples()
        
        phase = st.session_state.debug_phase
        sample_idx = st.session_state.debug_sample_idx
        samples = st.session_state.debugging_samples
        
        if sample_idx >= len(samples):
            if phase == 'without_ra':
                # Move to second phase
                st.session_state.debug_phase = 'with_ra'
                st.session_state.debug_sample_idx = 0
                st.rerun()
            else:
                self._show_debugging_completion()
                return
        
        # Show current phase
        phase_name = "Without Explanations" if phase == 'without_ra' else "With RA Explanations"
        st.subheader(f"Phase: {phase_name}")
        st.write(f"Debug Task {sample_idx + 1} of {len(samples)}")
        
        sample = samples[sample_idx]
        
        # Show the debugging scenario
        self._show_debugging_scenario(sample, phase)
        
        # Timing controls
        if st.session_state.current_task_id is None:
            if st.button(f"Start Debugging Task {sample_idx + 1}"):
                task_id = self.current_session.start_task_timer()
                st.session_state.current_task_id = task_id
                st.rerun()
        else:
            # Show timer
            if f'task_start_time_{sample_idx}' not in st.session_state:
                st.session_state[f'task_start_time_{sample_idx}'] = time.time()
            
            elapsed = time.time() - st.session_state[f'task_start_time_{sample_idx}']
            st.info(f"⏱️ Elapsed time: {elapsed:.1f} seconds")
            
            # Task completion
            task_solution = st.text_area(
                "Describe what you think is wrong with this prediction:",
                key=f"solution_{phase}_{sample_idx}"
            )
            
            confidence = st.slider(
                "How confident are you in your diagnosis?",
                1.0, 5.0, 3.0, 0.1,
                key=f"confidence_{phase}_{sample_idx}"
            )
            
            if st.button("Submit Solution"):
                # End timing
                self.current_session.end_task_timer(st.session_state.current_task_id)
                
                # Record results
                self._record_debugging_result(sample_idx, phase, task_solution, confidence)
                
                # Reset for next task
                st.session_state.current_task_id = None
                st.session_state.debug_sample_idx += 1
                st.rerun()
    
    def _prepare_debugging_samples(self, n_samples: int = 10) -> List[Dict[str, Any]]:
        """Prepare misclassified samples for debugging study."""
        # This should contain actual misclassified examples from your model
        samples = []
        for i in range(n_samples):
            samples.append({
                'id': i,
                'text': f"Misclassified sample {i}...",
                'true_label': 'positive',
                'predicted_label': 'negative',
                'confidence': 0.73,
                'ground_truth_issue': f"Issue description for sample {i}"
            })
        return samples
    
    def _show_debugging_scenario(self, sample: Dict[str, Any], phase: str):
        """Display debugging scenario to user."""
        st.markdown("### Debugging Scenario")
        
        # Show the sample
        st.write("**Text:**", sample['text'])
        st.write(f"**Model predicted:** {sample['predicted_label']} (confidence: {sample['confidence']:.2%})")
        st.write(f"**Ground truth:** {sample['true_label']}")
        st.error("❌ This prediction is incorrect!")
        
        # Show explanations if in RA phase
        if phase == 'with_ra':
            st.markdown("### Available Explanations")
            
            with st.expander("🔬 Reverse Attribution Analysis"):
                st.info("Counter-evidence features detected:")
                st.write("• Word 'not' at position 15: -0.34 suppression")
                st.write("• Phrase 'terrible acting': -0.22 suppression")  
                st.write("• Context negation pattern: -0.18 suppression")
                st.warning("⚠️ High attribution instability: 0.67")
                st.write("**Diagnosis:** Model may be missing negation context")
    
    def _record_debugging_result(
        self, 
        sample_idx: int, 
        phase: str, 
        solution: str, 
        confidence: float
    ):
        """Record debugging task results."""
        # This would store the debugging results for later analysis
        pass
    
    def _show_debugging_completion(self):
        """Show debugging study completion."""
        st.success("🏁 Debugging Study Complete!")
        
        if st.button("Export Debugging Data"):
            if self.current_session:
                export_path = self.current_session.export_session_data()
                st.success(f"Debugging data exported to: {export_path}")


class UserStudyAnalyzer:
    """
    Analyzes collected user study data and computes JMLR paper metrics.
    """
    
    def __init__(self, data_directory: str = "./user_study_data"):
        self.data_directory = data_directory
    
    def load_all_study_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load all user study data files."""
        trust_data = []
        debug_data = []
        
        for filename in os.listdir(self.data_directory):
            if filename.endswith('.json'):
                with open(os.path.join(self.data_directory, filename), 'r') as f:
                    session_data = json.load(f)
                    
                if session_data['study_type'] == 'trust_calibration':
                    trust_data.append(session_data)
                elif session_data['study_type'] == 'debugging_time':
                    debug_data.append(session_data)
        
        return {
            'trust_studies': trust_data,
            'debugging_studies': debug_data
        }
    
    def analyze_trust_calibration(self, trust_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trust calibration results."""
        all_responses = []
        for study in trust_data:
            all_responses.extend(study['responses'])
        
        if not all_responses:
            return {'error': 'No trust data available'}
        
        # Extract trust changes
        trust_changes = [r['trust_change'] for r in all_responses]
        pre_trust = [r['pre_explanation_trust'] for r in all_responses]
        post_trust = [r['post_explanation_trust'] for r in all_responses]
        
        analysis = {
            'num_participants': len(trust_data),
            'num_responses': len(all_responses),
            'avg_trust_change': np.mean(trust_changes),
            'std_trust_change': np.std(trust_changes),
            'positive_changes': sum(1 for tc in trust_changes if tc > 0),
            'negative_changes': sum(1 for tc in trust_changes if tc < 0),
            'neutral_changes': sum(1 for tc in trust_changes if tc == 0),
            'avg_pre_trust': np.mean(pre_trust),
            'avg_post_trust': np.mean(post_trust)
        }
        
        # Statistical significance
        from scipy import stats
        if len(trust_changes) > 1:
            t_stat, p_value = stats.ttest_1samp(trust_changes, 0)
            analysis.update({
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            })
        
        return analysis
    
    def analyze_debugging_time(self, debug_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze debugging time results."""
        times_without_ra = []
        times_with_ra = []
        
        for study in debug_data:
            for task in study['timing_data']:
                if task['duration'] is not None:
                    # This logic would need to be refined based on actual data structure
                    if 'without_ra' in str(task):
                        times_without_ra.append(task['duration'])
                    else:
                        times_with_ra.append(task['duration'])
        
        if not times_without_ra or not times_with_ra:
            return {'error': 'Insufficient debugging time data'}
        
        analysis = debug_time_improvement(times_with_ra, times_without_ra)
        analysis['num_participants'] = len(debug_data)
        
        return analysis
    
    def generate_study_report(self) -> str:
        """Generate comprehensive user study report."""
        study_data = self.load_all_study_data()
        
        report = ["# User Study Analysis Report\n"]
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Trust calibration analysis
        if study_data['trust_studies']:
            trust_analysis = self.analyze_trust_calibration(study_data['trust_studies'])
            report.append("## Trust Calibration Results\n")
            report.append(f"- **Participants:** {trust_analysis['num_participants']}\n")
            report.append(f"- **Total Responses:** {trust_analysis['num_responses']}\n")
            report.append(f"- **Average Trust Change:** {trust_analysis['avg_trust_change']:.3f}\n")
            report.append(f"- **Positive Changes:** {trust_analysis['positive_changes']}\n")
            report.append(f"- **Negative Changes:** {trust_analysis['negative_changes']}\n")
            
            if 'significant' in trust_analysis:
                significance = "Yes" if trust_analysis['significant'] else "No"
                report.append(f"- **Statistically Significant:** {significance} (p={trust_analysis['p_value']:.4f})\n")
            
            report.append("\n")
        
        # Debugging time analysis
        if study_data['debugging_studies']:
            debug_analysis = self.analyze_debugging_time(study_data['debugging_studies'])
            if 'error' not in debug_analysis:
                report.append("## Debugging Time Results\n")
                report.append(f"- **Participants:** {debug_analysis['num_participants']}\n")
                report.append(f"- **Avg Time Without RA:** {debug_analysis['avg_time_without_ra']:.1f}s\n")
                report.append(f"- **Avg Time With RA:** {debug_analysis['avg_time_with_ra']:.1f}s\n")
                report.append(f"- **Time Improvement:** {debug_analysis['improvement_seconds']:.1f}s ({debug_analysis['improvement_percent']:.1f}%)\n")
                
                significance = "Yes" if debug_analysis['is_significant'] else "No"
                report.append(f"- **Statistically Significant:** {significance} (p={debug_analysis['p_value']:.4f})\n")
        
        report_text = ''.join(report)
        
        # Save report
        report_path = os.path.join(self.data_directory, f"study_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        return report_path


def create_user_study_dashboard():
    """
    Main Streamlit dashboard for running user studies.
    """
    st.set_page_config(page_title="RA User Study Dashboard", layout="wide")
    
    st.title("🧪 Reverse Attribution User Study Dashboard")
    
    # Sidebar for study selection
    st.sidebar.title("Study Selection")
    study_type = st.sidebar.selectbox(
        "Choose Study Type:",
        ["Trust Calibration", "Debugging Time", "Data Analysis"]
    )
    
    # Initialize components (these would be loaded with actual models)
    @st.cache_resource
    def load_study_components():
        # Placeholder - would load actual models and explainers
        model = None
        ra_explainer = None
        baseline_explainers = None
        visualizer = None
        return model, ra_explainer, baseline_explainers, visualizer
    
    model, ra_explainer, baseline_explainers, visualizer = load_study_components()
    
    if study_type == "Trust Calibration":
        study = TrustCalibrationStudy(model, ra_explainer, baseline_explainers, visualizer)
        study.create_study_interface()
    
    elif study_type == "Debugging Time":
        study = DebuggingTimeStudy(model, ra_explainer, baseline_explainers)
        study.create_debugging_interface()
    
    elif study_type == "Data Analysis":
        st.subheader("📊 User Study Data Analysis")
        
        analyzer = UserStudyAnalyzer()
        
        if st.button("Analyze Study Data"):
            try:
                study_data = analyzer.load_all_study_data()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Trust Studies")
                    if study_data['trust_studies']:
                        trust_analysis = analyzer.analyze_trust_calibration(study_data['trust_studies'])
                        st.json(trust_analysis)
                    else:
                        st.info("No trust calibration data found")
                
                with col2:
                    st.markdown("### Debugging Studies")
                    if study_data['debugging_studies']:
                        debug_analysis = analyzer.analyze_debugging_time(study_data['debugging_studies'])
                        st.json(debug_analysis)
                    else:
                        st.info("No debugging time data found")
                
                # Generate report
                if st.button("Generate Report"):
                    report_path = analyzer.generate_study_report()
                    st.success(f"Report generated: {report_path}")
                    
            except Exception as e:
                st.error(f"Analysis failed: {e}")


if __name__ == "__main__":
    create_user_study_dashboard()
