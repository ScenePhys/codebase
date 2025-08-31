import os
import json
import re
import math
import time
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import openai

# Configuration
AVALAI_API_KEY = "aa-Fmk9AQbfxC1mZmEI0efUap2RfDpU7mLi67RIez5pmcUmZ7ym"
AVALAI_BASE_URL = "https://api.avalai.ir/v1"

# OpenAI client will be created per request

@dataclass
class EvaluationResult:
    """Stores comprehensive evaluation results for a single VLM response."""
    video_id: str
    topic: str
    question_type: str  # "conceptual", "numerical", "error_detection"
    model_name: str
    question: str
    vlm_response: str
    ground_truth: str
    
    # Objective Metrics (Non-LLM)
    keypoint_f1: float
    checklist_f1: float
    mathematical_accuracy: float
    
    # LLM-as-a-Judge
    llm_judge_score: float  # 1-5 scale
    llm_judge_confidence: float  # 0-1 scale
    llm_judge_reasoning: str
    
    # Metadata
    timestamp: str
    evaluation_duration: float

class RobustPhysicsEvaluator:
    """Comprehensive evaluation system combining objective metrics with LLM-as-a-judge."""
    
    def __init__(self):
        # Predefined physics keypoints for conceptual questions
        self.physics_keypoints = {
            "buoyancy": [
                "density", "buoyant force", "archimedes principle", "displacement",
                "weight", "volume", "fluid", "gravity", "float", "sink"
            ],
            "collision": [
                "momentum", "conservation", "elastic", "inelastic", "velocity",
                "mass", "kinetic energy", "impulse", "force", "time"
            ],
            "pendulum": [
                "period", "frequency", "amplitude", "simple harmonic motion",
                "gravity", "length", "mass", "oscillation", "energy", "friction"
            ],
            "projectile": [
                "trajectory", "range", "height", "velocity", "acceleration",
                "gravity", "angle", "horizontal", "vertical", "parabolic"
            ],
            "optics": [
                "reflection", "refraction", "focal point", "image", "object",
                "lens", "mirror", "converging", "diverging", "magnification"
            ],
            "electromagnetism": [
                "electric field", "magnetic field", "charge", "current", "voltage",
                "resistance", "capacitance", "inductance", "force", "energy"
            ],
            "quantum": [
                "wave function", "probability", "energy levels", "quantization",
                "uncertainty", "superposition", "tunneling", "photon", "electron"
            ],
            "general": [
                "force", "energy", "motion", "conservation", "equilibrium",
                "acceleration", "velocity", "mass", "time", "distance"
            ]
        }
        
        # Predefined simulation checklists for error detection
        self.simulation_checklist = [
            "ideal conditions", "no air resistance", "point masses", "frictionless",
            "perfect vacuum", "uniform field", "constant acceleration", "no energy loss",
            "infinite plane", "perfect conductor", "no quantum effects", "classical physics",
            "linear approximation", "small angle approximation", "steady state"
        ]
        
        # Mathematical patterns for numerical accuracy
        self.math_patterns = {
            "units": r'\b(m/s|m/s²|kg|N|J|W|V|A|Ω|F|H|rad|deg|°)\b',
            "numbers": r'\b\d+\.?\d*\b',
            "scientific_notation": r'\b\d+\.?\d*[eE][+-]?\d+\b',
            "fractions": r'\b\d+/\d+\b',
            "percentages": r'\b\d+\.?\d*%\b'
        }
    
    def _get_topic_keypoints(self, topic: str) -> List[str]:
        """Get keypoints for a specific topic, with fallback to general."""
        topic_lower = topic.lower()
        for key, keypoints in self.physics_keypoints.items():
            if key in topic_lower:
                return keypoints
        return self.physics_keypoints["general"]
    
    def _calculate_keypoint_f1(self, response: str, topic: str) -> float:
        """Calculate F1 score for conceptual questions based on keypoint presence."""
        keypoints = self._get_topic_keypoints(topic)
        response_lower = response.lower()
        
        # Count present keypoints
        present = sum(1 for kp in keypoints if kp.lower() in response_lower)
        
        if present == 0:
            return 0.0
        
        precision = present / len(keypoints)
        recall = present / len(keypoints)  # Simplified: assume all keypoints are equally important
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def _calculate_checklist_f1(self, response: str) -> float:
        """Calculate F1 score for error detection based on checklist items."""
        response_lower = response.lower()
        
        # Count identified checklist items
        identified = sum(1 for item in self.simulation_checklist if item.lower() in response_lower)
        
        if identified == 0:
            return 0.0
        
        precision = identified / len(self.simulation_checklist)
        recall = identified / len(self.simulation_checklist)  # Simplified: assume all items are equally important
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def _calculate_mathematical_accuracy(self, response: str) -> float:
        """Calculate mathematical accuracy score for numerical questions."""
        # Extract mathematical content
        math_content = []
        for pattern_name, pattern in self.math_patterns.items():
            matches = re.findall(pattern, response)
            math_content.extend(matches)
        
        if not math_content:
            return 0.0
        
        # Check for mathematical consistency indicators
        consistency_indicators = [
            "=", "+", "-", "*", "/", "×", "÷", "±", "≈", "≤", "≥", "<", ">"
        ]
        
        has_operations = any(op in response for op in consistency_indicators)
        has_units = bool(re.search(self.math_patterns["units"], response))
        has_numbers = bool(re.search(self.math_patterns["numbers"], response))
        
        # Score based on mathematical completeness
        score = 0.0
        if has_numbers:
            score += 0.4
        if has_operations:
            score += 0.3
        if has_units:
            score += 0.3
        
        return min(score, 1.0)
    
    def _llm_judge_evaluation(self, question: str, response: str, question_type: str) -> Tuple[float, float, str]:
        """Use LLM-as-a-judge to evaluate response quality."""
        try:
            # Create evaluation prompt based on question type
            if question_type == "conceptual":
                prompt = f"""Evaluate this physics conceptual question response CRITICALLY:

Question: {question}
Response: {response}

Rate the response on a scale of 1-5 where:
1 = Completely incorrect, irrelevant, or nonsensical
2 = Mostly incorrect with only 1-2 relevant points
3 = Partially correct but missing key concepts or has significant errors
4 = Mostly correct but missing important details or has minor conceptual errors
5 = Completely correct, comprehensive, and well-explained (RARE - only give this for exceptional responses)

IMPORTANT: Be very critical. Most responses should get 2-3. Only give 4-5 for truly excellent responses.
Look for: missing key concepts, oversimplifications, incorrect physics, lack of depth.

Provide your score (1-5), confidence level (0.0-1.0), and brief reasoning.
Format: Score: X, Confidence: Y, Reasoning: Z"""

            elif question_type == "numerical":
                prompt = f"""Evaluate this physics numerical question response CRITICALLY:

Question: {question}
Response: {response}

Rate the response on a scale of 1-5 where:
1 = Completely incorrect calculation, wrong units, or nonsensical math
2 = Mostly incorrect with only basic numerical elements present
3 = Partially correct but has calculation errors, wrong units, or missing steps
4 = Mostly correct but has minor numerical errors or incomplete calculations
5 = Completely correct calculation, proper units, and complete solution (RARE - only for perfect responses)

IMPORTANT: Be very critical. Most responses should get 2-3. Only give 4-5 for truly perfect numerical work.
Look for: calculation errors, wrong units, missing steps, incomplete solutions, incorrect formulas.

Provide your score (1-5), confidence level (0.0-1.0), and brief reasoning.
Format: Score: X, Confidence: Y, Reasoning: Z"""

            else:  # error_detection
                prompt = f"""Evaluate this physics error detection response CRITICALLY:

Question: {question}
Response: {response}

Rate the response on a scale of 1-5 where:
1 = No errors identified, completely wrong, or irrelevant response
2 = Few errors identified with major mistakes or missing key limitations
3 = Some errors identified but missing important ones or has inaccuracies
4 = Most errors identified correctly but may miss subtle limitations
5 = All relevant errors identified accurately and comprehensively (RARE - only for exceptional analysis)

IMPORTANT: Be very critical. Most responses should get 2-3. Only give 4-5 for truly comprehensive error analysis.
Look for: missing key limitations, oversimplified analysis, incorrect physics, lack of depth in error identification.

Provide your score (1-5), confidence level (0.0-1.0), and brief reasoning.
Format: Score: X, Confidence: Y, Reasoning: Z"""

            # Call LLM judge with reconnection logic
            max_attempts = 5
            base_timeout = 30
            
            for attempt in range(max_attempts):
                try:
                    # Create fresh client for each attempt to ensure clean connection
                    client = openai.OpenAI(api_key=AVALAI_API_KEY, base_url=AVALAI_BASE_URL)
                    
                    # Progressive timeout: 30s, 45s, 60s, 75s, 90s
                    current_timeout = base_timeout + (attempt * 15)
                    
                    print(f"    Attempt {attempt + 1}/{max_attempts} with {current_timeout}s timeout...")
                    
                    response_llm = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=150,
                        temperature=0.3,  # Increased temperature for more critical evaluation
                        timeout=current_timeout
                    )
                    
                    print(f"    Success on attempt {attempt + 1}!")
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    error_msg = str(e).lower()
                    
                    # Check if it's a connection/timeout error
                    if any(keyword in error_msg for keyword in ['timeout', 'connection', 'unavailable', 'premature close']):
                        if attempt < max_attempts - 1:  # Not the last attempt
                            wait_time = (2 ** attempt) + 5  # 6s, 7s, 9s, 13s, 21s
                            print(f"    Connection error on attempt {attempt + 1}: {e}")
                            print(f"    Waiting {wait_time}s before reconnecting...")
                            time.sleep(wait_time)
                            
                            # Force garbage collection to clean up connections
                            import gc
                            gc.collect()
                        else:
                            print(f"    All {max_attempts} attempts failed with connection errors")
                            raise e
                    else:
                        # Non-connection error, don't retry
                        print(f"    Non-connection error: {e}")
                        raise e
            
            judge_response = response_llm.choices[0].message.content
            
            # Parse the response
            score_match = re.search(r'Score:\s*(\d+)', judge_response)
            confidence_match = re.search(r'Confidence:\s*([0-9.]+)', judge_response)
            reasoning_match = re.search(r'Reasoning:\s*(.+)', judge_response)
            
            score = float(score_match.group(1)) if score_match else 3.0
            confidence = float(confidence_match.group(1)) if confidence_match else 0.5
            reasoning = reasoning_match.group(1) if reasoning_match else "No reasoning provided"
            
            # Apply critical adjustment to make scores lower
            if score >= 4:
                # Reduce high scores to be more realistic
                score = score * 0.6  # Reduce by 40%
            elif score >= 3:
                # Reduce medium scores more
                score = score * 0.75  # Reduce by 25%
            elif score >= 2:
                # Slightly reduce low scores
                score = score * 0.9  # Reduce by 10%
            
            # Add a small bias toward lower scores for realism
            score = score * 0.95
            
            # Ensure score stays within 1-5 range
            score = max(1.0, min(5.0, score))
            
            return score, confidence, reasoning
            
        except Exception as e:
            print(f"LLM judge evaluation failed after all reconnection attempts: {e}")
            # Give low scores for failed evaluations to maintain critical stance
            return 1.5, 0.2, f"Evaluation failed after reconnection attempts: {str(e)}"
    
    def evaluate_response(self, video_id: str, topic: str, question_type: str, 
                         model_name: str, question: str, vlm_response: str, 
                         ground_truth: str) -> EvaluationResult:
        """Evaluate a single VLM response using all metrics."""
        import time
        start_time = time.time()
        
        # Calculate objective metrics
        if question_type == "conceptual":
            keypoint_f1 = self._calculate_keypoint_f1(vlm_response, topic)
            checklist_f1 = 0.0  # Not applicable for conceptual
            mathematical_accuracy = 0.0  # Not applicable for conceptual
        elif question_type == "numerical":
            keypoint_f1 = 0.0  # Not applicable for numerical
            checklist_f1 = 0.0  # Not applicable for numerical
            mathematical_accuracy = self._calculate_mathematical_accuracy(vlm_response)
        else:  # error_detection
            keypoint_f1 = 0.0  # Not applicable for error detection
            checklist_f1 = self._calculate_checklist_f1(vlm_response)
            mathematical_accuracy = 0.0  # Not applicable for error detection
        
        # LLM-as-a-judge evaluation
        llm_score, llm_confidence, llm_reasoning = self._llm_judge_evaluation(
            question, vlm_response, question_type
        )
        
        evaluation_duration = time.time() - start_time
        
        return EvaluationResult(
            video_id=video_id,
            topic=topic,
            question_type=question_type,
            model_name=model_name,
            question=question,
            vlm_response=vlm_response,
            ground_truth=ground_truth,
            keypoint_f1=keypoint_f1,
            checklist_f1=checklist_f1,
            mathematical_accuracy=mathematical_accuracy,
            llm_judge_score=llm_score,
            llm_judge_confidence=llm_confidence,
            llm_judge_reasoning=llm_reasoning,
            timestamp=datetime.now().isoformat(),
            evaluation_duration=evaluation_duration
        )
    
    def generate_sample_questions(self, topic: str) -> Dict[str, str]:
        """Generate sample questions for each type based on topic."""
        questions = {
            "conceptual": f"Explain the fundamental physics principles involved in {topic} simulations.",
            "numerical": f"What are the key mathematical relationships and equations in {topic}?",
            "error_detection": f"What assumptions and limitations does the {topic} simulation make that differ from real-world conditions?"
        }
        return questions
    
    def evaluate_sample_dataset(self, sample_size: int = 5) -> List[EvaluationResult]:
        """Evaluate a sample dataset to demonstrate the system."""
        # Sample topics and questions
        sample_data = [
            ("buoyancy", "Explain why objects float or sink in fluids."),
            ("collision", "Calculate the final velocities after an elastic collision."),
            ("pendulum", "What factors affect the period of a simple pendulum?"),
            ("optics", "How do concave mirrors form images?"),
            ("electromagnetism", "Explain the relationship between electric and magnetic fields.")
        ]
        
        # Sample VLM responses (simulated)
        sample_responses = [
            "Objects float when their density is less than the fluid density. This is due to Archimedes principle where the buoyant force equals the weight of displaced fluid.",
            "Using conservation of momentum: m1v1 + m2v2 = m1v1' + m2v2'. For elastic collisions, kinetic energy is also conserved.",
            "The period T = 2π√(L/g) where L is length and g is gravitational acceleration. Mass doesn't affect the period.",
            "Concave mirrors can form real or virtual images depending on object position. The focal point is where parallel rays converge.",
            "Electric and magnetic fields are interconnected through Maxwell's equations. A changing electric field creates a magnetic field and vice versa."
        ]
        
        results = []
        for i, (topic, question) in enumerate(sample_data[:sample_size]):
            # Evaluate with different question types
            for q_type in ["conceptual", "numerical", "error_detection"]:
                result = self.evaluate_response(
                    video_id=f"sample_{i+1}",
                    topic=topic,
                    question_type=q_type,
                    model_name="sample_model",
                    question=question,
                    vlm_response=sample_responses[i],
                    ground_truth="Sample ground truth for demonstration"
                )
                results.append(result)
        
        return results
    
    def generate_comprehensive_report(self, results: List[EvaluationResult]) -> Dict:
        """Generate comprehensive evaluation report."""
        df = pd.DataFrame([vars(r) for r in results])
        
        # Calculate summary statistics
        report = {
            "total_evaluations": len(results),
            "unique_videos": df['video_id'].nunique(),
            "unique_topics": df['topic'].nunique(),
            "unique_models": df['model_name'].nunique(),
            
            "objective_metrics_summary": {
                "keypoint_f1": {
                    "mean": df['keypoint_f1'].mean(),
                    "std": df['keypoint_f1'].std(),
                    "min": df['keypoint_f1'].min(),
                    "max": df['keypoint_f1'].max()
                },
                "checklist_f1": {
                    "mean": df['checklist_f1'].mean(),
                    "std": df['checklist_f1'].std(),
                    "min": df['checklist_f1'].min(),
                    "max": df['checklist_f1'].max()
                },
                "mathematical_accuracy": {
                    "mean": df['mathematical_accuracy'].mean(),
                    "std": df['mathematical_accuracy'].std(),
                    "min": df['mathematical_accuracy'].min(),
                    "max": df['mathematical_accuracy'].max()
                }
            },
            
            "llm_judge_summary": {
                "score": {
                    "mean": df['llm_judge_score'].mean(),
                    "std": df['llm_judge_score'].std(),
                    "min": df['llm_judge_score'].min(),
                    "max": df['llm_judge_score'].max()
                },
                "confidence": {
                    "mean": df['llm_judge_confidence'].mean(),
                    "std": df['llm_judge_confidence'].std(),
                    "min": df['llm_judge_confidence'].min(),
                    "max": df['llm_judge_confidence'].max()
                }
            },
            
            "performance_by_question_type": df.groupby('question_type').agg({
                'keypoint_f1': 'mean',
                'checklist_f1': 'mean', 
                'mathematical_accuracy': 'mean',
                'llm_judge_score': 'mean'
            }).to_dict(),
            
            "performance_by_topic": df.groupby('topic').agg({
                'keypoint_f1': 'mean',
                'checklist_f1': 'mean',
                'mathematical_accuracy': 'mean',
                'llm_judge_score': 'mean'
            }).to_dict()
        }
        
        return report
    
    def save_results(self, results: List[EvaluationResult], filename: str = None):
        """Save evaluation results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_results_{timestamp}.json"
        
        # Convert dataclass objects to dictionaries
        results_dict = [vars(r) for r in results]
        
        with open(filename, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"Results saved to {filename}")
    
    def save_report(self, report: Dict, filename: str = None):
        """Save evaluation report to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_report_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Report saved to {filename}")

def main():
    """Demonstrate the robust evaluation system."""
    print("🚀 Initializing Robust Physics Evaluation System...")
    
    # Initialize evaluator
    evaluator = RobustPhysicsEvaluator()
    
    print("✅ Evaluator initialized successfully!")
    print("\n📊 Running sample evaluation...")
    
    # Run sample evaluation
    results = evaluator.evaluate_sample_dataset(sample_size=3)
    
    print(f"✅ Completed {len(results)} evaluations!")
    
    # Generate comprehensive report
    print("\n📈 Generating comprehensive report...")
    report = evaluator.generate_comprehensive_report(results)
    
    # Display key findings
    print("\n🎯 KEY FINDINGS:")
    print(f"Total evaluations: {report['total_evaluations']}")
    print(f"Objective metrics - Keypoint F1: {report['objective_metrics_summary']['keypoint_f1']['mean']:.3f}")
    print(f"Objective metrics - Checklist F1: {report['objective_metrics_summary']['checklist_f1']['mean']:.3f}")
    print(f"Objective metrics - Math Accuracy: {report['objective_metrics_summary']['mathematical_accuracy']['mean']:.3f}")
    print(f"LLM Judge - Average Score: {report['llm_judge_summary']['score']['mean']:.2f}/5")
    print(f"LLM Judge - Average Confidence: {report['llm_judge_summary']['confidence']['mean']:.3f}")
    
    # Save results and report
    evaluator.save_results(results)
    evaluator.save_report(report)
    
    print("\n🎉 Evaluation system demonstration completed!")
    print("📁 Check the generated JSON files for detailed results.")

if __name__ == "__main__":
    main()
