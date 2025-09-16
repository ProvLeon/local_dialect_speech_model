#!/usr/bin/env python3
"""
Model Testing and Validation Script

This script provides comprehensive testing and validation for the Twi Speech Model,
including unit tests, integration tests, performance benchmarks, and quality assurance.

Features:
- Model functionality testing
- Performance benchmarking
- API endpoint testing
- Audio processing validation
- Load testing
- Memory usage monitoring
- Accuracy validation
- Error handling testing
"""

import os
import sys
import json
import time
import asyncio
import logging
import argparse
import tempfile
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import torch
    import numpy as np
    import librosa
    import soundfile as sf
    import requests
    from tqdm import tqdm
    import psutil
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError as e:
    print(f"Error importing required packages: {e}")
    print("Please install requirements: pip install -r requirements.txt")
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelTester:
    """Comprehensive model testing and validation."""

    def __init__(self, model_path: str, api_url: Optional[str] = None):
        self.model_path = Path(model_path)
        self.api_url = api_url or "http://localhost:8000"
        self.test_results = {}
        self.performance_metrics = {}

    def run_all_tests(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run all configured tests."""
        logger.info("Starting comprehensive model testing...")

        test_suite = {
            "model_loading": self.test_model_loading,
            "basic_functionality": self.test_basic_functionality,
            "api_endpoints": self.test_api_endpoints,
            "audio_processing": self.test_audio_processing,
            "performance_benchmarks": self.test_performance_benchmarks,
            "load_testing": self.test_load_testing,
            "memory_usage": self.test_memory_usage,
            "error_handling": self.test_error_handling,
            "accuracy_validation": self.test_accuracy_validation
        }

        results = {
            "timestamp": time.time(),
            "config": config,
            "tests": {},
            "summary": {}
        }

        for test_name, test_func in test_suite.items():
            if config.get("tests", {}).get(test_name, True):
                logger.info(f"Running {test_name}...")
                try:
                    start_time = time.time()
                    test_result = test_func(config.get(test_name, {}))
                    end_time = time.time()

                    results["tests"][test_name] = {
                        "status": "passed" if test_result.get("success", False) else "failed",
                        "duration": end_time - start_time,
                        "details": test_result
                    }
                    logger.info(f"✓ {test_name} completed")

                except Exception as e:
                    results["tests"][test_name] = {
                        "status": "error",
                        "error": str(e),
                        "details": {}
                    }
                    logger.error(f"✗ {test_name} failed: {e}")
            else:
                logger.info(f"Skipping {test_name} (disabled in config)")

        # Generate summary
        results["summary"] = self._generate_test_summary(results["tests"])

        return results

    def test_model_loading(self, config: Dict) -> Dict[str, Any]:
        """Test model loading and initialization."""
        results = {"success": False, "details": {}}

        try:
            # Test direct model loading
            from deployable_twi_speech_model.utils.inference import ModelInference

            model = ModelInference(str(self.model_path))
            results["details"]["model_loaded"] = True

            # Test model info retrieval
            model_info = model.get_model_info()
            results["details"]["model_info"] = model_info
            results["details"]["model_info_valid"] = bool(model_info)

            # Test device placement
            results["details"]["device"] = str(model.device)
            results["details"]["cuda_available"] = torch.cuda.is_available()

            # Test model architecture
            if hasattr(model, 'model'):
                results["details"]["model_parameters"] = sum(p.numel() for p in model.model.parameters())
                results["details"]["model_mode"] = "training" if model.model.training else "inference"

            results["success"] = True

        except Exception as e:
            results["details"]["error"] = str(e)
            results["success"] = False

        return results

    def test_basic_functionality(self, config: Dict) -> Dict[str, Any]:
        """Test basic model functionality."""
        results = {"success": False, "details": {}}

        try:
            from deployable_twi_speech_model.utils.inference import ModelInference

            model = ModelInference(str(self.model_path))

            # Create test audio
            test_audio_path = self._create_test_audio()

            # Test single prediction
            intent, confidence = model.predict(test_audio_path)
            results["details"]["prediction"] = {
                "intent": intent,
                "confidence": float(confidence)
            }
            results["details"]["prediction_valid"] = isinstance(intent, str) and 0 <= confidence <= 1

            # Test top-k predictions if available
            if hasattr(model, 'predict_topk'):
                intent_topk, confidence_topk, top_predictions = model.predict_topk(test_audio_path, top_k=5)
                results["details"]["topk_prediction"] = {
                    "intent": intent_topk,
                    "confidence": float(confidence_topk),
                    "top_predictions": top_predictions
                }
                results["details"]["topk_valid"] = len(top_predictions) <= 5

            # Clean up
            os.unlink(test_audio_path)

            results["success"] = True

        except Exception as e:
            results["details"]["error"] = str(e)
            results["success"] = False

        return results

    def test_api_endpoints(self, config: Dict) -> Dict[str, Any]:
        """Test API endpoints."""
        results = {"success": False, "details": {}}

        endpoints = [
            {"path": "/health", "method": "GET"},
            {"path": "/model-info", "method": "GET"},
            {"path": "/test-intent", "method": "POST", "requires_file": True}
        ]

        endpoint_results = {}

        for endpoint in endpoints:
            endpoint_name = endpoint["path"]
            try:
                url = f"{self.api_url}{endpoint['path']}"

                if endpoint["method"] == "GET":
                    response = requests.get(url, timeout=10)
                elif endpoint["method"] == "POST" and endpoint.get("requires_file"):
                    # Create test audio file
                    test_audio_path = self._create_test_audio()

                    with open(test_audio_path, 'rb') as f:
                        files = {'file': f}
                        response = requests.post(f"{url}?top_k=3", files=files, timeout=30)

                    os.unlink(test_audio_path)

                endpoint_results[endpoint_name] = {
                    "status_code": response.status_code,
                    "response_time": response.elapsed.total_seconds(),
                    "success": response.status_code == 200
                }

                if response.status_code == 200:
                    try:
                        endpoint_results[endpoint_name]["response_data"] = response.json()
                    except json.JSONDecodeError:
                        endpoint_results[endpoint_name]["response_data"] = response.text
                else:
                    endpoint_results[endpoint_name]["error"] = response.text

            except Exception as e:
                endpoint_results[endpoint_name] = {
                    "success": False,
                    "error": str(e)
                }

        results["details"]["endpoints"] = endpoint_results
        results["success"] = all(ep.get("success", False) for ep in endpoint_results.values())

        return results

    def test_audio_processing(self, config: Dict) -> Dict[str, Any]:
        """Test audio processing capabilities."""
        results = {"success": False, "details": {}}

        try:
            from deployable_twi_speech_model.utils.inference import ModelInference

            model = ModelInference(str(self.model_path))

            # Test different audio formats and properties
            test_cases = [
                {"name": "16kHz_wav", "sample_rate": 16000, "duration": 2.0, "format": "wav"},
                {"name": "22kHz_wav", "sample_rate": 22050, "duration": 3.0, "format": "wav"},
                {"name": "44kHz_wav", "sample_rate": 44100, "duration": 1.5, "format": "wav"},
                {"name": "short_audio", "sample_rate": 16000, "duration": 0.5, "format": "wav"},
                {"name": "long_audio", "sample_rate": 16000, "duration": 10.0, "format": "wav"}
            ]

            processing_results = {}

            for test_case in test_cases:
                try:
                    # Create test audio with specific properties
                    audio_path = self._create_test_audio(
                        sample_rate=test_case["sample_rate"],
                        duration=test_case["duration"],
                        format=test_case["format"]
                    )

                    # Test processing
                    start_time = time.time()
                    intent, confidence = model.predict(audio_path)
                    processing_time = time.time() - start_time

                    processing_results[test_case["name"]] = {
                        "success": True,
                        "processing_time": processing_time,
                        "intent": intent,
                        "confidence": float(confidence),
                        "audio_properties": test_case
                    }

                    os.unlink(audio_path)

                except Exception as e:
                    processing_results[test_case["name"]] = {
                        "success": False,
                        "error": str(e),
                        "audio_properties": test_case
                    }

            results["details"]["audio_processing"] = processing_results
            results["success"] = any(case.get("success", False) for case in processing_results.values())

        except Exception as e:
            results["details"]["error"] = str(e)
            results["success"] = False

        return results

    def test_performance_benchmarks(self, config: Dict) -> Dict[str, Any]:
        """Test performance benchmarks."""
        results = {"success": False, "details": {}}

        try:
            from deployable_twi_speech_model.utils.inference import ModelInference

            model = ModelInference(str(self.model_path))

            # Performance test configuration
            num_iterations = config.get("num_iterations", 50)
            batch_sizes = config.get("batch_sizes", [1])

            benchmark_results = {}

            for batch_size in batch_sizes:
                batch_name = f"batch_size_{batch_size}"
                batch_results = {
                    "times": [],
                    "memory_usage": [],
                    "predictions": []
                }

                # Create test audio files
                test_audio_paths = []
                for i in range(batch_size):
                    audio_path = self._create_test_audio()
                    test_audio_paths.append(audio_path)

                # Run benchmark iterations
                for iteration in range(num_iterations):
                    # Monitor memory before
                    memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB

                    # Time the prediction
                    start_time = time.time()

                    if batch_size == 1:
                        intent, confidence = model.predict(test_audio_paths[0])
                        predictions = [(intent, confidence)]
                    else:
                        # Batch prediction if available
                        if hasattr(model, 'predict_batch'):
                            predictions = model.predict_batch(test_audio_paths)
                        else:
                            # Sequential prediction
                            predictions = []
                            for audio_path in test_audio_paths:
                                intent, confidence = model.predict(audio_path)
                                predictions.append((intent, confidence))

                    end_time = time.time()

                    # Monitor memory after
                    memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB

                    batch_results["times"].append(end_time - start_time)
                    batch_results["memory_usage"].append(memory_after - memory_before)
                    batch_results["predictions"].append(len(predictions))

                # Clean up test files
                for audio_path in test_audio_paths:
                    os.unlink(audio_path)

                # Calculate statistics
                times = batch_results["times"]
                benchmark_results[batch_name] = {
                    "mean_time": statistics.mean(times),
                    "median_time": statistics.median(times),
                    "min_time": min(times),
                    "max_time": max(times),
                    "std_time": statistics.stdev(times) if len(times) > 1 else 0,
                    "mean_memory_delta": statistics.mean(batch_results["memory_usage"]),
                    "throughput": batch_size / statistics.mean(times),  # predictions per second
                    "iterations": num_iterations,
                    "batch_size": batch_size
                }

            results["details"]["benchmarks"] = benchmark_results
            results["success"] = True

        except Exception as e:
            results["details"]["error"] = str(e)
            results["success"] = False

        return results

    def test_load_testing(self, config: Dict) -> Dict[str, Any]:
        """Test load handling capabilities."""
        results = {"success": False, "details": {}}

        if not self._check_api_available():
            results["details"]["error"] = "API not available for load testing"
            return results

        try:
            # Load test configuration
            num_concurrent = config.get("num_concurrent", 10)
            num_requests = config.get("num_requests", 100)
            ramp_up_time = config.get("ramp_up_time", 5)  # seconds

            # Create test audio
            test_audio_path = self._create_test_audio()

            # Load testing function
            def make_request():
                try:
                    with open(test_audio_path, 'rb') as f:
                        files = {'file': f}
                        start_time = time.time()
                        response = requests.post(
                            f"{self.api_url}/test-intent?top_k=3",
                            files=files,
                            timeout=30
                        )
                        end_time = time.time()

                        return {
                            "success": response.status_code == 200,
                            "status_code": response.status_code,
                            "response_time": end_time - start_time,
                            "timestamp": start_time
                        }
                except Exception as e:
                    return {
                        "success": False,
                        "error": str(e),
                        "timestamp": time.time()
                    }

            # Execute load test
            load_results = []
            start_time = time.time()

            with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
                # Submit requests with ramp-up
                futures = []
                for i in range(num_requests):
                    if ramp_up_time > 0:
                        time.sleep(ramp_up_time / num_requests)
                    future = executor.submit(make_request)
                    futures.append(future)

                # Collect results
                for future in as_completed(futures):
                    result = future.result()
                    load_results.append(result)

            end_time = time.time()

            # Clean up
            os.unlink(test_audio_path)

            # Analyze results
            successful_requests = [r for r in load_results if r.get("success", False)]
            failed_requests = [r for r in load_results if not r.get("success", False)]

            response_times = [r["response_time"] for r in successful_requests if "response_time" in r]

            results["details"]["load_test"] = {
                "total_requests": num_requests,
                "successful_requests": len(successful_requests),
                "failed_requests": len(failed_requests),
                "success_rate": len(successful_requests) / num_requests,
                "total_duration": end_time - start_time,
                "requests_per_second": num_requests / (end_time - start_time),
                "response_times": {
                    "mean": statistics.mean(response_times) if response_times else 0,
                    "median": statistics.median(response_times) if response_times else 0,
                    "min": min(response_times) if response_times else 0,
                    "max": max(response_times) if response_times else 0,
                    "p95": np.percentile(response_times, 95) if response_times else 0
                },
                "error_types": {}
            }

            # Categorize errors
            error_types = {}
            for req in failed_requests:
                error_key = req.get("error", "unknown_error")
                error_types[error_key] = error_types.get(error_key, 0) + 1

            results["details"]["load_test"]["error_types"] = error_types
            results["success"] = len(successful_requests) > num_requests * 0.8  # 80% success rate threshold

        except Exception as e:
            results["details"]["error"] = str(e)
            results["success"] = False

        return results

    def test_memory_usage(self, config: Dict) -> Dict[str, Any]:
        """Test memory usage patterns."""
        results = {"success": False, "details": {}}

        try:
            from deployable_twi_speech_model.utils.inference import ModelInference

            # Monitor memory during model loading
            memory_before_load = psutil.Process().memory_info().rss / 1024 / 1024  # MB

            model = ModelInference(str(self.model_path))

            memory_after_load = psutil.Process().memory_info().rss / 1024 / 1024  # MB

            # Monitor memory during predictions
            num_predictions = config.get("num_predictions", 20)
            memory_readings = []

            for i in range(num_predictions):
                test_audio_path = self._create_test_audio()

                memory_before = psutil.Process().memory_info().rss / 1024 / 1024
                intent, confidence = model.predict(test_audio_path)
                memory_after = psutil.Process().memory_info().rss / 1024 / 1024

                memory_readings.append({
                    "iteration": i,
                    "memory_before": memory_before,
                    "memory_after": memory_after,
                    "memory_delta": memory_after - memory_before
                })

                os.unlink(test_audio_path)

            # Calculate memory statistics
            memory_deltas = [r["memory_delta"] for r in memory_readings]
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024

            results["details"]["memory_analysis"] = {
                "model_loading_memory": memory_after_load - memory_before_load,
                "base_memory_usage": memory_after_load,
                "final_memory_usage": final_memory,
                "prediction_memory_stats": {
                    "mean_delta": statistics.mean(memory_deltas),
                    "max_delta": max(memory_deltas),
                    "min_delta": min(memory_deltas),
                    "total_leak": final_memory - memory_after_load
                },
                "memory_readings": memory_readings
            }

            # Check for memory leaks
            memory_leak_threshold = config.get("memory_leak_threshold", 50)  # MB
            has_memory_leak = (final_memory - memory_after_load) > memory_leak_threshold

            results["details"]["memory_leak_detected"] = has_memory_leak
            results["success"] = not has_memory_leak

        except Exception as e:
            results["details"]["error"] = str(e)
            results["success"] = False

        return results

    def test_error_handling(self, config: Dict) -> Dict[str, Any]:
        """Test error handling capabilities."""
        results = {"success": False, "details": {}}

        try:
            from deployable_twi_speech_model.utils.inference import ModelInference

            model = ModelInference(str(self.model_path))

            error_test_cases = [
                {
                    "name": "nonexistent_file",
                    "test": lambda: model.predict("nonexistent_file.wav"),
                    "expected_error": True
                },
                {
                    "name": "empty_file",
                    "test": lambda: self._test_empty_file(model),
                    "expected_error": True
                },
                {
                    "name": "corrupted_audio",
                    "test": lambda: self._test_corrupted_audio(model),
                    "expected_error": True
                },
                {
                    "name": "very_short_audio",
                    "test": lambda: self._test_very_short_audio(model),
                    "expected_error": False  # Should handle gracefully
                },
                {
                    "name": "very_long_audio",
                    "test": lambda: self._test_very_long_audio(model),
                    "expected_error": False  # Should handle gracefully
                }
            ]

            error_results = {}

            for test_case in error_test_cases:
                try:
                    result = test_case["test"]()
                    error_results[test_case["name"]] = {
                        "error_occurred": False,
                        "result": result,
                        "expected_error": test_case["expected_error"],
                        "test_passed": not test_case["expected_error"]
                    }
                except Exception as e:
                    error_results[test_case["name"]] = {
                        "error_occurred": True,
                        "error_message": str(e),
                        "expected_error": test_case["expected_error"],
                        "test_passed": test_case["expected_error"]
                    }

            results["details"]["error_handling"] = error_results
            results["success"] = all(case["test_passed"] for case in error_results.values())

        except Exception as e:
            results["details"]["error"] = str(e)
            results["success"] = False

        return results

    def test_accuracy_validation(self, config: Dict) -> Dict[str, Any]:
        """Test model accuracy with validation data."""
        results = {"success": False, "details": {}}

        # This would require validation dataset
        validation_data_path = config.get("validation_data_path")

        if not validation_data_path or not Path(validation_data_path).exists():
            results["details"]["error"] = "Validation data not available"
            results["success"] = True  # Don't fail if validation data is not provided
            return results

        try:
            from deployable_twi_speech_model.utils.inference import ModelInference

            model = ModelInference(str(self.model_path))

            # Load validation data (assuming CSV format with audio_path, true_intent columns)
            import pandas as pd
            validation_df = pd.read_csv(validation_data_path)

            predictions = []
            true_labels = []
            processing_times = []

            for _, row in validation_df.iterrows():
                audio_path = row['audio_path']
                true_intent = row['true_intent']

                if Path(audio_path).exists():
                    start_time = time.time()
                    predicted_intent, confidence = model.predict(audio_path)
                    end_time = time.time()

                    predictions.append(predicted_intent)
                    true_labels.append(true_intent)
                    processing_times.append(end_time - start_time)

            # Calculate accuracy metrics
            correct_predictions = sum(1 for p, t in zip(predictions, true_labels) if p == t)
            accuracy = correct_predictions / len(predictions) if predictions else 0

            # Calculate per-class metrics
            unique_labels = list(set(true_labels))
            per_class_metrics = {}

            for label in unique_labels:
                true_positives = sum(1 for p, t in zip(predictions, true_labels) if p == label and t == label)
                false_positives = sum(1 for p, t in zip(predictions, true_labels) if p == label and t != label)
                false_negatives = sum(1 for p, t in zip(predictions, true_labels) if p != label and t == label)

                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

                per_class_metrics[label] = {
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1_score,
                    "support": sum(1 for t in true_labels if t == label)
                }

            results["details"]["accuracy_validation"] = {
                "overall_accuracy": accuracy,
                "total_samples": len(predictions),
                "correct_predictions": correct_predictions,
                "mean_processing_time": statistics.mean(processing_times),
                "per_class_metrics": per_class_metrics,
                "macro_avg_f1": statistics.mean([m["f1_score"] for m in per_class_metrics.values()])
            }

            # Check if accuracy meets threshold
            accuracy_threshold = config.get("accuracy_threshold", 0.8)
            results["success"] = accuracy >= accuracy_threshold

        except Exception as e:
            results["details"]["error"] = str(e)
            results["success"] = False

        return results

    def _create_test_audio(self, sample_rate: int = 16000, duration: float = 2.0, format: str = "wav") -> str:
        """Create a test audio file."""
        # Generate test audio (sine wave with noise)
        t = np.linspace(0, duration, int(sample_rate * duration))
        frequency = 440  # A4 note
        audio = 0.3 * np.sin(2 * np.pi * frequency * t) + 0.1 * np.random.randn(len(t))

        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{format}")
        temp_file.close()

        # Save audio
        sf.write(temp_file.name, audio, sample_rate)

        return temp_file.name

    def _test_empty_file(self, model) -> Any:
        """Test with empty audio file."""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_file.close()

        try:
            return model.predict(temp_file.name)
        finally:
            os.unlink(temp_file.name)

    def _test_corrupted_audio(self, model) -> Any:
        """Test with corrupted audio file."""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_file.write(b"corrupted audio data")
        temp_file.close()

        try:
            return model.predict(temp_file.name)
        finally:
            os.unlink(temp_file.name)

    def _test_very_short_audio(self, model) -> Any:
        """Test with very short audio."""
        audio_path = self._create_test_audio(duration=0.1)
        try:
            return model.predict(audio_path)
        finally:
            os.unlink(audio_path)

    def _test_very_long_audio(self, model) -> Any:
        """Test with very long audio."""
        audio_path = self._create_test_audio(duration=60.0)
        try:
            return model.predict(audio_path)
        finally:
            os.unlink(audio_path)

    def _check_api_available(self) -> bool:
        """Check if API is available."""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False

    def _generate_test_summary(self, test_results: Dict) -> Dict[str, Any]:
        """Generate test summary."""
        total_tests = len(test_results)
        passed_tests = sum(1 for result in test_results.values() if result["status"] == "passed")
        failed_tests = sum(1 for result in test_results.values() if result["status"] == "failed")
        error_tests = sum(1 for result in test_results.values() if result["status"] == "error")

        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "error_tests": error_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "overall_status": "passed" if failed_tests == 0 and error_tests == 0 else "failed"
        }

    def generate_report(self, results: Dict[str, Any], output_path: Optional[str] = None) -> str:
        """Generate comprehensive test report."""
        report = []

        report.append("=" * 80)
        report.append("TWISPEECH MODEL TEST REPORT")
        report.append("=" * 80)
        report.append(f"Timestamp: {time.ctime(results['timestamp'])}")
        report.append(f"Model Path: {self.model_path}")
        report.append(f"API URL: {self.api_url}")
        report.append("")

        # Summary
        summary = results["summary"]
        report.append("SUMMARY")
        report.append("-" * 40)
        report.append(f"Total Tests: {summary['total_tests']}")
        report.append(f"Passed: {summary['passed_tests']}")
        report.append(f"Failed: {summary['failed_tests']}")
        report.append(f"Errors: {summary['error_tests']}")
        report.append(f"Success Rate: {summary['success_rate']:.2%}")
        report.append(f"Overall Status: {summary['overall_status'].upper()}")
        report.append("")

        # Detailed results
        for test_name, test_result in results["tests"].items():
            report.append(f"{test_name.upper()}")
            report.append("-" * len(test_name))
            report.append(f"Status: {test_result['status'].upper()}")
            report.append(f"Duration: {test_result.get('duration', 0):.2f}s")

            if test_result["status"] == "error":
                report.append(f"Error: {test_result.get('error', 'Unknown error')}")
            elif "details" in test_result:
                details = test_result["details"]
                for key, value in details.items():
                    if isinstance(value, dict):
                        report.append(f"{key}: {json.dumps(value, indent=2)}")
                    else:
                        report.append(f"{key}: {value}")

            report.append("")

        report_text = "\n".join(report)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Report saved to {output_path}")

        return report_text


def load_test_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load test configuration."""
    default_config = {
        "tests": {
            "model_loading": True,
            "basic_functionality": True,
            "api_endpoints": True,
            "audio_processing": True,
            "performance_benchmarks": True,
            "load_testing": False,  # Disabled by default
            "memory_usage": True,
            "error_handling": True,
            "accuracy_validation": False  # Requires validation data
        },
        "performance_benchmarks": {
            "num_iterations": 20,
            "batch_sizes": [1]
        },
        "load_testing": {
            "num_concurrent": 5,
            "num_requests": 50,
            "ramp_up_time": 2
        },
        "memory_usage": {
            "num_predictions": 10,
            "memory_leak_threshold": 50
        },
        "accuracy_validation": {
            "validation_data_path": None,
            "accuracy_threshold": 0.8
        }
    }

    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            user_config = json.load(f)
        # Merge configurations
        default_config.update(user_config)

    return default_config


def main():
    """Main entry point for the testing script."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Model Testing and Validation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default="deployable_twi_speech_model",
        help="Path to the model package"
    )

    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000",
        help="API URL for endpoint testing"
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to test configuration file"
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Output path for test report"
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick tests only"
    )

    parser.add_argument(
        "--load-test",
        action="store_true",
        help="Include load testing"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load configuration
    config = load_test_config(args.config)

    # Adjust configuration based on arguments
    if args.quick:
        config["tests"]["load_testing"] = False
        config["tests"]["accuracy_validation"] = False
        config["performance_benchmarks"]["num_iterations"] = 5

    if args.load_test:
        config["tests"]["load_testing"] = True

    try:
        # Initialize tester
        tester = ModelTester(args.model_path, args.api_url)

        # Run tests
        logger.info("Starting model testing...")
        results = tester.run_all_tests(config)

        # Generate report
        output_path = args.output or f"test_report_{int(time.time())}.txt"
        report = tester.generate_report(results, output_path)

        # Print summary
        summary = results["summary"]
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed_tests']}")
        print(f"Failed: {summary['failed_tests']}")
        print(f"Errors: {summary['error_tests']}")
        print(f"Success Rate: {summary['success_rate']:.2%}")
        print(f"Overall Status: {summary['overall_status'].upper()}")
        print(f"Report saved to: {output_path}")
        print("="*60)

        # Exit with appropriate code
        sys.exit(0 if summary["overall_status"] == "passed" else 1)

    except KeyboardInterrupt:
        logger.info("Testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Testing failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
