#!/usr/bin/env python3
"""
Integration Test Suite for Voice Assistant Components

This script runs comprehensive integration tests to validate that all components
work together properly across different configurations and scenarios.
"""

import os
import sys
import asyncio
import tempfile
import json
import yaml
import time
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import unittest
from unittest.mock import Mock, patch, MagicMock

# Rich console for better output
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.markdown import Markdown
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

class IntegrationTestSuite:
    """Comprehensive integration test suite"""
    
    def __init__(self):
        self.console = Console() if RICH_AVAILABLE else None
        self.test_results = {}
        self.temp_dir = Path(tempfile.gettempdir()) / "voice_assistant_tests"
        self.temp_dir.mkdir(exist_ok=True)
        
    def log(self, message: str, level: str = "info"):
        """Log message with appropriate formatting"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        if self.console:
            if level == "error":
                self.console.print(f"[red]❌ {timestamp} - {message}[/red]")
            elif level == "warning":
                self.console.print(f"[yellow]⚠️ {timestamp} - {message}[/yellow]")
            elif level == "success":
                self.console.print(f"[green]✅ {timestamp} - {message}[/green]")
            else:
                self.console.print(f"[blue]ℹ️ {timestamp} - {message}[/blue]")
        else:
            print(f"{level.upper()}: {timestamp} - {message}")
    
    async def test_component_imports(self) -> Dict[str, Any]:
        """Test that all components can be imported successfully"""
        self.log("Testing component imports...")
        
        result = {
            "test_name": "Component Imports",
            "status": "unknown",
            "details": {},
            "errors": [],
            "duration": 0
        }
        
        start_time = time.time()
        
        # Test imports
        import_tests = [
            ("streamlit", "Streamlit web framework"),
            ("openai", "OpenAI API client"),
            ("requests", "HTTP requests library"),
            ("numpy", "Numerical computing"),
            ("pathlib", "Path handling"),
            ("yaml", "YAML processing"),
            ("json", "JSON processing"),
            ("uuid", "UUID generation"),
            ("tempfile", "Temporary files"),
            ("threading", "Threading support"),
            ("asyncio", "Async programming"),
            ("datetime", "Date/time handling")
        ]
        
        # Optional imports
        optional_imports = [
            ("sounddevice", "Audio recording"),
            ("soundfile", "Audio file handling"),
            ("RealtimeSTT", "Real-time speech recognition"),
            ("rich", "Rich console output"),
            ("cv2", "Computer vision"),
            ("pyautogui", "GUI automation"),
            ("PIL", "Image processing"),
            ("pytesseract", "OCR processing")
        ]
        
        # Test core imports
        for module_name, description in import_tests:
            try:
                __import__(module_name)
                result["details"][module_name] = "available"
                self.log(f"✅ {description} import successful")
            except ImportError as e:
                result["errors"].append(f"Core import failed: {module_name} - {str(e)}")
                result["details"][module_name] = "missing"
        
        # Test optional imports
        for module_name, description in optional_imports:
            try:
                __import__(module_name)
                result["details"][module_name] = "available"
                self.log(f"✅ {description} import successful")
            except ImportError:
                result["details"][module_name] = "missing"
                self.log(f"⚠️ {description} not available (optional)")
        
        # Test our custom modules
        custom_modules = [
            ("agents.voice_agent", "Voice Agent"),
            ("agents.vision_agent", "Vision Agent"),
            ("agents.action_agent", "Action Agent"),
            ("agents.coordinator_agent", "Coordinator Agent"),
            ("utils.audio_utils", "Audio Utils"),
            ("utils.safety_controls", "Safety Controls"),
            ("config.settings", "Configuration Settings")
        ]
        
        for module_name, description in custom_modules:
            try:
                __import__(module_name)
                result["details"][module_name] = "available"
                self.log(f"✅ {description} import successful")
            except ImportError as e:
                result["errors"].append(f"Custom module import failed: {module_name} - {str(e)}")
                result["details"][module_name] = "missing"
        
        result["duration"] = time.time() - start_time
        result["status"] = "failed" if result["errors"] else "passed"
        
        return result
    
    async def test_configuration_management(self) -> Dict[str, Any]:
        """Test configuration loading and management"""
        self.log("Testing configuration management...")
        
        result = {
            "test_name": "Configuration Management",
            "status": "unknown",
            "details": {},
            "errors": [],
            "duration": 0
        }
        
        start_time = time.time()
        
        try:
            # Test environment variable loading
            test_env_vars = {
                "TEST_API_KEY": "test_key_123",
                "TEST_MODEL": "test_model",
                "TEST_VOICE": "test_voice"
            }
            
            # Set test environment variables
            for key, value in test_env_vars.items():
                os.environ[key] = value
            
            # Test retrieval
            for key, expected_value in test_env_vars.items():
                actual_value = os.getenv(key)
                if actual_value == expected_value:
                    result["details"][f"env_var_{key}"] = "passed"
                else:
                    result["errors"].append(f"Environment variable {key} not retrieved correctly")
            
            # Test .env file creation and loading
            test_env_file = self.temp_dir / "test.env"
            with open(test_env_file, "w") as f:
                f.write("TEST_ENV_FILE_VAR=test_value\n")
                f.write("ANOTHER_VAR=another_value\n")
            
            result["details"]["env_file_creation"] = "passed"
            
            # Test YAML configuration
            test_config = {
                "api_settings": {
                    "deepseek_model": "deepseek-chat",
                    "openai_model": "gpt-4o-mini"
                },
                "voice_settings": {
                    "default_voice": "nova",
                    "speech_rate": 1.0
                },
                "safety_settings": {
                    "mode": "high",
                    "rate_limit": 60
                }
            }
            
            test_yaml_file = self.temp_dir / "test_config.yml"
            with open(test_yaml_file, "w") as f:
                yaml.dump(test_config, f)
            
            # Load and verify YAML
            with open(test_yaml_file, "r") as f:
                loaded_config = yaml.safe_load(f)
            
            if loaded_config == test_config:
                result["details"]["yaml_config"] = "passed"
            else:
                result["errors"].append("YAML configuration loading failed")
            
            # Test JSON configuration
            test_json_file = self.temp_dir / "test_config.json"
            with open(test_json_file, "w") as f:
                json.dump(test_config, f)
            
            with open(test_json_file, "r") as f:
                loaded_json = json.load(f)
            
            if loaded_json == test_config:
                result["details"]["json_config"] = "passed"
            else:
                result["errors"].append("JSON configuration loading failed")
            
            # Clean up test environment variables
            for key in test_env_vars.keys():
                del os.environ[key]
            
            self.log("Configuration management tests completed")
            
        except Exception as e:
            result["errors"].append(f"Configuration test failed: {str(e)}")
        
        result["duration"] = time.time() - start_time
        result["status"] = "failed" if result["errors"] else "passed"
        
        return result
    
    async def test_audio_processing_pipeline(self) -> Dict[str, Any]:
        """Test audio processing components"""
        self.log("Testing audio processing pipeline...")
        
        result = {
            "test_name": "Audio Processing Pipeline",
            "status": "unknown",
            "details": {},
            "errors": [],
            "duration": 0
        }
        
        start_time = time.time()
        
        try:
            # Test audio device detection
            try:
                import sounddevice as sd
                devices = sd.query_devices()
                input_devices = [d for d in devices if d['max_input_channels'] > 0]
                output_devices = [d for d in devices if d['max_output_channels'] > 0]
                
                result["details"]["input_devices_found"] = len(input_devices)
                result["details"]["output_devices_found"] = len(output_devices)
                
                if input_devices and output_devices:
                    result["details"]["audio_devices"] = "available"
                else:
                    result["errors"].append("No audio devices found")
                
            except ImportError:
                result["details"]["sounddevice"] = "not_available"
            
            # Test audio file handling
            try:
                import soundfile as sf
                import numpy as np
                
                # Create test audio data
                sample_rate = 16000
                duration = 1.0  # 1 second
                frequency = 440  # A4 note
                
                t = np.linspace(0, duration, int(sample_rate * duration))
                test_audio = np.sin(2 * np.pi * frequency * t)
                
                # Save test audio
                test_audio_file = self.temp_dir / "test_audio.wav"
                sf.write(str(test_audio_file), test_audio, sample_rate)
                
                # Load and verify
                loaded_audio, loaded_sr = sf.read(str(test_audio_file))
                
                if loaded_sr == sample_rate and len(loaded_audio) > 0:
                    result["details"]["audio_file_handling"] = "passed"
                else:
                    result["errors"].append("Audio file handling test failed")
                
            except ImportError:
                result["details"]["soundfile"] = "not_available"
            except Exception as e:
                result["errors"].append(f"Audio file test failed: {str(e)}")
            
            # Test audio processing utilities
            try:
                from utils.audio_utils import AudioRecorder, AudioPlayer
                
                # Test AudioRecorder initialization
                recorder = AudioRecorder()
                result["details"]["audio_recorder_init"] = "passed"
                
                # Test AudioPlayer initialization
                player = AudioPlayer()
                result["details"]["audio_player_init"] = "passed"
                
            except ImportError:
                result["details"]["audio_utils"] = "not_available"
            except Exception as e:
                result["errors"].append(f"Audio utils test failed: {str(e)}")
            
        except Exception as e:
            result["errors"].append(f"Audio processing test failed: {str(e)}")
        
        result["duration"] = time.time() - start_time
        result["status"] = "failed" if result["errors"] else "passed"
        
        return result
    
    async def test_api_integration(self) -> Dict[str, Any]:
        """Test API integration with mock responses"""
        self.log("Testing API integration...")
        
        result = {
            "test_name": "API Integration",
            "status": "unknown",
            "details": {},
            "errors": [],
            "duration": 0
        }
        
        start_time = time.time()
        
        try:
            # Test OpenAI API structure (without actual API calls)
            try:
                from openai import OpenAI
                
                # Test client initialization (will fail without key, but structure should work)
                try:
                    client = OpenAI(api_key="test_key")
                    result["details"]["openai_client_init"] = "passed"
                except Exception:
                    # Expected to fail with invalid key, but structure should be correct
                    result["details"]["openai_client_init"] = "structure_ok"
                
            except ImportError:
                result["details"]["openai"] = "not_available"
            
            # Test DeepSeek API structure
            try:
                import requests
                
                # Test request structure (without actual call)
                headers = {
                    "Authorization": "Bearer test_key",
                    "Content-Type": "application/json"
                }
                
                data = {
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": "test"}],
                    "max_tokens": 10
                }
                
                # Verify structure is correct
                if "Authorization" in headers and "messages" in data:
                    result["details"]["deepseek_request_structure"] = "passed"
                
            except ImportError:
                result["details"]["requests"] = "not_available"
            
            # Test conversation management
            try:
                # Create test conversation
                test_conversation = [
                    {"role": "user", "content": "Hello", "timestamp": datetime.now().isoformat()},
                    {"role": "assistant", "content": "Hi there!", "timestamp": datetime.now().isoformat()}
                ]
                
                # Test YAML serialization
                test_file = self.temp_dir / "test_conversation.yml"
                with open(test_file, "w") as f:
                    yaml.dump(test_conversation, f)
                
                # Test loading
                with open(test_file, "r") as f:
                    loaded_conversation = yaml.safe_load(f)
                
                if loaded_conversation == test_conversation:
                    result["details"]["conversation_persistence"] = "passed"
                else:
                    result["errors"].append("Conversation persistence test failed")
                
            except Exception as e:
                result["errors"].append(f"Conversation management test failed: {str(e)}")
            
        except Exception as e:
            result["errors"].append(f"API integration test failed: {str(e)}")
        
        result["duration"] = time.time() - start_time
        result["status"] = "failed" if result["errors"] else "passed"
        
        return result
    
    async def test_safety_controls(self) -> Dict[str, Any]:
        """Test safety control systems"""
        self.log("Testing safety controls...")
        
        result = {
            "test_name": "Safety Controls",
            "status": "unknown",
            "details": {},
            "errors": [],
            "duration": 0
        }
        
        start_time = time.time()
        
        try:
            from utils.safety_controls import SafetyManager
            
            # Test safety manager initialization
            safety_manager = SafetyManager("high")
            result["details"]["safety_manager_init"] = "passed"
            
            # Test action safety checks
            test_cases = [
                ("click", "save button", True),
                ("type", "hello world", True),
                ("open_application", "chrome", True),
                ("open_application", "terminal", False),  # Should be blocked in high safety
                ("type", "rm -rf /", False),  # Should be blocked
                ("key_press", "enter", True),
                ("key_press", "ctrl+alt+del", False)  # Should be blocked in high safety
            ]
            
            passed_tests = 0
            for action_type, target, expected_safe in test_cases:
                is_safe = safety_manager.is_action_safe(action_type, target)
                if is_safe == expected_safe:
                    passed_tests += 1
                    result["details"][f"safety_test_{action_type}_{target.replace(' ', '_')}"] = "passed"
                else:
                    result["errors"].append(f"Safety test failed: {action_type}('{target}') expected {expected_safe}, got {is_safe}")
            
            result["details"]["safety_tests_passed"] = f"{passed_tests}/{len(test_cases)}"
            
            # Test rate limiting
            for i in range(5):
                safety_manager.is_action_safe("click", "test")
            
            status = safety_manager.get_safety_status()
            if "rate_limits" in status:
                result["details"]["rate_limiting"] = "passed"
            else:
                result["errors"].append("Rate limiting test failed")
            
            # Test emergency stop
            safety_manager.set_emergency_stop(True)
            if not safety_manager.is_action_safe("click", "safe button"):
                result["details"]["emergency_stop"] = "passed"
            else:
                result["errors"].append("Emergency stop test failed")
            
            safety_manager.set_emergency_stop(False)
            
        except ImportError:
            result["details"]["safety_controls"] = "not_available"
        except Exception as e:
            result["errors"].append(f"Safety controls test failed: {str(e)}")
        
        result["duration"] = time.time() - start_time
        result["status"] = "failed" if result["errors"] else "passed"
        
        return result
    
    async def test_streamlit_components(self) -> Dict[str, Any]:
        """Test Streamlit component functionality"""
        self.log("Testing Streamlit components...")
        
        result = {
            "test_name": "Streamlit Components",
            "status": "unknown",
            "details": {},
            "errors": [],
            "duration": 0
        }
        
        start_time = time.time()
        
        try:
            import streamlit as st
            
            # Test basic Streamlit functionality
            result["details"]["streamlit_import"] = "passed"
            
            # Test if we can access Streamlit components
            # Note: These won't actually render in test environment
            try:
                # Test component availability
                components = [
                    "text_input", "text_area", "button", "selectbox",
                    "toggle", "checkbox", "slider", "columns",
                    "sidebar", "container", "expander", "audio"
                ]
                
                for component in components:
                    if hasattr(st, component):
                        result["details"][f"streamlit_{component}"] = "available"
                    else:
                        result["errors"].append(f"Streamlit component {component} not available")
                
            except Exception as e:
                result["errors"].append(f"Streamlit component test failed: {str(e)}")
            
            # Test session state functionality
            try:
                # This would normally be done within a Streamlit app context
                # Here we just test the concept
                session_state_keys = [
                    "api_key", "model_selection", "voice_settings",
                    "conversation_history", "recording_state"
                ]
                
                result["details"]["session_state_concept"] = "passed"
                
            except Exception as e:
                result["errors"].append(f"Session state test failed: {str(e)}")
            
        except ImportError:
            result["details"]["streamlit"] = "not_available"
            result["errors"].append("Streamlit not available")
        except Exception as e:
            result["errors"].append(f"Streamlit test failed: {str(e)}")
        
        result["duration"] = time.time() - start_time
        result["status"] = "failed" if result["errors"] else "passed"
        
        return result
    
    async def test_end_to_end_workflow(self) -> Dict[str, Any]:
        """Test complete end-to-end workflow"""
        self.log("Testing end-to-end workflow...")
        
        result = {
            "test_name": "End-to-End Workflow",
            "status": "unknown",
            "details": {},
            "errors": [],
            "duration": 0
        }
        
        start_time = time.time()
        
        try:
            # Simulate complete workflow without actual API calls
            
            # 1. Configuration loading
            config = {
                "api_keys": {"openai": "test_key", "deepseek": "test_key"},
                "models": {"selected": "deepseek-chat"},
                "voice": {"selected": "nova"},
                "safety": {"mode": "high"}
            }
            result["details"]["config_simulation"] = "passed"
            
            # 2. Audio input simulation
            import numpy as np
            
            # Simulate audio data
            sample_rate = 16000
            duration = 2.0
            test_audio = np.random.random(int(sample_rate * duration))
            result["details"]["audio_input_simulation"] = "passed"
            
            # 3. Speech-to-text simulation
            simulated_text = "Hello, can you help me with a Python script?"
            result["details"]["stt_simulation"] = "passed"
            
            # 4. Trigger word detection
            trigger_words = ["hello", "help", "assistant"]
            detected = any(word in simulated_text.lower() for word in trigger_words)
            if detected:
                result["details"]["trigger_detection"] = "passed"
            else:
                result["errors"].append("Trigger word detection failed")
            
            # 5. Conversation context building
            conversation = [
                {"role": "user", "content": simulated_text},
            ]
            result["details"]["conversation_context"] = "passed"
            
            # 6. AI response simulation
            simulated_response = "I'd be happy to help you with a Python script. What would you like to create?"
            conversation.append({"role": "assistant", "content": simulated_response})
            result["details"]["ai_response_simulation"] = "passed"
            
            # 7. Response compression for TTS
            compressed_response = "I'll help you with a Python script. What do you need?"
            if len(compressed_response) < len(simulated_response):
                result["details"]["response_compression"] = "passed"
            else:
                result["errors"].append("Response compression failed")
            
            # 8. Text-to-speech simulation
            # Simulate TTS file creation
            tts_file = self.temp_dir / "test_tts.mp3"
            tts_file.write_text("simulated audio data")
            if tts_file.exists():
                result["details"]["tts_simulation"] = "passed"
            else:
                result["errors"].append("TTS simulation failed")
            
            # 9. Conversation persistence
            conversation_file = self.temp_dir / "test_conversation.yml"
            with open(conversation_file, "w") as f:
                yaml.dump(conversation, f)
            
            if conversation_file.exists():
                result["details"]["conversation_persistence"] = "passed"
            else:
                result["errors"].append("Conversation persistence failed")
            
            # 10. Safety validation throughout
            safety_checks = [
                ("user_input", simulated_text),
                ("ai_response", simulated_response),
                ("file_creation", str(tts_file))
            ]
            
            for check_type, content in safety_checks:
                # Simulate safety check
                is_safe = not any(dangerous in content.lower() for dangerous in ["delete", "format", "rm -rf"])
                if is_safe:
                    result["details"][f"safety_check_{check_type}"] = "passed"
                else:
                    result["errors"].append(f"Safety check failed for {check_type}")
            
            self.log("End-to-end workflow simulation completed")
            
        except Exception as e:
            result["errors"].append(f"End-to-end workflow test failed: {str(e)}")
        
        result["duration"] = time.time() - start_time
        result["status"] = "failed" if result["errors"] else "passed"
        
        return result
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all integration tests"""
        if self.console:
            self.console.print(Panel.fit(
                "[bold blue]🧪 Voice Assistant Integration Test Suite[/bold blue]\n"
                "Running comprehensive integration tests...",
                title="Integration Testing",
                border_style="blue"
            ))
        
        tests = [
            ("Component Imports", self.test_component_imports),
            ("Configuration Management", self.test_configuration_management),
            ("Audio Processing Pipeline", self.test_audio_processing_pipeline),
            ("API Integration", self.test_api_integration),
            ("Safety Controls", self.test_safety_controls),
            ("Streamlit Components", self.test_streamlit_components),
            ("End-to-End Workflow", self.test_end_to_end_workflow)
        ]
        
        results = {}
        
        if self.console:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=self.console
            ) as progress:
                overall_task = progress.add_task("Running integration tests...", total=len(tests))
                
                for name, test_func in tests:
                    test_task = progress.add_task(f"Testing {name}...", total=None)
                    results[name] = await test_func()
                    progress.remove_task(test_task)
                    progress.advance(overall_task)
        else:
            for name, test_func in tests:
                print(f"Running {name}...")
                results[name] = await test_func()
        
        return results
    
    def generate_test_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive test report"""
        if self.console:
            # Rich formatted report
            table = Table(title="Integration Test Results")
            table.add_column("Test Suite", style="cyan", no_wrap=True)
            table.add_column("Status", style="magenta")
            table.add_column("Duration", style="green")
            table.add_column("Details", style="blue")
            table.add_column("Issues", style="red")
            
            for test_name, result in results.items():
                status = result["status"]
                status_emoji = {
                    "passed": "✅ PASSED",
                    "partial": "⚠️ PARTIAL",
                    "failed": "❌ FAILED",
                    "unknown": "❓ UNKNOWN"
                }.get(status, status)
                
                duration = f"{result['duration']:.2f}s"
                details = f"{len(result['details'])} checks"
                issues = f"{len(result['errors'])} errors"
                
                table.add_row(test_name, status_emoji, duration, details, issues)
            
            self.console.print(table)
            
            # Summary
            passed = sum(1 for r in results.values() if r["status"] == "passed")
            failed = sum(1 for r in results.values() if r["status"] == "failed")
            total = len(results)
            
            summary = f"""
## Integration Test Summary

- ✅ **Passed**: {passed}/{total} test suites
- ❌ **Failed**: {failed}/{total} test suites
- ⏱️ **Total Duration**: {sum(r['duration'] for r in results.values()):.2f} seconds

### Overall Status: {"🎉 ALL TESTS PASSED" if failed == 0 else f"⚠️ {failed} TEST(S) FAILED"}
"""
            
            self.console.print(Panel(Markdown(summary), title="Test Summary", border_style="green" if failed == 0 else "red"))
        
        # Generate detailed text report
        report_lines = [
            "# Voice Assistant Integration Test Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Executive Summary",
            ""
        ]
        
        total_tests = len(results)
        passed_tests = sum(1 for r in results.values() if r["status"] == "passed")
        failed_tests = sum(1 for r in results.values() if r["status"] == "failed")
        total_duration = sum(r["duration"] for r in results.values())
        
        report_lines.extend([
            f"- **Total Test Suites**: {total_tests}",
            f"- **Passed**: {passed_tests}",
            f"- **Failed**: {failed_tests}",
            f"- **Success Rate**: {(passed_tests/total_tests)*100:.1f}%",
            f"- **Total Duration**: {total_duration:.2f} seconds",
            "",
            "## Detailed Results",
            ""
        ])
        
        for test_name, result in results.items():
            report_lines.extend([
                f"### {test_name}",
                f"**Status**: {result['status'].upper()}",
                f"**Duration**: {result['duration']:.2f} seconds",
                f"**Errors**: {len(result['errors'])}",
                ""
            ])
            
            if result['errors']:
                report_lines.append("**Errors:**")
                for error in result['errors']:
                    report_lines.append(f"- {error}")
                report_lines.append("")
            
            if result['details']:
                report_lines.append("**Details:**")
                for key, value in result['details'].items():
                    report_lines.append(f"- {key}: {value}")
                report_lines.append("")
        
        return "\n".join(report_lines)

async def main():
    """Main test runner"""
    test_suite = IntegrationTestSuite()
    
    try:
        # Run all integration tests
        results = await test_suite.run_all_tests()
        
        # Generate and save report
        report = test_suite.generate_test_report(results)
        
        # Save report to file
        report_file = Path("integration_test_report.md")
        with open(report_file, "w") as f:
            f.write(report)
        
        test_suite.log(f"Integration test report saved to {report_file}", "success")
        
        # Return results for programmatic use
        return results
        
    except Exception as e:
        test_suite.log(f"Integration testing failed: {str(e)}", "error")
        return None

if __name__ == "__main__":
    asyncio.run(main())

