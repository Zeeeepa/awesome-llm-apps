#!/usr/bin/env python3
"""
Component Validator for Voice Assistant Systems

This script validates all components of the voice assistant implementations
and ensures they work together properly.
"""

import os
import sys
import asyncio
import importlib
import subprocess
import tempfile
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import time
from datetime import datetime

# Rich console for better output
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.markdown import Markdown
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Rich not available - using basic output")

class ComponentValidator:
    """Validates all voice assistant components"""
    
    def __init__(self):
        self.console = Console() if RICH_AVAILABLE else None
        self.results = {}
        self.errors = []
        self.warnings = []
        
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
    
    def validate_python_environment(self) -> Dict[str, Any]:
        """Validate Python environment and version"""
        self.log("Validating Python environment...")
        
        result = {
            "component": "Python Environment",
            "status": "unknown",
            "details": {},
            "errors": [],
            "warnings": []
        }
        
        try:
            # Check Python version
            version = sys.version_info
            result["details"]["python_version"] = f"{version.major}.{version.minor}.{version.micro}"
            
            if version.major < 3 or (version.major == 3 and version.minor < 8):
                result["errors"].append(f"Python 3.8+ required, found {version.major}.{version.minor}")
                result["status"] = "failed"
                return result
            
            # Check pip availability
            try:
                import pip
                result["details"]["pip_available"] = True
            except ImportError:
                result["warnings"].append("pip not available as module")
                result["details"]["pip_available"] = False
            
            # Check virtual environment
            result["details"]["virtual_env"] = hasattr(sys, 'real_prefix') or (
                hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
            )
            
            result["status"] = "passed"
            self.log("Python environment validation passed", "success")
            
        except Exception as e:
            result["errors"].append(f"Environment validation failed: {str(e)}")
            result["status"] = "failed"
            self.log(f"Python environment validation failed: {e}", "error")
        
        return result
    
    def validate_dependencies(self) -> Dict[str, Any]:
        """Validate all required dependencies"""
        self.log("Validating dependencies...")
        
        result = {
            "component": "Dependencies",
            "status": "unknown",
            "details": {},
            "errors": [],
            "warnings": []
        }
        
        # Core dependencies
        core_deps = [
            "streamlit", "openai", "requests", "python-dotenv",
            "numpy", "pathlib", "uuid", "tempfile", "threading"
        ]
        
        # Audio dependencies
        audio_deps = [
            "sounddevice", "soundfile", "pyaudio"
        ]
        
        # Advanced dependencies
        advanced_deps = [
            "RealtimeSTT", "rich", "opencv-python", "pillow",
            "pytesseract", "pyautogui", "psutil"
        ]
        
        # Optional dependencies
        optional_deps = [
            "yaml", "markdown", "pydantic"
        ]
        
        all_deps = {
            "core": core_deps,
            "audio": audio_deps,
            "advanced": advanced_deps,
            "optional": optional_deps
        }
        
        for category, deps in all_deps.items():
            result["details"][category] = {}
            
            for dep in deps:
                try:
                    if dep == "pathlib":
                        # pathlib is built-in in Python 3.4+
                        import pathlib
                        result["details"][category][dep] = "built-in"
                    elif dep in ["uuid", "tempfile", "threading"]:
                        # These are built-in modules
                        importlib.import_module(dep)
                        result["details"][category][dep] = "built-in"
                    else:
                        # Try to import the module
                        module = importlib.import_module(dep.replace("-", "_"))
                        
                        # Try to get version if available
                        version = getattr(module, "__version__", "unknown")
                        result["details"][category][dep] = version
                        
                except ImportError as e:
                    result["details"][category][dep] = "missing"
                    
                    if category == "core":
                        result["errors"].append(f"Core dependency missing: {dep}")
                    elif category == "audio":
                        result["warnings"].append(f"Audio dependency missing: {dep}")
                    elif category == "advanced":
                        result["warnings"].append(f"Advanced feature unavailable: {dep}")
                    # Optional dependencies don't generate warnings
        
        # Determine overall status
        if result["errors"]:
            result["status"] = "failed"
            self.log("Dependency validation failed", "error")
        elif result["warnings"]:
            result["status"] = "partial"
            self.log("Dependency validation passed with warnings", "warning")
        else:
            result["status"] = "passed"
            self.log("All dependencies validated successfully", "success")
        
        return result
    
    def validate_api_keys(self) -> Dict[str, Any]:
        """Validate API key configuration"""
        self.log("Validating API keys...")
        
        result = {
            "component": "API Keys",
            "status": "unknown",
            "details": {},
            "errors": [],
            "warnings": []
        }
        
        # Check for .env file
        env_file = Path(".env")
        if env_file.exists():
            result["details"]["env_file_exists"] = True
            self.log("Found .env file", "success")
        else:
            result["warnings"].append("No .env file found")
            result["details"]["env_file_exists"] = False
        
        # Check API keys
        api_keys = {
            "OPENAI_API_KEY": "OpenAI (required for TTS)",
            "DEEPSEEK_API_KEY": "DeepSeek (required for DeepSeek assistant)",
            "ANTHROPIC_API_KEY": "Anthropic (required for Claude integration)"
        }
        
        for key, description in api_keys.items():
            value = os.getenv(key)
            if value:
                # Mask the key for security
                masked_value = f"{value[:8]}...{value[-4:]}" if len(value) > 12 else "***"
                result["details"][key] = f"configured ({masked_value})"
                self.log(f"{description} key found", "success")
            else:
                result["details"][key] = "missing"
                result["warnings"].append(f"{description} key not found")
        
        # Determine status
        if not any(os.getenv(key) for key in api_keys.keys()):
            result["status"] = "failed"
            result["errors"].append("No API keys configured")
        elif os.getenv("OPENAI_API_KEY"):
            result["status"] = "partial"  # At least basic functionality available
        else:
            result["status"] = "failed"
            result["errors"].append("OpenAI API key required for basic functionality")
        
        return result
    
    def validate_audio_system(self) -> Dict[str, Any]:
        """Validate audio recording and playback capabilities"""
        self.log("Validating audio system...")
        
        result = {
            "component": "Audio System",
            "status": "unknown",
            "details": {},
            "errors": [],
            "warnings": []
        }
        
        try:
            # Test sounddevice
            try:
                import sounddevice as sd
                
                # Get audio devices
                devices = sd.query_devices()
                input_devices = [d for d in devices if d['max_input_channels'] > 0]
                output_devices = [d for d in devices if d['max_output_channels'] > 0]
                
                result["details"]["input_devices"] = len(input_devices)
                result["details"]["output_devices"] = len(output_devices)
                result["details"]["default_input"] = sd.query_devices(kind='input')['name']
                result["details"]["default_output"] = sd.query_devices(kind='output')['name']
                
                if not input_devices:
                    result["errors"].append("No audio input devices found")
                if not output_devices:
                    result["errors"].append("No audio output devices found")
                
                # Test basic recording capability (without actually recording)
                try:
                    with sd.InputStream(samplerate=16000, channels=1, dtype='float32'):
                        result["details"]["recording_test"] = "passed"
                except Exception as e:
                    result["errors"].append(f"Recording test failed: {str(e)}")
                    result["details"]["recording_test"] = "failed"
                
                self.log("sounddevice validation passed", "success")
                
            except ImportError:
                result["warnings"].append("sounddevice not available")
                result["details"]["sounddevice"] = "missing"
            
            # Test soundfile
            try:
                import soundfile as sf
                result["details"]["soundfile"] = "available"
                self.log("soundfile validation passed", "success")
            except ImportError:
                result["warnings"].append("soundfile not available")
                result["details"]["soundfile"] = "missing"
            
            # Test pyaudio
            try:
                import pyaudio
                result["details"]["pyaudio"] = "available"
                self.log("pyaudio validation passed", "success")
            except ImportError:
                result["warnings"].append("pyaudio not available")
                result["details"]["pyaudio"] = "missing"
            
            # Test RealtimeSTT
            try:
                import RealtimeSTT
                result["details"]["RealtimeSTT"] = "available"
                self.log("RealtimeSTT validation passed", "success")
            except ImportError:
                result["warnings"].append("RealtimeSTT not available")
                result["details"]["RealtimeSTT"] = "missing"
            
            # Determine overall status
            if result["errors"]:
                result["status"] = "failed"
            elif result["warnings"]:
                result["status"] = "partial"
            else:
                result["status"] = "passed"
                
        except Exception as e:
            result["errors"].append(f"Audio system validation failed: {str(e)}")
            result["status"] = "failed"
            self.log(f"Audio system validation failed: {e}", "error")
        
        return result
    
    def validate_api_connectivity(self) -> Dict[str, Any]:
        """Validate API connectivity"""
        self.log("Validating API connectivity...")
        
        result = {
            "component": "API Connectivity",
            "status": "unknown",
            "details": {},
            "errors": [],
            "warnings": []
        }
        
        # Test OpenAI API
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=openai_key)
                
                # Test with a simple completion
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=5
                )
                
                result["details"]["openai"] = "connected"
                self.log("OpenAI API connection successful", "success")
                
            except Exception as e:
                result["errors"].append(f"OpenAI API connection failed: {str(e)}")
                result["details"]["openai"] = "failed"
        else:
            result["details"]["openai"] = "no_key"
        
        # Test DeepSeek API
        deepseek_key = os.getenv("DEEPSEEK_API_KEY")
        if deepseek_key:
            try:
                import requests
                
                headers = {
                    "Authorization": f"Bearer {deepseek_key}",
                    "Content-Type": "application/json"
                }
                
                data = {
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 5
                }
                
                response = requests.post(
                    "https://api.deepseek.com/v1/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=10
                )
                
                if response.status_code == 200:
                    result["details"]["deepseek"] = "connected"
                    self.log("DeepSeek API connection successful", "success")
                else:
                    result["errors"].append(f"DeepSeek API returned {response.status_code}")
                    result["details"]["deepseek"] = "failed"
                    
            except Exception as e:
                result["errors"].append(f"DeepSeek API connection failed: {str(e)}")
                result["details"]["deepseek"] = "failed"
        else:
            result["details"]["deepseek"] = "no_key"
        
        # Determine overall status
        connected_apis = sum(1 for v in result["details"].values() if v == "connected")
        if connected_apis == 0:
            result["status"] = "failed"
            result["errors"].append("No API connections successful")
        elif result["errors"]:
            result["status"] = "partial"
        else:
            result["status"] = "passed"
        
        return result
    
    def validate_file_structure(self) -> Dict[str, Any]:
        """Validate project file structure"""
        self.log("Validating file structure...")
        
        result = {
            "component": "File Structure",
            "status": "unknown",
            "details": {},
            "errors": [],
            "warnings": []
        }
        
        # Expected files and directories
        expected_structure = {
            "files": [
                "voice_computer_assistant.py",
                "deepseek_voice_assistant.py",
                "requirements.txt",
                "deepseek_requirements.txt",
                "setup.py",
                "setup_deepseek_assistant.py",
                "demo.py",
                "demo_deepseek_assistant.py",
                "component_validator.py",
                "README.md",
                "README_DeepSeek_Assistant.md"
            ],
            "directories": [
                "agents",
                "utils",
                "config"
            ],
            "agent_files": [
                "agents/__init__.py",
                "agents/voice_agent.py",
                "agents/vision_agent.py",
                "agents/action_agent.py",
                "agents/coordinator_agent.py"
            ],
            "util_files": [
                "utils/__init__.py",
                "utils/audio_utils.py",
                "utils/safety_controls.py"
            ],
            "config_files": [
                "config/__init__.py",
                "config/settings.py"
            ]
        }
        
        # Check files
        for file_path in expected_structure["files"]:
            if Path(file_path).exists():
                result["details"][file_path] = "exists"
            else:
                result["warnings"].append(f"Missing file: {file_path}")
                result["details"][file_path] = "missing"
        
        # Check directories
        for dir_path in expected_structure["directories"]:
            if Path(dir_path).is_dir():
                result["details"][dir_path] = "exists"
            else:
                result["errors"].append(f"Missing directory: {dir_path}")
                result["details"][dir_path] = "missing"
        
        # Check agent files
        for file_path in expected_structure["agent_files"]:
            if Path(file_path).exists():
                result["details"][file_path] = "exists"
            else:
                result["errors"].append(f"Missing agent file: {file_path}")
                result["details"][file_path] = "missing"
        
        # Check util files
        for file_path in expected_structure["util_files"]:
            if Path(file_path).exists():
                result["details"][file_path] = "exists"
            else:
                result["errors"].append(f"Missing util file: {file_path}")
                result["details"][file_path] = "missing"
        
        # Check config files
        for file_path in expected_structure["config_files"]:
            if Path(file_path).exists():
                result["details"][file_path] = "exists"
            else:
                result["errors"].append(f"Missing config file: {file_path}")
                result["details"][file_path] = "missing"
        
        # Determine status
        if result["errors"]:
            result["status"] = "failed"
        elif result["warnings"]:
            result["status"] = "partial"
        else:
            result["status"] = "passed"
            self.log("File structure validation passed", "success")
        
        return result
    
    def validate_imports(self) -> Dict[str, Any]:
        """Validate that all modules can be imported"""
        self.log("Validating module imports...")
        
        result = {
            "component": "Module Imports",
            "status": "unknown",
            "details": {},
            "errors": [],
            "warnings": []
        }
        
        # Test importing our modules
        modules_to_test = [
            ("agents.voice_agent", "VoiceAgent"),
            ("agents.vision_agent", "VisionAgent"),
            ("agents.action_agent", "ActionAgent"),
            ("agents.coordinator_agent", "CoordinatorAgent"),
            ("utils.audio_utils", "AudioRecorder"),
            ("utils.safety_controls", "SafetyManager"),
            ("config.settings", "AppConfig")
        ]
        
        for module_name, class_name in modules_to_test:
            try:
                module = importlib.import_module(module_name)
                if hasattr(module, class_name):
                    result["details"][f"{module_name}.{class_name}"] = "importable"
                    self.log(f"Successfully imported {module_name}.{class_name}", "success")
                else:
                    result["errors"].append(f"Class {class_name} not found in {module_name}")
                    result["details"][f"{module_name}.{class_name}"] = "class_missing"
                    
            except ImportError as e:
                result["errors"].append(f"Failed to import {module_name}: {str(e)}")
                result["details"][f"{module_name}.{class_name}"] = "import_failed"
            except Exception as e:
                result["errors"].append(f"Error importing {module_name}: {str(e)}")
                result["details"][f"{module_name}.{class_name}"] = "error"
        
        # Determine status
        if result["errors"]:
            result["status"] = "failed"
        else:
            result["status"] = "passed"
            self.log("Module import validation passed", "success")
        
        return result
    
    async def run_all_validations(self) -> Dict[str, Any]:
        """Run all validation tests"""
        if self.console:
            self.console.print(Panel.fit(
                "[bold blue]🔍 Voice Assistant Component Validator[/bold blue]\n"
                "Validating all components of the voice assistant systems...",
                title="Component Validation",
                border_style="blue"
            ))
        
        validations = [
            ("Python Environment", self.validate_python_environment),
            ("Dependencies", self.validate_dependencies),
            ("API Keys", self.validate_api_keys),
            ("Audio System", self.validate_audio_system),
            ("API Connectivity", self.validate_api_connectivity),
            ("File Structure", self.validate_file_structure),
            ("Module Imports", self.validate_imports)
        ]
        
        results = {}
        
        if self.console:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:
                for name, validation_func in validations:
                    task = progress.add_task(f"Validating {name}...", total=None)
                    results[name] = validation_func()
                    progress.remove_task(task)
        else:
            for name, validation_func in validations:
                print(f"Validating {name}...")
                results[name] = validation_func()
        
        return results
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive validation report"""
        if self.console:
            # Rich formatted report
            table = Table(title="Component Validation Results")
            table.add_column("Component", style="cyan", no_wrap=True)
            table.add_column("Status", style="magenta")
            table.add_column("Details", style="green")
            table.add_column("Issues", style="red")
            
            for component, result in results.items():
                status = result["status"]
                status_emoji = {
                    "passed": "✅ PASSED",
                    "partial": "⚠️ PARTIAL",
                    "failed": "❌ FAILED",
                    "unknown": "❓ UNKNOWN"
                }.get(status, status)
                
                details = f"{len(result['details'])} items checked"
                issues = f"{len(result['errors'])} errors, {len(result['warnings'])} warnings"
                
                table.add_row(component, status_emoji, details, issues)
            
            self.console.print(table)
            
            # Summary
            passed = sum(1 for r in results.values() if r["status"] == "passed")
            partial = sum(1 for r in results.values() if r["status"] == "partial")
            failed = sum(1 for r in results.values() if r["status"] == "failed")
            total = len(results)
            
            summary = f"""
## Validation Summary

- ✅ **Passed**: {passed}/{total} components
- ⚠️ **Partial**: {partial}/{total} components  
- ❌ **Failed**: {failed}/{total} components

### Overall Status: {"🎉 READY" if failed == 0 else "⚠️ NEEDS ATTENTION" if passed > failed else "❌ CRITICAL ISSUES"}
"""
            
            self.console.print(Panel(Markdown(summary), title="Summary", border_style="green" if failed == 0 else "yellow"))
            
        # Generate text report
        report_lines = [
            "# Voice Assistant Component Validation Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            ""
        ]
        
        for component, result in results.items():
            report_lines.extend([
                f"### {component}",
                f"**Status**: {result['status'].upper()}",
                f"**Errors**: {len(result['errors'])}",
                f"**Warnings**: {len(result['warnings'])}",
                ""
            ])
            
            if result['errors']:
                report_lines.append("**Errors:**")
                for error in result['errors']:
                    report_lines.append(f"- {error}")
                report_lines.append("")
            
            if result['warnings']:
                report_lines.append("**Warnings:**")
                for warning in result['warnings']:
                    report_lines.append(f"- {warning}")
                report_lines.append("")
            
            if result['details']:
                report_lines.append("**Details:**")
                for key, value in result['details'].items():
                    report_lines.append(f"- {key}: {value}")
                report_lines.append("")
        
        return "\n".join(report_lines)

async def main():
    """Main validation function"""
    validator = ComponentValidator()
    
    try:
        # Run all validations
        results = await validator.run_all_validations()
        
        # Generate and save report
        report = validator.generate_report(results)
        
        # Save report to file
        report_file = Path("validation_report.md")
        with open(report_file, "w") as f:
            f.write(report)
        
        validator.log(f"Validation report saved to {report_file}", "success")
        
        # Return results for programmatic use
        return results
        
    except Exception as e:
        validator.log(f"Validation failed: {str(e)}", "error")
        return None

if __name__ == "__main__":
    asyncio.run(main())

