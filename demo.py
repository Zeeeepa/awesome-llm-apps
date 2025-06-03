#!/usr/bin/env python3
"""
Voice Computer Assistant Demo Script

This script demonstrates the key capabilities of the voice computer assistant
without requiring the full Streamlit interface.
"""

import asyncio
import os
from pathlib import Path

# Import our components
from config.settings import AppConfig
from agents.voice_agent import VoiceAgent
from agents.vision_agent import VisionAgent
from agents.action_agent import ActionAgent
from agents.coordinator_agent import CoordinatorAgent
from utils.safety_controls import SafetyManager

async def demo_voice_processing():
    """Demo voice processing capabilities"""
    print("🎤 Testing Voice Processing...")
    
    # Check if API key is available
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
        return False
    
    config = AppConfig(openai_api_key=api_key)
    voice_agent = VoiceAgent(config)
    
    # Test TTS
    print("  🔊 Testing Text-to-Speech...")
    audio_path = await voice_agent.text_to_speech("Hello! I am your voice computer assistant.")
    
    if audio_path:
        print(f"  ✅ TTS successful! Audio saved to: {audio_path}")
    else:
        print("  ❌ TTS failed")
        return False
    
    # Test intent analysis
    print("  🧠 Testing Intent Analysis...")
    test_commands = [
        "Click the save button",
        "Open Chrome browser",
        "Type hello world",
        "Scroll down on the page"
    ]
    
    for command in test_commands:
        result = await voice_agent._analyze_command_intent(command)
        print(f"    Command: '{command}' -> Intent: {result.get('intent', 'unknown')}")
    
    return True

async def demo_vision_capabilities():
    """Demo computer vision capabilities"""
    print("\n👁️ Testing Vision Capabilities...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OpenAI API key required for vision testing")
        return False
    
    config = AppConfig(openai_api_key=api_key)
    vision_agent = VisionAgent(config)
    
    # Test screen capture
    print("  📸 Testing Screen Capture...")
    screenshot_path = vision_agent.capture_screen()
    
    if screenshot_path:
        print(f"  ✅ Screenshot captured: {screenshot_path}")
        
        # Test AI analysis
        print("  🤖 Testing AI Screen Analysis...")
        analysis = await vision_agent.analyze_screen_with_ai(
            screenshot_path, 
            "Describe the main elements visible on this screen"
        )
        
        if analysis["success"]:
            print(f"  ✅ AI Analysis successful!")
            print(f"    Analysis preview: {analysis['analysis'][:100]}...")
        else:
            print(f"  ❌ AI Analysis failed: {analysis.get('error', 'Unknown error')}")
    else:
        print("  ❌ Screen capture failed")
        return False
    
    return True

async def demo_action_capabilities():
    """Demo action execution capabilities"""
    print("\n🤖 Testing Action Capabilities...")
    
    config = AppConfig(openai_api_key="dummy", safety_mode="high")
    safety_manager = SafetyManager("high")
    action_agent = ActionAgent(config, safety_manager)
    
    # Test mouse position
    print("  🖱️ Testing Mouse Position...")
    try:
        mouse_pos = await action_agent.get_mouse_position()
        print(f"  ✅ Current mouse position: {mouse_pos}")
    except Exception as e:
        print(f"  ❌ Mouse position test failed: {e}")
        return False
    
    # Test window list
    print("  🪟 Testing Window Detection...")
    try:
        windows = await action_agent.get_window_list()
        print(f"  ✅ Found {len(windows)} open windows")
        for window in windows[:3]:  # Show first 3 windows
            print(f"    - {window['title'][:50]}...")
    except Exception as e:
        print(f"  ❌ Window detection failed: {e}")
    
    return True

async def demo_safety_controls():
    """Demo safety control system"""
    print("\n🛡️ Testing Safety Controls...")
    
    safety_manager = SafetyManager("high")
    
    # Test various actions
    test_cases = [
        ("click", "save button", True),
        ("open_application", "chrome", True),
        ("open_application", "terminal", False),
        ("type", "hello world", True),
        ("type", "rm -rf /", False),
        ("key_press", "enter", True),
        ("key_press", "ctrl+alt+del", False)
    ]
    
    print("  🧪 Testing Safety Rules...")
    for action_type, target, expected_safe in test_cases:
        is_safe = safety_manager.is_action_safe(action_type, target)
        status = "✅" if is_safe == expected_safe else "❌"
        print(f"    {status} {action_type}('{target}') -> {'Safe' if is_safe else 'Blocked'}")
    
    # Test safety status
    status = safety_manager.get_safety_status()
    print(f"  📊 Safety Status: Mode={status['safety_mode']}, Emergency={status['emergency_stop_active']}")
    
    return True

async def demo_full_workflow():
    """Demo the complete voice-to-action workflow"""
    print("\n🔄 Testing Complete Workflow...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OpenAI API key required for full workflow demo")
        return False
    
    # Initialize all components
    config = AppConfig(openai_api_key=api_key, safety_mode="high")
    safety_manager = SafetyManager("high")
    
    voice_agent = VoiceAgent(config)
    vision_agent = VisionAgent(config)
    action_agent = ActionAgent(config, safety_manager)
    
    coordinator = CoordinatorAgent(voice_agent, vision_agent, action_agent, config)
    
    # Test command processing
    test_commands = [
        "Take a screenshot",
        "Get the current mouse position",
        "Find any buttons on the screen"
    ]
    
    for command in test_commands:
        print(f"  🎯 Processing: '{command}'")
        try:
            result = await coordinator.process_command(command)
            
            if result["success"]:
                print(f"    ✅ Success: {result['response']}")
            else:
                print(f"    ❌ Failed: {result.get('response', 'Unknown error')}")
        except Exception as e:
            print(f"    ❌ Error: {e}")
    
    return True

async def main():
    """Run all demo functions"""
    print("🎤💻 Voice Computer Assistant Demo")
    print("=" * 50)
    
    # Check dependencies
    print("🔍 Checking Dependencies...")
    
    missing_deps = []
    try:
        import cv2
        print("  ✅ OpenCV available")
    except ImportError:
        missing_deps.append("opencv-python")
        print("  ❌ OpenCV not available")
    
    try:
        import pyautogui
        print("  ✅ PyAutoGUI available")
    except ImportError:
        missing_deps.append("pyautogui")
        print("  ❌ PyAutoGUI not available")
    
    try:
        import PIL
        print("  ✅ Pillow available")
    except ImportError:
        missing_deps.append("pillow")
        print("  ❌ Pillow not available")
    
    if missing_deps:
        print(f"\n⚠️ Missing dependencies: {', '.join(missing_deps)}")
        print("Install with: pip install " + " ".join(missing_deps))
        print("\nContinuing with available features...\n")
    
    # Run demos
    demos = [
        ("Voice Processing", demo_voice_processing),
        ("Vision Capabilities", demo_vision_capabilities),
        ("Action Capabilities", demo_action_capabilities),
        ("Safety Controls", demo_safety_controls),
        ("Full Workflow", demo_full_workflow)
    ]
    
    results = {}
    
    for demo_name, demo_func in demos:
        try:
            print(f"\n{'='*20} {demo_name} {'='*20}")
            success = await demo_func()
            results[demo_name] = success
        except Exception as e:
            print(f"❌ Demo '{demo_name}' failed with error: {e}")
            results[demo_name] = False
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 Demo Results Summary:")
    print("=" * 50)
    
    for demo_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {demo_name}: {status}")
    
    passed = sum(results.values())
    total = len(results)
    print(f"\nOverall: {passed}/{total} demos passed")
    
    if passed == total:
        print("🎉 All demos passed! The voice computer assistant is ready to use.")
        print("\nTo start the full application, run:")
        print("  streamlit run voice_computer_assistant.py")
    else:
        print("⚠️ Some demos failed. Check the error messages above.")
        print("Make sure all dependencies are installed and API keys are configured.")

if __name__ == "__main__":
    asyncio.run(main())

