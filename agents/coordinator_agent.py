"""
Coordinator Agent that orchestrates between Voice, Vision, and Action agents
"""
import asyncio
from typing import Dict, Any, List, Optional
import json
import time
from datetime import datetime

from agents.voice_agent import VoiceAgent
from agents.vision_agent import VisionAgent, UIElement
from agents.action_agent import ActionAgent
from config.settings import AppConfig

class CoordinatorAgent:
    """Orchestrates the complete voice-to-action workflow"""
    
    def __init__(self, voice_agent: VoiceAgent, vision_agent: VisionAgent, 
                 action_agent: ActionAgent, config: AppConfig):
        self.voice_agent = voice_agent
        self.vision_agent = vision_agent
        self.action_agent = action_agent
        self.config = config
        self.conversation_context = []
        self.last_screenshot = None
        self.last_elements = []
    
    async def process_command(self, command: str, is_voice: bool = False) -> Dict[str, Any]:
        """Process a complete command from text/voice to action execution"""
        try:
            start_time = time.time()
            
            # Step 1: Analyze command intent
            intent_result = await self._analyze_command_intent(command)
            
            if not intent_result["success"]:
                return await self._create_error_response(
                    "I couldn't understand your command. Could you please rephrase it?",
                    command
                )
            
            intent = intent_result["intent"]
            target = intent_result.get("target", "")
            parameters = intent_result.get("parameters", {})
            
            # Step 2: Capture and analyze screen if needed
            screen_analysis = None
            if self._requires_screen_analysis(intent):
                screen_analysis = await self._analyze_current_screen(target)
                
                if not screen_analysis["success"]:
                    return await self._create_error_response(
                        "I couldn't analyze the screen. Please try again.",
                        command
                    )
            
            # Step 3: Plan and execute actions
            execution_result = await self._plan_and_execute_actions(
                intent, target, parameters, screen_analysis
            )
            
            # Step 4: Generate response
            response_text = await self._generate_response(
                command, intent, execution_result
            )
            
            # Step 5: Convert response to speech
            audio_path = await self.voice_agent.text_to_speech(response_text)
            
            # Step 6: Update conversation context
            self._update_conversation_context(command, intent, execution_result, response_text)
            
            processing_time = time.time() - start_time
            
            return {
                "success": True,
                "response": response_text,
                "audio_path": audio_path,
                "intent": intent,
                "target": target,
                "parameters": parameters,
                "action_performed": execution_result.get("success", False),
                "action_type": execution_result.get("action_type", "unknown"),
                "details": execution_result,
                "processing_time": processing_time,
                "status": "completed"
            }
            
        except Exception as e:
            error_response = f"I encountered an error while processing your command: {str(e)}"
            audio_path = await self.voice_agent.text_to_speech(error_response)
            
            return {
                "success": False,
                "response": error_response,
                "audio_path": audio_path,
                "error": str(e),
                "status": "error"
            }
    
    async def _analyze_command_intent(self, command: str) -> Dict[str, Any]:
        """Analyze the command to extract intent and parameters"""
        try:
            # Use the voice agent's intent analysis
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=self.config.openai_api_key)
            
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a computer command analyzer. Analyze user commands and extract:
                        1. Intent (the main action to perform)
                        2. Target (what to interact with)
                        3. Parameters (specific details like text, coordinates, etc.)
                        4. Confidence (0.0-1.0)
                        
                        Common intents:
                        - click: Click on something
                        - type: Type text
                        - open: Open application/file/website
                        - navigate: Go to a location/URL
                        - scroll: Scroll up/down
                        - drag: Drag and drop
                        - select: Select text/items
                        - copy/paste: Clipboard operations
                        - save: Save file/document
                        - close: Close window/application
                        - search: Search for something
                        - find: Find element on screen
                        
                        Respond in JSON format:
                        {
                            "success": true,
                            "intent": "action_type",
                            "target": "element_description",
                            "parameters": {"key": "value"},
                            "confidence": 0.95,
                            "requires_screen": true/false
                        }"""
                    },
                    {
                        "role": "user",
                        "content": f"Analyze this command: '{command}'"
                    }
                ],
                temperature=0.1
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            print(f"Error analyzing command intent: {e}")
            return {
                "success": False,
                "error": str(e),
                "intent": "unknown",
                "target": command,
                "parameters": {},
                "confidence": 0.0
            }
    
    def _requires_screen_analysis(self, intent: str) -> bool:
        """Check if the intent requires screen analysis"""
        screen_required_intents = [
            "click", "find", "select", "drag", "scroll", "read", "describe"
        ]
        return intent.lower() in screen_required_intents
    
    async def _analyze_current_screen(self, target: str = "") -> Dict[str, Any]:
        """Capture and analyze the current screen"""
        try:
            # Capture screenshot
            screenshot_path = self.vision_agent.capture_screen()
            if not screenshot_path:
                return {"success": False, "error": "Failed to capture screen"}
            
            self.last_screenshot = screenshot_path
            
            # Get UI elements
            elements = await self.vision_agent.get_screen_elements(refresh=True)
            self.last_elements = elements
            
            # If we have a specific target, try to find it
            target_element = None
            if target:
                target_element = await self.vision_agent.find_element_by_description(target, screenshot_path)
            
            # Get AI analysis of the screen
            ai_analysis = await self.vision_agent.analyze_screen_with_ai(
                screenshot_path, 
                f"Analyze this screen. Focus on finding: {target}" if target else "Describe the main UI elements visible."
            )
            
            return {
                "success": True,
                "screenshot_path": screenshot_path,
                "elements": [elem.to_dict() for elem in elements],
                "target_element": target_element.to_dict() if target_element else None,
                "ai_analysis": ai_analysis.get("analysis", ""),
                "element_count": len(elements)
            }
            
        except Exception as e:
            print(f"Error analyzing screen: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _plan_and_execute_actions(self, intent: str, target: str, 
                                      parameters: Dict[str, Any], 
                                      screen_analysis: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Plan and execute the appropriate actions based on intent"""
        try:
            action_type = intent.lower()
            
            if action_type == "click":
                return await self._execute_click_action(target, parameters, screen_analysis)
            
            elif action_type == "type":
                return await self._execute_type_action(target, parameters)
            
            elif action_type == "open":
                return await self._execute_open_action(target, parameters)
            
            elif action_type == "scroll":
                return await self._execute_scroll_action(target, parameters, screen_analysis)
            
            elif action_type == "drag":
                return await self._execute_drag_action(target, parameters, screen_analysis)
            
            elif action_type in ["key_press", "press"]:
                return await self._execute_key_press_action(target, parameters)
            
            elif action_type == "navigate":
                return await self._execute_navigate_action(target, parameters)
            
            elif action_type in ["find", "search"]:
                return await self._execute_find_action(target, parameters, screen_analysis)
            
            else:
                return {
                    "success": False,
                    "error": f"Unknown action type: {action_type}",
                    "action_type": action_type
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "action_type": intent
            }
    
    async def _execute_click_action(self, target: str, parameters: Dict[str, Any], 
                                  screen_analysis: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute a click action"""
        try:
            # Check if we have specific coordinates
            if "x" in parameters and "y" in parameters:
                x, y = parameters["x"], parameters["y"]
            elif screen_analysis and screen_analysis.get("target_element"):
                # Use the found target element
                element = screen_analysis["target_element"]
                x, y = element["center"]
            else:
                # Try to find the element by description
                if not screen_analysis:
                    return {
                        "success": False,
                        "error": "No screen analysis available and no coordinates provided",
                        "action_type": "click"
                    }
                
                # Look for elements that match the target description
                elements = screen_analysis.get("elements", [])
                matching_element = None
                
                for element in elements:
                    if target.lower() in element.get("text", "").lower() or \
                       target.lower() in element.get("description", "").lower():
                        matching_element = element
                        break
                
                if not matching_element:
                    return {
                        "success": False,
                        "error": f"Could not find '{target}' on the screen",
                        "action_type": "click"
                    }
                
                x, y = matching_element["center"]
            
            # Check if confirmation is required
            if self.config.requires_confirmation("click", target):
                # In a real implementation, this would show a confirmation dialog
                # For now, we'll proceed with high safety mode restrictions
                pass
            
            # Execute the click
            button = parameters.get("button", "left")
            clicks = parameters.get("clicks", 1)
            
            result = await self.action_agent.click(x, y, button, clicks)
            result["action_type"] = "click"
            result["target"] = target
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "action_type": "click"
            }
    
    async def _execute_type_action(self, target: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a typing action"""
        try:
            text = parameters.get("text", target)  # Use target as text if no specific text provided
            
            if not text:
                return {
                    "success": False,
                    "error": "No text provided to type",
                    "action_type": "type"
                }
            
            # Check if confirmation is required
            if self.config.requires_confirmation("type", text):
                pass  # Confirmation logic would go here
            
            result = await self.action_agent.type_text(text)
            result["action_type"] = "type"
            result["target"] = text
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "action_type": "type"
            }
    
    async def _execute_open_action(self, target: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an open application action"""
        try:
            app_name = target
            
            if not app_name:
                return {
                    "success": False,
                    "error": "No application name provided",
                    "action_type": "open"
                }
            
            result = await self.action_agent.open_application(app_name)
            result["action_type"] = "open"
            result["target"] = app_name
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "action_type": "open"
            }
    
    async def _execute_scroll_action(self, target: str, parameters: Dict[str, Any], 
                                   screen_analysis: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute a scroll action"""
        try:
            # Get scroll parameters
            direction = parameters.get("direction", "up")
            clicks = parameters.get("clicks", 3)
            
            # Use center of screen if no specific coordinates
            x = parameters.get("x", 960)  # Default to center
            y = parameters.get("y", 540)
            
            result = await self.action_agent.scroll(x, y, direction, clicks)
            result["action_type"] = "scroll"
            result["target"] = target
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "action_type": "scroll"
            }
    
    async def _execute_drag_action(self, target: str, parameters: Dict[str, Any], 
                                 screen_analysis: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute a drag action"""
        try:
            # Get drag coordinates
            start_x = parameters.get("start_x")
            start_y = parameters.get("start_y")
            end_x = parameters.get("end_x")
            end_y = parameters.get("end_y")
            
            if None in [start_x, start_y, end_x, end_y]:
                return {
                    "success": False,
                    "error": "Drag action requires start and end coordinates",
                    "action_type": "drag"
                }
            
            duration = parameters.get("duration", 1.0)
            
            result = await self.action_agent.drag(start_x, start_y, end_x, end_y, duration)
            result["action_type"] = "drag"
            result["target"] = target
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "action_type": "drag"
            }
    
    async def _execute_key_press_action(self, target: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a key press action"""
        try:
            key = parameters.get("key", target)
            presses = parameters.get("presses", 1)
            
            if not key:
                return {
                    "success": False,
                    "error": "No key specified",
                    "action_type": "key_press"
                }
            
            result = await self.action_agent.press_key(key, presses)
            result["action_type"] = "key_press"
            result["target"] = key
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "action_type": "key_press"
            }
    
    async def _execute_navigate_action(self, target: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a navigation action (e.g., open URL)"""
        try:
            url = target
            
            # If it looks like a URL, open it in browser
            if "." in url and not url.startswith("http"):
                url = f"https://{url}"
            
            # Open browser first, then navigate
            browser_result = await self.action_agent.open_application("chrome")
            
            if browser_result["success"]:
                # Wait a moment for browser to open
                await asyncio.sleep(2)
                
                # Type the URL in address bar (Ctrl+L to focus address bar)
                await self.action_agent.press_key("ctrl+l")
                await asyncio.sleep(0.5)
                await self.action_agent.type_text(url)
                await self.action_agent.press_key("enter")
                
                return {
                    "success": True,
                    "action_type": "navigate",
                    "target": url,
                    "details": "Opened browser and navigated to URL"
                }
            else:
                return browser_result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "action_type": "navigate"
            }
    
    async def _execute_find_action(self, target: str, parameters: Dict[str, Any], 
                                 screen_analysis: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute a find/search action"""
        try:
            if not screen_analysis:
                return {
                    "success": False,
                    "error": "No screen analysis available for search",
                    "action_type": "find"
                }
            
            # Look for the target in the screen analysis
            elements = screen_analysis.get("elements", [])
            ai_analysis = screen_analysis.get("ai_analysis", "")
            
            found_elements = []
            for element in elements:
                if target.lower() in element.get("text", "").lower() or \
                   target.lower() in element.get("description", "").lower():
                    found_elements.append(element)
            
            if found_elements:
                return {
                    "success": True,
                    "action_type": "find",
                    "target": target,
                    "found_elements": found_elements,
                    "count": len(found_elements),
                    "details": f"Found {len(found_elements)} matching elements"
                }
            else:
                return {
                    "success": False,
                    "action_type": "find",
                    "target": target,
                    "error": f"Could not find '{target}' on the screen",
                    "ai_analysis": ai_analysis
                }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "action_type": "find"
            }
    
    async def _generate_response(self, command: str, intent: str, execution_result: Dict[str, Any]) -> str:
        """Generate a natural language response about the action performed"""
        try:
            if execution_result.get("success", False):
                action_type = execution_result.get("action_type", intent)
                target = execution_result.get("target", "")
                
                # Generate contextual responses based on action type
                if action_type == "click":
                    return f"I clicked on {target} for you."
                elif action_type == "type":
                    return f"I typed '{target}' as requested."
                elif action_type == "open":
                    return f"I opened {target}."
                elif action_type == "scroll":
                    direction = execution_result.get("direction", "up")
                    return f"I scrolled {direction} on the screen."
                elif action_type == "navigate":
                    return f"I navigated to {target}."
                elif action_type == "find":
                    count = execution_result.get("count", 0)
                    if count > 0:
                        return f"I found {count} instances of '{target}' on the screen."
                    else:
                        return f"I couldn't find '{target}' on the current screen."
                else:
                    return f"I completed the {action_type} action successfully."
            else:
                error = execution_result.get("error", "Unknown error")
                return f"I couldn't complete that action. {error}"
                
        except Exception as e:
            return f"I tried to help, but encountered an issue: {str(e)}"
    
    async def _create_error_response(self, message: str, original_command: str) -> Dict[str, Any]:
        """Create an error response with audio"""
        audio_path = await self.voice_agent.text_to_speech(message)
        
        return {
            "success": False,
            "response": message,
            "audio_path": audio_path,
            "original_command": original_command,
            "status": "error"
        }
    
    def _update_conversation_context(self, command: str, intent: str, 
                                   execution_result: Dict[str, Any], response: str):
        """Update conversation context for future reference"""
        context_entry = {
            "timestamp": datetime.now(),
            "command": command,
            "intent": intent,
            "execution_result": execution_result,
            "response": response,
            "success": execution_result.get("success", False)
        }
        
        self.conversation_context.append(context_entry)
        
        # Keep only last 10 interactions
        if len(self.conversation_context) > 10:
            self.conversation_context = self.conversation_context[-10:]
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history"""
        return self.conversation_context
    
    async def test_coordinator_setup(self) -> Dict[str, Any]:
        """Test the coordinator setup and all agents"""
        try:
            # Test all agents
            voice_test = await self.voice_agent.test_voice_setup()
            vision_test = await self.vision_agent.test_vision_setup()
            action_test = await self.action_agent.test_action_setup()
            
            return {
                "success": True,
                "voice_agent": voice_test,
                "vision_agent": vision_test,
                "action_agent": action_test,
                "coordinator_ready": all([
                    voice_test.get("success", False),
                    vision_test.get("success", False),
                    action_test.get("success", False)
                ])
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

