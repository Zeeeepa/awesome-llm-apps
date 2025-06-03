"""
Action Agent for executing computer actions like clicking, typing, etc.
"""
import asyncio
import time
from typing import Dict, Any, Optional, Tuple, List
import threading

try:
    import pyautogui
    import pygetwindow as gw
    import psutil
except ImportError as e:
    print(f"Warning: Some action dependencies not available: {e}")
    print("Install with: pip install pyautogui pygetwindow psutil")

from config.settings import AppConfig, ActionConfig
from utils.safety_controls import SafetyManager

class ActionAgent:
    """Handles execution of computer actions"""
    
    def __init__(self, config: AppConfig, safety_manager: SafetyManager):
        self.config = config
        self.action_config = ActionConfig()
        self.safety_manager = safety_manager
        self.emergency_stop_flag = threading.Event()
        
        # Configure pyautogui safety settings
        if 'pyautogui' in globals():
            pyautogui.FAILSAFE = True  # Move mouse to corner to abort
            pyautogui.PAUSE = self.action_config.BETWEEN_ACTION_DELAY
    
    def emergency_stop(self):
        """Emergency stop all actions"""
        self.emergency_stop_flag.set()
        print("Emergency stop activated!")
    
    def reset_emergency_stop(self):
        """Reset emergency stop flag"""
        self.emergency_stop_flag.clear()
    
    def _check_emergency_stop(self):
        """Check if emergency stop is activated"""
        if self.emergency_stop_flag.is_set():
            raise Exception("Emergency stop activated")
    
    async def click(self, x: int, y: int, button: str = "left", clicks: int = 1) -> Dict[str, Any]:
        """Click at specified coordinates"""
        try:
            self._check_emergency_stop()
            
            # Safety check
            if not self.safety_manager.is_action_safe("click", f"coordinates ({x}, {y})"):
                return {
                    "success": False,
                    "error": "Action blocked by safety controls",
                    "action": "click"
                }
            
            # Validate coordinates
            screen_width, screen_height = pyautogui.size()
            if not (0 <= x <= screen_width and 0 <= y <= screen_height):
                return {
                    "success": False,
                    "error": f"Coordinates ({x}, {y}) are outside screen bounds",
                    "action": "click"
                }
            
            # Perform click
            if button == "left":
                pyautogui.click(x, y, clicks=clicks, duration=self.action_config.CLICK_DURATION)
            elif button == "right":
                pyautogui.rightClick(x, y)
            elif button == "middle":
                pyautogui.middleClick(x, y)
            
            # Add delay for safety
            await asyncio.sleep(self.action_config.BETWEEN_ACTION_DELAY)
            
            return {
                "success": True,
                "action": "click",
                "coordinates": (x, y),
                "button": button,
                "clicks": clicks
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "action": "click"
            }
    
    async def double_click(self, x: int, y: int) -> Dict[str, Any]:
        """Double click at specified coordinates"""
        return await self.click(x, y, clicks=2)
    
    async def type_text(self, text: str, interval: Optional[float] = None) -> Dict[str, Any]:
        """Type text with specified interval between characters"""
        try:
            self._check_emergency_stop()
            
            # Safety check
            if not self.safety_manager.is_action_safe("type", text):
                return {
                    "success": False,
                    "error": "Text input blocked by safety controls",
                    "action": "type"
                }
            
            # Use configured interval or default
            type_interval = interval or self.action_config.TYPE_INTERVAL
            
            # Type the text
            pyautogui.typewrite(text, interval=type_interval)
            
            # Add delay for safety
            await asyncio.sleep(self.action_config.BETWEEN_ACTION_DELAY)
            
            return {
                "success": True,
                "action": "type",
                "text": text,
                "length": len(text)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "action": "type"
            }
    
    async def press_key(self, key: str, presses: int = 1) -> Dict[str, Any]:
        """Press a specific key or key combination"""
        try:
            self._check_emergency_stop()
            
            # Safety check for sensitive keys
            sensitive_keys = ['delete', 'ctrl+alt+del', 'alt+f4', 'cmd+q']
            if key.lower() in sensitive_keys:
                if not self.safety_manager.is_action_safe("key_press", key):
                    return {
                        "success": False,
                        "error": f"Key press '{key}' blocked by safety controls",
                        "action": "key_press"
                    }
            
            # Handle key combinations
            if '+' in key:
                keys = key.split('+')
                pyautogui.hotkey(*keys)
            else:
                pyautogui.press(key, presses=presses)
            
            # Add delay for safety
            await asyncio.sleep(self.action_config.BETWEEN_ACTION_DELAY)
            
            return {
                "success": True,
                "action": "key_press",
                "key": key,
                "presses": presses
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "action": "key_press"
            }
    
    async def scroll(self, x: int, y: int, direction: str = "up", clicks: int = 3) -> Dict[str, Any]:
        """Scroll at specified coordinates"""
        try:
            self._check_emergency_stop()
            
            # Move mouse to position first
            pyautogui.moveTo(x, y)
            
            # Determine scroll direction
            scroll_amount = clicks if direction == "up" else -clicks
            
            # Perform scroll
            pyautogui.scroll(scroll_amount)
            
            # Add delay for safety
            await asyncio.sleep(self.action_config.SCROLL_PAUSE)
            
            return {
                "success": True,
                "action": "scroll",
                "coordinates": (x, y),
                "direction": direction,
                "clicks": clicks
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "action": "scroll"
            }
    
    async def drag(self, start_x: int, start_y: int, end_x: int, end_y: int, duration: float = 1.0) -> Dict[str, Any]:
        """Drag from start coordinates to end coordinates"""
        try:
            self._check_emergency_stop()
            
            # Safety check
            if not self.safety_manager.is_action_safe("drag", f"from ({start_x}, {start_y}) to ({end_x}, {end_y})"):
                return {
                    "success": False,
                    "error": "Drag action blocked by safety controls",
                    "action": "drag"
                }
            
            # Perform drag
            pyautogui.drag(end_x - start_x, end_y - start_y, duration=duration, button='left')
            
            # Add delay for safety
            await asyncio.sleep(self.action_config.BETWEEN_ACTION_DELAY)
            
            return {
                "success": True,
                "action": "drag",
                "start": (start_x, start_y),
                "end": (end_x, end_y),
                "duration": duration
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "action": "drag"
            }
    
    async def open_application(self, app_name: str) -> Dict[str, Any]:
        """Open an application by name"""
        try:
            self._check_emergency_stop()
            
            # Safety check
            if not self.safety_manager.is_action_safe("open_application", app_name):
                return {
                    "success": False,
                    "error": f"Opening '{app_name}' blocked by safety controls",
                    "action": "open_application"
                }
            
            # Try different methods to open the application
            import subprocess
            import platform
            
            system = platform.system()
            
            if system == "Windows":
                # Try to open using Windows start command
                subprocess.Popen(f'start {app_name}', shell=True)
            elif system == "Darwin":  # macOS
                subprocess.Popen(['open', '-a', app_name])
            else:  # Linux
                subprocess.Popen([app_name])
            
            # Wait for application to start
            await asyncio.sleep(self.action_config.APPLICATION_LAUNCH_TIMEOUT)
            
            return {
                "success": True,
                "action": "open_application",
                "application": app_name
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "action": "open_application"
            }
    
    async def get_window_list(self) -> List[Dict[str, Any]]:
        """Get list of open windows"""
        try:
            windows = []
            for window in gw.getAllWindows():
                if window.title:  # Only include windows with titles
                    windows.append({
                        "title": window.title,
                        "left": window.left,
                        "top": window.top,
                        "width": window.width,
                        "height": window.height,
                        "active": window.isActive
                    })
            
            return windows
            
        except Exception as e:
            print(f"Error getting window list: {e}")
            return []
    
    async def focus_window(self, window_title: str) -> Dict[str, Any]:
        """Focus on a specific window by title"""
        try:
            self._check_emergency_stop()
            
            # Find window by title (partial match)
            windows = gw.getWindowsWithTitle(window_title)
            
            if not windows:
                return {
                    "success": False,
                    "error": f"Window with title '{window_title}' not found",
                    "action": "focus_window"
                }
            
            # Focus on the first matching window
            window = windows[0]
            window.activate()
            
            # Add delay for window switching
            await asyncio.sleep(self.action_config.WINDOW_SWITCH_DELAY)
            
            return {
                "success": True,
                "action": "focus_window",
                "window_title": window.title
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "action": "focus_window"
            }
    
    async def get_mouse_position(self) -> Tuple[int, int]:
        """Get current mouse position"""
        try:
            return pyautogui.position()
        except Exception as e:
            print(f"Error getting mouse position: {e}")
            return (0, 0)
    
    async def move_mouse(self, x: int, y: int, duration: float = 0.5) -> Dict[str, Any]:
        """Move mouse to specified coordinates"""
        try:
            self._check_emergency_stop()
            
            pyautogui.moveTo(x, y, duration=duration)
            
            return {
                "success": True,
                "action": "move_mouse",
                "coordinates": (x, y),
                "duration": duration
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "action": "move_mouse"
            }
    
    async def take_screenshot_action(self, region: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, Any]:
        """Take a screenshot (action wrapper)"""
        try:
            if region:
                screenshot = pyautogui.screenshot(region=region)
            else:
                screenshot = pyautogui.screenshot()
            
            # Save screenshot
            import tempfile
            import uuid
            temp_path = f"{tempfile.gettempdir()}/action_screenshot_{uuid.uuid4()}.png"
            screenshot.save(temp_path)
            
            return {
                "success": True,
                "action": "screenshot",
                "path": temp_path,
                "region": region
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "action": "screenshot"
            }
    
    async def execute_action_sequence(self, actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute a sequence of actions"""
        results = []
        
        try:
            for i, action in enumerate(actions):
                self._check_emergency_stop()
                
                # Check if we've exceeded max actions per command
                max_actions = self.config.safety_config.get("max_actions_per_command", 5)
                if i >= max_actions:
                    results.append({
                        "success": False,
                        "error": f"Exceeded maximum actions per command ({max_actions})",
                        "action": "sequence_limit"
                    })
                    break
                
                action_type = action.get("type")
                params = action.get("parameters", {})
                
                # Execute based on action type
                if action_type == "click":
                    result = await self.click(params.get("x"), params.get("y"), 
                                            params.get("button", "left"), params.get("clicks", 1))
                elif action_type == "type":
                    result = await self.type_text(params.get("text", ""))
                elif action_type == "key_press":
                    result = await self.press_key(params.get("key", ""))
                elif action_type == "scroll":
                    result = await self.scroll(params.get("x", 0), params.get("y", 0), 
                                             params.get("direction", "up"), params.get("clicks", 3))
                elif action_type == "drag":
                    result = await self.drag(params.get("start_x"), params.get("start_y"),
                                           params.get("end_x"), params.get("end_y"))
                else:
                    result = {
                        "success": False,
                        "error": f"Unknown action type: {action_type}",
                        "action": action_type
                    }
                
                results.append(result)
                
                # Stop sequence if any action fails
                if not result.get("success", False):
                    break
                
                # Add delay between actions
                await asyncio.sleep(self.action_config.BETWEEN_ACTION_DELAY)
        
        except Exception as e:
            results.append({
                "success": False,
                "error": str(e),
                "action": "sequence_execution"
            })
        
        return results
    
    async def test_action_setup(self) -> Dict[str, Any]:
        """Test the action setup and return status"""
        try:
            # Test mouse position
            mouse_pos = await self.get_mouse_position()
            
            # Test screen size
            screen_size = pyautogui.size()
            
            return {
                "success": True,
                "mouse_position": mouse_pos,
                "screen_size": screen_size,
                "pyautogui_available": 'pyautogui' in globals(),
                "safety_manager_active": self.safety_manager is not None
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

