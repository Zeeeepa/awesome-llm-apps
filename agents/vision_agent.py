"""
Vision Agent for screen capture and UI element detection
"""
import asyncio
import tempfile
import uuid
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import base64
import io

try:
    import cv2
    import numpy as np
    from PIL import Image, ImageGrab
    import pytesseract
except ImportError as e:
    print(f"Warning: Some vision dependencies not available: {e}")
    print("Install with: pip install opencv-python pillow pytesseract")

from openai import AsyncOpenAI
from config.settings import AppConfig, VisionConfig

class UIElement:
    """Represents a UI element detected on screen"""
    
    def __init__(self, element_type: str, bounds: Tuple[int, int, int, int], 
                 text: str = "", confidence: float = 0.0, description: str = ""):
        self.element_type = element_type  # button, text_field, etc.
        self.bounds = bounds  # (x, y, width, height)
        self.text = text
        self.confidence = confidence
        self.description = description
        self.center = (bounds[0] + bounds[2] // 2, bounds[1] + bounds[3] // 2)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.element_type,
            "bounds": self.bounds,
            "center": self.center,
            "text": self.text,
            "confidence": self.confidence,
            "description": self.description
        }

class VisionAgent:
    """Handles screen capture and UI element detection"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.vision_config = VisionConfig()
        self.client = AsyncOpenAI(api_key=config.openai_api_key)
        self.temp_dir = Path(tempfile.gettempdir()) / "voice_assistant" / "screenshots"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.last_screenshot = None
        self.detected_elements = []
    
    def capture_screen(self, region: Optional[Tuple[int, int, int, int]] = None) -> str:
        """Capture screenshot and return file path"""
        try:
            # Capture screenshot
            if region:
                screenshot = ImageGrab.grab(bbox=region)
            else:
                screenshot = ImageGrab.grab()
            
            # Save screenshot
            screenshot_path = self.temp_dir / f"screenshot_{uuid.uuid4()}.png"
            screenshot.save(screenshot_path, format=self.vision_config.SCREENSHOT_FORMAT, 
                          quality=self.vision_config.SCREENSHOT_QUALITY)
            
            self.last_screenshot = str(screenshot_path)
            return self.last_screenshot
            
        except Exception as e:
            print(f"Error capturing screen: {e}")
            return None
    
    def capture_screen_as_image(self, region: Optional[Tuple[int, int, int, int]] = None) -> Optional[Image.Image]:
        """Capture screenshot and return PIL Image"""
        try:
            if region:
                return ImageGrab.grab(bbox=region)
            else:
                return ImageGrab.grab()
        except Exception as e:
            print(f"Error capturing screen as image: {e}")
            return None
    
    async def analyze_screen_with_ai(self, screenshot_path: str, query: str = "") -> Dict[str, Any]:
        """Use OpenAI Vision to analyze the screen"""
        try:
            # Read and encode image
            with open(screenshot_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Prepare the query
            if not query:
                query = """Analyze this screenshot and identify all interactive UI elements. 
                For each element, provide:
                1. Type (button, text_field, dropdown, link, etc.)
                2. Location description
                3. Text content if visible
                4. Purpose/function
                
                Focus on elements a user might want to click or interact with."""
            
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": query},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_data}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000
            )
            
            return {
                "success": True,
                "analysis": response.choices[0].message.content,
                "screenshot_path": screenshot_path
            }
            
        except Exception as e:
            print(f"Error in AI screen analysis: {e}")
            return {
                "success": False,
                "error": str(e),
                "analysis": None
            }
    
    def detect_text_with_ocr(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Detect text in image using OCR"""
        try:
            # Convert PIL image to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Use pytesseract to detect text
            data = pytesseract.image_to_data(cv_image, output_type=pytesseract.Output.DICT)
            
            text_elements = []
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > self.vision_config.OCR_CONFIDENCE * 100:
                    text = data['text'][i].strip()
                    if text:
                        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                        text_elements.append({
                            "text": text,
                            "bounds": (x, y, w, h),
                            "confidence": int(data['conf'][i]) / 100.0,
                            "type": "text"
                        })
            
            return text_elements
            
        except Exception as e:
            print(f"Error in OCR text detection: {e}")
            return []
    
    def detect_ui_elements(self, image: Image.Image) -> List[UIElement]:
        """Detect UI elements using computer vision"""
        try:
            # Convert to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            elements = []
            
            # Detect buttons using edge detection and contours
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Filter contours by area and aspect ratio
                area = cv2.contourArea(contour)
                if area > 500:  # Minimum area for UI elements
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    
                    # Classify based on aspect ratio and size
                    if 0.2 < aspect_ratio < 5.0:  # Reasonable aspect ratio
                        if w > 50 and h > 20:  # Minimum button size
                            element_type = "button" if aspect_ratio > 1.5 else "text_field"
                            
                            elements.append(UIElement(
                                element_type=element_type,
                                bounds=(x, y, w, h),
                                confidence=0.7,
                                description=f"{element_type} at ({x}, {y})"
                            ))
            
            # Add OCR-detected text as potential clickable elements
            text_elements = self.detect_text_with_ocr(image)
            for text_elem in text_elements:
                x, y, w, h = text_elem["bounds"]
                elements.append(UIElement(
                    element_type="text",
                    bounds=(x, y, w, h),
                    text=text_elem["text"],
                    confidence=text_elem["confidence"],
                    description=f"Text: {text_elem['text']}"
                ))
            
            return elements
            
        except Exception as e:
            print(f"Error detecting UI elements: {e}")
            return []
    
    async def find_element_by_description(self, description: str, screenshot_path: Optional[str] = None) -> Optional[UIElement]:
        """Find UI element by natural language description"""
        try:
            # Use current screenshot if none provided
            if not screenshot_path:
                screenshot_path = self.capture_screen()
            
            if not screenshot_path:
                return None
            
            # Use AI to locate the element
            query = f"""Find the UI element that matches this description: "{description}"
            
            Provide the approximate coordinates and bounds in this format:
            {{
                "found": true/false,
                "element_type": "button/text_field/link/etc",
                "bounds": [x, y, width, height],
                "center": [center_x, center_y],
                "confidence": 0.0-1.0,
                "description": "detailed description"
            }}
            
            If multiple elements match, choose the most likely one based on context."""
            
            result = await self.analyze_screen_with_ai(screenshot_path, query)
            
            if result["success"]:
                try:
                    import json
                    # Try to extract JSON from the response
                    response_text = result["analysis"]
                    
                    # Look for JSON in the response
                    start_idx = response_text.find('{')
                    end_idx = response_text.rfind('}') + 1
                    
                    if start_idx != -1 and end_idx != -1:
                        json_str = response_text[start_idx:end_idx]
                        element_data = json.loads(json_str)
                        
                        if element_data.get("found", False):
                            bounds = element_data.get("bounds", [0, 0, 0, 0])
                            return UIElement(
                                element_type=element_data.get("element_type", "unknown"),
                                bounds=tuple(bounds),
                                confidence=element_data.get("confidence", 0.5),
                                description=element_data.get("description", description)
                            )
                
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Error parsing AI response: {e}")
            
            return None
            
        except Exception as e:
            print(f"Error finding element by description: {e}")
            return None
    
    async def get_screen_elements(self, refresh: bool = True) -> List[UIElement]:
        """Get all detected UI elements on current screen"""
        try:
            if refresh or not self.detected_elements:
                # Capture fresh screenshot
                screenshot_path = self.capture_screen()
                if not screenshot_path:
                    return []
                
                # Get image for processing
                image = Image.open(screenshot_path)
                
                # Detect elements using computer vision
                self.detected_elements = self.detect_ui_elements(image)
                
                # Enhance with AI analysis
                ai_result = await self.analyze_screen_with_ai(screenshot_path)
                if ai_result["success"]:
                    # Store AI analysis for reference
                    self.last_ai_analysis = ai_result["analysis"]
            
            return self.detected_elements
            
        except Exception as e:
            print(f"Error getting screen elements: {e}")
            return []
    
    def get_element_at_coordinates(self, x: int, y: int) -> Optional[UIElement]:
        """Find UI element at specific coordinates"""
        for element in self.detected_elements:
            ex, ey, ew, eh = element.bounds
            if ex <= x <= ex + ew and ey <= y <= ey + eh:
                return element
        return None
    
    def find_elements_by_text(self, text: str, partial_match: bool = True) -> List[UIElement]:
        """Find UI elements containing specific text"""
        matching_elements = []
        search_text = text.lower()
        
        for element in self.detected_elements:
            element_text = element.text.lower()
            if partial_match and search_text in element_text:
                matching_elements.append(element)
            elif not partial_match and search_text == element_text:
                matching_elements.append(element)
        
        return matching_elements
    
    def cleanup_temp_files(self, max_age_hours: int = 24):
        """Clean up old screenshot files"""
        try:
            import time
            current_time = time.time()
            
            for file_path in self.temp_dir.glob("*.png"):
                file_age = current_time - file_path.stat().st_mtime
                if file_age > (max_age_hours * 3600):
                    file_path.unlink(missing_ok=True)
                    
        except Exception as e:
            print(f"Error cleaning up temp files: {e}")
    
    async def test_vision_setup(self) -> Dict[str, Any]:
        """Test the vision setup and return status"""
        try:
            # Test screen capture
            screenshot_path = self.capture_screen()
            
            # Test AI analysis
            ai_test = None
            if screenshot_path:
                ai_result = await self.analyze_screen_with_ai(screenshot_path, "Describe what you see in this screenshot.")
                ai_test = ai_result["success"]
            
            return {
                "success": True,
                "screen_capture_working": screenshot_path is not None,
                "ai_analysis_working": ai_test,
                "ocr_available": 'pytesseract' in globals(),
                "cv2_available": 'cv2' in globals()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

