"""
Safety controls and permission management for computer actions
"""
import re
import time
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timedelta
import threading
import json
from pathlib import Path

class SafetyManager:
    """Manages safety controls and permissions for computer actions"""
    
    def __init__(self, safety_mode: str = "high"):
        self.safety_mode = safety_mode
        self.action_history = []
        self.blocked_actions = set()
        self.allowed_applications = set()
        self.restricted_applications = set()
        self.action_count_limit = {}
        self.last_action_time = {}
        self.emergency_stop_active = False
        self.lock = threading.Lock()
        
        # Load default restrictions
        self._load_default_restrictions()
        
        # Rate limiting
        self.rate_limits = {
            "click": {"max_per_minute": 60, "current": 0, "reset_time": time.time()},
            "type": {"max_per_minute": 30, "current": 0, "reset_time": time.time()},
            "key_press": {"max_per_minute": 120, "current": 0, "reset_time": time.time()},
            "open_application": {"max_per_minute": 10, "current": 0, "reset_time": time.time()}
        }
    
    def _load_default_restrictions(self):
        """Load default safety restrictions based on safety mode"""
        if self.safety_mode == "high":
            self.restricted_applications.update([
                "terminal", "cmd", "powershell", "bash", "zsh",
                "system preferences", "control panel", "registry editor",
                "task manager", "activity monitor", "system monitor",
                "disk utility", "partition manager", "format",
                "sudo", "admin", "root"
            ])
            
            self.sensitive_patterns = [
                r"delete.*system",
                r"format.*drive",
                r"rm\s+-rf",
                r"del\s+/s",
                r"shutdown",
                r"restart",
                r"reboot",
                r"passwd",
                r"password",
                r"credit.*card",
                r"social.*security",
                r"bank.*account"
            ]
            
        elif self.safety_mode == "medium":
            self.restricted_applications.update([
                "terminal", "cmd", "powershell",
                "system preferences", "control panel", "registry editor",
                "format", "disk utility"
            ])
            
            self.sensitive_patterns = [
                r"delete.*system",
                r"format.*drive",
                r"rm\s+-rf",
                r"shutdown",
                r"restart"
            ]
            
        else:  # low safety mode
            self.restricted_applications.update([
                "format", "disk utility"
            ])
            
            self.sensitive_patterns = [
                r"format.*drive",
                r"rm\s+-rf\s+/"
            ]
    
    def is_action_safe(self, action_type: str, target: str = "", context: Dict[str, Any] = None) -> bool:
        """Check if an action is safe to execute"""
        with self.lock:
            try:
                # Check emergency stop
                if self.emergency_stop_active:
                    return False
                
                # Check rate limits
                if not self._check_rate_limit(action_type):
                    return False
                
                # Check application restrictions
                if action_type == "open_application":
                    if not self._is_application_allowed(target):
                        return False
                
                # Check for sensitive patterns
                if self._contains_sensitive_pattern(target):
                    return False
                
                # Check action-specific safety rules
                if not self._check_action_specific_safety(action_type, target, context):
                    return False
                
                # Log the action check
                self._log_action_check(action_type, target, True)
                
                return True
                
            except Exception as e:
                print(f"Error in safety check: {e}")
                return False  # Fail safe
    
    def _check_rate_limit(self, action_type: str) -> bool:
        """Check if action is within rate limits"""
        current_time = time.time()
        
        if action_type in self.rate_limits:
            limit_info = self.rate_limits[action_type]
            
            # Reset counter if a minute has passed
            if current_time - limit_info["reset_time"] >= 60:
                limit_info["current"] = 0
                limit_info["reset_time"] = current_time
            
            # Check if under limit
            if limit_info["current"] >= limit_info["max_per_minute"]:
                return False
            
            # Increment counter
            limit_info["current"] += 1
        
        return True
    
    def _is_application_allowed(self, app_name: str) -> bool:
        """Check if application is allowed to be opened"""
        app_lower = app_name.lower()
        
        # Check if explicitly blocked
        for restricted in self.restricted_applications:
            if restricted.lower() in app_lower:
                return False
        
        # Check if in allowed list (if any)
        if self.allowed_applications:
            for allowed in self.allowed_applications:
                if allowed.lower() in app_lower:
                    return True
            return False  # Not in allowed list
        
        return True  # No restrictions or not in blocked list
    
    def _contains_sensitive_pattern(self, text: str) -> bool:
        """Check if text contains sensitive patterns"""
        text_lower = text.lower()
        
        for pattern in self.sensitive_patterns:
            if re.search(pattern, text_lower):
                return True
        
        return False
    
    def _check_action_specific_safety(self, action_type: str, target: str, context: Dict[str, Any] = None) -> bool:
        """Check action-specific safety rules"""
        context = context or {}
        
        if action_type == "click":
            # Check for dangerous click targets
            dangerous_targets = ["delete", "remove", "uninstall", "format", "erase"]
            target_lower = target.lower()
            
            for dangerous in dangerous_targets:
                if dangerous in target_lower:
                    if self.safety_mode == "high":
                        return False
                    elif self.safety_mode == "medium":
                        # Would require confirmation in real implementation
                        pass
        
        elif action_type == "type":
            # Check for sensitive information
            if self._contains_sensitive_pattern(target):
                return False
            
            # Check for command injection attempts
            command_patterns = [r";\s*rm", r"&&\s*del", r"\|\s*format"]
            for pattern in command_patterns:
                if re.search(pattern, target.lower()):
                    return False
        
        elif action_type == "key_press":
            # Check for dangerous key combinations
            dangerous_keys = ["ctrl+alt+del", "cmd+option+esc", "alt+f4"]
            if target.lower() in dangerous_keys and self.safety_mode == "high":
                return False
        
        elif action_type == "drag":
            # Check for drag to dangerous locations (like trash/recycle bin)
            if context:
                end_x = context.get("end_x", 0)
                end_y = context.get("end_y", 0)
                # Would check if coordinates correspond to dangerous drop zones
        
        return True
    
    def _log_action_check(self, action_type: str, target: str, allowed: bool):
        """Log action safety check"""
        log_entry = {
            "timestamp": datetime.now(),
            "action_type": action_type,
            "target": target,
            "allowed": allowed,
            "safety_mode": self.safety_mode
        }
        
        self.action_history.append(log_entry)
        
        # Keep only last 1000 entries
        if len(self.action_history) > 1000:
            self.action_history = self.action_history[-1000:]
    
    def block_action_type(self, action_type: str):
        """Block a specific action type"""
        with self.lock:
            self.blocked_actions.add(action_type)
    
    def unblock_action_type(self, action_type: str):
        """Unblock a specific action type"""
        with self.lock:
            self.blocked_actions.discard(action_type)
    
    def add_allowed_application(self, app_name: str):
        """Add application to allowed list"""
        with self.lock:
            self.allowed_applications.add(app_name.lower())
    
    def remove_allowed_application(self, app_name: str):
        """Remove application from allowed list"""
        with self.lock:
            self.allowed_applications.discard(app_name.lower())
    
    def add_restricted_application(self, app_name: str):
        """Add application to restricted list"""
        with self.lock:
            self.restricted_applications.add(app_name.lower())
    
    def remove_restricted_application(self, app_name: str):
        """Remove application from restricted list"""
        with self.lock:
            self.restricted_applications.discard(app_name.lower())
    
    def set_emergency_stop(self, active: bool):
        """Set emergency stop state"""
        with self.lock:
            self.emergency_stop_active = active
            if active:
                print("🛑 EMERGENCY STOP ACTIVATED - All actions blocked")
            else:
                print("✅ Emergency stop deactivated")
    
    def get_safety_status(self) -> Dict[str, Any]:
        """Get current safety status"""
        with self.lock:
            return {
                "safety_mode": self.safety_mode,
                "emergency_stop_active": self.emergency_stop_active,
                "blocked_actions": list(self.blocked_actions),
                "allowed_applications": list(self.allowed_applications),
                "restricted_applications": list(self.restricted_applications),
                "recent_actions": len([
                    entry for entry in self.action_history 
                    if entry["timestamp"] > datetime.now() - timedelta(minutes=5)
                ]),
                "rate_limits": {
                    action: {
                        "current": info["current"],
                        "max": info["max_per_minute"],
                        "time_until_reset": max(0, 60 - (time.time() - info["reset_time"]))
                    }
                    for action, info in self.rate_limits.items()
                }
            }
    
    def get_action_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent action history"""
        with self.lock:
            return self.action_history[-limit:]
    
    def clear_action_history(self):
        """Clear action history"""
        with self.lock:
            self.action_history.clear()
    
    def export_safety_config(self) -> Dict[str, Any]:
        """Export current safety configuration"""
        with self.lock:
            return {
                "safety_mode": self.safety_mode,
                "blocked_actions": list(self.blocked_actions),
                "allowed_applications": list(self.allowed_applications),
                "restricted_applications": list(self.restricted_applications),
                "sensitive_patterns": self.sensitive_patterns,
                "rate_limits": {
                    action: {"max_per_minute": info["max_per_minute"]}
                    for action, info in self.rate_limits.items()
                }
            }
    
    def import_safety_config(self, config: Dict[str, Any]):
        """Import safety configuration"""
        with self.lock:
            try:
                if "safety_mode" in config:
                    self.safety_mode = config["safety_mode"]
                
                if "blocked_actions" in config:
                    self.blocked_actions = set(config["blocked_actions"])
                
                if "allowed_applications" in config:
                    self.allowed_applications = set(config["allowed_applications"])
                
                if "restricted_applications" in config:
                    self.restricted_applications = set(config["restricted_applications"])
                
                if "sensitive_patterns" in config:
                    self.sensitive_patterns = config["sensitive_patterns"]
                
                if "rate_limits" in config:
                    for action, limits in config["rate_limits"].items():
                        if action in self.rate_limits:
                            self.rate_limits[action]["max_per_minute"] = limits["max_per_minute"]
                
                print("Safety configuration imported successfully")
                
            except Exception as e:
                print(f"Error importing safety config: {e}")
    
    def save_config_to_file(self, file_path: str):
        """Save safety configuration to file"""
        try:
            config = self.export_safety_config()
            with open(file_path, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"Safety configuration saved to {file_path}")
            
        except Exception as e:
            print(f"Error saving safety config: {e}")
    
    def load_config_from_file(self, file_path: str):
        """Load safety configuration from file"""
        try:
            with open(file_path, 'r') as f:
                config = json.load(f)
            self.import_safety_config(config)
            print(f"Safety configuration loaded from {file_path}")
            
        except Exception as e:
            print(f"Error loading safety config: {e}")

class PermissionManager:
    """Manages user permissions and confirmations"""
    
    def __init__(self):
        self.pending_confirmations = {}
        self.user_preferences = {}
        self.auto_approve_patterns = set()
        self.auto_deny_patterns = set()
    
    def request_confirmation(self, action_type: str, target: str, timeout: int = 30) -> str:
        """Request user confirmation for an action"""
        confirmation_id = f"{action_type}_{target}_{int(time.time())}"
        
        self.pending_confirmations[confirmation_id] = {
            "action_type": action_type,
            "target": target,
            "timestamp": time.time(),
            "timeout": timeout,
            "status": "pending"
        }
        
        return confirmation_id
    
    def respond_to_confirmation(self, confirmation_id: str, response: str) -> bool:
        """Respond to a confirmation request"""
        if confirmation_id not in self.pending_confirmations:
            return False
        
        confirmation = self.pending_confirmations[confirmation_id]
        
        # Check if not expired
        if time.time() - confirmation["timestamp"] > confirmation["timeout"]:
            confirmation["status"] = "expired"
            return False
        
        # Update status
        if response.lower() in ["yes", "y", "approve", "allow", "ok"]:
            confirmation["status"] = "approved"
            return True
        else:
            confirmation["status"] = "denied"
            return False
    
    def cleanup_expired_confirmations(self):
        """Clean up expired confirmation requests"""
        current_time = time.time()
        expired_ids = []
        
        for conf_id, conf_data in self.pending_confirmations.items():
            if current_time - conf_data["timestamp"] > conf_data["timeout"]:
                expired_ids.append(conf_id)
        
        for conf_id in expired_ids:
            del self.pending_confirmations[conf_id]
    
    def get_pending_confirmations(self) -> List[Dict[str, Any]]:
        """Get list of pending confirmations"""
        self.cleanup_expired_confirmations()
        
        return [
            {
                "id": conf_id,
                "action_type": conf_data["action_type"],
                "target": conf_data["target"],
                "time_remaining": max(0, conf_data["timeout"] - (time.time() - conf_data["timestamp"]))
            }
            for conf_id, conf_data in self.pending_confirmations.items()
            if conf_data["status"] == "pending"
        ]

# Utility functions
def create_safety_manager(safety_mode: str = "high") -> SafetyManager:
    """Create a safety manager with specified mode"""
    return SafetyManager(safety_mode)

def validate_action_safety(action_type: str, target: str, safety_manager: SafetyManager) -> Dict[str, Any]:
    """Validate action safety and return detailed result"""
    is_safe = safety_manager.is_action_safe(action_type, target)
    
    return {
        "safe": is_safe,
        "action_type": action_type,
        "target": target,
        "safety_mode": safety_manager.safety_mode,
        "timestamp": datetime.now(),
        "reason": "Action allowed" if is_safe else "Action blocked by safety controls"
    }

# Test function
def test_safety_controls() -> Dict[str, Any]:
    """Test safety control functionality"""
    try:
        # Test different safety modes
        high_safety = SafetyManager("high")
        medium_safety = SafetyManager("medium")
        low_safety = SafetyManager("low")
        
        # Test some actions
        test_results = {
            "high_safety_terminal": high_safety.is_action_safe("open_application", "terminal"),
            "medium_safety_terminal": medium_safety.is_action_safe("open_application", "terminal"),
            "low_safety_terminal": low_safety.is_action_safe("open_application", "terminal"),
            "high_safety_chrome": high_safety.is_action_safe("open_application", "chrome"),
            "sensitive_text": high_safety.is_action_safe("type", "rm -rf /"),
            "normal_click": high_safety.is_action_safe("click", "save button")
        }
        
        return {
            "success": True,
            "test_results": test_results,
            "safety_managers_created": True
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

