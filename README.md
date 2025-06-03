# 🎤💻 Voice Computer Assistant

A comprehensive voice-controlled computer assistant that combines speech recognition, computer vision, and automation to enable natural language control of your computer. Built using patterns from the awesome-llm-apps voice agents with advanced computer control capabilities.

## ✨ Features

### 🗣️ **Voice Interaction**
- **Speech-to-Text**: Convert voice commands to text using OpenAI Whisper
- **Text-to-Speech**: Natural voice responses with 11 voice options
- **Real-time Audio**: Live recording with silence detection and auto-stop
- **Multi-modal Input**: Support both voice and text input

### 👁️ **Computer Vision**
- **Screen Capture**: Real-time screenshot analysis
- **UI Element Detection**: Automatic detection of buttons, text fields, and interactive elements
- **AI-Powered Analysis**: GPT-4 Vision for intelligent screen understanding
- **OCR Text Recognition**: Extract text from any part of the screen

### 🤖 **Computer Automation**
- **Mouse Control**: Click, double-click, right-click, drag and drop
- **Keyboard Control**: Type text, press keys, keyboard shortcuts
- **Application Management**: Open applications, switch windows, focus control
- **Scroll and Navigation**: Intelligent scrolling and page navigation

### 🛡️ **Safety & Security**
- **Multi-level Safety Modes**: High, Medium, Low safety configurations
- **Action Validation**: Prevent dangerous operations and system commands
- **Rate Limiting**: Prevent rapid-fire actions that could cause issues
- **Emergency Stop**: Instant halt of all operations
- **Permission Management**: User confirmation for sensitive actions

### 🎯 **Intelligent Coordination**
- **Multi-Agent Architecture**: Specialized agents for voice, vision, and actions
- **Context Awareness**: Maintains conversation history and screen context
- **Intent Recognition**: Natural language understanding for complex commands
- **Error Handling**: Graceful failure recovery and user feedback

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+** installed on your system
2. **OpenAI API Key** from [OpenAI Platform](https://platform.openai.com/)
3. **System Dependencies**:
   - **Windows**: No additional setup required
   - **macOS**: Install Xcode command line tools
   - **Linux**: Install audio and display dependencies

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd voice-computer-assistant
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install system dependencies**:

   **On macOS**:
   ```bash
   brew install portaudio
   ```

   **On Ubuntu/Debian**:
   ```bash
   sudo apt-get update
   sudo apt-get install portaudio19-dev python3-pyaudio tesseract-ocr
   ```

   **On Windows**:
   - Download and install [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)
   - Add Tesseract to your PATH

### Running the Application

1. **Start the Streamlit app**:
   ```bash
   streamlit run voice_computer_assistant.py
   ```

2. **Configure your settings**:
   - Enter your OpenAI API Key in the sidebar
   - Choose your preferred voice and safety mode
   - Click "Initialize System"

3. **Start using voice commands**:
   - Click the microphone button and speak your command
   - Or type commands in the text input field
   - View real-time screen analysis and action history

## 🎯 Example Commands

### Basic Navigation
- *"Click the save button"*
- *"Open Chrome"*
- *"Type 'Hello World'"*
- *"Scroll down"*
- *"Press Enter"*

### Advanced Operations
- *"Find the login button and click it"*
- *"Open Chrome and navigate to Google"*
- *"Type my email address in the username field"*
- *"Drag this file to the desktop"*
- *"Take a screenshot of the current window"*

### Application Control
- *"Switch to the browser window"*
- *"Close this application"*
- *"Minimize all windows"*
- *"Open the file menu"*

## 🏗️ Architecture

### Multi-Agent System

The assistant uses a sophisticated multi-agent architecture:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Voice Agent   │    │  Vision Agent    │    │  Action Agent   │
│                 │    │                  │    │                 │
│ • Speech-to-Text│    │ • Screen Capture │    │ • Mouse Control │
│ • Text-to-Speech│    │ • UI Detection   │    │ • Keyboard Input│
│ • Intent Analysis│   │ • AI Analysis    │    │ • App Management│
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────────┐
                    │ Coordinator Agent   │
                    │                     │
                    │ • Workflow Control  │
                    │ • Context Management│
                    │ • Safety Validation │
                    │ • Response Generation│
                    └─────────────────────┘
```

### Key Components

- **Voice Agent**: Handles all speech processing using OpenAI's Whisper and TTS
- **Vision Agent**: Captures and analyzes screen content using computer vision and GPT-4 Vision
- **Action Agent**: Executes computer actions with safety controls and validation
- **Coordinator Agent**: Orchestrates the complete workflow from voice input to action execution
- **Safety Manager**: Enforces security policies and prevents dangerous operations

## ⚙️ Configuration

### Safety Modes

- **High Safety**: Requires confirmation for all actions, blocks system commands
- **Medium Safety**: Confirms sensitive actions, allows basic file operations
- **Low Safety**: Minimal restrictions, suitable for experienced users

### Voice Settings

Choose from 11 OpenAI TTS voices:
- `alloy`, `ash`, `ballad`, `coral`, `echo`, `fable`, `onyx`, `nova`, `sage`, `shimmer`, `verse`

### Environment Variables

Create a `.env` file for configuration:

```env
OPENAI_API_KEY=your_openai_api_key_here
VOICE_ASSISTANT_LOG_LEVEL=INFO
VOICE_ASSISTANT_TEMP_DIR=/tmp/voice_assistant
```

## 🛡️ Safety Features

### Built-in Protections

- **Application Restrictions**: Blocks access to system utilities and dangerous applications
- **Command Filtering**: Prevents execution of harmful commands
- **Rate Limiting**: Limits action frequency to prevent system overload
- **Emergency Stop**: Immediate halt of all operations
- **Action Logging**: Complete audit trail of all actions

### Customizable Security

- Configure allowed/blocked applications
- Set custom safety patterns
- Adjust rate limits per action type
- Export/import security configurations

## 🔧 Development

### Project Structure

```
voice-computer-assistant/
├── voice_computer_assistant.py    # Main Streamlit application
├── agents/
│   ├── voice_agent.py            # Speech processing
│   ├── vision_agent.py           # Computer vision
│   ├── action_agent.py           # Computer automation
│   └── coordinator_agent.py      # Workflow orchestration
├── utils/
│   ├── audio_utils.py            # Audio recording/playback
│   └── safety_controls.py        # Security and permissions
├── config/
│   └── settings.py               # Configuration management
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

### Adding New Features

1. **New Action Types**: Extend the `ActionAgent` class
2. **Enhanced Vision**: Add new detection methods to `VisionAgent`
3. **Voice Improvements**: Customize the `VoiceAgent` for specific use cases
4. **Safety Rules**: Modify `SafetyManager` for custom restrictions

### Testing

Run the test functions to verify setup:

```python
# Test individual components
await voice_agent.test_voice_setup()
await vision_agent.test_vision_setup()
await action_agent.test_action_setup()

# Test complete system
await coordinator.test_coordinator_setup()
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built using patterns from [awesome-llm-apps](https://github.com/Shubhamsaboo/awesome-llm-apps)
- Powered by OpenAI's GPT-4, Whisper, and TTS models
- Computer vision capabilities using OpenCV and Tesseract
- Automation powered by PyAutoGUI

## ⚠️ Disclaimer

This software provides computer automation capabilities. Use responsibly and ensure you understand the safety implications. The developers are not responsible for any damage caused by misuse of this software. Always test in a safe environment first.

## 📞 Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check the documentation
- Review the safety guidelines

---

**Happy voice computing! 🎤💻**

