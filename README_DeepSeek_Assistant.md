# 🤖🎤 DeepSeek Voice Assistant

A modern voice-enabled AI assistant that integrates with DeepSeek's powerful language models, featuring an intuitive Streamlit interface with toggleable voice input/output controls.

## ✨ Features

### 🎛️ **Enhanced UI Controls**
- **API Key Input**: Secure input for DeepSeek and OpenAI API keys
- **Model Selection**: Choose from DeepSeek's latest models (chat, coder, reasoner)
- **🎤 Microphone Listen Toggle**: Enable/disable voice input
- **🔊 Read Response Toggle**: Enable/disable text-to-speech output
- **Real-time Status**: Live indicators for API connectivity and recording status

### 🧠 **DeepSeek Integration**
- **Multiple Models**: Support for DeepSeek Chat, Coder, and Reasoner models
- **Conversation History**: Maintains context across interactions
- **Streaming Responses**: Fast, efficient API communication
- **Error Handling**: Robust error management with user feedback

### 🎙️ **Voice Capabilities**
- **Voice Input**: Real-time speech recognition with recording controls
- **Voice Output**: High-quality text-to-speech with 11 voice options
- **Audio Controls**: Play, pause, and download audio responses
- **Silence Detection**: Automatic recording stop on silence

### 💬 **Conversation Management**
- **History Tracking**: View and manage conversation history
- **Context Preservation**: Maintains conversation context for better responses
- **Clear Conversations**: Easy conversation reset functionality
- **Timestamped Messages**: Track when each interaction occurred

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+** installed on your system
2. **DeepSeek API Key** from [DeepSeek Platform](https://platform.deepseek.com/)
3. **OpenAI API Key** from [OpenAI Platform](https://platform.openai.com/) (for TTS)

### Installation

#### Option 1: Automated Setup (Recommended)

```bash
# Run the setup script
python setup_deepseek_assistant.py
```

#### Option 2: Manual Setup

1. **Install dependencies**:
   ```bash
   pip install -r deepseek_requirements.txt
   ```

2. **Install audio dependencies**:
   
   **macOS**:
   ```bash
   brew install portaudio
   ```
   
   **Ubuntu/Debian**:
   ```bash
   sudo apt-get update
   sudo apt-get install portaudio19-dev python3-dev build-essential
   ```
   
   **Windows**:
   ```bash
   # Dependencies should install automatically with pip
   ```

3. **Configure environment**:
   ```bash
   cp .env.deepseek.example .env
   # Edit .env and add your API keys
   ```

### Running the Assistant

#### Option 1: Launch Script
```bash
# Windows
launch_deepseek_assistant.bat

# macOS/Linux
./launch_deepseek_assistant.sh
```

#### Option 2: Direct Command
```bash
streamlit run deepseek_voice_assistant.py
```

## 🎯 Usage Guide

### 1. **Initial Setup**
- Open the application in your browser (usually http://localhost:8501)
- Enter your DeepSeek API key in the sidebar
- Enter your OpenAI API key for text-to-speech functionality
- Select your preferred DeepSeek model
- Choose your preferred TTS voice

### 2. **Configure Voice Settings**
- **🎤 Microphone Listen**: Toggle to enable voice input
- **🔊 Read Response**: Toggle to enable spoken responses
- Check status indicators to ensure everything is working

### 3. **Interact with the Assistant**

#### Text Input:
- Type your message in the text area
- Click "📤 Send Message" to get a response

#### Voice Input (when enabled):
- Click "🎤 Start Recording" to begin voice input
- Speak your message clearly
- Click "⏹️ Stop Recording" when finished
- The system will process your speech and respond

### 4. **Manage Conversations**
- View conversation history in the right panel
- Use "🗑️ Clear Conversation" to start fresh
- Download audio responses when available

## 🎛️ Interface Overview

### Sidebar Controls
```
🔑 API Configuration
├── DeepSeek API Key
└── OpenAI API Key

🧠 Model Selection
└── DeepSeek Model (chat/coder/reasoner)

🎙️ Voice Settings
└── TTS Voice Selection

🎛️ Controls
├── 🎤 Microphone Listen Toggle
└── 🔊 Read Response Toggle

📊 Status
├── API Key Status
├── Audio Availability
└── Recording Status
```

### Main Interface
```
🎤 Voice Input
├── Start/Stop Recording Buttons
└── Recording Status

⌨️ Text Input
├── Message Text Area
└── Send Button

🤖 DeepSeek Response
├── Response Display
├── Audio Playback
└── Download Audio

💬 Conversation History
├── Message History
└── Timestamps
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file with your API keys:

```env
# DeepSeek API Key (required)
DEEPSEEK_API_KEY=your_deepseek_api_key_here

# OpenAI API Key (required for TTS)
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Logging level
LOG_LEVEL=INFO
```

### Model Options

**DeepSeek Models:**
- `deepseek-chat`: General conversation and Q&A
- `deepseek-coder`: Code generation and programming tasks
- `deepseek-reasoner`: Complex reasoning and analysis

**TTS Voices:**
- `alloy`, `ash`, `ballad`, `coral`, `echo`, `fable`, `onyx`, `nova`, `sage`, `shimmer`, `verse`

## 🛠️ Technical Details

### Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit UI  │    │  DeepSeek API    │    │   OpenAI TTS    │
│                 │    │                  │    │                 │
│ • Input Controls│◄──►│ • Chat Models    │    │ • Voice Synthesis│
│ • Voice Toggles │    │ • Code Models    │    │ • Audio Output  │
│ • Status Display│    │ • Reasoner Model │    │ • Multiple Voices│
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────────┐
                    │   Audio Processing  │
                    │                     │
                    │ • Speech Recording  │
                    │ • Audio Playback    │
                    │ • Format Conversion │
                    └─────────────────────┘
```

### Key Components

- **Streamlit Interface**: Modern web-based UI with real-time controls
- **DeepSeek Integration**: Direct API communication with error handling
- **Audio Processing**: Real-time recording and playback capabilities
- **Session Management**: Persistent conversation history and settings

### Dependencies

**Core:**
- `streamlit`: Web interface framework
- `requests`: HTTP client for DeepSeek API
- `openai`: OpenAI client for TTS

**Audio:**
- `sounddevice`: Audio recording and playback
- `soundfile`: Audio file handling
- `numpy`: Audio data processing

## 🔍 Troubleshooting

### Common Issues

**1. Audio not working:**
```bash
# Install audio dependencies
pip install sounddevice soundfile numpy

# macOS
brew install portaudio

# Ubuntu/Debian
sudo apt-get install portaudio19-dev
```

**2. API key errors:**
- Verify your API keys are correct
- Check your account has sufficient credits
- Ensure API keys have proper permissions

**3. Recording issues:**
- Check microphone permissions
- Verify audio input device is working
- Try different audio settings

**4. TTS not working:**
- Verify OpenAI API key is valid
- Check internet connection
- Try different voice options

### Debug Mode

Enable debug logging by setting:
```env
LOG_LEVEL=DEBUG
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built using patterns from [awesome-llm-apps](https://github.com/Shubhamsaboo/awesome-llm-apps)
- Powered by DeepSeek's advanced language models
- Voice capabilities provided by OpenAI's TTS technology
- Audio processing using sounddevice and soundfile libraries

## 📞 Support

For issues, questions, or contributions:
- Check the troubleshooting section above
- Review the configuration options
- Ensure all dependencies are properly installed

---

**Happy voice computing with DeepSeek! 🤖🎤**

