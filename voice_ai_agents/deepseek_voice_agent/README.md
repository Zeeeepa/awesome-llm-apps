# 🎙️ DeepSeek Voice Agent

A comprehensive voice-enabled AI assistant powered by DeepSeek models. This Streamlit application provides a seamless interface for interacting with DeepSeek's powerful language models using both voice and text input, with customizable text-to-speech responses.

## ✨ Features

- **🤖 DeepSeek Model Integration**: Support for multiple DeepSeek models including:
  - `deepseek-chat` - General conversation model
  - `deepseek-coder` - Specialized for coding tasks
  - `deepseek-reasoner` - Advanced reasoning capabilities
  - `deepseek-r1` and distilled variants - Latest reasoning models
  
- **🎤 Voice Input**: Real-time speech-to-text using Google Speech Recognition
- **🔊 Voice Output**: High-quality text-to-speech with multiple voice options
- **⚙️ Toggle Controls**: 
  - Enable/disable microphone listening
  - Enable/disable response reading
- **💬 Conversation Management**: Persistent chat history with timestamps
- **🎨 Modern UI**: Clean, intuitive Streamlit interface

## 🚀 Quick Start

### Prerequisites

1. **DeepSeek API Key**: Get your API key from [DeepSeek Platform](https://platform.deepseek.com/)
2. **OpenAI API Key**: Get your API key from [OpenAI Platform](https://platform.openai.com/) for text-to-speech functionality
3. **Python 3.8+**: Ensure you have Python installed
4. **Microphone**: For voice input functionality

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Zeeeepa/awesome-llm-apps.git
   cd awesome-llm-apps/voice_ai_agents/deepseek_voice_agent
   ```

2. **Install dependencies**:
   ```bash
   pip install streamlit openai speech-recognition pyttsx3 python-dotenv
   ```

3. **Additional system dependencies** (for speech recognition):
   
   **On Ubuntu/Debian**:
   ```bash
   sudo apt-get install portaudio19-dev python3-pyaudio
   pip install pyaudio
   ```
   
   **On macOS**:
   ```bash
   brew install portaudio
   pip install pyaudio
   ```
   
   **On Windows**:
   ```bash
   pip install pyaudio
   ```

### Running the Application

1. **Start the Streamlit app**:
   ```bash
   streamlit run deepseek_voice_agent.py
   ```

2. **Configure the application**:
   - Enter your DeepSeek API key in the sidebar
   - Enter your OpenAI API key in the sidebar (for text-to-speech)
   - Select your preferred DeepSeek model
   - Choose voice settings and toggle features
   - Click "Initialize System"

3. **Start conversing**:
   - Type messages in the text input
   - Use the "Voice Input" button for speech input
   - Listen to AI responses (if enabled)

## 🎛️ Configuration Options

### API Settings
- **DeepSeek API Key**: Your authentication key for DeepSeek services
- **OpenAI API Key**: Your authentication key for OpenAI services

### Model Selection
Choose from various DeepSeek models based on your needs:
- **deepseek-chat**: Best for general conversations
- **deepseek-coder**: Optimized for programming tasks
- **deepseek-reasoner**: Advanced reasoning and analysis
- **deepseek-r1**: Latest reasoning model with enhanced capabilities

### Voice Settings
- **TTS Voice**: Select from 6 different voice personalities:
  - `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`

### Feature Toggles
- **����️ Microphone Listen**: Enable/disable voice input
- **🔊 Read Response**: Enable/disable text-to-speech output

## 💡 Usage Examples

### Text Conversation
1. Type your question in the text input field
2. Press Enter to send
3. View the AI response in the chat interface

### Voice Conversation
1. Enable "Microphone Listen" in the sidebar
2. Click the "🎤 Voice Input" button
3. Speak your question clearly
4. The transcribed text will appear and be processed
5. Listen to the AI response (if "Read Response" is enabled)

### Programming Help
1. Select the "deepseek-coder" model
2. Ask coding questions like:
   - "Write a Python function to sort a list"
   - "Explain how to use async/await in JavaScript"
   - "Debug this code snippet: [paste code]"

### Advanced Reasoning
1. Select the "deepseek-reasoner" or "deepseek-r1" model
2. Ask complex questions requiring analysis:
   - "Analyze the pros and cons of different database architectures"
   - "Explain the economic implications of renewable energy adoption"

## 🔧 Technical Details

### Architecture
- **Frontend**: Streamlit web interface
- **Speech Recognition**: Google Speech Recognition API
- **Text-to-Speech**: OpenAI TTS API
- **AI Models**: DeepSeek API integration
- **Session Management**: Streamlit session state

### Dependencies
```
streamlit>=1.28.0
openai>=1.0.0
speech-recognition>=3.10.0
pyttsx3>=2.90
python-dotenv>=1.0.0
pyaudio>=0.2.11
```

### File Structure
```
deepseek_voice_agent/
├── deepseek_voice_agent.py    # Main application
├── README.md                  # This documentation
└── requirements.txt           # Python dependencies
```

## 🛠️ Troubleshooting

### Common Issues

1. **Microphone not working**:
   - Check microphone permissions in your browser/system
   - Ensure PyAudio is properly installed
   - Try running with administrator privileges

2. **Speech recognition errors**:
   - Check internet connection (Google Speech API requires internet)
   - Speak clearly and avoid background noise
   - Adjust microphone sensitivity

3. **API errors**:
   - Verify your DeepSeek API key is correct
   - Check API rate limits and usage
   - Ensure you have sufficient API credits

4. **Audio playback issues**:
   - Check system audio settings
   - Ensure browser allows audio playback
   - Try different voice options

### Performance Tips

- Use lighter models (like distilled variants) for faster responses
- Clear conversation history periodically for better performance
- Disable voice features if not needed to reduce resource usage

## 🔒 Privacy & Security

- **API Keys**: Stored only in session state, not persisted
- **Voice Data**: Processed via Google Speech Recognition (see their privacy policy)
- **Conversation History**: Stored locally in browser session only
- **Audio Files**: Temporary files are automatically cleaned up

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## 📄 License

This project is part of the awesome-llm-apps collection. Please refer to the main repository license.

## 🔗 Related Projects

- [Customer Support Voice Agent](../customer_support_voice_agent/) - RAG-powered voice support
- [Voice RAG OpenAI SDK](../voice_rag_openaisdk/) - Document-based voice assistant
- [DeepSeek Local RAG Agent](../../rag_tutorials/deepseek_local_rag_agent/) - Local DeepSeek with RAG

## 📞 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the [DeepSeek API documentation](https://platform.deepseek.com/api-docs/)
3. Open an issue in the main repository

---

**Happy conversing with DeepSeek! 🚀**
