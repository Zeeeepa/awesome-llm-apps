# 🎙️ DeepSeek Voice Agent - Single Deployment Package

A complete voice-to-voice AI assistant using DeepSeek models, consolidated into a single deployment folder for easy setup and use.

## ✨ Features

**Complete Voice-to-Voice Pipeline:**
- 🎤 **Voice Input** - Real-time speech recognition
- 🤖 **DeepSeek AI** - Multiple model options (chat, coder, reasoner, r1)
- 🔊 **Voice Output** - High-quality text-to-speech
- 💬 **Conversation History** - Persistent chat memory
- ⚙️ **Toggle Controls** - Enable/disable voice features
- 🎨 **Single UI** - Everything in one clean interface

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**System Dependencies (for speech recognition):**

**Ubuntu/Debian:**
```bash
sudo apt-get install portaudio19-dev python3-pyaudio
```

**macOS:**
```bash
brew install portaudio
```

**Windows:**
```bash
# PyAudio should install automatically with pip
```

### 2. Get API Keys

- **DeepSeek API Key**: [Get from DeepSeek Platform](https://platform.deepseek.com/)
- **OpenAI API Key**: [Get from OpenAI Platform](https://platform.openai.com/)

### 3. Run the Application

```bash
streamlit run deepseek_voice_app.py
```

### 4. Configure & Use

1. **Enter API Keys** in the sidebar
2. **Select Model** (deepseek-chat, deepseek-coder, etc.)
3. **Choose Voice** (alloy, nova, shimmer, etc.)
4. **Toggle Features** (Microphone Listen, Read Response)
5. **Initialize System** 
6. **Start Talking!** 🗣️

## 🎯 Usage Examples

### Voice Conversation
1. Click "🎤 Voice Input"
2. Speak your question
3. Listen to AI response

### Text + Voice
1. Type your message
2. Get voice response (if enabled)

### Programming Help
1. Select "deepseek-coder" model
2. Ask: "Write a Python function to sort a list"
3. Get code explanation via voice

## 📁 File Structure

```
deepseek_voice_deployment/
├── deepseek_voice_app.py    # Main application (single file)
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## 🔧 Configuration Options

### Models Available
- `deepseek-chat` - General conversation
- `deepseek-coder` - Programming tasks
- `deepseek-reasoner` - Advanced reasoning
- `deepseek-r1` - Latest reasoning model
- Multiple distilled variants (70b, 32b, 14b, 7b, 1.5b)

### Voice Options
- `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`

### Feature Toggles
- **🎙️ Microphone Listen** - Enable/disable voice input
- **🔊 Read Response** - Enable/disable voice output

## 🛠️ Troubleshooting

### Common Issues

**Microphone not working:**
- Check browser/system permissions
- Ensure PyAudio is installed correctly
- Try running with admin privileges

**Speech recognition errors:**
- Check internet connection (Google Speech API)
- Speak clearly, avoid background noise
- Adjust microphone sensitivity

**API errors:**
- Verify API keys are correct
- Check rate limits and credits
- Ensure proper API access

**Audio playback issues:**
- Check system audio settings
- Ensure browser allows audio
- Try different voice options

## 🔒 Security

- API keys stored only in session state
- No persistent storage of sensitive data
- Temporary audio files auto-cleaned
- Secure API communication

## 🚀 Deployment Options

### Local Development
```bash
streamlit run deepseek_voice_app.py
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# Install system dependencies
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

COPY deepseek_voice_app.py .

EXPOSE 8501

CMD ["streamlit", "run", "deepseek_voice_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Cloud Deployment
- **Streamlit Cloud**: Upload folder and deploy
- **Heroku**: Add buildpacks for audio dependencies
- **Railway**: Direct deployment from folder
- **Render**: Static site with Python runtime

## 💡 Advanced Features

### Environment Variables
Create a `.env` file:
```
DEEPSEEK_API_KEY=your_deepseek_key_here
OPENAI_API_KEY=your_openai_key_here
```

### Custom Configuration
Modify the constants in `deepseek_voice_app.py`:
```python
DEEPSEEK_MODELS = ["deepseek-chat", "your-custom-model"]
TTS_VOICES = ["nova", "alloy"]  # Limit voice options
```

## 📊 Performance Tips

- Use lighter models (distilled variants) for faster responses
- Clear conversation history periodically
- Disable voice features if not needed
- Adjust timeout settings for speech recognition

## 🤝 Contributing

This is a consolidated deployment package. For development:
1. Modify `deepseek_voice_app.py` directly
2. Test locally with `streamlit run deepseek_voice_app.py`
3. Update requirements if adding dependencies

## 📄 License

Part of the awesome-llm-apps collection. See main repository for license details.

---

**Ready to deploy! 🎉** This single folder contains everything needed for a complete voice-to-voice DeepSeek AI experience.

