# 🚀 DeepSeek Voice Agent - Deployment Guide

Complete deployment options for the DeepSeek Voice Agent single-folder package.

## 📁 Package Contents

```
deepseek_voice_deployment/
├── deepseek_voice_app.py    # Main application (single file)
├── requirements.txt         # Python dependencies
├── README.md               # User documentation
├── .env.example            # Environment variables template
├── run.py                  # Quick launcher script
├── setup.sh               # Automated setup script
├── Dockerfile             # Docker container
├── docker-compose.yml     # Docker Compose config
└── DEPLOYMENT_GUIDE.md    # This file
```

## 🎯 Deployment Options

### Option 1: Quick Local Setup (Recommended)

**1. Run the setup script:**
```bash
chmod +x setup.sh
./setup.sh
```

**2. Configure API keys:**
```bash
# Edit .env file with your keys
nano .env
```

**3. Launch the app:**
```bash
source venv/bin/activate
python run.py
```

### Option 2: Manual Local Setup

**1. Install dependencies:**
```bash
# System dependencies (Ubuntu/Debian)
sudo apt-get install portaudio19-dev python3-pyaudio

# Python dependencies
pip install -r requirements.txt
```

**2. Configure environment:**
```bash
cp .env.example .env
# Edit .env with your API keys
```

**3. Run the application:**
```bash
streamlit run deepseek_voice_app.py
```

### Option 3: Docker Deployment

**1. Build and run with Docker:**
```bash
# Build the image
docker build -t deepseek-voice-agent .

# Run the container
docker run -p 8501:8501 \
  -e DEEPSEEK_API_KEY=your_key \
  -e OPENAI_API_KEY=your_key \
  deepseek-voice-agent
```

**2. Or use Docker Compose:**
```bash
# Configure .env file first
cp .env.example .env
# Edit .env with your API keys

# Start the service
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the service
docker-compose down
```

### Option 4: Cloud Deployment

#### Streamlit Cloud
1. Upload the `deepseek_voice_deployment` folder to GitHub
2. Connect to Streamlit Cloud
3. Set environment variables in Streamlit Cloud dashboard
4. Deploy!

#### Railway
1. Create new project from folder
2. Set environment variables
3. Deploy automatically

#### Render
1. Create new web service
2. Connect repository
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `streamlit run deepseek_voice_app.py --server.port=$PORT --server.address=0.0.0.0`

#### Heroku
1. Create `Procfile`:
   ```
   web: streamlit run deepseek_voice_app.py --server.port=$PORT --server.address=0.0.0.0
   ```
2. Add buildpacks:
   - `heroku/python`
   - `https://github.com/jonathanong/heroku-buildpack-ffmpeg-latest.git`
3. Deploy with git

## 🔧 Configuration

### Required Environment Variables
```bash
DEEPSEEK_API_KEY=your_deepseek_api_key
OPENAI_API_KEY=your_openai_api_key
```

### Optional Configuration
Edit `deepseek_voice_app.py` to customize:

```python
# Available models
DEEPSEEK_MODELS = [
    "deepseek-chat",
    "deepseek-coder", 
    "deepseek-reasoner",
    # Add/remove models as needed
]

# Available voices
TTS_VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
```

## 🛠️ Troubleshooting

### Common Issues

**1. Audio dependencies not found:**
```bash
# Ubuntu/Debian
sudo apt-get install portaudio19-dev python3-pyaudio

# macOS
brew install portaudio

# Windows
# Usually works with pip install pyaudio
```

**2. Permission errors:**
```bash
# Make scripts executable
chmod +x setup.sh run.py

# Run with proper permissions
sudo ./setup.sh  # if needed
```

**3. Port already in use:**
```bash
# Change port in command
streamlit run deepseek_voice_app.py --server.port=8502
```

**4. API key errors:**
- Verify keys are correct in `.env` file
- Check API key permissions and credits
- Ensure no extra spaces in keys

### Performance Optimization

**1. Use lighter models:**
```python
# In the app, select:
# - deepseek-r1-distill-qwen-1.5b (fastest)
# - deepseek-r1-distill-qwen-7b (balanced)
```

**2. Disable features if not needed:**
- Turn off "Read Response" for text-only
- Turn off "Microphone Listen" for typing-only

**3. Clear conversation history regularly:**
- Use "Clear History" button in sidebar
- Reduces memory usage and API context

## 📊 Monitoring

### Health Checks
```bash
# Check if app is running
curl http://localhost:8501/_stcore/health

# Docker health check
docker-compose ps
```

### Logs
```bash
# Local logs
tail -f ~/.streamlit/logs/streamlit.log

# Docker logs
docker-compose logs -f deepseek-voice-agent
```

## 🔒 Security Considerations

### Production Deployment
1. **Use HTTPS** - Always deploy with SSL/TLS
2. **Environment Variables** - Never commit API keys to code
3. **Access Control** - Consider authentication if needed
4. **Rate Limiting** - Monitor API usage
5. **Firewall** - Restrict access to necessary ports only

### API Key Security
```bash
# Set restrictive permissions on .env
chmod 600 .env

# Use environment variables in production
export DEEPSEEK_API_KEY="your_key"
export OPENAI_API_KEY="your_key"
```

## 📈 Scaling

### Horizontal Scaling
- Deploy multiple instances behind a load balancer
- Use session affinity for conversation continuity
- Consider shared storage for conversation history

### Vertical Scaling
- Increase memory for better speech processing
- Use faster CPUs for real-time audio processing
- Consider GPU instances for enhanced performance

## 🆘 Support

### Getting Help
1. Check the troubleshooting section above
2. Review application logs
3. Test with minimal configuration
4. Check API service status

### Common Commands
```bash
# Restart the application
docker-compose restart

# View real-time logs
docker-compose logs -f

# Update the application
git pull && docker-compose up -d --build

# Clean up Docker resources
docker system prune -a
```

---

**Ready to deploy! 🎉** Choose the deployment option that best fits your needs and start voice chatting with DeepSeek!

