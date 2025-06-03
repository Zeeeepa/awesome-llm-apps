#!/bin/bash
# DeepSeek Voice Agent - Quick Setup Script

set -e

echo "🎙️ DeepSeek Voice Agent - Quick Setup"
echo "======================================"

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.9"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python 3.9+ required. Found: $python_version"
    exit 1
fi

echo "✅ Python version: $python_version"

# Install system dependencies
echo "📦 Installing system dependencies..."

if command -v apt-get &> /dev/null; then
    # Ubuntu/Debian
    sudo apt-get update
    sudo apt-get install -y portaudio19-dev python3-pyaudio python3-pip
elif command -v brew &> /dev/null; then
    # macOS
    brew install portaudio
elif command -v yum &> /dev/null; then
    # CentOS/RHEL
    sudo yum install -y portaudio-devel python3-pip
else
    echo "⚠️  Please install portaudio manually for your system"
fi

# Create virtual environment
echo "🐍 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
echo "📚 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    cp .env.example .env
    echo "⚠️  Please edit .env file and add your API keys!"
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file and add your API keys:"
echo "   - DeepSeek API key from https://platform.deepseek.com/"
echo "   - OpenAI API key from https://platform.openai.com/"
echo ""
echo "2. Run the application:"
echo "   source venv/bin/activate"
echo "   python run.py"
echo ""
echo "   Or directly:"
echo "   source venv/bin/activate"
echo "   streamlit run deepseek_voice_app.py"
echo ""
echo "🚀 Happy voice chatting with DeepSeek!"

