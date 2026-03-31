#!/bin/bash
set -e
cd "$(dirname "$0")"

if [ ! -f .env ]; then
  echo "❌ No .env file found. Copy .env.example to .env and fill in your tokens."
  exit 1
fi

echo "📦 Installing dependencies..."
pip3 install -q -r requirements.txt

LOCAL_IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo "localhost")
echo ""
echo "🚀 Starting PaperSwipe..."
echo "   Local:   http://localhost:8000"
echo "   Phone:   http://${LOCAL_IP}:8000"
echo ""

/Library/Developer/CommandLineTools/usr/bin/python3 -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
