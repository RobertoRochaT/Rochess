#!/bin/bash
# Script para iniciar el servidor API de ChessBot

echo "🚀 Iniciando ChessBot API Server..."
echo ""

# Verificar si el entorno virtual existe
if [ ! -d "venv" ]; then
    echo "❌ No se encontró el entorno virtual."
    echo "   Por favor ejecuta primero: ./install.sh"
    exit 1
fi

# Activar entorno virtual
source venv/bin/activate

# Verificar si las dependencias están instaladas
if ! python -c "import tensorflow" 2>/dev/null; then
    echo "❌ TensorFlow no está instalado."
    echo "   Por favor ejecuta primero: ./install.sh"
    exit 1
fi

# Verificar si el modelo existe
if [ ! -f "tensorflow_chessbot/saved_models/frozen_graph.pb" ]; then
    echo "⚠️  Advertencia: No se encontró el modelo pre-entrenado"
    echo "   Ubicación esperada: tensorflow_chessbot/saved_models/frozen_graph.pb"
    echo ""
fi

# Configurar variables de entorno opcionales
export FLASK_ENV=${FLASK_ENV:-production}
export FLASK_PORT=${FLASK_PORT:-5000}
export FLASK_HOST=${FLASK_HOST:-0.0.0.0}

echo "📍 Configuración:"
echo "   - Host: $FLASK_HOST"
echo "   - Puerto: $FLASK_PORT"
echo "   - Entorno: $FLASK_ENV"
echo ""

# Iniciar servidor
cd tensorflow_chessbot
echo "✅ Servidor iniciando..."
echo "   URL: http://localhost:$FLASK_PORT"
echo ""
python api_server.py --host $FLASK_HOST --port $FLASK_PORT
