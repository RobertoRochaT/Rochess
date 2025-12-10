#!/bin/bash
# Script de instalación para ChessBot API

echo "🚀 Instalando ChessBot API..."
echo ""

# Verificar Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 no está instalado. Por favor instala Python 3.10 o superior."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✅ Python $PYTHON_VERSION detectado"
echo ""

# Crear entorno virtual si no existe
if [ ! -d "venv" ]; then
    echo "📦 Creando entorno virtual..."
    python3 -m venv venv
    echo "✅ Entorno virtual creado"
else
    echo "✅ Entorno virtual ya existe"
fi

# Activar entorno virtual
echo "🔄 Activando entorno virtual..."
source venv/bin/activate

# Actualizar pip
echo "⬆️  Actualizando pip..."
pip install --upgrade pip

# Instalar dependencias principales
echo "📥 Instalando dependencias principales..."
pip install -r requirements_chessbot.txt

# Instalar dependencias de API
echo "📥 Instalando dependencias de API..."
pip install -r tensorflow_chessbot/requirements_api.txt

echo ""
echo "✅ ¡Instalación completada!"
echo ""
echo "Para usar ChessBot API:"
echo "  1. Activa el entorno: source venv/bin/activate"
echo "  2. Inicia el servidor: cd tensorflow_chessbot && python api_server.py"
echo "  3. O ejecuta scripts: python chess_board_to_fen.py <imagen>"
echo ""
