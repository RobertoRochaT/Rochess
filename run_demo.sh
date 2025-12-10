#!/bin/bash
# Script de ejemplo completo para usar el sistema de análisis FEN

# Colores para output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}🏁  Demo Completo - Sistema de Análisis de Tableros FEN${NC}"
echo -e "${BLUE}============================================================${NC}\n"

# Activar entorno virtual
echo -e "${YELLOW}📦 Activando entorno virtual...${NC}"
source venv/bin/activate

echo -e "\n${GREEN}✅ Entorno activado${NC}\n"

# 1. Generar tableros de ejemplo
echo -e "${YELLOW}1️⃣  Generando tableros de ejemplo...${NC}\n"
python generate_samples.py
echo ""

# 2. Crear tablero de demostración
echo -e "${YELLOW}2️⃣  Creando tablero de demostración...${NC}\n"
python chess_fen_analyzer.py --demo
echo ""

# 3. Visualizar posición inicial
echo -e "${YELLOW}3️⃣  Visualizando posición inicial...${NC}\n"
python chess_fen_analyzer.py --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" --output posicion_inicial_test.svg
echo ""

# 4. Visualizar apertura italiana
echo -e "${YELLOW}4️⃣  Visualizando Apertura Italiana...${NC}\n"
python chess_fen_analyzer.py --fen "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3" --output apertura_italiana.svg
echo ""

# 5. Mostrar archivos generados
echo -e "${YELLOW}5️⃣  Archivos SVG generados:${NC}\n"
ls -lh *.svg data/predictions/*.svg 2>/dev/null | grep -v "total"
echo ""

# Resumen final
echo -e "${BLUE}============================================================${NC}"
echo -e "${GREEN}✅ Demo completado exitosamente!${NC}"
echo -e "${BLUE}============================================================${NC}\n"

echo -e "${YELLOW}📂 Archivos generados:${NC}"
echo "   • Tableros de ejemplo en: data/predictions/"
echo "   • Tableros de demo en el directorio actual"
echo ""

echo -e "${YELLOW}🌐 Para visualizar los tableros:${NC}"
echo "   1. Abre cualquier archivo .svg con tu navegador web"
echo "   2. O usa: open demo_board.svg (en macOS)"
echo ""

echo -e "${YELLOW}📖 Para más información:${NC}"
echo "   • Lee GUIA_USO.md"
echo "   • Ejecuta: python chess_fen_analyzer.py --help"
echo ""

echo -e "${YELLOW}🚀 Próximos pasos:${NC}"
echo "   1. Descarga el modelo ONNX desde:"
echo "      https://github.com/davidmallasen/LiveChess2FEN/releases"
echo "   2. Guárdalo en: data/models/MobileNetV2_0p5_all.onnx"
echo "   3. Procesa tus propias imágenes de tableros!"
echo ""
