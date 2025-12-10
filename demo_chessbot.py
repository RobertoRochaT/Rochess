#!/usr/bin/env python3
"""
Script de demostración: procesa imágenes clave para mostrar las capacidades del sistema
"""

import os
import sys

# Ejemplos representativos de diferentes tipos de posiciones
example_images = [
    "images/01_starting_position.png",
    "images/02_e4_opening.png", 
    "images/08_ruy_lopez.png",
    "images/14_middlegame_4.png",
    "images/21_endgame_1.png",
    "images/31_tactics_1.png",
    "images/73_castling_3.png",
]

print("🎯 ChessBot - Demostración de Conversión de Tableros a FEN\n")
print(f"Procesando {len(example_images)} imágenes de ejemplo...\n")

for img in example_images:
    if os.path.exists(img):
        img_name = os.path.basename(img)
        print(f"▶️  Procesando: {img_name}")
        cmd = f"python chess_board_to_fen.py {img} --output-dir demo_resultados"
        os.system(cmd + " > /dev/null 2>&1")
        print(f"   ✅ Completado\n")
    else:
        print(f"   ⚠️  No encontrado: {img}\n")

print("=" * 60)
print("✅ Demostración completada!")
print("📁 Revisa la carpeta 'demo_resultados' para ver los resultados")
print("📊 Abre los archivos *_comparison.png para ver:")
print("   - Imagen original del tablero")
print("   - Representación detectada con colores")
print("   - FEN mapeado con piezas reales en tablero virtual")
print("=" * 60)
