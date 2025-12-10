#!/usr/bin/env python3
"""
Chess Board to FEN Converter
Utiliza tensorflow_chessbot para convertir imágenes de tableros de ajedrez a notación FEN
y genera visualizaciones del tablero detectado.
"""

import os
import sys
import argparse
import numpy as np
import cv2
from PIL import Image
import chess
import chess.svg
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Añadir el path del tensorflow_chessbot
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tensorflow_chessbot'))

from tensorflow_chessbot import ChessboardPredictor
from helper_functions import shortenFEN
import helper_image_loading
import chessboard_finder


class ChessBoardAnalyzer:
    """Analizador de tableros de ajedrez que convierte imágenes a FEN"""
    
    def __init__(self, model_path='tensorflow_chessbot/saved_models/frozen_graph.pb'):
        """Inicializa el predictor con el modelo entrenado"""
        self.model_path = model_path
        self.predictor = None
        
        if os.path.exists(model_path):
            print("🔍 Inicializando modelo de reconocimiento...")
            self.predictor = ChessboardPredictor(model_path)
            print("✅ Modelo cargado correctamente")
        else:
            print(f"❌ Error: No se encontró el modelo en {model_path}")
            print("   Por favor descarga el modelo desde:")
            print("   https://github.com/Elucidation/tensorflow_chessbot/tree/chessfenbot/saved_models")
            raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
    
    def process_image(self, image_path):
        """
        Procesa una imagen y extrae el FEN del tablero
        
        Args:
            image_path: Ruta a la imagen del tablero
            
        Returns:
            dict con 'fen', 'certainty', 'shortened_fen', 'board_image'
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Imagen no encontrada: {image_path}")
        
        print(f"\n📸 Procesando imagen: {os.path.basename(image_path)}")
        
        # Cargar imagen
        img = helper_image_loading.loadImageFromPath(image_path)
        
        # Redimensionar si es necesario
        img = helper_image_loading.resizeAsNeeded(img)
        
        if img is None:
            return {
                'success': False,
                'error': 'Imagen demasiado grande para procesar',
                'fen': None,
                'certainty': 0.0
            }
        
        # Extraer tiles del tablero
        tiles, corners = chessboard_finder.findGrayscaleTilesInImage(img)
        
        if tiles is None:
            return {
                'success': False,
                'error': 'No se pudo detectar el tablero en la imagen',
                'fen': None,
                'certainty': 0.0
            }
        
        # Predecir el FEN
        fen, tile_certainties = self.predictor.getPrediction(tiles)
        
        if fen is None:
            return {
                'success': False,
                'error': 'No se pudo generar FEN del tablero',
                'fen': None,
                'certainty': 0.0
            }
        
        # Calcular certeza promedio
        avg_certainty = np.mean(tile_certainties)
        min_certainty = np.min(tile_certainties)
        max_certainty = np.max(tile_certainties)
        
        # Acortar FEN (notación estándar)
        shortened_fen = shortenFEN(fen)
        
        print(f"📋 FEN detectado: {shortened_fen}")
        print(f"📊 Certeza: Promedio={avg_certainty*100:.1f}%, Mín={min_certainty*100:.1f}%, Máx={max_certainty*100:.1f}%")
        
        return {
            'success': True,
            'fen': fen,
            'shortened_fen': shortened_fen,
            'certainty': {
                'average': float(avg_certainty),
                'min': float(min_certainty),
                'max': float(max_certainty),
                'per_tile': tile_certainties.tolist()
            },
            'board_image': img
        }
    
    def visualize_board(self, fen, output_path=None, show=True):
        """
        Genera visualización del tablero desde un FEN
        
        Args:
            fen: String FEN del tablero
            output_path: Ruta donde guardar la imagen (opcional)
            show: Si mostrar la imagen (default: True)
        """
        try:
            # Crear tablero de chess
            board = chess.Board(fen)
            
            # Generar SVG del tablero
            svg_data = chess.svg.board(board, size=400)
            
            # Convertir SVG a imagen para guardar
            if output_path:
                # Usar cairosvg si está disponible, sino guardar SVG directamente
                try:
                    import cairosvg
                    cairosvg.svg2png(bytestring=svg_data.encode('utf-8'), 
                                    write_to=output_path)
                    print(f"💾 Tablero guardado en: {output_path}")
                except ImportError:
                    # Guardar como SVG si no tenemos cairosvg
                    svg_path = output_path.replace('.png', '.svg')
                    with open(svg_path, 'w') as f:
                        f.write(svg_data)
                    print(f"💾 Tablero guardado como SVG en: {svg_path}")
            
            return board, svg_data
            
        except Exception as e:
            print(f"❌ Error al visualizar tablero: {e}")
            return None, None
    
    def create_comparison_image(self, original_img, board, output_path):
        """
        Crea una imagen comparativa con tres paneles:
        1. Imagen original
        2. Tablero detectado con representación visual
        3. FEN mapeado a tablero virtual de ajedrez
        
        Args:
            original_img: Imagen original PIL
            board: Objeto chess.Board
            output_path: Ruta donde guardar la comparación
        """
        try:
            # Crear figura con tres subplots
            fig = plt.figure(figsize=(18, 6))
            gs = fig.add_gridspec(1, 3, hspace=0.3, wspace=0.3)
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            ax3 = fig.add_subplot(gs[0, 2])
            
            # Panel 1: Imagen original
            ax1.imshow(np.array(original_img))
            ax1.set_title('📸 Imagen Original', fontsize=14, fontweight='bold', pad=10)
            ax1.axis('off')
            
            # Panel 2: Tablero detectado con representación de piezas
            board_array = self._board_to_array(board)
            ax2.imshow(board_array)
            ax2.set_title('🔍 Tablero Detectado', fontsize=14, fontweight='bold', pad=10)
            ax2.axis('off')
            
            # Añadir etiquetas de coordenadas al tablero detectado
            for i in range(8):
                # Files (a-h)
                ax2.text(i, -0.5, chr(97 + i), ha='center', va='center', 
                        fontsize=10, fontweight='bold')
                # Ranks (1-8)
                ax2.text(-0.5, 7-i, str(i+1), ha='center', va='center', 
                        fontsize=10, fontweight='bold')
            
            # Panel 3: FEN mapeado a tablero virtual
            virtual_board_array = self._create_virtual_board(board)
            ax3.imshow(virtual_board_array)
            ax3.set_title('♟️  FEN Mapeado (Tablero Virtual)', fontsize=14, fontweight='bold', pad=10)
            ax3.axis('off')
            
            # Añadir etiquetas de coordenadas al tablero virtual
            for i in range(8):
                ax3.text(i, -0.5, chr(97 + i), ha='center', va='center', 
                        fontsize=10, fontweight='bold')
                ax3.text(-0.5, 7-i, str(i+1), ha='center', va='center', 
                        fontsize=10, fontweight='bold')
            
            # Añadir información del FEN y análisis en la parte inferior
            info_text = f'FEN: {board.fen()}\n'
            info_text += f'Turno: {"Blancas" if board.turn else "Negras"} | '
            info_text += f'Jaque: {"Sí" if board.is_check() else "No"} | '
            info_text += f'Movimientos legales: {board.legal_moves.count()}'
            
            fig.text(0.5, 0.02, info_text, 
                    ha='center', fontsize=11, fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"📊 Comparación de 3 paneles guardada en: {output_path}")
            
        except Exception as e:
            print(f"⚠️  Error al crear imagen de comparación: {e}")
            import traceback
            traceback.print_exc()
    
    def _board_to_array(self, board):
        """Convierte un tablero de chess a un array numpy para visualización"""
        # Crear array 8x8 para representar el tablero
        board_array = np.zeros((8, 8, 3), dtype=np.uint8)
        
        piece_colors = {
            'P': [255, 255, 200], 'N': [255, 255, 150], 'B': [255, 255, 100],
            'R': [255, 255, 50],  'Q': [255, 255, 0],   'K': [255, 200, 0],
            'p': [100, 100, 255], 'n': [80, 80, 255],   'b': [60, 60, 255],
            'r': [40, 40, 255],   'q': [20, 20, 255],   'k': [0, 0, 200]
        }
        
        for i in range(64):
            rank = 7 - (i // 8)
            file = i % 8
            square = chess.square(file, rank)
            piece = board.piece_at(square)
            
            if piece:
                symbol = piece.symbol()
                board_array[rank, file] = piece_colors.get(symbol, [128, 128, 128])
            else:
                # Casillas vacías alternadas
                if (rank + file) % 2 == 0:
                    board_array[rank, file] = [240, 217, 181]  # Casilla clara
                else:
                    board_array[rank, file] = [181, 136, 99]   # Casilla oscura
        
        return board_array
    
    def _create_virtual_board(self, board):
        """
        Crea una representación visual del tablero usando python-chess
        con piezas reales y notación de coordenadas
        """
        # Generar SVG del tablero con coordenadas
        svg_data = chess.svg.board(
            board, 
            size=512,
            coordinates=True,
            colors={
                'square light': '#f0d9b5',
                'square dark': '#b58863',
            }
        )
        
        # Convertir SVG a imagen
        try:
            from io import BytesIO
            from PIL import Image
            import cairosvg
            
            # Convertir SVG a PNG en memoria
            png_data = cairosvg.svg2png(bytestring=svg_data.encode('utf-8'))
            img = Image.open(BytesIO(png_data))
            return np.array(img)
            
        except ImportError:
            # Si cairosvg no está disponible, usar matplotlib para renderizar texto
            import matplotlib.patches as mpatches
            from matplotlib.patches import FancyBboxPatch
            
            # Crear figura temporal
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_xlim(0, 8)
            ax.set_ylim(0, 8)
            ax.set_aspect('equal')
            ax.axis('off')
            
            # Colores del tablero
            light_color = '#f0d9b5'
            dark_color = '#b58863'
            
            # Símbolos Unicode de piezas
            piece_unicode = {
                'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
                'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚'
            }
            
            # Dibujar casillas y piezas
            for rank in range(8):
                for file in range(8):
                    # Color de casilla
                    is_light = (rank + file) % 2 == 0
                    color = light_color if is_light else dark_color
                    
                    # Dibujar casilla
                    rect = FancyBboxPatch(
                        (file, 7 - rank), 1, 1,
                        boxstyle="square,pad=0",
                        facecolor=color,
                        edgecolor='none'
                    )
                    ax.add_patch(rect)
                    
                    # Obtener pieza
                    square = chess.square(file, rank)
                    piece = board.piece_at(square)
                    
                    if piece:
                        symbol = piece.symbol()
                        unicode_piece = piece_unicode.get(symbol, symbol)
                        piece_color = 'white' if piece.color else 'black'
                        
                        # Dibujar pieza con mayor tamaño
                        ax.text(
                            file + 0.5, 7 - rank + 0.5,
                            unicode_piece,
                            fontsize=48,
                            ha='center',
                            va='center',
                            color=piece_color,
                            weight='bold',
                            family='DejaVu Sans'
                        )
            
            # Añadir coordenadas
            for i in range(8):
                # Files (a-h) en la parte inferior
                ax.text(i + 0.5, -0.2, chr(97 + i), 
                       ha='center', va='top', fontsize=12, weight='bold')
                # Ranks (1-8) en el lado izquierdo
                ax.text(-0.2, i + 0.5, str(8 - i),
                       ha='right', va='center', fontsize=12, weight='bold')
            
            # Convertir figura a array
            fig.canvas.draw()
            img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            
            return img_array


def main():
    parser = argparse.ArgumentParser(
        description='Convierte imágenes de tableros de ajedrez a notación FEN'
    )
    parser.add_argument('image', help='Ruta a la imagen del tablero')
    parser.add_argument('--output-dir', '-o', default='resultados_chessbot',
                       help='Directorio para guardar resultados')
    parser.add_argument('--model', '-m', 
                       default='tensorflow_chessbot/saved_models/frozen_graph.pb',
                       help='Ruta al modelo congelado')
    parser.add_argument('--no-viz', action='store_true',
                       help='No generar visualizaciones')
    
    args = parser.parse_args()
    
    # Crear directorio de salida
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Inicializar analizador
        analyzer = ChessBoardAnalyzer(args.model)
        
        # Procesar imagen
        result = analyzer.process_image(args.image)
        
        if not result['success']:
            print(f"❌ {result['error']}")
            return 1
        
        # Nombre base para archivos de salida
        base_name = Path(args.image).stem
        
        # Guardar FEN y metadatos
        json_path = output_dir / f"{base_name}_fen.json"
        with open(json_path, 'w') as f:
            json.dump({
                'image': str(args.image),
                'fen': result['shortened_fen'],
                'certainty': result['certainty']
            }, f, indent=2)
        print(f"💾 Metadatos guardados en: {json_path}")
        
        # Guardar FEN en archivo de texto
        fen_path = output_dir / f"{base_name}.fen"
        with open(fen_path, 'w') as f:
            f.write(result['shortened_fen'])
        print(f"💾 FEN guardado en: {fen_path}")
        
        # Generar visualizaciones si no está deshabilitado
        if not args.no_viz:
            # Visualizar tablero
            board, svg_data = analyzer.visualize_board(
                result['shortened_fen'],
                output_path=str(output_dir / f"{base_name}_board.png"),
                show=False
            )
            
            # Crear imagen de comparación
            if board and result.get('board_image'):
                analyzer.create_comparison_image(
                    result['board_image'],
                    board,
                    str(output_dir / f"{base_name}_comparison.png")
                )
        
        print(f"\n✅ Procesamiento completado exitosamente")
        print(f"📁 Resultados guardados en: {output_dir}")
        
        # Imprimir análisis del tablero
        if result.get('shortened_fen'):
            try:
                board = chess.Board(result['shortened_fen'])
                print(f"\n♟️  Análisis del tablero:")
                print(f"   - Turno: {'Blancas' if board.turn else 'Negras'}")
                print(f"   - Jaque: {'Sí' if board.is_check() else 'No'}")
                print(f"   - Jaque mate: {'Sí' if board.is_checkmate() else 'No'}")
                print(f"   - Movimientos legales: {board.legal_moves.count()}")
            except:
                pass
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
