#!/usr/bin/env python3
"""
Analizador Unificado de Tableros - Virtuales y Reales
Usa tensorflow_chessbot que funciona para ambos tipos
"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Agregar tensorflow_chessbot al path
sys.path.insert(0, str(Path(__file__).parent / 'tensorflow_chessbot'))

from tensorflow_chessbot import ChessboardPredictor
from helper_image_loading import loadImageGrayscale
import chessboard_finder
import chess
import chess.svg
import PIL.Image


class UnifiedBoardAnalyzer:
    """Analizador para tableros virtuales y reales"""
    
    def __init__(self):
        """Inicializar predictor de tensorflow_chessbot"""
        logger.info("🔄 Cargando modelo tensorflow_chessbot...")
        try:
            frozen_graph_path = str(Path(__file__).parent / 'tensorflow_chessbot' / 'saved_models' / 'frozen_graph.pb')
            self.predictor = ChessboardPredictor(frozen_graph_path)
            logger.info("✅ Modelo cargado correctamente\n")
        except Exception as e:
            logger.error(f"❌ Error cargando modelo: {e}")
            raise
    
    def process_image(self, image_path, output_dir='real_board_results', auto_rotate=True):
        """
        Procesar una imagen de tablero (virtual o real)
        
        Args:
            image_path: Ruta a la imagen
            output_dir: Directorio de salida
            auto_rotate: Intentar rotaciones automáticas si falla
        
        Returns:
            dict con resultados o None si falla
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            logger.error(f"❌ Imagen no encontrada: {image_path}")
            return None
        
        logger.info(f"📸 Procesando: {image_path.name}")
        
        try:
            # Leer imagen como PIL Image (requerido por chessboard_finder)
            img = PIL.Image.open(str(image_path))
            
            if img is None:
                logger.error(f"   ❌ No se pudo leer la imagen")
                return None
            
            # Intentar encontrar tablero
            tiles, corners = chessboard_finder.findGrayscaleTilesInImage(img)
            
            if tiles is None or len(tiles) != 64:
                logger.warning(f"   ⚠️  No se detectó tablero completo en orientación original")
                
                if auto_rotate:
                    # Intentar con diferentes rotaciones y ajustes
                    for angle in [90, 180, 270]:
                        logger.info(f"   🔄 Intentando rotación {angle}°...")
                        rotated = self._rotate_image(img, angle)
                        tiles, corners = chessboard_finder.findGrayscaleTilesInImage(rotated)
                        if tiles is not None and len(tiles) == 64:
                            img = rotated
                            logger.info(f"   ✅ Tablero detectado con rotación {angle}°")
                            break
                    
                    if tiles is None or len(tiles) != 64:
                        logger.error(f"   ❌ No se pudo detectar el tablero en ninguna orientación")
                        return {
                            'filename': image_path.name,
                            'error': 'No se detectó tablero',
                            'status': 'failed'
                        }
            
            # Predecir piezas
            predictions = self.predictor.getPrediction(tiles)
            
            # Convertir a FEN
            fen = self._predictions_to_fen(predictions)
            
            # Calcular certeza
            certainties = [pred.max() for pred in predictions]
            certainty = np.mean(certainties) * 100
            
            logger.info(f"   ✅ FEN detectado: {fen}")
            logger.info(f"   📊 Certeza: {certainty:.2f}%\n")
            
            # Crear directorio de salida
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            base_name = image_path.stem
            
            # Guardar resultados
            self._save_results(img, tiles, corners, fen, certainties, 
                             output_path, base_name)
            
            return {
                'filename': image_path.name,
                'fen': fen,
                'certainty': certainty,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return {
                'filename': image_path.name,
                'error': str(e),
                'status': 'failed'
            }
    
    def _rotate_image(self, img, angle):
        """Rotar imagen PIL"""
        if angle == 90:
            return img.transpose(PIL.Image.ROTATE_270)
        elif angle == 180:
            return img.transpose(PIL.Image.ROTATE_180)
        elif angle == 270:
            return img.transpose(PIL.Image.ROTATE_90)
        return img
    
    def _predictions_to_fen(self, predictions):
        """Convertir predicciones a FEN"""
        piece_chars = '_KQRBNP__kqrbnp'
        
        fen_rows = []
        for row in range(8):
            fen_row = ''
            empty_count = 0
            
            for col in range(8):
                idx = row * 8 + col
                pred = predictions[idx]
                piece_idx = np.argmax(pred)
                piece_char = piece_chars[piece_idx]
                
                if piece_char == '_':
                    empty_count += 1
                else:
                    if empty_count > 0:
                        fen_row += str(empty_count)
                        empty_count = 0
                    fen_row += piece_char
            
            if empty_count > 0:
                fen_row += str(empty_count)
            
            fen_rows.append(fen_row)
        
        return '/'.join(fen_rows) + ' w - - 0 1'
    
    def _save_results(self, img, tiles, corners, fen, certainties, output_path, base_name):
        """Guardar resultados visuales"""
        
        # Convertir PIL Image a numpy/OpenCV si es necesario
        if isinstance(img, PIL.Image.Image):
            img_cv = np.array(img.convert('RGB'))
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
        else:
            img_cv = img
        
        # 1. Imagen con tablero detectado
        detected_img = img_cv.copy()
        if corners is not None:
            cv2.polylines(detected_img, [corners.astype(np.int32)], True, (0, 255, 0), 3)
        
        cv2.imwrite(str(output_path / f"{base_name}_detected.png"), detected_img)
        
        # 2. Tablero virtual con piezas
        try:
            board = chess.Board(fen)
            svg_data = chess.svg.board(board, size=400)
            
            try:
                import cairosvg
                png_data = cairosvg.svg2png(bytestring=svg_data.encode('utf-8'))
                with open(output_path / f"{base_name}_virtual.png", 'wb') as f:
                    f.write(png_data)
            except ImportError:
                logger.warning("   ⚠️  cairosvg no disponible")
        except Exception as e:
            logger.warning(f"   ⚠️  Error generando virtual board: {e}")
        
        # 3. Comparación de 3 paneles
        self._create_comparison(img_cv, detected_img, fen, certainties,
                               output_path / f"{base_name}_comparison.png")
        
        # 4. Guardar FEN en archivo de texto
        with open(output_path / f"{base_name}_fen.txt", 'w') as f:
            f.write(f"FEN: {fen}\n")
            f.write(f"Certeza: {np.mean(certainties)*100:.2f}%\n")
    
    def _create_comparison(self, original, detected, fen, certainties, output_path):
        """Crear imagen de comparación de 3 paneles"""
        try:
            height = 400
            
            # Redimensionar original
            orig_h, orig_w = original.shape[:2]
            ratio = height / orig_h
            orig_resized = cv2.resize(original, (int(orig_w * ratio), height))
            
            # Redimensionar detected
            det_h, det_w = detected.shape[:2]
            ratio = height / det_h
            det_resized = cv2.resize(detected, (int(det_w * ratio), height))
            
            # Tablero virtual
            try:
                board = chess.Board(fen)
                svg_data = chess.svg.board(board, size=height)
                
                import cairosvg
                png_data = cairosvg.svg2png(bytestring=svg_data.encode('utf-8'))
                nparr = np.frombuffer(png_data, np.uint8)
                virtual = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except:
                virtual = np.ones((height, height, 3), dtype=np.uint8) * 255
                cv2.putText(virtual, "Virtual Board", (50, height//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            
            # Agregar títulos
            def add_title(img, title, subtitle=""):
                title_bar = np.ones((60, img.shape[1], 3), dtype=np.uint8) * 50
                cv2.putText(title_bar, title, (10, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                if subtitle:
                    cv2.putText(title_bar, subtitle, (10, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                return np.vstack([title_bar, img])
            
            certainty_text = f"Certeza: {np.mean(certainties)*100:.2f}%"
            orig_with_title = add_title(orig_resized, "Imagen Original")
            det_with_title = add_title(det_resized, "Tablero Detectado", certainty_text)
            virtual_with_title = add_title(virtual, "Tablero Virtual", fen.split()[0])
            
            # Combinar
            comparison = np.hstack([orig_with_title, det_with_title, virtual_with_title])
            cv2.imwrite(str(output_path), comparison)
            
        except Exception as e:
            logger.warning(f"   ⚠️  Error creando comparación: {e}")


def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analizador Unificado de Tableros')
    parser.add_argument('images', nargs='+', help='Imagen(es) a procesar')
    parser.add_argument('--output-dir', '-o', default='real_board_results',
                       help='Directorio de salida')
    parser.add_argument('--no-rotate', action='store_true',
                       help='No intentar rotaciones automáticas')
    
    args = parser.parse_args()
    
    print("="*60)
    print("🔍 ANALIZADOR UNIFICADO DE TABLEROS")
    print("   Tableros Virtuales y Reales")
    print("="*60 + "\n")
    
    try:
        analyzer = UnifiedBoardAnalyzer()
        
        results = []
        for image_path in args.images:
            result = analyzer.process_image(
                image_path,
                output_dir=args.output_dir,
                auto_rotate=not args.no_rotate
            )
            if result:
                results.append(result)
        
        # Resumen
        successful = [r for r in results if r['status'] == 'success']
        failed = [r for r in results if r['status'] == 'failed']
        
        print("\n" + "="*60)
        print("📊 RESUMEN")
        print("="*60)
        print(f"Total: {len(results)}")
        print(f"Exitosas: {len(successful)} ({len(successful)/len(results)*100:.1f}%)")
        print(f"Fallidas: {len(failed)} ({len(failed)/len(results)*100:.1f}%)")
        
        if successful:
            avg_cert = np.mean([r['certainty'] for r in successful])
            print(f"Certeza promedio: {avg_cert:.2f}%")
        
        print(f"\n✅ Resultados en: {args.output_dir}")
        print("="*60 + "\n")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
