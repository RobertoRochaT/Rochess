#!/usr/bin/env python3
"""
Ejecuta benchmark REAL usando el modelo de TensorFlow ChessBot
Procesa imágenes reales y genera métricas verdaderas
"""

import os
import sys
import json
import time
from pathlib import Path
from collections import defaultdict
import numpy as np

# Añadir path del tensorflow_chessbot
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tensorflow_chessbot'))

try:
    from tensorflow_chessbot import ChessboardPredictor
    import helper_image_loading
    import chessboard_finder
except ImportError as e:
    print(f"❌ Error importando módulos: {e}")
    print("   Asegúrate de que tensorflow_chessbot esté disponible")
    sys.exit(1)

class RealBenchmark:
    """Ejecuta benchmark real con el modelo"""
    
    def __init__(self, model_path, images_dir):
        """Inicializar con modelo e imágenes"""
        self.model_path = model_path
        self.images_dir = Path(images_dir)
        self.predictor = None
        
        # Métricas
        self.total_boards = 0
        self.successful_predictions = 0
        self.failed_predictions = 0
        self.inference_times = []
        
        # Métricas por pieza
        self.piece_stats = defaultdict(lambda: {
            'total': 0,
            'high_confidence': 0,
            'low_confidence': 0,
            'confidences': []
        })
        
        # Clases de piezas
        self.classes = ['empty', 'wp', 'wn', 'wb', 'wr', 'wq', 'wk', 
                       'bp', 'bn', 'bb', 'br', 'bq', 'bk']
        
    def load_model(self):
        """Cargar modelo de TensorFlow"""
        print("🔄 Cargando modelo TensorFlow ChessBot...")
        try:
            self.predictor = ChessboardPredictor(self.model_path)
            print("✅ Modelo cargado exitosamente")
            return True
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            return False
    
    def process_image(self, image_path):
        """Procesar una imagen y extraer métricas"""
        try:
            start_time = time.time()
            
            # Cargar imagen directamente como PIL Image
            import PIL.Image
            img = PIL.Image.open(str(image_path))
            
            # Resize si es necesario
            img = helper_image_loading.resizeAsNeeded(img)
            if img is None:
                raise Exception("Imagen demasiado grande")
            
            # Buscar tablero
            tiles, corners = chessboard_finder.findGrayscaleTilesInImage(img)
            
            if tiles is None:
                raise Exception("No se encontró tablero en la imagen")
            
            # Hacer predicción
            fen, tile_certainties = self.predictor.getPrediction(tiles)
            # Hacer predicción
            fen, tile_certainties = self.predictor.getPrediction(tiles)
            
            inference_time = (time.time() - start_time) * 1000  # ms
            self.inference_times.append(inference_time)
            
            if fen and tile_certainties is not None:
                self.successful_predictions += 1
                
                # Obtener predicciones individuales por casilla
                # Ejecutar de nuevo para obtener probabilidades
                validation_set = np.swapaxes(np.reshape(tiles, [32*32, 64]),0,1)
                guess_prob, guessed = self.predictor.sess.run(
                    [self.predictor.probabilities, self.predictor.prediction], 
                    feed_dict={self.predictor.x: validation_set, self.predictor.keep_prob: 1.0})
                
                # Analizar predicciones (64 casillas)
                for prob_dist, pred_idx in zip(guess_prob, guessed):
                    confidence = prob_dist[pred_idx]
                    piece = self.classes[pred_idx]
                    
                    # Actualizar estadísticas
                    self.piece_stats[piece]['total'] += 1
                    self.piece_stats[piece]['confidences'].append(float(confidence))
                    
                    if confidence >= 0.9:
                        self.piece_stats[piece]['high_confidence'] += 1
                    elif confidence < 0.7:
                        self.piece_stats[piece]['low_confidence'] += 1
                
                return True, fen, inference_time
            else:
                self.failed_predictions += 1
                return False, None, inference_time
                
        except Exception as e:
            self.failed_predictions += 1
            print(f"   ⚠️  Error procesando {image_path.name}: {e}")
            return False, None, 0
    
    def run_benchmark(self, max_images=None):
        """Ejecutar benchmark en todas las imágenes"""
        print(f"\n🎯 Iniciando benchmark real...")
        print(f"📁 Directorio: {self.images_dir}")
        
        # Obtener imágenes
        image_files = list(self.images_dir.glob("*.png")) + \
                     list(self.images_dir.glob("*.jpg"))
        
        if max_images:
            image_files = image_files[:max_images]
        
        self.total_boards = len(image_files)
        print(f"📊 Total de imágenes: {self.total_boards}\n")
        
        # Procesar cada imagen
        results = []
        for i, img_path in enumerate(image_files, 1):
            print(f"[{i}/{self.total_boards}] Procesando: {img_path.name}...", end=' ')
            
            success, fen, inference_time = self.process_image(img_path)
            
            if success:
                print(f"✅ {inference_time:.1f}ms")
                results.append({
                    'image': img_path.name,
                    'success': True,
                    'fen': fen,
                    'inference_time_ms': inference_time
                })
            else:
                print(f"❌ Falló")
                results.append({
                    'image': img_path.name,
                    'success': False,
                    'fen': None,
                    'inference_time_ms': inference_time
                })
        
        return results
    
    def calculate_metrics(self):
        """Calcular métricas finales"""
        metrics = {
            'model_info': {
                'name': 'TensorFlow ChessBot',
                'model_path': str(self.model_path),
                'architecture': 'Custom CNN',
                'framework': 'TensorFlow'
            },
            'overall_metrics': {
                'total_boards': self.total_boards,
                'successful_predictions': self.successful_predictions,
                'failed_predictions': self.failed_predictions,
                'success_rate': self.successful_predictions / self.total_boards if self.total_boards > 0 else 0,
                'avg_inference_time_ms': np.mean(self.inference_times) if self.inference_times else 0,
                'min_inference_time_ms': np.min(self.inference_times) if self.inference_times else 0,
                'max_inference_time_ms': np.max(self.inference_times) if self.inference_times else 0,
                'std_inference_time_ms': np.std(self.inference_times) if self.inference_times else 0
            },
            'per_piece_stats': {}
        }
        
        # Métricas por pieza
        white_pieces = ['wp', 'wn', 'wb', 'wr', 'wq', 'wk']
        black_pieces = ['bp', 'bn', 'bb', 'br', 'bq', 'bk']
        
        white_confidences = []
        black_confidences = []
        empty_confidences = []
        
        for piece, stats in self.piece_stats.items():
            if stats['total'] > 0:
                avg_conf = np.mean(stats['confidences'])
                std_conf = np.std(stats['confidences'])
                min_conf = np.min(stats['confidences'])
                max_conf = np.max(stats['confidences'])
                
                metrics['per_piece_stats'][piece] = {
                    'total_detections': stats['total'],
                    'avg_confidence': float(avg_conf),
                    'std_confidence': float(std_conf),
                    'min_confidence': float(min_conf),
                    'max_confidence': float(max_conf),
                    'high_confidence_rate': stats['high_confidence'] / stats['total'],
                    'low_confidence_rate': stats['low_confidence'] / stats['total']
                }
                
                # Agrupar por color
                if piece in white_pieces:
                    white_confidences.extend(stats['confidences'])
                elif piece in black_pieces:
                    black_confidences.extend(stats['confidences'])
                elif piece == 'empty':
                    empty_confidences.extend(stats['confidences'])
        
        # Métricas por color
        metrics['metrics_by_color'] = {
            'white_pieces': {
                'avg_confidence': float(np.mean(white_confidences)) if white_confidences else 0,
                'std_confidence': float(np.std(white_confidences)) if white_confidences else 0,
                'total_detections': len(white_confidences)
            },
            'black_pieces': {
                'avg_confidence': float(np.mean(black_confidences)) if black_confidences else 0,
                'std_confidence': float(np.std(black_confidences)) if black_confidences else 0,
                'total_detections': len(black_confidences)
            },
            'empty_squares': {
                'avg_confidence': float(np.mean(empty_confidences)) if empty_confidences else 0,
                'std_confidence': float(np.std(empty_confidences)) if empty_confidences else 0,
                'total_detections': len(empty_confidences)
            }
        }
        
        return metrics
    
    def print_summary(self, metrics):
        """Imprimir resumen de resultados"""
        print("\n" + "="*70)
        print("  RESUMEN DE BENCHMARK REAL")
        print("="*70)
        
        overall = metrics['overall_metrics']
        print(f"\n📊 MÉTRICAS GENERALES:")
        print(f"   Tableros procesados:       {overall['total_boards']}")
        print(f"   Predicciones exitosas:     {overall['successful_predictions']} ({overall['success_rate']:.1%})")
        print(f"   Predicciones fallidas:     {overall['failed_predictions']}")
        print(f"   Tiempo promedio:           {overall['avg_inference_time_ms']:.1f}ms")
        print(f"   Tiempo mín/máx:            {overall['min_inference_time_ms']:.1f}ms / {overall['max_inference_time_ms']:.1f}ms")
        
        color_metrics = metrics['metrics_by_color']
        print(f"\n🎨 CONFIANZA POR COLOR:")
        print(f"   Piezas Blancas:  {color_metrics['white_pieces']['avg_confidence']:.2%} "
              f"({color_metrics['white_pieces']['total_detections']} detecciones)")
        print(f"   Piezas Negras:   {color_metrics['black_pieces']['avg_confidence']:.2%} "
              f"({color_metrics['black_pieces']['total_detections']} detecciones)")
        print(f"   Casillas Vacías: {color_metrics['empty_squares']['avg_confidence']:.2%} "
              f"({color_metrics['empty_squares']['total_detections']} detecciones)")
        
        # Top 5 piezas por confianza
        piece_stats = metrics['per_piece_stats']
        sorted_pieces = sorted(piece_stats.items(), 
                              key=lambda x: x[1]['avg_confidence'], 
                              reverse=True)
        
        print(f"\n🏆 TOP 5 PIEZAS (Mayor Confianza):")
        for i, (piece, stats) in enumerate(sorted_pieces[:5], 1):
            print(f"   {i}. {piece:6s} - {stats['avg_confidence']:.2%} "
                  f"({stats['total_detections']} detecciones)")
        
        print(f"\n⚠️  BOTTOM 5 PIEZAS (Menor Confianza):")
        for i, (piece, stats) in enumerate(sorted_pieces[-5:], 1):
            print(f"   {i}. {piece:6s} - {stats['avg_confidence']:.2%} "
                  f"({stats['total_detections']} detecciones)")
        
        print("\n" + "="*70)


def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Ejecutar benchmark REAL del modelo TensorFlow ChessBot'
    )
    parser.add_argument('--model', '-m',
                       default='tensorflow_chessbot/saved_models/frozen_graph.pb',
                       help='Ruta al modelo congelado (.pb)')
    parser.add_argument('--images', '-i',
                       default='../LiveChess2FEN/images',
                       help='Directorio con imágenes de prueba')
    parser.add_argument('--output', '-o',
                       default='real_benchmark_results.json',
                       help='Archivo de salida para resultados')
    parser.add_argument('--max-images', '-n',
                       type=int,
                       default=None,
                       help='Número máximo de imágenes a procesar')
    
    args = parser.parse_args()
    
    print("="*70)
    print("  BENCHMARK REAL - TensorFlow ChessBot")
    print("="*70)
    
    # Verificar archivos
    model_path = Path(args.model)
    images_dir = Path(args.images)
    
    if not model_path.exists():
        print(f"❌ Modelo no encontrado: {model_path}")
        sys.exit(1)
    
    if not images_dir.exists():
        print(f"❌ Directorio de imágenes no encontrado: {images_dir}")
        sys.exit(1)
    
    # Crear benchmark
    benchmark = RealBenchmark(model_path, images_dir)
    
    # Cargar modelo
    if not benchmark.load_model():
        sys.exit(1)
    
    # Ejecutar benchmark
    results = benchmark.run_benchmark(max_images=args.max_images)
    
    # Calcular métricas
    metrics = benchmark.calculate_metrics()
    
    # Agregar resultados detallados
    metrics['detailed_results'] = results
    
    # Guardar resultados
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n💾 Resultados guardados en: {output_path}")
    
    # Imprimir resumen
    benchmark.print_summary(metrics)
    
    print(f"\n✅ Benchmark completado exitosamente!")
    print(f"   Archivo JSON: {output_path}")
    print(f"   Para visualizar: python visualize_real_benchmark.py")


if __name__ == '__main__':
    main()
