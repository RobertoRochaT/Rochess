#!/usr/bin/env python3
"""
Procesamiento por lotes de imágenes de tableros de ajedrez a FEN
"""

import os
import sys
import argparse
from pathlib import Path
import json
from datetime import datetime
import glob

# Importar el analizador
from chess_board_to_fen import ChessBoardAnalyzer


def process_batch(image_paths, output_dir, model_path, generate_viz=True):
    """
    Procesa un lote de imágenes
    
    Args:
        image_paths: Lista de rutas a imágenes
        output_dir: Directorio de salida
        model_path: Ruta al modelo
        generate_viz: Generar visualizaciones
        
    Returns:
        dict con estadísticas del procesamiento
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Inicializar analizador
    print("🚀 Inicializando ChessBot...")
    analyzer = ChessBoardAnalyzer(model_path)
    
    results = []
    success_count = 0
    fail_count = 0
    
    print(f"\n📦 Procesando {len(image_paths)} imágenes...\n")
    
    for idx, img_path in enumerate(image_paths, 1):
        print(f"[{idx}/{len(image_paths)}] Procesando: {Path(img_path).name}")
        
        try:
            # Procesar imagen
            result = analyzer.process_image(img_path)
            
            if result['success']:
                success_count += 1
                base_name = Path(img_path).stem
                
                # Guardar FEN
                fen_path = output_dir / f"{base_name}.fen"
                with open(fen_path, 'w') as f:
                    f.write(result['shortened_fen'])
                
                # Guardar JSON con metadatos
                json_path = output_dir / f"{base_name}_fen.json"
                with open(json_path, 'w') as f:
                    json.dump({
                        'image': str(img_path),
                        'fen': result['shortened_fen'],
                        'certainty': result['certainty'],
                        'processed_at': datetime.now().isoformat()
                    }, f, indent=2)
                
                # Generar visualizaciones
                if generate_viz:
                    board, _ = analyzer.visualize_board(
                        result['shortened_fen'],
                        output_path=str(output_dir / f"{base_name}_board.png"),
                        show=False
                    )
                    
                    # Crear imagen de comparación de 3 paneles
                    if board and result.get('board_image'):
                        analyzer.create_comparison_image(
                            result['board_image'],
                            board,
                            str(output_dir / f"{base_name}_comparison.png")
                        )
                
                results.append({
                    'image': str(img_path),
                    'status': 'success',
                    'fen': result['shortened_fen'],
                    'certainty': result['certainty']['average']
                })
                
                print(f"   ✅ FEN: {result['shortened_fen'][:40]}...")
                
            else:
                fail_count += 1
                results.append({
                    'image': str(img_path),
                    'status': 'failed',
                    'error': result.get('error', 'Unknown error')
                })
                print(f"   ❌ Error: {result.get('error')}")
                
        except Exception as e:
            fail_count += 1
            results.append({
                'image': str(img_path),
                'status': 'error',
                'error': str(e)
            })
            print(f"   ❌ Excepción: {e}")
        
        print()
    
    # Guardar resumen
    summary = {
        'total': len(image_paths),
        'success': success_count,
        'failed': fail_count,
        'success_rate': (success_count / len(image_paths) * 100) if image_paths else 0,
        'processed_at': datetime.now().isoformat(),
        'results': results
    }
    
    summary_path = output_dir / 'batch_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Crear reporte HTML
    create_html_report(summary, output_dir)
    
    return summary


def create_html_report(summary, output_dir):
    """Crea un reporte HTML con los resultados"""
    html_content = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reporte ChessBot - {summary['processed_at']}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }}
        .container {{
            background: white;
            border-radius: 10px;
            padding: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .stat-value {{
            font-size: 2.5em;
            font-weight: bold;
            margin: 10px 0;
        }}
        .stat-label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        .results {{
            margin-top: 30px;
        }}
        .result-item {{
            background: #f8f9fa;
            border-left: 4px solid #667eea;
            padding: 15px;
            margin: 10px 0;
            border-radius: 4px;
        }}
        .result-item.success {{
            border-left-color: #28a745;
        }}
        .result-item.failed {{
            border-left-color: #dc3545;
        }}
        .fen-code {{
            font-family: 'Courier New', monospace;
            background: #2c3e50;
            color: #ecf0f1;
            padding: 10px;
            border-radius: 4px;
            margin: 10px 0;
            overflow-x: auto;
        }}
        .certainty {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: bold;
        }}
        .certainty.high {{
            background: #d4edda;
            color: #155724;
        }}
        .certainty.medium {{
            background: #fff3cd;
            color: #856404;
        }}
        .certainty.low {{
            background: #f8d7da;
            color: #721c24;
        }}
        .image-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .image-card {{
            border: 1px solid #ddd;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .image-card img {{
            width: 100%;
            height: 300px;
            object-fit: cover;
        }}
        .image-info {{
            padding: 15px;
            background: white;
        }}
        footer {{
            margin-top: 40px;
            text-align: center;
            color: #7f8c8d;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>♟️ Reporte de Análisis de Tableros de Ajedrez</h1>
        <p><strong>Fecha:</strong> {summary['processed_at']}</p>
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-label">Total Imágenes</div>
                <div class="stat-value">{summary['total']}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Éxitos</div>
                <div class="stat-value">{summary['success']}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Fallos</div>
                <div class="stat-value">{summary['failed']}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Tasa de Éxito</div>
                <div class="stat-value">{summary['success_rate']:.1f}%</div>
            </div>
        </div>
        
        <div class="results">
            <h2>Resultados Detallados</h2>
"""
    
    # Añadir resultados individuales
    for result in summary['results']:
        status_class = 'success' if result['status'] == 'success' else 'failed'
        
        if result['status'] == 'success':
            certainty = result['certainty'] * 100
            certainty_class = 'high' if certainty > 95 else 'medium' if certainty > 85 else 'low'
            
            html_content += f"""
            <div class="result-item {status_class}">
                <strong>📁 {Path(result['image']).name}</strong>
                <span class="certainty {certainty_class}">Certeza: {certainty:.1f}%</span>
                <div class="fen-code">{result['fen']}</div>
            </div>
"""
        else:
            html_content += f"""
            <div class="result-item {status_class}">
                <strong>📁 {Path(result['image']).name}</strong>
                <p style="color: #dc3545; margin: 10px 0;">❌ Error: {result.get('error', 'Unknown')}</p>
            </div>
"""
    
    html_content += """
        </div>
        
        <footer>
            <p>Generado por ChessBot - TensorFlow Chess Board Recognition</p>
            <p>Basado en <a href="https://github.com/Elucidation/tensorflow_chessbot" target="_blank">tensorflow_chessbot</a></p>
        </footer>
    </div>
</body>
</html>
"""
    
    html_path = output_dir / 'reporte.html'
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"📊 Reporte HTML generado: {html_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Procesamiento por lotes de tableros de ajedrez a FEN'
    )
    parser.add_argument('images', nargs='+', help='Imágenes o patrón (ej: images/*.png)')
    parser.add_argument('--output-dir', '-o', default='resultados_batch',
                       help='Directorio para guardar resultados')
    parser.add_argument('--model', '-m',
                       default='tensorflow_chessbot/saved_models/frozen_graph.pb',
                       help='Ruta al modelo congelado')
    parser.add_argument('--no-viz', action='store_true',
                       help='No generar visualizaciones')
    
    args = parser.parse_args()
    
    # Expandir patrones glob
    image_paths = []
    for pattern in args.images:
        matches = glob.glob(pattern)
        if matches:
            image_paths.extend(matches)
        elif os.path.exists(pattern):
            image_paths.append(pattern)
    
    # Filtrar solo imágenes
    image_paths = [p for p in image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    
    if not image_paths:
        print("❌ No se encontraron imágenes válidas")
        return 1
    
    print(f"📸 Encontradas {len(image_paths)} imágenes para procesar")
    
    try:
        # Procesar lote
        summary = process_batch(
            image_paths,
            args.output_dir,
            args.model,
            generate_viz=not args.no_viz
        )
        
        # Mostrar resumen
        print("\n" + "="*60)
        print("📊 RESUMEN DEL PROCESAMIENTO")
        print("="*60)
        print(f"Total:        {summary['total']}")
        print(f"Éxitos:       {summary['success']} ✅")
        print(f"Fallos:       {summary['failed']} ❌")
        print(f"Tasa éxito:   {summary['success_rate']:.1f}%")
        print("="*60)
        print(f"\n📁 Resultados guardados en: {args.output_dir}")
        print(f"📊 Abre el reporte HTML: {args.output_dir}/reporte.html")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error en procesamiento por lotes: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
