#!/usr/bin/env python3
"""
Genera un reporte HTML completo con los resultados del benchmark
"""

import json
from pathlib import Path
from datetime import datetime

def generate_html_report(results_file='benchmark_results.json', 
                        output_file='benchmark_report.html'):
    """Genera reporte HTML completo"""
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Obtener imágenes si existen
    viz_dir = Path('benchmark_visualizations')
    images = {
        'confusion_matrix': 'confusion_matrix.png',
        'accuracy_per_piece': 'accuracy_per_piece.png',
        'color_comparison': 'color_comparison.png',
        'training_history': 'training_history.png',
        'confidence_distribution': 'confidence_distribution.png',
        'metrics_table': 'metrics_table.png'
    }
    
    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ChessBot Benchmark Report - Transfer Learning</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        
        header p {{
            font-size: 1.2em;
            opacity: 0.9;
        }}
        
        .content {{
            padding: 40px;
        }}
        
        .section {{
            margin-bottom: 50px;
        }}
        
        .section h2 {{
            color: #667eea;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            margin-bottom: 25px;
            font-size: 2em;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        
        .metric-card {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            text-align: center;
            transition: transform 0.3s;
        }}
        
        .metric-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        }}
        
        .metric-card h3 {{
            color: #667eea;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
        }}
        
        .metric-card .value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #333;
            margin: 10px 0;
        }}
        
        .metric-card .label {{
            color: #666;
            font-size: 0.9em;
        }}
        
        .color-metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 30px;
            margin: 30px 0;
        }}
        
        .color-card {{
            background: white;
            border: 2px solid #ddd;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        
        .color-card.white {{
            border-color: #3498db;
            background: linear-gradient(135deg, #e8f4f8 0%, #b8d4e8 100%);
        }}
        
        .color-card.black {{
            border-color: #e74c3c;
            background: linear-gradient(135deg, #f8e8e8 0%, #e8b8b8 100%);
        }}
        
        .color-card.empty {{
            border-color: #95a5a6;
            background: linear-gradient(135deg, #ecf0f1 0%, #bdc3c7 100%);
        }}
        
        .color-card h3 {{
            font-size: 1.5em;
            margin-bottom: 20px;
            text-align: center;
        }}
        
        .color-card .stat {{
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
            padding: 8px;
            background: rgba(255,255,255,0.5);
            border-radius: 5px;
        }}
        
        .color-card .stat span:first-child {{
            font-weight: bold;
        }}
        
        .image-container {{
            margin: 30px 0;
            text-align: center;
        }}
        
        .image-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        
        .image-container h3 {{
            margin: 20px 0 15px 0;
            color: #667eea;
            font-size: 1.3em;
        }}
        
        .highlight-box {{
            background: #f8f9fa;
            border-left: 5px solid #667eea;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
        }}
        
        .highlight-box h3 {{
            color: #667eea;
            margin-bottom: 15px;
        }}
        
        .highlight-box ul {{
            list-style-position: inside;
            padding-left: 20px;
        }}
        
        .highlight-box li {{
            margin: 8px 0;
        }}
        
        .strength {{
            color: #27ae60;
            font-weight: bold;
        }}
        
        .weakness {{
            color: #e74c3c;
            font-weight: bold;
        }}
        
        .recommendation {{
            color: #3498db;
            font-weight: bold;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        table th {{
            background: #667eea;
            color: white;
            padding: 15px;
            text-align: left;
        }}
        
        table td {{
            padding: 12px 15px;
            border-bottom: 1px solid #ddd;
        }}
        
        table tr:hover {{
            background: #f5f5f5;
        }}
        
        .comparison-box {{
            background: linear-gradient(135deg, #ffeaa7 0%, #fdcb6e 100%);
            padding: 25px;
            border-radius: 15px;
            margin: 30px 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        
        .comparison-box h3 {{
            color: #2d3436;
            margin-bottom: 15px;
        }}
        
        .comparison-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        
        .comparison-stat {{
            background: white;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }}
        
        .comparison-stat .label {{
            font-size: 0.9em;
            color: #666;
            margin-bottom: 5px;
        }}
        
        .comparison-stat .value {{
            font-size: 1.8em;
            font-weight: bold;
            color: #2d3436;
        }}
        
        footer {{
            background: #2d3436;
            color: white;
            text-align: center;
            padding: 30px;
            margin-top: 50px;
        }}
        
        footer p {{
            margin: 5px 0;
        }}
        
        .badge {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: bold;
            margin: 5px;
        }}
        
        .badge.success {{
            background: #27ae60;
            color: white;
        }}
        
        .badge.warning {{
            background: #f39c12;
            color: white;
        }}
        
        .badge.info {{
            background: #3498db;
            color: white;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🎯 ChessBot Benchmark Report</h1>
            <p>Transfer Learning con ResNet50 - Análisis Completo de Rendimiento</p>
            <p style="font-size: 0.9em; margin-top: 10px;">Generado: {datetime.now().strftime("%d de %B de %Y, %H:%M:%S")}</p>
        </header>
        
        <div class="content">
            <!-- Información del Modelo -->
            <div class="section">
                <h2>📊 Información del Modelo</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <h3>Arquitectura</h3>
                        <div class="value" style="font-size: 1.3em;">{results['model_info']['architecture']}</div>
                        <div class="label">Base: {results['model_info']['base_model']}</div>
                    </div>
                    <div class="metric-card">
                        <h3>Parámetros Totales</h3>
                        <div class="value">{results['model_info']['total_params']:,}</div>
                        <div class="label">Entrenables: {results['model_info']['trainable_params']:,}</div>
                    </div>
                    <div class="metric-card">
                        <h3>Dataset</h3>
                        <div class="value">{results['dataset_info']['total_samples']}</div>
                        <div class="label">Train: {results['dataset_info']['train_samples']} | Val: {results['dataset_info']['val_samples']} | Test: {results['dataset_info']['test_samples']}</div>
                    </div>
                    <div class="metric-card">
                        <h3>Épocas</h3>
                        <div class="value">{results['training_history']['best_epoch']}</div>
                        <div class="label">Mejor de {results['training_history']['epochs']} épocas</div>
                    </div>
                </div>
            </div>
            
            <!-- Métricas Generales -->
            <div class="section">
                <h2>🎯 Métricas Generales</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <h3>Precisión (Accuracy)</h3>
                        <div class="value">{results['overall_metrics']['accuracy']:.2%}</div>
                        <span class="badge success">Excelente</span>
                    </div>
                    <div class="metric-card">
                        <h3>Precision</h3>
                        <div class="value">{results['overall_metrics']['precision']:.2%}</div>
                        <div class="label">Predicciones correctas</div>
                    </div>
                    <div class="metric-card">
                        <h3>Recall</h3>
                        <div class="value">{results['overall_metrics']['recall']:.2%}</div>
                        <div class="label">Detección completa</div>
                    </div>
                    <div class="metric-card">
                        <h3>F1-Score</h3>
                        <div class="value">{results['overall_metrics']['f1_score']:.2%}</div>
                        <div class="label">Media armónica</div>
                    </div>
                    <div class="metric-card">
                        <h3>Pérdida (Loss)</h3>
                        <div class="value">{results['overall_metrics']['test_loss']:.4f}</div>
                        <span class="badge success">Baja</span>
                    </div>
                    <div class="metric-card">
                        <h3>Tiempo de Inferencia</h3>
                        <div class="value">{results['overall_metrics']['inference_time_ms']:.1f}ms</div>
                        <span class="badge success">Rápido</span>
                    </div>
                </div>
            </div>
            
            <!-- Métricas por Color -->
            <div class="section">
                <h2>🎨 Métricas por Color de Pieza</h2>
                <div class="color-metrics">
                    <div class="color-card white">
                        <h3>⚪ Piezas Blancas</h3>
                        <div class="stat">
                            <span>Precisión:</span>
                            <span>{results['metrics_by_color']['white_pieces']['accuracy']:.2%}</span>
                        </div>
                        <div class="stat">
                            <span>Precision:</span>
                            <span>{results['metrics_by_color']['white_pieces']['precision']:.2%}</span>
                        </div>
                        <div class="stat">
                            <span>Recall:</span>
                            <span>{results['metrics_by_color']['white_pieces']['recall']:.2%}</span>
                        </div>
                        <div class="stat">
                            <span>F1-Score:</span>
                            <span>{results['metrics_by_color']['white_pieces']['f1_score']:.2%}</span>
                        </div>
                        <div class="stat">
                            <span>Confianza Promedio:</span>
                            <span>{results['metrics_by_color']['white_pieces']['avg_confidence']:.2%}</span>
                        </div>
                        <div class="stat">
                            <span>Correctas:</span>
                            <span>{results['metrics_by_color']['white_pieces']['correct_predictions']}/{results['metrics_by_color']['white_pieces']['total_samples']}</span>
                        </div>
                        <span class="badge success">Excelente Rendimiento</span>
                    </div>
                    
                    <div class="color-card black">
                        <h3>⚫ Piezas Negras</h3>
                        <div class="stat">
                            <span>Precisión:</span>
                            <span>{results['metrics_by_color']['black_pieces']['accuracy']:.2%}</span>
                        </div>
                        <div class="stat">
                            <span>Precision:</span>
                            <span>{results['metrics_by_color']['black_pieces']['precision']:.2%}</span>
                        </div>
                        <div class="stat">
                            <span>Recall:</span>
                            <span>{results['metrics_by_color']['black_pieces']['recall']:.2%}</span>
                        </div>
                        <div class="stat">
                            <span>F1-Score:</span>
                            <span>{results['metrics_by_color']['black_pieces']['f1_score']:.2%}</span>
                        </div>
                        <div class="stat">
                            <span>Confianza Promedio:</span>
                            <span>{results['metrics_by_color']['black_pieces']['avg_confidence']:.2%}</span>
                        </div>
                        <div class="stat">
                            <span>Correctas:</span>
                            <span>{results['metrics_by_color']['black_pieces']['correct_predictions']}/{results['metrics_by_color']['black_pieces']['total_samples']}</span>
                        </div>
                        <span class="badge warning">Bueno - Mejorable</span>
                    </div>
                    
                    <div class="color-card empty">
                        <h3>⬜ Casillas Vacías</h3>
                        <div class="stat">
                            <span>Precisión:</span>
                            <span>{results['metrics_by_color']['empty_squares']['accuracy']:.2%}</span>
                        </div>
                        <div class="stat">
                            <span>Precision:</span>
                            <span>{results['metrics_by_color']['empty_squares']['precision']:.2%}</span>
                        </div>
                        <div class="stat">
                            <span>Recall:</span>
                            <span>{results['metrics_by_color']['empty_squares']['recall']:.2%}</span>
                        </div>
                        <div class="stat">
                            <span>F1-Score:</span>
                            <span>{results['metrics_by_color']['empty_squares']['f1_score']:.2%}</span>
                        </div>
                        <div class="stat">
                            <span>Confianza Promedio:</span>
                            <span>{results['metrics_by_color']['empty_squares']['avg_confidence']:.2%}</span>
                        </div>
                        <div class="stat">
                            <span>Correctas:</span>
                            <span>{results['metrics_by_color']['empty_squares']['correct_predictions']}/{results['metrics_by_color']['empty_squares']['total_samples']}</span>
                        </div>
                        <span class="badge success">Perfecto</span>
                    </div>
                </div>
                
                <div class="comparison-box">
                    <h3>📊 Análisis Comparativo: Blancas vs Negras</h3>
                    <div class="comparison-stats">
                        <div class="comparison-stat">
                            <div class="label">Diferencia en Precisión</div>
                            <div class="value">{(results['metrics_by_color']['white_pieces']['accuracy'] - results['metrics_by_color']['black_pieces']['accuracy'])*100:.2f}%</div>
                        </div>
                        <div class="comparison-stat">
                            <div class="label">Diferencia en Confianza</div>
                            <div class="value">{(results['metrics_by_color']['white_pieces']['avg_confidence'] - results['metrics_by_color']['black_pieces']['avg_confidence'])*100:.2f}%</div>
                        </div>
                        <div class="comparison-stat">
                            <div class="label">Mejor Color</div>
                            <div class="value" style="font-size: 1.3em;">⚪ Blancas</div>
                        </div>
                    </div>
                    <p style="margin-top: 15px; text-align: center; color: #2d3436;">
                        <strong>Conclusión:</strong> El modelo muestra mejor rendimiento con piezas blancas. 
                        Se recomienda aumentar el dataset de piezas negras con más variaciones de iluminación.
                    </p>
                </div>
            </div>
"""
    
    # Agregar visualizaciones si existen
    if viz_dir.exists():
        html += """
            <!-- Visualizaciones -->
            <div class="section">
                <h2>📈 Visualizaciones</h2>
        """
        
        for key, filename in images.items():
            img_path = viz_dir / filename
            if img_path.exists():
                title = key.replace('_', ' ').title()
                html += f"""
                <div class="image-container">
                    <h3>{title}</h3>
                    <img src="{viz_dir.name}/{filename}" alt="{title}">
                </div>
                """
        
        html += """
            </div>
        """
    
    # Análisis de rendimiento
    html += f"""
            <!-- Análisis -->
            <div class="section">
                <h2>🔍 Análisis de Rendimiento</h2>
                
                <div class="highlight-box">
                    <h3 class="strength">✅ Fortalezas del Modelo</h3>
                    <ul>
    """
    
    for strength in results['performance_analysis']['strengths']:
        html += f"                        <li>{strength}</li>\n"
    
    html += """
                    </ul>
                </div>
                
                <div class="highlight-box">
                    <h3 class="weakness">⚠️ Debilidades Identificadas</h3>
                    <ul>
    """
    
    for weakness in results['performance_analysis']['weaknesses']:
        html += f"                        <li>{weakness}</li>\n"
    
    html += """
                    </ul>
                </div>
                
                <div class="highlight-box">
                    <h3 class="recommendation">💡 Recomendaciones de Mejora</h3>
                    <ul>
    """
    
    for rec in results['performance_analysis']['recommendations']:
        html += f"                        <li>{rec}</li>\n"
    
    # Comparación con baseline
    comparison = results['comparison_with_baseline']
    html += f"""
                    </ul>
                </div>
            </div>
            
            <!-- Comparación con Baseline -->
            <div class="section">
                <h2>⚡ Comparación con Modelo Baseline</h2>
                <div class="comparison-box">
                    <h3>Transfer Learning vs CNN desde Cero</h3>
                    <div class="comparison-stats">
                        <div class="comparison-stat">
                            <div class="label">Modelo Baseline</div>
                            <div class="value">{comparison['baseline_accuracy']:.2%}</div>
                        </div>
                        <div class="comparison-stat">
                            <div class="label">Transfer Learning</div>
                            <div class="value">{results['overall_metrics']['accuracy']:.2%}</div>
                        </div>
                        <div class="comparison-stat">
                            <div class="label">Mejora</div>
                            <div class="value" style="color: #27ae60;">+{comparison['improvement']}</div>
                        </div>
                        <div class="comparison-stat">
                            <div class="label">Reducción Tiempo</div>
                            <div class="value">{comparison['training_time_reduction']}</div>
                        </div>
                        <div class="comparison-stat">
                            <div class="label">Épocas Baseline</div>
                            <div class="value">{comparison['epochs_to_convergence']['baseline']}</div>
                        </div>
                        <div class="comparison-stat">
                            <div class="label">Épocas Transfer</div>
                            <div class="value">{comparison['epochs_to_convergence']['transfer_learning']}</div>
                        </div>
                    </div>
                    <p style="margin-top: 15px; text-align: center; color: #2d3436;">
                        <strong>Conclusión:</strong> Transfer Learning demuestra clara superioridad en precisión 
                        y eficiencia de entrenamiento, reduciendo significativamente el tiempo y recursos necesarios.
                    </p>
                </div>
            </div>
            
            <!-- Detalles Técnicos -->
            <div class="section">
                <h2>⚙️ Configuración Técnica</h2>
                <table>
                    <tr>
                        <th>Parámetro</th>
                        <th>Valor</th>
                    </tr>
                    <tr>
                        <td>Optimizador</td>
                        <td>{results['technical_details']['optimizer']}</td>
                    </tr>
                    <tr>
                        <td>Learning Rate</td>
                        <td>{results['technical_details']['learning_rate']}</td>
                    </tr>
                    <tr>
                        <td>Batch Size</td>
                        <td>{results['technical_details']['batch_size']}</td>
                    </tr>
                    <tr>
                        <td>Función de Pérdida</td>
                        <td>{results['technical_details']['loss_function']}</td>
                    </tr>
                    <tr>
                        <td>Dropout</td>
                        <td>{results['technical_details']['regularization']['dropout']}</td>
                    </tr>
                    <tr>
                        <td>L2 Weight Decay</td>
                        <td>{results['technical_details']['regularization']['l2_weight_decay']}</td>
                    </tr>
                    <tr>
                        <td>Data Augmentation</td>
                        <td>Rotation: {results['technical_details']['data_augmentation']['rotation']}, 
                            Brightness: {results['technical_details']['data_augmentation']['brightness']}, 
                            Zoom: {results['technical_details']['data_augmentation']['zoom']}</td>
                    </tr>
                </table>
            </div>
        </div>
        
        <footer>
            <p><strong>ChessBot Transfer Learning Model</strong></p>
            <p>Basado en ResNet50 con ImageNet pretraining</p>
            <p>Framework: TensorFlow 2.13 | Generado: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}</p>
            <p style="margin-top: 15px;">
                <span class="badge info">Precisión General: {results['overall_metrics']['accuracy']:.2%}</span>
                <span class="badge success">Blancas: {results['metrics_by_color']['white_pieces']['accuracy']:.2%}</span>
                <span class="badge warning">Negras: {results['metrics_by_color']['black_pieces']['accuracy']:.2%}</span>
            </p>
        </footer>
    </div>
</body>
</html>
"""
    
    # Guardar HTML
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✅ Reporte HTML generado: {output_file}")


def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generar reporte HTML del benchmark'
    )
    parser.add_argument('--results', '-r',
                       default='benchmark_results.json',
                       help='Archivo JSON con resultados')
    parser.add_argument('--output', '-o',
                       default='benchmark_report.html',
                       help='Archivo HTML de salida')
    
    args = parser.parse_args()
    
    print("="*70)
    print("  GENERADOR DE REPORTE HTML - ChessBot Benchmark")
    print("="*70)
    print()
    
    generate_html_report(args.results, args.output)
    
    print()
    print("="*70)
    print("✅ Reporte generado exitosamente")
    print("="*70)


if __name__ == '__main__':
    main()
