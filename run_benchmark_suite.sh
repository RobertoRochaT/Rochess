#!/bin/bash
# Script maestro para ejecutar el benchmark completo

echo "════════════════════════════════════════════════════════════════════"
echo "  ChessBot Benchmark Suite - Transfer Learning Analysis"
echo "════════════════════════════════════════════════════════════════════"
echo ""

# Verificar Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 no está instalado."
    exit 1
fi

echo "✅ Python detectado: $(python3 --version)"
echo ""

# Verificar dependencias
echo "📦 Verificando dependencias..."
python3 -c "import matplotlib, seaborn, numpy, pandas, json" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ Todas las dependencias están instaladas"
else
    echo "⚠️  Instalando dependencias faltantes..."
    pip3 install matplotlib seaborn numpy pandas
fi

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  Paso 1/3: Generando visualizaciones"
echo "════════════════════════════════════════════════════════════════════"
echo ""

python3 visualize_benchmark.py

if [ $? -ne 0 ]; then
    echo "❌ Error al generar visualizaciones"
    exit 1
fi

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  Paso 2/3: Generando reporte HTML"
echo "════════════════════════════════════════════════════════════════════"
echo ""

python3 generate_report.py

if [ $? -ne 0 ]; then
    echo "❌ Error al generar reporte HTML"
    exit 1
fi

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  Paso 3/3: Resumen de Resultados"
echo "════════════════════════════════════════════════════════════════════"
echo ""

# Extraer métricas clave del JSON
python3 << 'EOF'
import json

with open('benchmark_results.json', 'r') as f:
    results = json.load(f)

print("📊 RESUMEN DE MÉTRICAS")
print("─" * 70)
print(f"Precisión General:     {results['overall_metrics']['accuracy']:.2%}")
print(f"Precision:             {results['overall_metrics']['precision']:.2%}")
print(f"Recall:                {results['overall_metrics']['recall']:.2%}")
print(f"F1-Score:              {results['overall_metrics']['f1_score']:.2%}")
print(f"Pérdida:               {results['overall_metrics']['test_loss']:.4f}")
print(f"Tiempo de Inferencia:  {results['overall_metrics']['inference_time_ms']:.1f}ms")
print()
print("🎨 MÉTRICAS POR COLOR")
print("─" * 70)
print(f"Piezas Blancas:        {results['metrics_by_color']['white_pieces']['accuracy']:.2%} "
      f"(confianza: {results['metrics_by_color']['white_pieces']['avg_confidence']:.2%})")
print(f"Piezas Negras:         {results['metrics_by_color']['black_pieces']['accuracy']:.2%} "
      f"(confianza: {results['metrics_by_color']['black_pieces']['avg_confidence']:.2%})")
print(f"Casillas Vacías:       {results['metrics_by_color']['empty_squares']['accuracy']:.2%} "
      f"(confianza: {results['metrics_by_color']['empty_squares']['avg_confidence']:.2%})")
print()
print("📈 COMPARACIÓN CON BASELINE")
print("─" * 70)
comp = results['comparison_with_baseline']
print(f"Modelo Baseline:       {comp['baseline_accuracy']:.2%}")
print(f"Transfer Learning:     {results['overall_metrics']['accuracy']:.2%}")
print(f"Mejora:                {comp['improvement']}")
print(f"Reducción tiempo:      {comp['training_time_reduction']}")
print()
print("🏆 MEJORES Y PEORES PIEZAS")
print("─" * 70)

# Encontrar mejores y peores
pieces_metrics = results['per_piece_metrics']
sorted_pieces = sorted(pieces_metrics.items(), 
                      key=lambda x: x[1]['accuracy'], 
                      reverse=True)

print("Top 3 Mejores:")
for i, (piece, metrics) in enumerate(sorted_pieces[:3], 1):
    print(f"  {i}. {piece:6s} - {metrics['accuracy']:.2%} (confianza: {metrics['avg_confidence']:.2%})")

print()
print("Top 3 Peores:")
for i, (piece, metrics) in enumerate(sorted_pieces[-3:], 1):
    print(f"  {i}. {piece:6s} - {metrics['accuracy']:.2%} (confianza: {metrics['avg_confidence']:.2%})")
EOF

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  ✅ BENCHMARK COMPLETADO"
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "📁 Archivos generados:"
echo "   • benchmark_results.json          - Resultados en formato JSON"
echo "   • benchmark_report.html           - Reporte interactivo HTML"
echo "   • benchmark_visualizations/       - Carpeta con todos los gráficos"
echo "     ├── confusion_matrix.png"
echo "     ├── accuracy_per_piece.png"
echo "     ├── color_comparison.png"
echo "     ├── training_history.png"
echo "     ├── confidence_distribution.png"
echo "     ├── metrics_table.png"
echo "     └── metrics_summary.csv"
echo ""
echo "🌐 Para ver el reporte completo:"
echo "   open benchmark_report.html"
echo ""
