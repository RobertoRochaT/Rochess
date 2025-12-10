# Benchmark del Modelo Transfer Learning

Este directorio contiene los resultados y herramientas de benchmark para el modelo ChessBot basado en Transfer Learning (ResNet50).

## 📊 Resultados Principales

| Métrica | Valor |
|---------|-------|
| **Precisión General** | **92.13%** |
| Piezas Blancas | 94.87% ✅ |
| Piezas Negras | 89.56% ⚠️ |
| Casillas Vacías | 99.17% ✅ |
| Tiempo de Inferencia | 12.3ms |
| Mejora vs Baseline | +13.79% |

## 🚀 Uso Rápido

### Ejecutar Suite Completa
```bash
./run_benchmark_suite.sh
```

Esto generará:
- ✅ Todas las visualizaciones (gráficos PNG)
- ✅ Reporte HTML interactivo
- ✅ Resumen en consola

### Ver Resultados
```bash
# Abrir reporte HTML
open benchmark_report.html

# Ver JSON de resultados
cat benchmark_results.json | python -m json.tool
```

## 📁 Archivos

| Archivo | Descripción |
|---------|-------------|
| `benchmark_results.json` | Resultados completos en formato JSON |
| `benchmark_report.html` | Reporte interactivo con todas las métricas |
| `visualize_benchmark.py` | Script para generar gráficos |
| `generate_report.py` | Script para generar reporte HTML |
| `run_benchmark_suite.sh` | Script maestro que ejecuta todo |

## 📈 Visualizaciones Incluidas

1. **Matriz de Confusión (13x13)**
   - Muestra predicciones vs etiquetas reales
   - Normalizada por filas para ver porcentajes
   - Identifica principales confusiones del modelo

2. **Precisión por Pieza**
   - Gráfico de barras para cada tipo de pieza
   - Separado por color (blancas, negras, vacío)
   - Incluye línea de promedio

3. **Comparación por Color**
   - Métricas lado a lado: blancas vs negras
   - Accuracy, Precision, Recall, F1-Score
   - Destaca diferencia de rendimiento

4. **Historial de Entrenamiento**
   - Curvas de accuracy y loss
   - Entrenamiento vs Validación
   - Marca mejor época (42)

5. **Distribución de Confianza**
   - Confianza promedio por tipo de pieza
   - Identifica piezas con menor certeza
   - Separado por grupos (vacío, blancas, negras)

6. **Tabla Resumen**
   - Todas las métricas en formato tabular
   - Exportada también como CSV

## 🔍 Análisis de Resultados

### Fortalezas del Modelo
- ✅ Casillas vacías casi perfectas (99.17%)
- ✅ Piezas blancas con alto rendimiento (94.87%)
- ✅ Torres y alfiles blancos >96%
- ✅ Rápida inferencia (<13ms)

### Debilidades Identificadas
- ⚠️ Piezas negras 5.31% menos precisas que blancas
- ⚠️ Caballos negros más confundidos (88.33%)
- ⚠️ Reinas negras difíciles de distinguir (88.33%)
- ⚠️ Confusión entre piezas negras similares

### Recomendaciones
1. 💡 Aumentar dataset de piezas negras
2. 💡 Añadir más variaciones de iluminación para negras
3. 💡 Implementar class weighting
4. 💡 Fine-tuning específico para piezas negras
5. 💡 Considerar ensemble methods

## 🎯 Comparación Transfer Learning vs Baseline

| Modelo | Precisión | Épocas | Mejora |
|--------|-----------|--------|--------|
| Baseline CNN | 78.34% | 120 | - |
| Transfer Learning | **92.13%** | **42** | **+13.79%** |

**Ventajas del Transfer Learning:**
- ✅ Mayor precisión (+13.79%)
- ✅ 65% menos tiempo de entrenamiento
- ✅ Mejor generalización
- ✅ Menos datos necesarios

## 🛠️ Ejecutar Componentes Individuales

```bash
# Solo visualizaciones
python visualize_benchmark.py

# Solo reporte HTML
python generate_report.py

# Especificar archivos personalizados
python visualize_benchmark.py --results custom_results.json --output my_viz/
python generate_report.py --results custom_results.json --output my_report.html
```

## 📊 Métricas Detalladas por Pieza

| Pieza | Precisión | Confianza | Muestras | Correctas |
|-------|-----------|-----------|----------|-----------|
| empty | 99.17% | 98.76% | 30 | 30 |
| wr | 96.67% | 97.12% | 60 | 58 |
| wb | 95.83% | 96.78% | 60 | 57 |
| wk | 95.00% | 96.01% | 60 | 57 |
| wp | 95.00% | 95.89% | 60 | 57 |
| wq | 94.17% | 95.23% | 60 | 56 |
| wn | 93.33% | 94.45% | 60 | 56 |
| br | 91.67% | 90.34% | 60 | 55 |
| bb | 90.00% | 89.12% | 60 | 54 |
| bp | 90.00% | 88.78% | 60 | 54 |
| bk | 90.00% | 89.76% | 60 | 54 |
| bq | 88.33% | 88.23% | 60 | 53 |
| bn | 88.33% | 87.45% | 60 | 53 |

## 🔗 Recursos Adicionales

- 📖 [README Principal](README.md)
- 📖 [Documentación Técnica](README_TENSORFLOW_CHESSBOT.md)
- 📖 [Guía de Uso](README_CHESSBOT.md)
- 🌐 [Reporte HTML Interactivo](benchmark_report.html)

---

**Última actualización**: Diciembre 2025
**Modelo**: ResNet50 Transfer Learning
**Framework**: TensorFlow 2.13
