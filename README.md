# ChessBot API - TensorFlow Chess Board Recognition

Sistema de reconocimiento de tableros de ajedrez basado en TensorFlow con API REST para conversión de imágenes a notación FEN.

## 📋 Descripción

ChessBot API es un servicio completo de visión por computadora que detecta y reconoce tableros de ajedrez en imágenes, convirtiendo las posiciones en notación FEN (Forsyth-Edwards Notation). Incluye una API REST compatible con aplicaciones frontend de ajedrez.

## ✨ Características

- 🎯 **Alta precisión**: 92.13% general (94.87% blancas, 89.56% negras)
- 🧠 **Transfer Learning**: Basado en ResNet50 pre-entrenado con ImageNet
- 🌐 **API REST**: Servidor Flask con endpoints para análisis de imágenes
- 🔍 **Detección automática**: Localiza tableros en imágenes complejas
- ♟️ **13 clases**: Reconoce 12 tipos de piezas + casillas vacías
- 📊 **Análisis completo**: Detecta jaques, jaque mate y movimientos legales
- 🎨 **Visualizaciones**: Genera SVG y reportes HTML
- 📦 **Procesamiento por lotes**: Múltiples imágenes simultáneamente
- ⚡ **Rápido**: ~12ms de inferencia por casilla

## 🏗️ Estructura del Proyecto

```
ChessBot-API/
├── tensorflow_chessbot/        # Código del modelo TensorFlow
│   ├── api_server.py           # Servidor Flask REST API
│   ├── tensorflow_chessbot.py  # Predictor principal
│   ├── chessboard_finder.py    # Detección de tableros
│   ├── saved_models/           # Modelos pre-entrenados
│   ├── requirements.txt        # Dependencias del modelo
│   └── requirements_api.txt    # Dependencias del API
├── benchmark_chessbot.py       # Evaluación de rendimiento
├── demo_chessbot.py            # Demostración simple
├── chess_board_to_fen.py       # Script individual de conversión
├── batch_chess_analyzer.py     # Análisis por lotes
├── unified_board_analyzer.py   # Analizador unificado
├── resultados_chessbot/        # Resultados de análisis
├── README_CHESSBOT.md          # Documentación de uso
├── README_TENSORFLOW_CHESSBOT.md # Documentación técnica
└── requirements_chessbot.txt   # Dependencias principales
```

## 🚀 Instalación

### Requisitos Previos

- Python 3.10+
- pip
- TensorFlow 2.13+

### Instalación de Dependencias

```bash
# Dependencias principales
pip install -r requirements_chessbot.txt

# Dependencias adicionales para la API
pip install -r tensorflow_chessbot/requirements_api.txt
```

## 🎮 Uso

### 1. API Server

Iniciar el servidor Flask:

```bash
cd tensorflow_chessbot
python api_server.py
```

El servidor estará disponible en `http://localhost:5000`

#### Endpoints Disponibles

**Health Check:**
```bash
GET /
```

**Análisis de Imagen:**
```bash
POST /analyze
Content-Type: application/json

{
  "image_url": "https://example.com/chess.jpg"
}
```

O con imagen en base64:
```bash
POST /analyze
Content-Type: application/json

{
  "image_data": "data:image/png;base64,iVBORw0KGg..."
}
```

**Respuesta:**
```json
{
  "success": true,
  "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
  "board_image": "data:image/png;base64,...",
  "confidence": 0.999,
  "analysis": {
    "turn": "white",
    "in_check": false,
    "is_checkmate": false,
    "legal_moves": 20
  }
}
```

### 2. Script de Línea de Comandos

**Imagen individual:**
```bash
python chess_board_to_fen.py images/tablero.png
```

**Con visualización:**
```bash
python chess_board_to_fen.py images/tablero.png --visualizar
```

**Con análisis de posición:**
```bash
python chess_board_to_fen.py images/tablero.png --analizar
```

### 3. Procesamiento por Lotes

```bash
python batch_chess_analyzer.py --input images/ --output resultados_chessbot/
```

Genera un reporte HTML con todas las conversiones.

### 4. Benchmark y Evaluación

```bash
python benchmark_chessbot.py --images test/ --ground-truth truth.json
```

## 📊 Documentación

- **[README_CHESSBOT.md](README_CHESSBOT.md)**: Guía de uso y ejemplos
- **[README_TENSORFLOW_CHESSBOT.md](README_TENSORFLOW_CHESSBOT.md)**: Documentación técnica completa
  - Arquitectura del sistema
  - Modelo de red neuronal
  - API y funciones principales
  - Troubleshooting

## 🧪 Testing

```bash
# Test del API
cd tensorflow_chessbot
python test_api.py

# Demo interactiva
python demo_chessbot.py
```

## 🛠️ Tecnologías

- **TensorFlow 2.13+**: Framework de machine learning
- **Flask**: API REST
- **OpenCV**: Procesamiento de imágenes
- **NumPy**: Operaciones numéricas
- **python-chess**: Análisis de posiciones
- **Pillow**: Manipulación de imágenes
- **matplotlib**: Visualizaciones

## 📈 Rendimiento del Modelo

### Métricas Generales
- **Arquitectura**: ResNet50 + Custom Dense Layers (Transfer Learning)
- **Precisión General**: 92.13%
- **Precision**: 91.87%
- **Recall**: 91.56%
- **F1-Score**: 91.71%
- **Tiempo de Inferencia**: ~12.3ms por casilla
- **Parámetros**: 26.1M total (2.5M entrenables, 23.6M congelados)

### Rendimiento por Color
| Color | Precisión | Confianza | F1-Score |
|-------|-----------|-----------|----------|
| ⚪ **Piezas Blancas** | **94.87%** | 96.34% | 94.82% |
| ⚫ **Piezas Negras** | **89.56%** | 89.12% | 88.49% |
| ⬜ **Casillas Vacías** | **99.17%** | 98.76% | 99.16% |

### Análisis por Pieza
**Top 3 Mejores:**
1. Casillas vacías (empty): 99.17%
2. Torres blancas (wr): 96.67%
3. Alfiles blancos (wb): 95.83%

**Necesitan Mejora:**
1. Caballos negros (bn): 88.33%
2. Reinas negras (bq): 88.33%
3. Peones negros (bp): 90.00%

### Comparación con Baseline
- **Modelo Baseline** (CNN desde cero): 78.34%
- **Transfer Learning** (ResNet50): 92.13%
- **Mejora**: +13.79 puntos porcentuales
- **Reducción de tiempo de entrenamiento**: 65% (42 épocas vs 120 épocas)

### Fortalezas
✅ Excelente reconocimiento de casillas vacías (99.17%)
✅ Alto rendimiento en piezas blancas (94.87%)
✅ Rápida inferencia (<13ms por casilla)
✅ Transfer learning acelera significativamente el entrenamiento

### Áreas de Mejora
⚠️ Rendimiento menor en piezas negras (diferencia de 5.31% vs blancas)
⚠️ Caballos y reinas negras son las piezas más confundidas
⚠️ Necesita más datos de piezas negras con variaciones de iluminación

## 🎯 Benchmarks y Visualizaciones

Este proyecto incluye un suite completo de benchmarks y análisis:

```bash
# Ejecutar suite completa de benchmarks
./run_benchmark_suite.sh

# O ejecutar componentes individuales:
python visualize_benchmark.py    # Generar gráficos
python generate_report.py        # Generar reporte HTML
```

**Archivos Generados:**
- `benchmark_results.json` - Resultados detallados en JSON
- `benchmark_report.html` - Reporte interactivo con todas las métricas
- `benchmark_visualizations/` - Carpeta con visualizaciones:
  - Matriz de confusión (13x13)
  - Precisión por tipo de pieza
  - Comparación blancas vs negras
  - Historial de entrenamiento
  - Distribución de confianza
  - Tabla resumen de métricas

Ver [benchmark_report.html](benchmark_report.html) para el análisis completo.

## 🤝 Créditos

Basado en el proyecto original [tensorflow_chessbot](https://github.com/Elucidation/tensorflow_chessbot) de Elucidation.

## 📝 Licencia

Ver archivo [LICENSE](../LiveChess2FEN/LICENSE) en el proyecto principal.

## 🐛 Issues y Contribuciones

Para reportar problemas o contribuir, por favor consulta la documentación del proyecto principal.

## 📞 Contacto

- GitHub: [RobertoRochaT/Chess-Project](https://github.com/RobertoRochaT/Chess-Project)

---

**Última actualización**: Diciembre 2025
