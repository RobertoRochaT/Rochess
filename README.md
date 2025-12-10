# ChessBot API - TensorFlow Chess Board Recognition

Sistema de reconocimiento de tableros de ajedrez basado en TensorFlow con API REST para conversión de imágenes a notación FEN.

## 📋 Descripción

ChessBot API es un servicio completo de visión por computadora que detecta y reconoce tableros de ajedrez en imágenes, convirtiendo las posiciones en notación FEN (Forsyth-Edwards Notation). Incluye una API REST compatible con aplicaciones frontend de ajedrez.

## ✨ Características

- 🎯 **Alta precisión**: >99.9% de certeza en reconocimiento de piezas
- 🌐 **API REST**: Servidor Flask con endpoints para análisis de imágenes
- 🔍 **Detección automática**: Localiza tableros en imágenes complejas
- ♟️ **13 clases**: Reconoce 12 tipos de piezas + casillas vacías
- 📊 **Análisis completo**: Detecta jaques, jaque mate y movimientos legales
- 🎨 **Visualizaciones**: Genera SVG y reportes HTML
- 📦 **Procesamiento por lotes**: Múltiples imágenes simultáneamente

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

## 📈 Rendimiento

- **Precisión**: >99.9% en piezas individuales
- **Velocidad**: ~100-200ms por imagen (CPU)
- **Soporte**: Tableros físicos y virtuales
- **Perspectivas**: Múltiples ángulos y perspectivas

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
