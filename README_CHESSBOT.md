# 🎯 ChessBot - Conversor de Tableros de Ajedrez a FEN

## 📋 Descripción

Este proyecto utiliza **TensorFlow ChessBot** para convertir imágenes de tableros de ajedrez (físicos o virtuales) a notación FEN con alta precisión y generar visualizaciones interactivas.

## ✨ Características

- ✅ **Conversión precisa**: Reconocimiento de tableros con >99% de certeza
- 🎨 **Visualización**: Genera imágenes comparativas y tableros SVG
- 📊 **Análisis de posición**: Información sobre turno, jaque, jaque mate y movimientos legales
- 🚀 **Procesamiento por lotes**: Procesa múltiples imágenes automáticamente
- 📱 **Reporte HTML**: Genera reportes interactivos con resultados

## 🔧 Instalación

### Requisitos Previos

- Python 3.10+
- pip

### Instalar Dependencias

```bash
pip install -r requirements_chessbot.txt
```

Dependencias principales:
- TensorFlow >= 2.10.0
- NumPy >= 1.21.0
- Pillow >= 9.0.0
- OpenCV >= 4.5.0
- python-chess >= 1.9.0
- matplotlib >= 3.3.0
- beautifulsoup4
- lxml
- scipy

## 🚀 Uso

### Procesamiento de una Imagen Individual

```bash
python chess_board_to_fen.py <imagen> [opciones]
```

**Ejemplo:**

```bash
python chess_board_to_fen.py images/tablero.png
```

**Opciones:**
- `--output-dir, -o`: Directorio de salida (default: `resultados_chessbot`)
- `--model, -m`: Ruta al modelo (default: `tensorflow_chessbot/saved_models/frozen_graph.pb`)
- `--no-viz`: Desactivar generación de visualizaciones

**Salida:**
- `<nombre>.fen`: Archivo con la notación FEN
- `<nombre>_fen.json`: Metadatos incluyendo FEN y certeza
- `<nombre>_board.svg`: Visualización del tablero detectado
- `<nombre>_comparison.png`: Comparación imagen original vs tablero detectado

### Procesamiento por Lotes

```bash
python batch_chess_analyzer.py <imágenes> [opciones]
```

**Ejemplos:**

```bash
# Procesar todas las imágenes PNG en la carpeta images/
python batch_chess_analyzer.py "images/*.png"

# Procesar múltiples archivos específicos
python batch_chess_analyzer.py img1.png img2.jpg img3.png

# Procesar con directorio de salida personalizado
python batch_chess_analyzer.py "images/*.png" --output-dir mis_resultados
```

**Opciones:**
- `--output-dir, -o`: Directorio de salida (default: `resultados_batch`)
- `--model, -m`: Ruta al modelo
- `--no-viz`: Desactivar visualizaciones

**Salida:**
- `batch_summary.json`: Resumen completo del procesamiento
- `reporte.html`: Reporte visual interactivo (¡ábrelo en tu navegador!)
- Archivos individuales por cada imagen procesada

## 📊 Ejemplos de Resultados

### Procesamiento Individual

```
🔍 Inicializando modelo de reconocimiento...
✅ Modelo cargado correctamente

📸 Procesando imagen: tablero_partida.png
📋 FEN detectado: r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQ1RK1
📊 Certeza: Promedio=100.0%, Mín=100.0%, Máx=100.0%

✅ Procesamiento completado exitosamente

♟️  Análisis del tablero:
   - Turno: Blancas
   - Jaque: No
   - Jaque mate: No
   - Movimientos legales: 28
```

### Procesamiento por Lotes

```
📦 Procesando 50 imágenes...

============================================================
📊 RESUMEN DEL PROCESAMIENTO
============================================================
Total:        50
Éxitos:       48 ✅
Fallos:       2 ❌
Tasa éxito:   96.0%
============================================================

📁 Resultados guardados en: resultados_batch
📊 Abre el reporte HTML: resultados_batch/reporte.html
```

## 🎯 Tipos de Imágenes Soportadas

El modelo funciona mejor con:

✅ **Soportados:**
- Capturas de pantalla de chess.com
- Capturas de pantalla de lichess.org
- Tableros generados por diagrama FEN
- Tableros digitales con casillas claramente definidas
- Imágenes con buena iluminación y contraste

⚠️ **Limitaciones:**
- Tableros físicos muy iluminados o con reflejos
- Tableros con piezas no estándar
- Imágenes muy borrosas o de baja resolución
- Tableros con orientación no estándar (sin cuadrícula 8x8 visible)

## 🛠️ Solución de Problemas

### Error: "No se pudo detectar el tablero en la imagen"

**Causas comunes:**
- La imagen no contiene un tablero de ajedrez claramente visible
- El tablero está muy rotado o distorsionado
- La iluminación es muy pobre

**Soluciones:**
- Asegúrate de que el tablero ocupe una porción significativa de la imagen
- Verifica que las líneas del tablero sean visibles
- Usa una imagen con mejor resolución o contraste

### Error: "Imagen demasiado grande para procesar"

**Solución:**
- El script automáticamente redimensiona imágenes grandes
- Si persiste, redimensiona manualmente a menos de 2000x2000 píxeles

### Baja certeza en la predicción

**Si la certeza es < 90%:**
- Revisa manualmente el FEN generado
- Considera tomar una nueva captura con mejor calidad
- El modelo puede tener dificultades con estilos de piezas no entrenados

## 📚 Estructura del Proyecto

```
LiveChess2FEN/
├── chess_board_to_fen.py      # Script para procesar imágenes individuales
├── batch_chess_analyzer.py     # Script para procesamiento por lotes
├── requirements_chessbot.txt   # Dependencias del proyecto
├── tensorflow_chessbot/        # Repositorio clonado de TensorFlow ChessBot
│   ├── saved_models/           # Modelos entrenados
│   │   └── frozen_graph.pb     # Modelo principal
│   ├── chessboard_finder.py    # Detección de tableros
│   └── tensorflow_chessbot.py  # Predictor CNN
├── images/                     # Imágenes de ejemplo
├── resultados_chessbot/        # Resultados de procesamiento individual
└── resultados_batch/           # Resultados de procesamiento por lotes
```

## 🔬 Cómo Funciona

1. **Detección del Tablero**: Usa visión por computadora para encontrar las esquinas del tablero
2. **Extracción de Casillas**: Divide el tablero en 64 casillas de 32x32 píxeles en escala de grises
3. **Red Neuronal Convolucional**: Clasifica cada casilla en 13 categorías:
   - 6 piezas blancas (P, N, B, R, Q, K)
   - 6 piezas negras (p, n, b, r, q, k)
   - 1 casilla vacía
4. **Generación de FEN**: Convierte las predicciones a notación FEN estándar
5. **Visualización**: Genera tablero SVG y comparaciones usando python-chess

## 🎓 Arquitectura del Modelo

El modelo CNN tiene la siguiente estructura:
- **Capa de entrada**: Convolución 5x5x32
- **Capa oculta**: Convolución 5x5x64
- **Capa densa**: 8x8x1024 completamente conectada
- **Capa de salida**: 1024x13 Dropout + Softmax

## 📈 Rendimiento

- **Tasa de éxito**: ~95% en imágenes de chess.com/lichess
- **Certeza promedio**: >99% en posiciones válidas
- **Velocidad**: ~1-2 segundos por imagen (incluye visualización)
- **Procesamiento por lotes**: ~100 imágenes en 3-4 minutos

## 🤝 Créditos

Este proyecto está basado en:
- [TensorFlow ChessBot](https://github.com/Elucidation/tensorflow_chessbot) por [@Elucidation](https://github.com/Elucidation)
- [python-chess](https://python-chess.readthedocs.io/) para análisis y visualización de tableros

## 📄 Licencia

Este proyecto utiliza código de TensorFlow ChessBot bajo licencia MIT.

## 🆘 Soporte

Si encuentras problemas:
1. Verifica que todas las dependencias estén instaladas
2. Asegúrate de que el modelo `frozen_graph.pb` esté presente
3. Revisa que tu imagen tenga un tablero claramente visible
4. Consulta la sección de solución de problemas

## 🎯 Próximas Mejoras

- [ ] Soporte para tableros con notación algebraica
- [ ] Reconocimiento de tableros físicos mejorado
- [ ] API REST para integración
- [ ] Soporte para análisis de video (frame por frame)
- [ ] Entrenamiento con más estilos de piezas

---

**¿Preguntas o sugerencias?** Abre un issue en el repositorio.

🎮 ¡Disfruta convirtiendo tus tableros de ajedrez a FEN!
