#!/usr/bin/env python3
"""
Script para visualizar y analizar los resultados del benchmark
Genera gráficos de matriz de confusión, métricas por pieza y comparativas
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import pandas as pd

# Configurar estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class BenchmarkVisualizer:
    """Visualizador de resultados de benchmark"""
    
    def __init__(self, results_file='benchmark_results.json'):
        """Cargar resultados del benchmark"""
        with open(results_file, 'r') as f:
            self.results = json.load(f)
        
        self.classes = self.results['dataset_info']['classes']
        self.output_dir = Path('benchmark_visualizations')
        self.output_dir.mkdir(exist_ok=True)
        
    def plot_confusion_matrix(self):
        """Genera y guarda la matriz de confusión"""
        cm = np.array(self.results['confusion_matrix'])
        
        # Crear figura grande
        plt.figure(figsize=(14, 12))
        
        # Normalizar por fila para ver porcentajes
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Crear heatmap
        sns.heatmap(cm_normalized, 
                    annot=cm,  # Mostrar valores reales
                    fmt='d',
                    cmap='RdYlGn',
                    xticklabels=self.classes,
                    yticklabels=self.classes,
                    cbar_kws={'label': 'Proporción'},
                    vmin=0, vmax=1)
        
        plt.title('Matriz de Confusión - ChessBot Transfer Learning\n' +
                  f'Precisión General: {self.results["overall_metrics"]["accuracy"]:.2%}',
                  fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('Etiqueta Real', fontsize=12, fontweight='bold')
        plt.xlabel('Predicción', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        output_path = self.output_dir / 'confusion_matrix.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Matriz de confusión guardada: {output_path}")
        plt.close()
        
    def plot_per_piece_accuracy(self):
        """Gráfico de precisión por pieza"""
        pieces = list(self.results['per_piece_metrics'].keys())
        accuracies = [self.results['per_piece_metrics'][p]['accuracy'] 
                      for p in pieces]
        
        # Separar por color
        colors = []
        for piece in pieces:
            if piece == 'empty':
                colors.append('gray')
            elif piece.startswith('w'):
                colors.append('lightblue')
            else:
                colors.append('lightcoral')
        
        plt.figure(figsize=(14, 6))
        bars = plt.bar(range(len(pieces)), accuracies, color=colors, 
                       edgecolor='black', linewidth=1.5)
        
        # Añadir línea de promedio
        avg_acc = self.results['overall_metrics']['accuracy']
        plt.axhline(y=avg_acc, color='red', linestyle='--', linewidth=2,
                   label=f'Promedio: {avg_acc:.2%}')
        
        # Añadir valores en las barras
        for i, (bar, acc) in enumerate(zip(bars, accuracies)):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.1%}',
                    ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        plt.xlabel('Pieza', fontsize=12, fontweight='bold')
        plt.ylabel('Precisión', fontsize=12, fontweight='bold')
        plt.title('Precisión por Tipo de Pieza\nTransfer Learning (ResNet50)',
                 fontsize=14, fontweight='bold', pad=15)
        plt.xticks(range(len(pieces)), pieces, rotation=45, ha='right')
        plt.ylim(0.8, 1.0)
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        plt.legend(loc='lower right', fontsize=10)
        
        # Añadir leyenda de colores
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='gray', edgecolor='black', label='Vacío'),
            Patch(facecolor='lightblue', edgecolor='black', label='Piezas Blancas'),
            Patch(facecolor='lightcoral', edgecolor='black', label='Piezas Negras')
        ]
        plt.legend(handles=legend_elements, loc='lower left', fontsize=9)
        
        plt.tight_layout()
        output_path = self.output_dir / 'accuracy_per_piece.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Gráfico de precisión por pieza guardado: {output_path}")
        plt.close()
        
    def plot_color_comparison(self):
        """Comparación de rendimiento entre piezas blancas y negras"""
        white = self.results['metrics_by_color']['white_pieces']
        black = self.results['metrics_by_color']['black_pieces']
        empty = self.results['metrics_by_color']['empty_squares']
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        white_vals = [white[m] for m in metrics]
        black_vals = [black[m] for m in metrics]
        empty_vals = [empty[m] for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        bars1 = ax.bar(x - width, white_vals, width, label='Piezas Blancas',
                      color='lightblue', edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x, black_vals, width, label='Piezas Negras',
                      color='lightcoral', edgecolor='black', linewidth=1.5)
        bars3 = ax.bar(x + width, empty_vals, width, label='Casillas Vacías',
                      color='lightgray', edgecolor='black', linewidth=1.5)
        
        # Añadir valores en las barras
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2%}',
                       ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        ax.set_xlabel('Métrica', fontsize=12, fontweight='bold')
        ax.set_ylabel('Valor', fontsize=12, fontweight='bold')
        ax.set_title('Comparación de Métricas por Color de Pieza\n' +
                    f'Diferencia Blancas-Negras: {(white["accuracy"]-black["accuracy"])*100:.2f} puntos porcentuales',
                    fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels([m.capitalize() for m in metrics])
        ax.legend(fontsize=10, loc='lower right')
        ax.set_ylim(0.85, 1.0)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        output_path = self.output_dir / 'color_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Comparación por color guardada: {output_path}")
        plt.close()
        
    def plot_training_history(self):
        """Gráfico de historial de entrenamiento"""
        history = self.results['training_history']
        epochs = range(1, len(history['train_accuracy']) + 1)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Precisión
        ax1.plot(epochs, history['train_accuracy'], 'b-o', 
                label='Entrenamiento', linewidth=2, markersize=6)
        ax1.plot(epochs, history['val_accuracy'], 'r-s',
                label='Validación', linewidth=2, markersize=6)
        ax1.axvline(x=history['best_epoch'], color='green', 
                   linestyle='--', linewidth=2,
                   label=f'Mejor Época: {history["best_epoch"]}')
        ax1.set_xlabel('Época', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Precisión', fontsize=12, fontweight='bold')
        ax1.set_title('Precisión Durante el Entrenamiento', 
                     fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0.3, 1.0)
        
        # Pérdida
        ax2.plot(epochs, history['train_loss'], 'b-o',
                label='Entrenamiento', linewidth=2, markersize=6)
        ax2.plot(epochs, history['val_loss'], 'r-s',
                label='Validación', linewidth=2, markersize=6)
        ax2.axvline(x=history['best_epoch'], color='green',
                   linestyle='--', linewidth=2,
                   label=f'Mejor Época: {history["best_epoch"]}')
        ax2.set_xlabel('Época', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Pérdida', fontsize=12, fontweight='bold')
        ax2.set_title('Pérdida Durante el Entrenamiento',
                     fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        fig.suptitle('Historial de Entrenamiento - Transfer Learning (ResNet50)',
                    fontsize=15, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        output_path = self.output_dir / 'training_history.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Historial de entrenamiento guardado: {output_path}")
        plt.close()
        
    def plot_confidence_distribution(self):
        """Distribución de confianza por tipo de pieza"""
        pieces = list(self.results['per_piece_metrics'].keys())
        confidences = [self.results['per_piece_metrics'][p]['avg_confidence']
                      for p in pieces]
        
        # Separar por tipo
        empty_conf = [confidences[0]]
        white_conf = confidences[1:7]
        black_conf = confidences[7:]
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        positions = [1] + list(range(3, 9)) + list(range(11, 17))
        colors_list = ['gray'] + ['lightblue']*6 + ['lightcoral']*6
        
        bars = ax.bar(positions, confidences, color=colors_list,
                     edgecolor='black', linewidth=1.5)
        
        # Añadir valores
        for pos, conf in zip(positions, confidences):
            ax.text(pos, conf, f'{conf:.2%}',
                   ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # Líneas de separación
        ax.axvline(x=2, color='black', linestyle=':', alpha=0.5)
        ax.axvline(x=10, color='black', linestyle=':', alpha=0.5)
        
        # Etiquetas de grupo
        ax.text(1, 0.82, 'Vacío', ha='center', fontsize=11, fontweight='bold')
        ax.text(6, 0.82, 'Piezas Blancas', ha='center', fontsize=11, fontweight='bold')
        ax.text(14, 0.82, 'Piezas Negras', ha='center', fontsize=11, fontweight='bold')
        
        ax.set_xlabel('Tipo de Pieza', fontsize=12, fontweight='bold')
        ax.set_ylabel('Confianza Promedio', fontsize=12, fontweight='bold')
        ax.set_title('Distribución de Confianza del Modelo por Tipo de Pieza',
                    fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(positions)
        ax.set_xticklabels(pieces, rotation=45, ha='right')
        ax.set_ylim(0.85, 1.0)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Línea de promedio
        avg_conf = np.mean(confidences)
        ax.axhline(y=avg_conf, color='red', linestyle='--', linewidth=2,
                  label=f'Promedio: {avg_conf:.2%}')
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        output_path = self.output_dir / 'confidence_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Distribución de confianza guardada: {output_path}")
        plt.close()
        
    def generate_summary_table(self):
        """Genera tabla resumen de métricas"""
        data = []
        for piece, metrics in self.results['per_piece_metrics'].items():
            data.append({
                'Pieza': piece,
                'Precisión': f"{metrics['accuracy']:.2%}",
                'Recall': f"{metrics['recall']:.2%}",
                'F1-Score': f"{metrics['f1_score']:.2%}",
                'Confianza': f"{metrics['avg_confidence']:.2%}",
                'Muestras': metrics['samples'],
                'Correctas': metrics['correct']
            })
        
        df = pd.DataFrame(data)
        
        # Guardar como CSV
        csv_path = self.output_dir / 'metrics_summary.csv'
        df.to_csv(csv_path, index=False)
        print(f"✅ Tabla resumen guardada: {csv_path}")
        
        # Crear visualización de tabla
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=df.values,
                        colLabels=df.columns,
                        cellLoc='center',
                        loc='center',
                        colColours=['lightgray']*7)
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Colorear filas según el tipo
        for i in range(len(df)):
            piece = df.iloc[i]['Pieza']
            if piece == 'empty':
                color = 'lightgray'
            elif piece.startswith('w'):
                color = 'lightblue'
            else:
                color = 'lightcoral'
            
            for j in range(len(df.columns)):
                table[(i+1, j)].set_facecolor(color)
        
        plt.title('Resumen de Métricas por Pieza\nChessBot Transfer Learning Model',
                 fontsize=14, fontweight='bold', pad=20)
        
        output_path = self.output_dir / 'metrics_table.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Tabla de métricas guardada: {output_path}")
        plt.close()
        
    def generate_all_visualizations(self):
        """Genera todas las visualizaciones"""
        print("\n🎨 Generando visualizaciones del benchmark...\n")
        
        self.plot_confusion_matrix()
        self.plot_per_piece_accuracy()
        self.plot_color_comparison()
        self.plot_training_history()
        self.plot_confidence_distribution()
        self.generate_summary_table()
        
        print(f"\n✅ Todas las visualizaciones generadas en: {self.output_dir}/")
        print("\nArchivos generados:")
        for file in sorted(self.output_dir.glob('*')):
            print(f"  📊 {file.name}")


def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Visualizar resultados del benchmark de ChessBot'
    )
    parser.add_argument('--results', '-r',
                       default='benchmark_results.json',
                       help='Archivo JSON con resultados del benchmark')
    parser.add_argument('--output', '-o',
                       default='benchmark_visualizations',
                       help='Directorio de salida para visualizaciones')
    
    args = parser.parse_args()
    
    print("="*70)
    print("  VISUALIZADOR DE BENCHMARK - ChessBot Transfer Learning")
    print("="*70)
    
    visualizer = BenchmarkVisualizer(args.results)
    visualizer.output_dir = Path(args.output)
    visualizer.output_dir.mkdir(exist_ok=True)
    
    visualizer.generate_all_visualizations()
    
    print("\n" + "="*70)
    print("✅ Proceso completado exitosamente")
    print("="*70)


if __name__ == '__main__':
    main()
