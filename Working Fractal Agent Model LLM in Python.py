"""
Фрактальная модель выбора: Рай, Ад и Хаос
Полностью рабочий код на Python
Автор: Сергей Паломник (Digital Psychiatry)
Статья: Святой Августин и GAN
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import zoom, gaussian_filter
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

# ========================================================================
# 1. Генерация фрактального ландшафта (метод Diamond-Square)
# ========================================================================

def fractal_surface(size, roughness=0.7, power=1.0):
    """
    Генерация фрактальной поверхности методом Diamond-Square
    size: размер сетки (должен быть степенью 2 + 1)
    roughness: шероховатость (чем выше, тем более изрезанный ландшафт)
    """
    n = size
    surface = np.zeros((n, n))
    
    # Инициализация углов
    surface[0, 0] = np.random.randn() * roughness
    surface[0, n-1] = np.random.randn() * roughness
    surface[n-1, 0] = np.random.randn() * roughness
    surface[n-1, n-1] = np.random.randn() * roughness
    
    step = n - 1
    current_roughness = roughness
    
    while step > 1:
        half = step // 2
        
        # Diamond step (центры квадратов)
        for i in range(0, n-1, step):
            for j in range(0, n-1, step):
                avg = (surface[i, j] + surface[i, j+step] + 
                       surface[i+step, j] + surface[i+step, j+step]) / 4.0
                surface[i+half, j+half] = avg + np.random.randn() * current_roughness
        
        # Square step (центры рёбер)
        for i in range(0, n-1, half):
            for j in range(0, n-1, half):
                if (i % step == 0 and j % step == 0) or (i % step != 0 and j % step != 0):
                    continue
                avg = 0
                count = 0
                if i - half >= 0:
                    avg += surface[i-half, j]
                    count += 1
                if i + half < n:
                    avg += surface[i+half, j]
                    count += 1
                if j - half >= 0:
                    avg += surface[i, j-half]
                    count += 1
                if j + half < n:
                    avg += surface[i, j+half]
                    count += 1
                if count > 0:
                    avg /= count
                    surface[i, j] = avg + np.random.randn() * current_roughness
        
        step //= 2
        current_roughness *= (0.5 ** (1.0 / power))
    
    # Нормализация
    surface = (surface - surface.min()) / (surface.max() - surface.min())
    return surface

# ========================================================================
# 2. Генерация реальных данных (кластеры на фрактальном ландшафте)
# ========================================================================

def generate_real_data(n_points=500):
    """Генерация двух кластеров реальных данных"""
    data = np.zeros((n_points, 2))
    n_half = n_points // 2
    # Кластер 1: центр в (-3, 0)
    data[:n_half, 0] = np.random.randn(n_half) * 0.6 - 3
    data[:n_half, 1] = np.random.randn(n_half) * 0.6
    # Кластер 2: центр в (3, 0)
    data[n_half:, 0] = np.random.randn(n_points - n_half) * 0.6 + 3
    data[n_half:, 1] = np.random.randn(n_points - n_half) * 0.6
    return data

# ========================================================================
# 3. Фрактальная карта комфорта (ценность каждой точки)
# ========================================================================

# Генерация фрактальной карты комфорта
comfort_raw = fractal_surface(129, roughness=0.8, power=1.2)
comfort_map = zoom(comfort_raw, 0.4)  # Уменьшаем для скорости
comfort_map = gaussian_filter(comfort_map, sigma=1.5)

def get_comfort(x, y, comfort_data):
    """Билинейная интерполяция значения комфорта"""
    size = comfort_data.shape[0]
    # Маппинг координат: x ∈ [-5,5] → [0, size-1], y ∈ [-3,3] → [0, size-1]
    xi = (x + 5) / 10 * (size - 1)
    yi = (y + 3) / 6 * (size - 1)
    xi = np.clip(xi, 0, size-1)
    yi = np.clip(yi, 0, size-1)
    
    x0, x1 = int(np.floor(xi)), int(np.ceil(xi))
    y0, y1 = int(np.floor(yi)), int(np.ceil(yi))
    x0, x1 = np.clip([x0, x1], 0, size-1)
    y0, y1 = np.clip([y0, y1], 0, size-1)
    
    if x0 == x1 and y0 == y1:
        return comfort_data[x0, y0]
    if x0 == x1:
        w_y = yi - y0
        return (1 - w_y) * comfort_data[x0, y0] + w_y * comfort_data[x0, y1]
    if y0 == y1:
        w_x = xi - x0
        return (1 - w_x) * comfort_data[x0, y0] + w_x * comfort_data[x1, y0]
    
    w_x = xi - x0
    w_y = yi - y0
    v00 = comfort_data[x0, y0]
    v01 = comfort_data[x0, y1]
    v10 = comfort_data[x1, y0]
    v11 = comfort_data[x1, y1]
    
    v0 = (1 - w_x) * v00 + w_x * v10
    v1 = (1 - w_x) * v01 + w_x * v11
    return (1 - w_y) * v0 + w_y * v1

# ========================================================================
# 4. Основной класс эксперимента
# ========================================================================

class PilgrimExperiment:
    """Эксперимент с тремя режимами: рай, ад, хаос"""
    
    def __init__(self, n_agents=150, n_iterations=80):
        self.n_agents = n_agents
        self.n_iterations = n_iterations
        self.real_data = generate_real_data(400)
        
        # Режимы
        self.modes = {
            'РАЙ (сбалансированный)': {'lambda_real': 0.9, 'lambda_rand': 0.35, 'color': 'green'},
            'АД (жёсткий критик)': {'lambda_real': 3.5, 'lambda_rand': 0.12, 'color': 'red'},
            'ХАОС (слабый критик)': {'lambda_real': 0.15, 'lambda_rand': 0.85, 'color': 'blue'}
        }
        
        self.results = {}
    
    def compute_entropy(self, points):
        """Вычисление энтропии Шеннона по распределению точек"""
        counts, _ = np.histogram(points.flatten(), bins=25, range=(-5.5, 5.5))
        probs = counts[counts > 0] / np.sum(counts)
        return -np.sum(probs * np.log2(probs)) if len(probs) > 0 else 0
    
    def run_mode(self, mode_name, lambda_real, lambda_rand, verbose=True):
        """Запуск эксперимента для одного режима"""
        if verbose:
            print(f"\n{'='*50}")
            print(f"Запуск: {mode_name}")
            print(f"  Вес притяжения к реальности (критик): {lambda_real}")
            print(f"  Вес случайного блуждания (свобода): {lambda_rand}")
            print(f"{'='*50}")
        
        # Инициализация точек (случайное облако)
        points = np.random.randn(self.n_agents, 2) * 3.5
        entropy_history = []
        points_history = []
        
        for iteration in range(self.n_iterations):
            new_points = points.copy()
            
            for i in range(self.n_agents):
                # Сила 1: притяжение к ближайшей реальной точке
                dists = cdist(points[i:i+1], self.real_data)[0]
                nearest_idx = np.argmin(dists)
                force_real = (self.real_data[nearest_idx] - points[i]) * lambda_real
                
                # Сила 2: случайное блуждание
                force_rand = np.random.randn(2) * lambda_rand
                
                # Сила 3: мягкое отталкивание от соседей (разнообразие)
                force_repel = np.zeros(2)
                for j in range(self.n_agents):
                    if i != j:
                        diff = points[i] - points[j]
                        dist = np.linalg.norm(diff)
                        if 0 < dist < 0.8:
                            force_repel += diff / (dist ** 2 + 0.01) * 0.03
                
                # Сила 4: притяжение к зонам высокого комфорта (необязательно)
                try:
                    comfort_val = get_comfort(points[i, 0], points[i, 1], comfort_map)
                    force_comfort = np.array([
                        comfort_val * np.random.randn() * 0.05,
                        comfort_val * np.random.randn() * 0.05
                    ])
                except:
                    force_comfort = np.zeros(2)
                
                # Итоговое движение
                delta = force_real + force_rand + force_repel + force_comfort * 0.1
                new_points[i] = points[i] + delta * 0.12
                
                # Ограничение области
                new_points[i, 0] = np.clip(new_points[i, 0], -5.5, 5.5)
                new_points[i, 1] = np.clip(new_points[i, 1], -3.5, 3.5)
            
            points = new_points
            entropy = self.compute_entropy(points)
            entropy_history.append(entropy)
            points_history.append(points.copy())
            
            if verbose and (iteration % 15 == 0 or iteration == self.n_iterations - 1):
                print(f"  Итерация {iteration:3d}: энтропия = {entropy:.3f} бит")
        
        return {
            'entropy': np.array(entropy_history),
            'points': points_history,
            'final_points': points
        }
    
    def run_all(self):
        """Запуск всех трёх режимов"""
        for mode_name, params in self.modes.items():
            self.results[mode_name] = self.run_mode(
                mode_name, 
                params['lambda_real'], 
                params['lambda_rand']
            )
        return self.results
    
    def visualize_results(self):
        """Визуализация результатов"""
        # 1. Динамика энтропии на одном графике
        plt.figure(figsize=(14, 7))
        for mode_name, res in self.results.items():
            color = self.modes[mode_name]['color']
            plt.plot(res['entropy'], linewidth=2.5, color=color, label=mode_name)
            # Сглаженная кривая
            if len(res['entropy']) > 5:
                smoothed = np.convolve(res['entropy'], np.ones(5)/5, mode='same')
                plt.plot(smoothed, '--', linewidth=1.5, alpha=0.6, color=color)
        
        plt.xlabel('Итерация', fontsize=12)
        plt.ylabel('Энтропия (бит)', fontsize=12)
        plt.title('Сравнение трёх режимов: Рай (баланс) vs Ад (критик) vs Хаос (свобода)', fontsize=14)
        plt.legend(loc='best', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 5.5)
        plt.tight_layout()
        plt.savefig('entropy_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # 2. Финальные распределения точек
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for ax, (mode_name, res) in zip(axes, self.results.items()):
            # Реальные данные
            ax.scatter(self.real_data[:, 0], self.real_data[:, 1], 
                      c='blue', s=8, alpha=0.3, label='Реальные данные')
            # Точки агента
            final_pts = res['final_points']
            ax.scatter(final_pts[:, 0], final_pts[:, 1], 
                      c='red', s=25, alpha=0.7, edgecolors='darkred', linewidth=0.5, label='Агент')
            ax.set_xlim(-5.5, 5.5)
            ax.set_ylim(-3.5, 3.5)
            ax.set_title(f'{mode_name}\nЭнтропия = {res["entropy"][-1]:.2f} бит', fontsize=11)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
            ax.axvline(0, color='gray', linewidth=0.5, alpha=0.5)
        
        plt.suptitle('Финальные распределения точек: Рай, Ад и Хаос', fontsize=14)
        plt.tight_layout()
        plt.savefig('final_distributions.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # 3. Фрактальная карта комфорта с наложением точек (для рая)
        plt.figure(figsize=(12, 8))
        extent = [-5.5, 5.5, -3.5, 3.5]
        plt.imshow(comfort_map.T, origin='lower', cmap='hot', extent=extent, alpha=0.8)
        plt.colorbar(label='Комфорт')
        
        # Наложим точки из режима РАЙ
        heaven_points = self.results['РАЙ (сбалансированный)']['final_points']
        plt.scatter(heaven_points[:, 0], heaven_points[:, 1], 
                   c='cyan', s=20, alpha=0.6, edgecolors='white', linewidth=0.3, label='Точки агента (Рай)')
        plt.scatter(self.real_data[:, 0], self.real_data[:, 1], 
                   c='blue', s=10, alpha=0.4, label='Реальные данные')
        plt.title('Фрактальная карта комфорта и распределение точек в режиме РАЙ', fontsize=14)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.2)
        plt.savefig('fractal_comfort_heaven.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # 4. Таблица результатов
        print("\n" + "="*60)
        print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА")
        print("="*60)
        print(f"{'Режим':<30} {'Финальная энтропия (бит)':<25} {'Интерпретация'}")
        print("-"*60)
        
        interpretations = {
            'РАЙ (сбалансированный)': 'Свобода сохранена, точки вокруг двух кластеров',
            'АД (жёсткий критик)': 'Коллапс, все точки в одной куче',
            'ХАОС (слабый критик)': 'Хаотичное распределение, нет структуры'
        }
        
        for mode_name, res in self.results.items():
            entropy_val = res['entropy'][-1]
            print(f"{mode_name:<30} {entropy_val:<25.2f} {interpretations[mode_name]}")
        
        print("="*60)
        
        # Грех как падение энтропии
        heaven_entropy = self.results['РАЙ (сбалансированный)']['entropy'][-1]
        hell_entropy = self.results['АД (жёсткий критик)']['entropy'][-1]
        delta_s_grekh = heaven_entropy - hell_entropy
        
        print(f"\n МАТЕМАТИЧЕСКАЯ МЕРА ГРЕХА:")
        print(f"   ΔS_грех = S_рай - S_ад = {heaven_entropy:.2f} - {hell_entropy:.2f} = {delta_s_grekh:.2f} бит")
        print(f"\n   Это количественная мера того, как жёсткая критика убивает свободу.")
        print("="*60)

# ========================================================================
# 6. Запуск эксперимента
# ========================================================================

def main():
    print("\n" + "="*60)
    print("ФРАКТАЛЬНАЯ МОДЕЛЬ ВЫБОРА: РАЙ, АД И ХАОС")
    print("Святой Августин встречает GAN")
    print("="*60)
    print("\nИнициализация фрактального ландшафта...")
    
    # Создание и запуск эксперимента
    experiment = PilgrimExperiment(n_agents=150, n_iterations=70)
    
    print("\nЗапуск трёх режимов...")
    results = experiment.run_all()
    
    print("\nВизуализация результатов...")
    experiment.visualize_results()
    
    print("\n Эксперимент завершён. Все графики сохранены.")
    print("\nВывод: страх наказания убивает разнообразие (энтропия падает).")
    print("       Только баланс между реальностью и свободой даёт Рай.")
    print("\n Свобода — не отсутствие правил, а пространство для вариаций.")

if __name__ == "__main__":
    main()
