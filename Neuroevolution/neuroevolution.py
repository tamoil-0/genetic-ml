import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Configurar semillas para reproducibilidad
np.random.seed(42)
tf.random.set_seed(42)

# Generar dataset de ejemplo
X, y = make_classification(n_samples=1000, n_features=20, n_informative=8, 
                          n_redundant=5, random_state=42)
X = StandardScaler().fit_transform(X)

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class GeneticNeuroevolution:
    def __init__(self, n_population=10, n_generations=5, crossover_rate=0.6, 
                 mutation_rate=0.2, elitism=0.2):
        # Parámetros del algoritmo genético (reducidos para demo)
        self.n_population = n_population
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism = elitism
        self.input_dim = X.shape[1]
        
        # Historial para gráficos
        self.best_fitness_history = []
        self.avg_fitness_history = []
        
    def initialize_population(self):
        """Inicializa la población con arquitecturas aleatorias"""
        population = []
        for _ in range(self.n_population):
            # Representación del cromosoma: 
            # [n_layers, units_layer1, units_layer2, dropout_rate, learning_rate_log]
            n_layers = np.random.randint(1, 3)  # 1-2 capas ocultas (simplificado)
            individual = [n_layers]
            
            # Unidades por capa (simplificado)
            for _ in range(n_layers):
                individual.append(np.random.choice([32, 64, 128]))
            
            # Dropout rate (0.0 - 0.5)
            individual.append(np.random.uniform(0.0, 0.5))
            
            # Learning rate (en escala logarítmica: 1e-4 to 1e-2)
            individual.append(np.random.uniform(-4, -2))
            
            population.append(individual)
        return population
    
    def build_model(self, individual):
        """Construye un modelo de Keras basado en el individuo"""
        try:
            n_layers = individual[0]
            units = individual[1:1+n_layers]
            dropout_rate = individual[1+n_layers]
            learning_rate = 10 ** individual[2+n_layers]  # Convertir de log scale
            
            # Validar valores
            if dropout_rate < 0 or dropout_rate > 1:
                return None
            if learning_rate <= 0:
                return None
            
            model = Sequential()
            model.add(Dense(units[0], activation='relu', input_shape=(self.input_dim,)))
            model.add(Dropout(dropout_rate))
            
            for u in units[1:]:
                model.add(Dense(u, activation='relu'))
                model.add(Dropout(dropout_rate))
            
            model.add(Dense(1, activation='sigmoid'))
            
            model.compile(
                optimizer=Adam(learning_rate=learning_rate),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            return model
        except:
            return None
    
    def fitness_function(self, individual):
        """Evalúa la aptitud de una arquitectura de red neuronal"""
        model = self.build_model(individual)
        if model is None:
            return -1.0  # Penalizar individuos inválidos
        
        try:
            # Entrenamiento rápido para evaluación
            history = model.fit(
                X_train, y_train,
                epochs=5,  # Reducido para mayor velocidad
                batch_size=32,
                validation_split=0.2,
                verbose=0
            )
            
            # Fitness basado en accuracy de validación
            val_accuracy = max(history.history['val_accuracy'])
            
            # Penalizar arquitecturas complejas
            complexity_penalty = 0.001 * sum(individual[1:1+individual[0]]) / 100
            
            return val_accuracy - complexity_penalty
            
        except Exception as e:
            print(f"Error en entrenamiento: {e}")
            return -1.0
    
    def selection(self, population, fitness_scores):
        """Selección por método de ruleta"""
        # Reemplazar valores negativos con 0
        fitness_scores = np.array([max(0, f) for f in fitness_scores])
        
        if np.sum(fitness_scores) == 0:
            probabilities = np.ones(len(population)) / len(population)
        else:
            probabilities = fitness_scores / np.sum(fitness_scores)
        
        selected_indices = np.random.choice(
            len(population), 
            size=self.n_population, 
            p=probabilities,
            replace=True
        )
        return [population[i] for i in selected_indices]
    
    def crossover(self, parent1, parent2):
        """Operador de cruzamiento de un punto"""
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        if np.random.random() < self.crossover_rate:
            # Cruzar número de capas
            if np.random.random() < 0.5:
                child1[0], child2[0] = child2[0], parent1[0]
            
            # Ajustar longitud de hijos según número de capas
            for child in [child1, child2]:
                n_layers = child[0]
                expected_length = 1 + n_layers + 2  # n_layers + units + dropout + lr
                if len(child) > expected_length:
                    # Eliminar elementos extras
                    child = child[:expected_length]
                elif len(child) < expected_length:
                    # Añadir elementos faltantes
                    while len(child) < expected_length:
                        if len(child) < 1 + n_layers:
                            child.append(np.random.choice([32, 64, 128]))
                        elif len(child) == 1 + n_layers:
                            child.append(np.random.uniform(0.0, 0.5))
                        else:
                            child.append(np.random.uniform(-4, -2))
            
        return child1, child2
    
    def mutation(self, individual):
        """Operador de mutación"""
        individual = individual.copy()
        
        # Mutación del número de capas
        if np.random.random() < self.mutation_rate:
            new_n_layers = np.random.randint(1, 3)
            if new_n_layers != individual[0]:
                individual[0] = new_n_layers
                # Ajustar unidades
                current_units = individual[1:1+new_n_layers]
                if len(current_units) < new_n_layers:
                    # Añadir capas si es necesario
                    for _ in range(new_n_layers - len(current_units)):
                        individual.insert(1, np.random.choice([32, 64, 128]))
                else:
                    # Remover capas si es necesario
                    individual = individual[:1+new_n_layers] + individual[1+len(current_units):]
        
        # Mutación de unidades en capas
        for i in range(1, 1 + individual[0]):
            if np.random.random() < self.mutation_rate/2:
                individual[i] = np.random.choice([32, 64, 128])
        
        # Asegurar que tenemos todos los parámetros necesarios
        expected_length = 1 + individual[0] + 2
        if len(individual) < expected_length:
            # Añadir parámetros faltantes
            while len(individual) < expected_length:
                if len(individual) == 1 + individual[0]:
                    individual.append(np.random.uniform(0.0, 0.5))
                else:
                    individual.append(np.random.uniform(-4, -2))
        
        # Mutación de dropout rate
        dropout_idx = 1 + individual[0]
        if np.random.random() < self.mutation_rate/2:
            individual[dropout_idx] = np.random.uniform(0.0, 0.5)
        
        # Mutación de learning rate
        lr_idx = 2 + individual[0]
        if np.random.random() < self.mutation_rate/2:
            individual[lr_idx] = np.random.uniform(-4, -2)
        
        return individual
    
    def evolve(self):
        """Ejecuta el proceso evolutivo"""
        population = self.initialize_population()
        best_individual_history = []
        
        print("Iniciando proceso de neuroevolución...")
        
        for generation in range(self.n_generations):
            # Evaluar fitness de toda la población
            fitness_scores = []
            for individual in population:
                fitness = self.fitness_function(individual)
                fitness_scores.append(fitness)
            
            # Guardar estadísticas
            best_idx = np.argmax(fitness_scores)
            best_fitness = fitness_scores[best_idx]
            avg_fitness = np.mean(fitness_scores)
            
            self.best_fitness_history.append(best_fitness)
            self.avg_fitness_history.append(avg_fitness)
            best_individual_history.append(population[best_idx])
            
            print(f"Generación {generation + 1}: Mejor fitness = {best_fitness:.4f}, Fitness promedio = {avg_fitness:.4f}")
            
            # Selección
            selected_population = self.selection(population, fitness_scores)
            
            # Cruzamiento y mutación
            new_population = []
            
            # Elitismo: mantener los mejores individuos
            n_elite = max(1, int(self.elitism * self.n_population))
            elite_indices = np.argsort(fitness_scores)[-n_elite:]
            new_population.extend([population[i] for i in elite_indices])
            
            # Generar nueva población
            while len(new_population) < self.n_population:
                # Seleccionar padres aleatorios
                idx1, idx2 = np.random.choice(len(selected_population), 2, replace=False)
                parent1, parent2 = selected_population[idx1], selected_population[idx2]
                
                # Cruzamiento
                child1, child2 = self.crossover(parent1, parent2)
                
                # Mutación
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)
                
                # Añadir hijos si hay espacio
                if len(new_population) < self.n_population:
                    new_population.append(child1)
                if len(new_population) < self.n_population:
                    new_population.append(child2)
            
            population = new_population[:self.n_population]
        
        # Encontrar el mejor individuo de todas las generaciones
        best_overall_idx = np.argmax(self.best_fitness_history)
        return best_individual_history[best_overall_idx]
    
    def plot_evolution(self):
        """Visualiza el progreso de la evolución"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.best_fitness_history, label='Mejor Fitness', marker='o')
        plt.plot(self.avg_fitness_history, label='Fitness Promedio', marker='s')
        plt.xlabel('Generación')
        plt.ylabel('Fitness')
        plt.title('Progreso de la Neuroevolución')
        plt.legend()
        plt.grid(True)
        plt.savefig('evolution_progress.png')
        plt.show()

# Ejecutar neuroevolución
if __name__ == "__main__":
    print("=== NEUROEVOLUTION CON ALGORITMOS GENÉTICOS ===")
    print("Configurando algoritmo genético...")
    
    # Parámetros reducidos para evitar errores
    neuro_evo = GeneticNeuroevolution(n_population=8, n_generations=4)
    best_architecture = neuro_evo.evolve()

    print(f"\n=== MEJOR ARQUITECTURA ENCONTRADA ===")
    print(f"Número de capas ocultas: {best_architecture[0]}")
    print(f"Unidades por capa: {best_architecture[1:1+best_architecture[0]]}")
    print(f"Dropout rate: {best_architecture[1+best_architecture[0]]:.3f}")
    print(f"Learning rate: {10**best_architecture[2+best_architecture[0]]:.6f}")

    # Visualizar progreso
    print("\nGenerando gráfica de evolución...")
    neuro_evo.plot_evolution()

    # Entrenar y evaluar el mejor modelo
    print("\n=== ENTRENANDO MEJOR MODELO ===")
    best_model = neuro_evo.build_model(best_architecture)
    if best_model is not None:
        history = best_model.fit(X_train, y_train, epochs=15, validation_data=(X_test, y_test), verbose=1)
        
        # Evaluar rendimiento final
        test_loss, test_accuracy = best_model.evaluate(X_test, y_test, verbose=0)
        print(f"\nPrecisión final en prueba: {test_accuracy:.4f}")
        