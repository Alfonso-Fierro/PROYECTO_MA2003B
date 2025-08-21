# Integratory Project for MA2003B

_Authors: Santiago, Mateo, Oscar, Alfonso_


# MA2014 - Proyecto de Análisis de Datos

## 📋 Descripción del Proyecto
Breve descripción de lo que hace el proyecto...

## 🛠️ Configuración del Entorno

### Prerrequisitos
- Python 3.8 o superior
- Git

### Instalación

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/Alfonso-Fierro/MA2014.git
   cd MA2014
   ```

2. **Crear y activar el entorno virtual:**
   
   **En Linux/Mac:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
   
   **En Windows:**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verificar instalación:**
   ```bash
   python -c "import pandas, numpy, matplotlib; print('¡Todo listo!')"
   ```

## 📁 Estructura del Proyecto

```
MA2014/
├── data/
│   ├── raw/                 # Datos originales (no modificar)
│   ├── processed/          # Datos procesados
│   └── external/           # Datos externos
├── notebooks/              # Jupyter notebooks
├── reports/                # Comunicaciones
|   ├── figures/                # Plots
├── requirements.txt        # Dependencias de Python
├── .gitignore             # Archivos a ignorar por Git
└── README.md              # Este archivo
```

## 🔄 Flujo de Trabajo con Git

### Para trabajar en una nueva funcionalidad:

1. **Actualizar el repositorio:**
   ```bash
   git pull origin main
   ```

2. **Crear una nueva rama:**
   ```bash
   git checkout -b feature/nombre-descriptivo
   ```

3. **Realizar cambios y confirmar:**
   ```bash
   git add .
   git commit -m "Descripción clara de los cambios"
   git push -u origin feature/nombre-descriptivo
   ```

4. **Crear Pull Request en GitHub**

### Convenciones de nombres de ramas:
- `feature/nueva-funcionalidad` - Para nuevas características
- `fix/correccion-bug` - Para corrección de errores
- `docs/actualizacion-readme` - Para documentación
- `analysis/exploratorio-datos` - Para análisis específicos

## 📊 Uso de Notebooks

Los notebooks están organizados por propósito:
- `01_exploracion_datos.ipynb` - Análisis exploratorio inicial
- `02_limpieza_datos.ipynb` - Preprocesamiento y limpieza
- `03_analisis_principal.ipynb` - Análisis principal

## 🤝 Colaboración

### Antes de hacer push:
1. Asegúrate de que tu código funciona
2. Ejecuta todos los notebooks sin errores
3. Actualiza la documentación si es necesario
4. Usa mensajes de commit descriptivos

### Resolución de conflictos:
```bash
git pull origin main
# Resolver conflictos manualmente
git add .
git commit -m "Resuelve conflictos con main"
git push
```

## 📦 Dependencias Principales

- **pandas**: Manipulación de datos
- **numpy**: Operaciones numéricas
- **matplotlib/seaborn**: Visualización
- **jupyter**: Notebooks interactivos
- **scikit-learn**: Machine Learning (si aplica)

## 🐛 Problemas Comunes

### El notebook no encuentra archivos:
```python
import os
print("Directorio actual:", os.getcwd())
# Cambiar al directorio raíz del proyecto si es necesario
os.chdir('..')
```

### Error de módulos no encontrados:
```bash
# Verificar que el venv está activado
which python
# Debe apuntar a tu venv

# Reinstalar requirements
pip install -r requirements.txt
```

## 👥 Colaboradores

- Alfonso Fierro (@Alfonso-Fierro) - Maintainer
- Santiago Rodriguez
- Óscar Casas

## 📞 Contacto

Si tienes problemas o preguntas, crea un issue en GitHub o contacta al equipo.