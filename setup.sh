#!/bin/bash

# Nombre del entorno virtual
ENV_NAME="env"

echo "Creando el entorno virtual '$ENV_NAME'..."
# Crea el entorno virtual usando el intérprete python3 predeterminado del sistema
python3 -m venv "$ENV_NAME"

# Verifica si la creación fue exitosa
if [ $? -ne 0 ]; then
    echo "Error: No se pudo crear el entorno virtual."
fi

echo "Activando el entorno virtual..."
# Activa el entorno virtual
source "$ENV_NAME"/bin/activate

# Verifica si la activación fue exitosa
if [ $? -ne 0 ]; then
    echo "Error: No se pudo activar el entorno virtual."
    exit 1
fi

echo "Instalando paquetes..."
# Instala los paquetes dentro del entorno virtual
# Usamos --upgrade para asegurar que sean las últimas versiones compatibles
pip install --upgrade mne matplotlib joblib autoreject

# Verifica si la instalación fue exitosa
if [ $? -ne 0 ]; then
    echo "Error: Falló la instalación de uno o más paquetes."
    # Desactiva el entorno antes de salir en caso de error
    deactivate
    exit 1
fi

echo ""
echo "Instalación completada con éxito en el entorno '$ENV_NAME'."
echo "Para activar este entorno en el futuro, ejecuta:"
echo "source ./$ENV_NAME/bin/activate"
echo ""
echo "Para desactivarlo cuando hayas terminado, ejecuta:"
echo "deactivate"