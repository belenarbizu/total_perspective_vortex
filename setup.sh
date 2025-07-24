#!/bin/bash

# Nombre del entorno virtual
ENV_NAME="env"

echo "Creando el entorno virtual '$ENV_NAME'..."
python3 -m venv "$ENV_NAME"

if [ $? -ne 0 ]; then
    echo "Error: No se pudo crear el entorno virtual."
fi

echo "Activando el entorno virtual..."
source "$ENV_NAME"/bin/activate

if [ $? -ne 0 ]; then
    echo "Error: No se pudo activar el entorno virtual."
    exit 1
fi

echo "Instalando paquetes..."
pip install --upgrade mne matplotlib joblib pandas scikit-learn

if [ $? -ne 0 ]; then
    echo "Error: Falló la instalación de uno o más paquetes."
    deactivate
    exit 1
fi

echo ""
echo "Instalación completada con éxito en el entorno '$ENV_NAME'."
echo "Para activar este entorno, ejecuta:"
echo "source ./$ENV_NAME/bin/activate"
echo ""
echo "Para desactivarlo cuando hayas terminado, ejecuta:"
echo "deactivate"