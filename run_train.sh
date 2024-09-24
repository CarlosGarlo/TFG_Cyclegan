# run train_exp1.py, train_exp2.py, train_exp3.py


#!/bin/bash

# Ejecutar el script de prueba
python3 train_night_cloudy.py

# Verificar si el segundo script se ejecutó correctamente
if [ $? -ne 0 ]; then
  echo "Error al ejecutar train_night_cloudy.py"
  exit 1
fi

# Ejecutar el script de prueba
python3 train_sunny_cloudy.py

# Verificar si el segundo script se ejecutó correctamente
if [ $? -ne 0 ]; then
  echo "Error al ejecutar train_sunny_cloudy.py"
  exit 1
fi

# Ejecutar el script de prueba
python3 train_sunny_night.py

# Verificar si el segundo script se ejecutó correctamente
if [ $? -ne 0 ]; then
  echo "Error al ejecutar train_sunny_cloudy.py"
  exit 1
fi

echo "Ejecución completada con éxito"