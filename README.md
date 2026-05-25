# Proyecto 6: Computer Vision

Sistema de inicio de sesion seguro con verificacion facial usando DeepFace. El usuario se autentica comparando su rostro con un repositorio de personas autorizadas.

## Requisitos

- Python 3.9+
- Camara (opcional, solo si usas el modo webcam)

## Instalacion

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Preparar datos

Crea subcarpetas dentro de `data/authorized` con el nombre de cada persona y coloca ahi varias imagenes de su rostro.

Ejemplo de estructura:

```
data/authorized/
	ana/
		1.jpg
		2.jpg
	juan/
		1.jpg
```

## Uso

### Verificacion con imagen

```bash
python app.py --login-image data/login/ana_test.jpg
```

### Verificacion con webcam (opcional)

```bash
python app.py --webcam
```

## Notas

- Si la deteccion de rostro falla con alguna imagen, puedes intentar `--no-enforce-detection`.
- Puedes cambiar modelo o detector con `--model` y `--detector`.