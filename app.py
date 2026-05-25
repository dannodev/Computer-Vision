import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import cv2
from deepface import DeepFace


def list_authorized_images(authorized_dir: str) -> Dict[str, List[str]]:
    authorized: Dict[str, List[str]] = {}
    if not os.path.isdir(authorized_dir):
        return authorized

    for person_name in sorted(os.listdir(authorized_dir)):
        person_dir = os.path.join(authorized_dir, person_name)
        if not os.path.isdir(person_dir):
            continue
        images = []
        for file_name in sorted(os.listdir(person_dir)):
            if file_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                images.append(os.path.join(person_dir, file_name))
        if images:
            authorized[person_name] = images

    return authorized


def capture_webcam_frame(output_path: str, device_index: int = 0) -> None:
    capture = cv2.VideoCapture(device_index)
    if not capture.isOpened():
        raise RuntimeError("No se pudo abrir la camara")

    time.sleep(1.0)
    ret, frame = capture.read()
    capture.release()

    if not ret or frame is None:
        raise RuntimeError("No se pudo capturar un frame de la camara")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, frame)


def verify_login(
    login_image: str,
    authorized: Dict[str, List[str]],
    model_name: str,
    detector_backend: str,
    distance_metric: str,
    enforce_detection: bool,
) -> Tuple[Optional[str], Optional[dict]]:
    for person_name, images in authorized.items():
        for image_path in images:
            try:
                result = DeepFace.verify(
                    img1_path=login_image,
                    img2_path=image_path,
                    model_name=model_name,
                    detector_backend=detector_backend,
                    distance_metric=distance_metric,
                    enforce_detection=enforce_detection,
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"Aviso: error verificando contra {image_path}: {exc}",
                    file=sys.stderr,
                )
                continue

            if result.get("verified"):
                return person_name, result

    return None, None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inicio de sesion con verificacion facial usando DeepFace"
    )
    parser.add_argument(
        "--authorized-dir",
        default="data/authorized",
        help="Directorio con subcarpetas por persona",
    )
    parser.add_argument(
        "--login-image",
        help="Ruta de la imagen para login",
    )
    parser.add_argument(
        "--webcam",
        action="store_true",
        help="Captura una imagen desde la camara para el login",
    )
    parser.add_argument(
        "--webcam-output",
        default="data/captures/login.jpg",
        help="Ruta donde se guarda la captura de la camara",
    )
    parser.add_argument("--model", default="VGG-Face", help="Modelo de DeepFace")
    parser.add_argument(
        "--detector",
        default="opencv",
        help="Backend de deteccion (opencv, mtcnn, retinaface, etc)",
    )
    parser.add_argument(
        "--metric",
        default="cosine",
        help="Metrica de distancia (cosine, euclidean, euclidean_l2)",
    )
    parser.add_argument(
        "--no-enforce-detection",
        action="store_true",
        help="Desactiva la deteccion obligatoria de rostro",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    authorized = list_authorized_images(args.authorized_dir)
    if not authorized:
        print(
            "No se encontraron imagenes autorizadas. "
            "Crea subcarpetas en data/authorized con imagenes por persona.",
            file=sys.stderr,
        )
        return 1

    login_image = args.login_image
    if args.webcam:
        try:
            capture_webcam_frame(args.webcam_output)
        except RuntimeError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        login_image = args.webcam_output

    if not login_image:
        print("Debes proporcionar --login-image o usar --webcam", file=sys.stderr)
        return 1

    if not os.path.isfile(login_image):
        print(f"No existe la imagen de login: {login_image}", file=sys.stderr)
        return 1

    person, result = verify_login(
        login_image,
        authorized,
        model_name=args.model,
        detector_backend=args.detector,
        distance_metric=args.metric,
        enforce_detection=not args.no_enforce_detection,
    )

    if person:
        distance = result.get("distance") if result else None
        threshold = result.get("threshold") if result else None
        print("Acceso permitido")
        print(f"Persona verificada: {person}")
        if distance is not None and threshold is not None:
            print(f"Distancia: {distance:.4f} | Umbral: {threshold:.4f}")
        return 0

    print("Acceso denegado")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
