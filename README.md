# Presión Arterial

**Estimación experimental de presión arterial (PA) a partir de la onda de pulso rPPG con deep learning.**
Complemento de investigación para [VitaSense](https://github.com/braytwing/VitaSense).

> **Estado: prototipo de investigación. No es un dispositivo médico.** No usar para diagnóstico, vigilancia ni
> decisiones de salud. Hoy el repositorio entrega el *pipeline* completo y probado con datos sintéticos; **no incluye
> ningún modelo entrenado con datos reales ni demuestra que una webcam pueda medir PA.**

## Idea y límite fundamental

Una webcam a ~30 fps no registra la morfología fina de la onda de pulso que los modelos de PPG de contacto
(125 Hz) usan para estimar PA. El deep learning puede aprender mejor lo que *sí* está en la señal, pero no
inventa lo que la cámara no captó. Por eso este proyecto está diseñado para **responder honestamente si hay
señal de PA utilizable**: toda evaluación se compara con un predictor trivial y se separa por persona.

## Cómo funciona

```
PPG de contacto (125 Hz) + PA de referencia        <- datasets públicos (PulseDB, MIMIC, VitalDB...)
        │  degrade.py: remuestreo a 30 Hz, pasa-banda 0,7–3 Hz, ruido en banda,
        │              movimiento y modulación respiratoria  (simula la webcam)
        ▼
 ventana de 6 s × 30 Hz (180 muestras) × 3 canales: pulso, 1ª derivada, 2ª derivada
        │  model.py: ResNet 1D
        ▼
 [SBP, DBP] en mmHg  ──►  calibración por persona (VitaSense: BPCalibration)
```

| Archivo | Función |
|---|---|
| `presion_arterial/degrade.py` | Convierte PPG de contacto en señal "tipo webcam" (aumento de datos) |
| `presion_arterial/data.py` | Formato `.npz`, partición **por sujeto**, `Dataset` de PyTorch |
| `presion_arterial/model.py` | `BPNet`, ResNet 1D (≈ 0,1–1 M parámetros según `--base`) |
| `presion_arterial/train.py` | Entrenamiento y evaluación contra el predictor trivial |
| `presion_arterial/evaluate.py` | Validación externa de un checkpoint en otro dataset |
| `presion_arterial/metrics.py` | ME ± DE, MAE, r, BHS, criterio 5 ± 8 mmHg, evaluación calibrada |
| `presion_arterial/vitasense_adapter.py` | `DeepBPModel`: se conecta a `BloodPressureEstimator` de VitaSense |
| `presion_arterial/synthetic.py` | Datos de juguete **solo** para probar el pipeline |
| `scripts/prepare_dataset.py` | Convierte HDF5 / MAT v7.3 al formato `.npz` |
| `docs/DATASETS.md` | Qué datasets conseguir y cómo |

## Instalación

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest -q                                    # 8 tests, ~15 s en CPU
```

## Uso

**1. Prueba de humo (datos sintéticos, no dice nada de PA real):**
```bash
python -m presion_arterial.train --synthetic --epochs 10 --out runs/demo
```

**2. Con datos reales** (ver `docs/DATASETS.md`):
```bash
python scripts/prepare_dataset.py inspect PulseDB_subset.mat
python scripts/prepare_dataset.py convert PulseDB_subset.mat --ppg-key ... --sbp-key ... \
    --dbp-key ... --subject-key ... --fs 125 --out data/pulsedb.npz
python -m presion_arterial.train --data data/pulsedb.npz --epochs 30 --out runs/exp1
```

**3. Validación externa** (otro dataset o personas nunca usadas):
```bash
python -m presion_arterial.evaluate --ckpt runs/exp1/best.pt --data data/otro.npz --json runs/exp1/externa.json
```

**4. Integrar con VitaSense:**
```python
from presion_arterial.vitasense_adapter import DeepBPModel
from rppg import BloodPressureEstimator, BPCalibration   # de VitaSense

est = BloodPressureEstimator(DeepBPModel("runs/exp1/best.pt"),
                             BPCalibration(sbp_offset=0.0, dbp_offset=0.0))
# agrega `est` a la lista de estimadores de RPPGPipeline
```
`DeepBPModel` cumple el contrato `predict(feats, window)` (ventana de 750 muestras, 6 s a 125 Hz) que ya usa
`rppg/bp.py`. VitaSense seguirá sin mostrar un número si falta modelo **o** calibración.

## Cómo leer los resultados (importante)

- **Compara siempre con el predictor trivial** (media del entrenamiento). Si no lo supera con margen claro
  (> 1 mmHg de MAE), el modelo no aprendió PA: el código lo marca con `NO`.
- **Calibrado con 1 lectura:** el predictor trivial pasa a ser *"arrastrar la lectura del tensiómetro"*. Es la
  vara justa: si la señal no mejora eso, no aporta nada.
- En modo calibrado la **correlación r sale inflada** (entra la varianza entre personas); fíjate en la ganancia de MAE.
- La partición es **por sujeto**. Nunca mezcles ventanas de una misma persona entre entrenamiento y prueba.
- Referencias: 5 ± 8 mmHg (AAMI/ISO 81060-2, exige decenas de sujetos) y grados BHS A/B/C/D.

### Qué se verificó hasta ahora

Solo con **datos sintéticos** (CPU, 40 sujetos, 12 épocas), para comprobar la mecánica del pipeline:

| Caso | Resultado esperado | Observado (sujetos nuevos) |
|---|---|---|
| La señal contiene información de PA (`--signal-strength 1`) | El modelo supera al trivial | SBP: MAE 4,6 vs 12,8 mmHg |
| La señal NO contiene información (`--signal-strength 0`) | El modelo **no** debe superar al trivial | SBP: MAE 13,6 vs 12,8 → `NO` |

Esto prueba que el código separa bien ambos casos. **No** prueba nada sobre PA humana real.

## Lo que falta (hoja de ruta honesta)

1. **Entrenar con datos reales** (PulseDB / MIMIC / VitalDB) y reportar el resultado, sea cual sea.
2. **Adaptar el conversor** a los archivos reales: `prepare_dataset.py` se escribió sin acceso a ellos y no está probado.
3. **Etapa 1 no incluida:** mejorar *video → pulso* con una red rPPG (PhysNet, TS-CAN, EfficientPhys…), p. ej. con
   [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox). Hoy se usa el pulso POS/CHROM de VitaSense.
4. **Validación con video real + PA de referencia** (BP4D+/V4V o datos propios), con personas de distinta edad,
   tono de piel y rango de PA, incluyendo hipertensos. Es el paso decisivo.
5. Entrenamiento de extremo a extremo (video → PA) con suficiente dato propio.

## Privacidad y ética

El video facial es un dato biométrico y de salud. Para recolectar datos propios se necesita consentimiento
informado, aprobación de un comité de ética y política de almacenamiento y borrado. Evita subir videos, `.npz`
o modelos con datos de personas a repositorios públicos (el `.gitignore` ya excluye `*.npz`, `*.pt`, `data/`, `runs/`).
Verifica la licencia y los términos de acceso de cada dataset antes de usarlo.
