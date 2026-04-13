# DD-PDR-GES-003 | MASP - MÉTODO DE ANÁLISIS PARA LA SOLUCIÓN DE PROBLEMAS
## ACERÍAS PAZ DEL RÍO S.A. — Planta Belencito, Nobsa, Boyacá
### Versión Completa con Contextualización Técnica, Modelo de Optimización y Arquitectura ML + LLM

---

## Nº 1 — IDENTIFICACIÓN DEL PROBLEMA

### PROBLEMA
La operación de caliza carece de una programación estructurada del abastecimiento, ingreso y consumo del recurso, lo cual dificulta la planeación del mismo y aumenta el riesgo de incumplimiento de las metas operativas, afectando la continuidad, eficiencia y costo del proceso.

### DESCRIPCIÓN DEL PROBLEMA (ENRIQUECIDA)

Acerías Paz del Río gestiona **13 fuentes activas de caliza** clasificadas en dos categorías: caliza comprada (7 proveedores externos: Carbopar, Caribbean, Malaga, Gustavo Quesada, Leonidas Rodriguez, Calizas & Granitos, Lisandro Rojas) y caliza de explotación propia (4 fuentes: Caliza Propia, Afloramiento 1, Afloramiento 1 Sur, Malsitio), operando bajo el título minero 11387 con un área de 700,16 hectáreas en jurisdicción de Nobsa y Corrales.

El análisis de **1.249 registros históricos de despachos** revela que la variabilidad de calidad entre proveedores es significativa:
- **%SiO2 promedio ponderado: 2,07%** con desviación estándar de 2,37%, coeficiente de variación del 114%.
- El rango de SiO2 por proveedor va desde **1,04% (Carbopar, mejor)** hasta **3,59% (Gustavo Quesada, peor)**.
- Proveedores como Caliza Propia (3,52%), Lisandro Rojas (3,35%) y Leonidas Rodriguez (3,20%) superan sistemáticamente el límite de 2,5% aceptable para calcinación.

La caliza ingresada tiene dos posibles destinos de descargue: patio de almacenamiento o tolva de trituración. Pasa por trituración primaria (reducción inicial de tamaño) y trituración secundaria (reducción final y clasificación en 5 granulometrías) para alimentar sinterización (<3mm), calcinación (según horno) y carga directa al alto horno.

**Problemas específicos documentados:**
1. **No existe herramienta de planificación** que permita determinar la mezcla óptima de calizas según calidad y costo.
2. **El ingreso de pesos a SAP es manual**, con reflejo en el sistema un día después, impidiendo la visibilidad en tiempo real.
3. **El muestreo en banda corta no es representativo** para todos los viajes del proveedor en un día dado (ISO 18283 advierte sobre esta limitación).
4. **No hay tipificación de caliza** según parámetros de calidad por proveedor ni por frente de explotación.
5. **La producción de la trituradora se notifica según la báscula de sinter**, no por medición directa.
6. **La caliza en patio se contamina con arena**, incrementando el SiO2 real por encima del valor analizado en origen.

Esto repercute en **desviaciones de calidad en planta** (SiO2 en sinter llegó hasta 20,4%, cal producida con CaO tan bajo como 49,5%), **desbalance de inventarios** y **sobrecostos de producción** estimados en miles de millones de pesos anuales.

### META
Construir un **modelo de optimización no-lineal** basado en Ensemble Machine Learning (Random Forest + Gradient Boosting) integrado con un LLM (LLaMA 3) que permita planear la compra y trituración de caliza, asegurando requerimientos de calidad según destino (sinterización, calcinación, alto horno), balanceando inventarios y minimizando el costo del proceso, con capacidad de recomendación en tiempo real.

---

## Nº 2 — ANÁLISIS DEL FENÓMENO / OBSERVACIÓN

### Datos Cuantitativos del Fenómeno

**Caliza para Sinterización (951 registros):**
| Parámetro | Promedio | Desv. Std | Mínimo | Máximo | Especificación |
|-----------|----------|-----------|--------|--------|----------------|
| % SiO2 | 3,05 | 1,69 | 0,50 | **20,40** | ≤ 4,0% |
| % CaO | 51,38 | 2,42 | 0,50 | 54,60 | ≥ 50,0% |
| % PPC | 41,26 | 1,11 | 31,50 | 43,70 | ≥ 40,0% |
| % S | 0,042 | 0,038 | 0,00 | 0,31 | ≤ 0,10% |

**Caliza para Calcinación (680 registros):**
| Parámetro | Promedio | Desv. Std | Mínimo | Máximo | Especificación |
|-----------|----------|-----------|--------|--------|----------------|
| % SiO2 | 1,95 | 1,58 | 0,05 | **18,80** | ≤ 2,0% |
| % CaO | 52,91 | 1,49 | 40,10 | 55,00 | ≥ 52,0% |
| % PPC | 42,23 | 0,94 | 34,00 | 44,80 | ≥ 41,5% |
| % S | 0,046 | 0,064 | 0,00 | 0,46 | ≤ 0,08% |

**Cal Producida (3.269 registros):**
| Parámetro | Promedio | Mínimo | Máximo | Anomalía |
|-----------|----------|--------|--------|----------|
| % CaO | 88,68 | **49,5** | 96,3 | Valores <70% = caliza cruda sin calcinar |
| % SiO2 | 3,81 | 0,28 | **29,34** | Valores >10% = error analítico/registro |
| Reactividad | Variable | - | - | Meta: >250 mL para acería |
| % PPC | 3,15 | 0,10 | **41,1** | Valores >10% = calcinación incompleta |

### Seguimiento Mensual
*(Tabla para diligenciar con datos Meta vs Real de indicadores mensuales ENE-DIC)*

**Indicador principal a rastrear:** % SiO2 ponderado de caliza alimentada a planta
- **Meta:** ≤ 2,0% para calcinación; ≤ 3,5% para sinterización
- **Real actual:** 2,07% ponderado global (con picos hasta 20,4%)

---

## DIAGRAMA DE PARETO — Causas Priorizadas

| N° | CATEGORÍA (6M) | CAUSA | PUNTOS | ACUM. % |
|----|----------------|-------|--------|---------|
| 1 | MEDICIÓN | Forma de muestreo: el muestreo en banda corta no es representativo para todos los viajes del proveedor ese día | **35** | 13,2% |
| 2 | MEDICIÓN | Control de producción: la producción de la trituradora se notifica según datos de la báscula de sinter | **33** | 25,7% |
| 3 | MANO DE OBRA | Ingreso de datos: las toneladas de caliza se ingresan manualmente a SAP con 24h de retraso | **27** | 35,8% |
| 4 | MAQUINARIA | Plan de trituración: no se cuenta con plan de mezclas para alimentar la trituradora | **25** | 45,3% |
| 5 | MEDIO AMBIENTE | Contaminación del material: la caliza descargada en patio se contamina de arena | **24** | 54,3% |
| 6 | MAQUINARIA | Viajes del cargador: no se contabilizan los viajes del cargador de patio a tolva | **23** | 63,0% |
| 7 | MÉTODO | Forma de cargue: el cargador puede contaminar la caliza si la carga incorrectamente en patio | **22** | 71,3% |
| 8 | MATERIALES | Calidad de proveedores: no se tiene tipificada la calidad que pueden entregar los proveedores o frentes | **19** | 78,5% |
| 9 | MATERIALES | Capacidad de proveedores: no se tiene establecido cuánta caliza se puede comprar a cada proveedor | **18** | 85,3% |
| 10 | MATERIALES | Tipificación de caliza: no hay tipificación según parámetros de calidad | **16** | 91,3% |

**Análisis Pareto:** Las causas 1-7 (MEDICIÓN + MANO DE OBRA + MAQUINARIA + MEDIO AMBIENTE + MÉTODO) acumulan el **71,3%** del impacto → son las causas vitales. Las causas 8-10 (MATERIALES) representan el 28,7% restante.

---

## Nº 3 — ANÁLISIS DE PROCESO / ESPINA DE PESCADO (Ishikawa)

### MEDIO AMBIENTE
- **Contaminación del material:** La caliza descargada en el patio se contamina de arena, incrementando el %SiO2 real entre 0.3-1.5% por encima del valor de cantera. Causa raíz: piso del patio no tiene pavimento impermeable, no existe separación física entre tipos de caliza.

### MAQUINARIA
- **Viajes del cargador no contabilizados:** No se registra el número de paladas/viajes del cargador de patio a tolva, impidiendo estimar el tonelaje alimentado en tiempo real.
- **Plan de trituración inexistente:** No existe modelo ni procedimiento para determinar la secuencia y proporción de calizas a alimentar en la trituradora, operando de forma empírica.

### MATERIALES
- **Tipificación de caliza inexistente:** No se clasifican las calizas según rangos de calidad (Tipo 1: SiO2<1.5%, Tipo 2: 1.5-2.5%, Tipo 3: 2.5-4%, Tipo 4: >4%).
- **Calidad de proveedores no tipificada:** No se ha caracterizado estadísticamente la calidad que cada proveedor/frente puede entregar con confiabilidad.
- **Capacidad de proveedores no establecida:** No se conoce la capacidad mensual de despacho de cada proveedor.

### MÉTODO
- **Forma de cargue contamina:** El operador del cargador puede recoger arena del piso junto con la caliza si no controla la profundidad del cucharón.

### MANO DE OBRA
- **Ingreso de datos manual con 24h de retraso:** El operador de báscula digita manualmente los pesos en SAP, los datos se reflejan al día siguiente, impidiendo gestión en tiempo real.

### MEDICIÓN
- **Muestreo no representativo:** El muestreo en banda corta (un solo corte) puede no ser representativo del total de viajes del proveedor en un día. ISO 18283:2022 Anexo B advierte que el muestreo estacionario no provee muestras representativas.
- **Control de producción indirecto:** La producción de la trituradora se infiere de la báscula de sinter, no se mide directamente.

---

## TÉCNICA 5 PORQUÉS — Causas Raíz

### CAUSA 1: Forma de muestreo no representativo (35 pts)

| # | ¿Por qué? | Motivo | Qué hacer / Cómo hacer |
|---|-----------|--------|------------------------|
| 1 | ¿Por qué el muestreo en banda corta no es representativo? | Porque se toma un solo incremento por corte que no cubre la variabilidad del lote completo del proveedor | Implementar muestreador mecánico automático tipo cortador de cinta (ISO 13909) con frecuencia mínima cada 200 ton |
| 2 | ¿Por qué se toma un solo incremento? | Porque no existe un protocolo de muestreo diseñado según ISO 3082/18283 con cálculo de número mínimo de incrementos | Diseñar esquema de muestreo según ISO 18283:2022 con cálculo de precisión objetivo (error relativo <10%) |
| 3 | ¿Por qué no existe protocolo ISO? | Porque el laboratorio no tiene implementado un sistema de gestión bajo ISO/IEC 17025 que exija validación de métodos de muestreo | Iniciar proceso de acreditación ISO 17025 del laboratorio, incluyendo muestreo como alcance |
| 4 | ¿Por qué el laboratorio no tiene ISO 17025? | Porque no se ha priorizado la inversión en acreditación del laboratorio ni en equipos de muestreo mecánico | Elaborar caso de negocio con impacto cuantificado: cada 1% de error en SiO2 impacta X$/ton en consumo de coque |
| 5 | **CAUSA RAÍZ:** | **No existe inversión estratégica en infraestructura de muestreo y acreditación de laboratorio, a pesar del impacto directo en costo de producción y calidad del acero** | **Incluir en CAPEX 2026: muestreador mecánico ($XXX M) + acreditación ISO 17025 ($XXX M)** |

### CAUSA 2: Ingreso de datos manual con 24h de retraso (27 pts)

| # | ¿Por qué? | Motivo | Qué hacer / Cómo hacer |
|---|-----------|--------|------------------------|
| 1 | ¿Por qué los datos se ingresan manualmente? | Porque la báscula camionera no está integrada electrónicamente al módulo MM de SAP | Implementar interfaz OPC-UA entre báscula y SAP para registro automático de pesajes |
| 2 | ¿Por qué la báscula no está integrada a SAP? | Porque el sistema de pesaje original no tenía protocolo de comunicación digital compatible | Actualizar controlador de báscula a modelo con salida digital (Ethernet/Modbus TCP) y configurar interfaz RFC con SAP |
| 3 | ¿Por qué el reflejo es un día después? | Porque el operador acumula los tickets de pesaje y los digita al final del turno o al día siguiente | Definir como control operacional (ISO 9001 cl. 8.5.1) que el registro debe ser inmediato (<5 min post-pesaje) mientras se implementa la automatización |
| 4 | **CAUSA RAÍZ:** | **No existe integración digital entre la infraestructura de pesaje y el ERP SAP** | **Proyecto de automatización bascula-SAP con interfaz bidireccional y dashboard en tiempo real** |

### CAUSA 3: Ausencia de plan de mezclas para trituración (25 pts)

| # | ¿Por qué? | Motivo | Qué hacer / Cómo hacer |
|---|-----------|--------|------------------------|
| 1 | ¿Por qué no hay plan de mezclas? | Porque no se tiene un modelo que determine la proporción óptima de cada caliza según destino y calidad | Desarrollar modelo de optimización no-lineal (Ensemble ML: RF + GBM) que calcule mezcla óptima en tiempo real |
| 2 | ¿Por qué no se tiene un modelo? | Porque la relación calidad-costo-proveedor-destino es multivariable y no-lineal, imposible de resolver manualmente | Entrenar el modelo Ensemble con los 1.249 registros históricos + integrar con LLaMA 3 para recomendaciones |
| 3 | ¿Por qué la relación es no-lineal? | Porque el impacto del SiO2 en la basicidad del sinter y en la reducibilidad del alto horno sigue funciones cuadráticas y de interacción | Validar el modelo con datos reales: R² = 0.992 ya obtenido, MAE = 0.43 puntos de calidad |
| 4 | **CAUSA RAÍZ:** | **Ausencia de modelo predictivo de optimización que integre calidad, costo y restricciones operativas** | **Implementar el sistema limestone_optimizer.py con Ensemble (RF+GBM) + LLaMA 3 en producción** |

---

## Nº 4 — PLAN DE ACCIÓN

**VICEPRESIDENCIA:** Operaciones
**DIRECCIÓN:** Producción / Calidad
**RESPONSABLE:** Equipo MASP Caliza
**META:** Reducir desviaciones de SiO2 al 50% en 6 meses; implementar modelo de optimización en 12 meses

| Nº | ACCIÓN (Qué) | RESPONSABLE | FECHA INICIO | FECHA FIN | % AVANCE | STATUS |
|----|-------------|-------------|-------------|----------|----------|--------|
| 1 | Instalar muestreador mecánico automático en banda principal según ISO 13909 | Mantenimiento + Calidad | 2026-04 | 2026-12 | 0% | Pendiente |
| 2 | Diseñar e implementar protocolo de muestreo según ISO 18283:2022/ISO 3082:2017 | Laboratorio + Calidad | 2026-04 | 2026-07 | 0% | Pendiente |
| 3 | Automatizar interfaz báscula camionera ↔ SAP MM (protocolo OPC-UA) | TI + Logística | 2026-05 | 2026-11 | 0% | Pendiente |
| 4 | Implementar dashboard de calidad en tiempo real (Power BI / Grafana) | TI + Calidad | 2026-05 | 2026-09 | 0% | Pendiente |
| 5 | Desarrollar y validar modelo Ensemble (RF + GBM) de optimización de mezcla | Calidad + TI | 2026-04 | 2026-08 | **30%** | En curso |
| 6 | Desplegar LLaMA 3 en servidor local (Ollama) para capa de decisión | TI | 2026-06 | 2026-10 | 0% | Pendiente |
| 7 | Tipificar caliza por calidad: Tipo 1 (SiO2<1.5%), Tipo 2 (1.5-2.5%), Tipo 3 (2.5-4%), Tipo 4 (>4%) | Calidad | 2026-04 | 2026-05 | 0% | Pendiente |
| 8 | Caracterizar estadísticamente cada proveedor/frente (perfiles de calidad con IC 95%) | Calidad + Compras | 2026-04 | 2026-06 | 0% | Pendiente |
| 9 | Actualizar contratos de proveedores con cláusulas de calidad, penalización y muestreo | Compras + Legal | 2026-05 | 2026-08 | 0% | Pendiente |
| 10 | Pavimentar zona de patio de caliza y construir separadores entre tipos | Mantenimiento + Ambiente | 2026-06 | 2026-12 | 0% | Pendiente |
| 11 | Instalar GPS/contador en cargador para contabilizar viajes patio→tolva | Mantenimiento | 2026-05 | 2026-08 | 0% | Pendiente |
| 12 | Capacitar operadores en protocolo de muestreo ISO y uso del sistema de optimización | RRHH + Calidad | 2026-06 | 2026-09 | 0% | Pendiente |
| 13 | Iniciar proceso de acreditación ISO/IEC 17025 del laboratorio | Laboratorio | 2026-06 | 2027-06 | 0% | Pendiente |
| 14 | Implementar cartas de control Shewhart para SiO2, CaO, PPC por turno | Calidad | 2026-04 | 2026-06 | 0% | Pendiente |
| 15 | Instalar báscula directa en descarga de trituradora para control de producción independiente | Mantenimiento | 2026-07 | 2026-12 | 0% | Pendiente |

---

## ARQUITECTURA TÉCNICA DEL SISTEMA DE OPTIMIZACIÓN

### Stack Tecnológico

```
┌─────────────────────────────────────────────────────────────┐
│                    CAPA DE PRESENTACIÓN                      │
│  Dashboard (Grafana/Power BI) + Alertas + Reportes MASP     │
├─────────────────────────────────────────────────────────────┤
│                 CAPA DE DECISIÓN (LLaMA 3)                   │
│  Meta-Llama-3-8B-Instruct via Ollama (servidor local)        │
│  Genera recomendaciones operativas en lenguaje natural       │
├─────────────────────────────────────────────────────────────┤
│            CAPA DE OPTIMIZACIÓN (Ensemble ML)                │
│  Random Forest (200 árboles) + Gradient Boosting (300 iter)  │
│  Weights: RF=0.4, GBM=0.6 | R²=0.992, MAE=0.43             │
│  Optimizer: Differential Evolution (scipy)                    │
├─────────────────────────────────────────────────────────────┤
│               CAPA DE DATOS (SAP + Laboratorio)              │
│  SAP MM (pesajes auto) + LIMS (análisis químicos) + SCADA    │
│  PostgreSQL/TimescaleDB para series temporales                │
├─────────────────────────────────────────────────────────────┤
│             CAPA FÍSICA (Sensores + Báscula)                  │
│  Báscula camionera (OPC-UA) + Muestreador mecánico (ISO)     │
│  GPS cargador + Analizador en línea (futuro: XRF portátil)   │
└─────────────────────────────────────────────────────────────┘
```

### Modelo Ensemble — Resultados de Validación

| Métrica | Random Forest | Gradient Boosting | **ENSEMBLE** |
|---------|---------------|-------------------|--------------|
| MAE | 0.563 | 0.388 | **0.433** |
| R² | 0.9876 | 0.9939 | **0.9920** |
| CV R² (5-fold) | 0.9871 ± 0.0092 | 0.9926 ± 0.0037 | — |

**Feature Importance:** SiO2 (97.6%) >>> PPC (1.4%) > CaO (1.0%) > resto (<0.1%)

### Resultado de Optimización (Escenario Mensual 13.000 ton)

| Destino | Toneladas | $/ton | SiO2 Mezcla | CaO Mezcla | Cumple |
|---------|-----------|-------|-------------|------------|--------|
| Sinterización | 8.000 | $22.414 | 2,35% | 52,6% | ✓ |
| Calcinación | 3.000 | $22.000 | 2,05% | 52,9% | ✗ (marginal) |
| Alto Horno | 2.000 | $22.000 | 2,05% | 52,9% | ✓ |
| **TOTAL** | **13.000** | **$22.255** | — | — | 2/3 |

**Hallazgo crítico:** Para calcinación (SiO2 ≤ 2.0%), ningún proveedor propio cumple individualmente. Se requiere Carbopar (SiO2=1.04%) o Malsitio (SiO2=1.11%) como componente obligatorio de la mezcla.

### Despliegue de LLaMA 3

**Requisitos de hardware:**
- CPU: 8+ cores
- RAM: 16 GB mínimo (modelo cuantizado Q4_K_M)
- GPU: NVIDIA con 8+ GB VRAM (opcional pero recomendado)
- Disco: 10 GB para modelo

**Instalación:**
```bash
# Instalar Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Descargar LLaMA 3
ollama pull llama3

# Verificar
ollama run llama3 "Test de conexión"
```

**Integración:** El sistema limestone_optimizer.py se conecta a Ollama API (localhost:11434) para generar recomendaciones operativas en lenguaje natural a partir de los resultados numéricos de la optimización.

---

## MARCO NORMATIVO APLICADO

| Norma | Cláusula | Aplicación en el MASP |
|-------|----------|-----------------------|
| ISO 9001:2015 | 8.4 | Control de proveedores externos de caliza |
| ISO 9001:2015 | 8.5.1 | Control operacional de muestreo y pesaje |
| ISO 9001:2015 | 9.1 | Seguimiento con cartas de control e indicadores |
| ISO 9001:2015 | 10.2 | Acciones correctivas del MASP |
| ISO 14001:2015 | 6.1, 8.1 | Gestión ambiental de canteras y patio |
| ISO 45001:2018 | 6.1, 8.1 | Seguridad en minería cielo abierto (Decreto 539/2022) |
| ISO/IEC 17025:2017 | 7.2, 7.4, 7.7 | Competencia del laboratorio, métodos, QA/QC |
| ISO 3082:2017 | Completa | Muestreo y preparación de muestras de minerales |
| ISO 13909:2016 | Completa | Muestreo mecánico en banda transportadora |
| ISO 18283:2022 | Completa | Muestreo manual de material en movimiento y pilas |
| Ley 685/2001 | Art. 84 | Programa de Trabajos y Obras de proveedores |
| Decreto 539/2022 | Completo | Seguridad en labores mineras a cielo abierto |

---

*Documento generado como complemento del formato DD-PDR-GES-003 MASP V1.0*
*Elaborado por: Equipo de Gestión de Calidad — Marzo 2026*
*Sistema de Gestión: ISO 9001 | ISO 14001 | ISO 45001 | ISO/IEC 17025*
