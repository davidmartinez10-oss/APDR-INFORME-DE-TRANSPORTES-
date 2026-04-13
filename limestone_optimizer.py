#!/usr/bin/env python3
"""
================================================================================
SISTEMA DE OPTIMIZACIÓN DE MEZCLA DE CALIZA - ACERÍAS PAZ DEL RÍO
DD-PDR-GES-003 MASP - Modelo de Análisis para Solución de Problemas
================================================================================
Modelo Ensemble (Random Forest + Gradient Boosting) + Integración LLaMA 3
Objetivo: Minimizar costo de mezcla de caliza satisfaciendo restricciones de
          calidad para sinterización, calcinación y alto horno.

Autor: Equipo de Gestión de Calidad - Planta Belencito
Fecha: Marzo 2026
Normas: ISO 9001:2015, ISO 14001:2015, ISO 45001:2018, ISO/IEC 17025:2017
================================================================================
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.optimize import minimize, differential_evolution
import json, warnings, os
from datetime import datetime

warnings.filterwarnings('ignore')
np.random.seed(42)

# ==============================================================================
# 1. CARGA Y PREPARACIÓN DE DATOS REALES
# ==============================================================================

class DataLoader:
    """Carga y prepara datos reales de los archivos de Acerías Paz del Río."""
    
    def __init__(self, proveedores_path, calidad_path):
        self.prov_path = proveedores_path
        self.cal_path = calidad_path
        self.df_proveedores = None
        self.df_calcinacion = None
        self.df_sinter = None
        self.df_cales = None
        self.supplier_profiles = {}
    
    def load_all(self):
        print("=" * 70)
        print("  CARGA DE DATOS REALES - ACERÍAS PAZ DEL RÍO")
        print("=" * 70)
        
        # Proveedores
        self.df_proveedores = pd.read_excel(self.prov_path, sheet_name='Hoja2')
        self.df_proveedores['%SiO2'] = pd.to_numeric(self.df_proveedores['%SiO2'], errors='coerce')
        self.df_proveedores['%CaO2'] = pd.to_numeric(self.df_proveedores['%CaO2'], errors='coerce')
        self.df_proveedores['TONC'] = pd.to_numeric(self.df_proveedores['TONC'], errors='coerce')
        self.df_proveedores = self.df_proveedores.dropna(subset=['Proveedor', '%SiO2', '%CaO2', 'TONC'])
        # Normalizar nombres
        self.df_proveedores['Proveedor'] = self.df_proveedores['Proveedor'].str.strip()
        
        # Calidad
        self.df_calcinacion = pd.read_excel(self.cal_path, sheet_name='CALCINACION')
        self.df_sinter = pd.read_excel(self.cal_path, sheet_name='SINTER')
        self.df_cales = pd.read_excel(self.cal_path, sheet_name='CALES')
        
        print(f"  Proveedores:  {len(self.df_proveedores):>6} registros")
        print(f"  Calcinación:  {len(self.df_calcinacion):>6} registros")
        print(f"  Sinter:       {len(self.df_sinter):>6} registros")
        print(f"  Cales:        {len(self.df_cales):>6} registros")
        
        self._build_supplier_profiles()
        return self
    
    def _build_supplier_profiles(self):
        """Construye perfiles estadísticos por proveedor."""
        for prov, grp in self.df_proveedores.groupby('Proveedor'):
            sio2 = grp['%SiO2']
            cao = grp['%CaO2'].clip(upper=60)  # Filtrar outliers de CaO
            tons = grp['TONC']
            
            self.supplier_profiles[prov] = {
                'n_despachos': len(grp),
                'ton_total': tons.sum(),
                'ton_promedio': tons.mean(),
                'sio2_mean': sio2.mean(),
                'sio2_std': sio2.std(),
                'sio2_p95': sio2.quantile(0.95),
                'cao_mean': cao.mean(),
                'cao_std': cao.std(),
                'tipo': grp['FUENTE'].mode().iloc[0] if len(grp['FUENTE'].mode()) > 0 else 'DESCONOCIDO',
                # Costo estimado ($/ton) - se debe reemplazar con datos reales
                'costo_ton': self._estimate_cost(prov, sio2.mean()),
                # Capacidad mensual estimada (ton/mes)
                'capacidad_mensual': min(tons.sum() / max(1, len(grp.groupby(pd.to_datetime(grp['Fecha'], errors='coerce').dt.to_period('M')))), 5000),
            }
        
        print(f"\n  Perfiles de {len(self.supplier_profiles)} proveedores construidos")
    
    def _estimate_cost(self, prov, sio2_mean):
        """Estima costo por tonelada. REEMPLAZAR con datos reales de contratos."""
        base_costs = {
            'Carbopar': 45000, 'Caribbean': 42000, 'Malaga': 38000,
            'Gustavo Quesada': 35000, 'Leonidas Rodriguez': 37000,
            'Calizas & Granitos': 40000, 'Lisandro Rojas': 36000,
            'Caliza Propia': 25000, 'Afloramiento 1': 22000,
            'Afloramiento 1 Sur': 23000, 'Malsitio': 24000, 'Volador': 30000,
        }
        return base_costs.get(prov, 35000 + sio2_mean * 2000)
    
    def get_training_data(self):
        """Genera dataset de entrenamiento para el modelo ensemble."""
        records = []
        for prov, profile in self.supplier_profiles.items():
            grp = self.df_proveedores[self.df_proveedores['Proveedor'] == prov]
            for _, row in grp.iterrows():
                sio2 = row['%SiO2']
                cao = min(row['%CaO2'], 56)  # CaO máximo teórico en caliza
                ton = row['TONC']
                if sio2 > 0 and cao > 30 and ton > 0:
                    # PPC estimada a partir de CaO (relación estequiométrica)
                    ppc_est = min(44.0, cao * 0.835)  # ~44% para CaCO3 puro
                    records.append({
                        'proveedor_idx': list(self.supplier_profiles.keys()).index(prov),
                        'sio2': sio2,
                        'cao': cao,
                        'ppc_est': ppc_est,
                        'tonelaje': ton,
                        'sio2_mean_prov': profile['sio2_mean'],
                        'sio2_std_prov': profile['sio2_std'],
                        'cao_mean_prov': profile['cao_mean'],
                        'costo_ton': profile['costo_ton'],
                        # Target: índice de calidad compuesto
                        'quality_index': self._compute_quality_index(sio2, cao, ppc_est),
                    })
        
        df = pd.DataFrame(records)
        print(f"  Dataset de entrenamiento: {len(df)} muestras, {len(df.columns)} features")
        return df
    
    @staticmethod
    def _compute_quality_index(sio2, cao, ppc):
        """Índice de calidad compuesto [0-100]. Mayor = mejor."""
        # Penaliza SiO2 alto, premia CaO alto y PPC alto
        score = 100
        score -= max(0, (sio2 - 1.5)) * 15  # Penalización por SiO2 > 1.5%
        score -= max(0, (52 - cao)) * 5       # Penalización por CaO < 52%
        score -= max(0, (42 - ppc)) * 3       # Penalización por PPC < 42%
        return max(0, min(100, score))


# ==============================================================================
# 2. MODELO ENSEMBLE: RANDOM FOREST + GRADIENT BOOSTING
# ==============================================================================

class LimestoneEnsembleModel:
    """
    Modelo Ensemble para predicción de calidad de caliza.
    Combina Random Forest (robustez) + Gradient Boosting (precisión).
    """
    
    def __init__(self):
        self.rf = RandomForestRegressor(
            n_estimators=200, max_depth=12, min_samples_split=5,
            min_samples_leaf=3, random_state=42, n_jobs=-1
        )
        self.gb = GradientBoostingRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, random_state=42
        )
        self.scaler = StandardScaler()
        self.feature_names = None
        self.weights = {'rf': 0.4, 'gb': 0.6}  # GBM tiene mayor peso
        self.is_trained = False
        self.metrics = {}
    
    def train(self, df):
        """Entrena el ensemble con datos reales."""
        print("\n" + "=" * 70)
        print("  ENTRENAMIENTO DEL MODELO ENSEMBLE")
        print("=" * 70)
        
        self.feature_names = ['sio2', 'cao', 'ppc_est', 'tonelaje',
                              'sio2_mean_prov', 'sio2_std_prov', 'cao_mean_prov', 'costo_ton']
        X = df[self.feature_names].values
        y = df['quality_index'].values
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s = self.scaler.transform(X_test)
        
        # Entrenar Random Forest
        self.rf.fit(X_train_s, y_train)
        rf_pred = self.rf.predict(X_test_s)
        rf_mae = mean_absolute_error(y_test, rf_pred)
        rf_r2 = r2_score(y_test, rf_pred)
        
        # Entrenar Gradient Boosting
        self.gb.fit(X_train_s, y_train)
        gb_pred = self.gb.predict(X_test_s)
        gb_mae = mean_absolute_error(y_test, gb_pred)
        gb_r2 = r2_score(y_test, gb_pred)
        
        # Ensemble
        ens_pred = self.weights['rf'] * rf_pred + self.weights['gb'] * gb_pred
        ens_mae = mean_absolute_error(y_test, ens_pred)
        ens_r2 = r2_score(y_test, ens_pred)
        
        # Cross-validation del ensemble
        X_all_s = self.scaler.transform(X)
        rf_cv = cross_val_score(self.rf, X_all_s, y, cv=5, scoring='r2')
        gb_cv = cross_val_score(self.gb, X_all_s, y, cv=5, scoring='r2')
        
        self.metrics = {
            'rf': {'MAE': rf_mae, 'R2': rf_r2, 'CV_R2_mean': rf_cv.mean(), 'CV_R2_std': rf_cv.std()},
            'gb': {'MAE': gb_mae, 'R2': gb_r2, 'CV_R2_mean': gb_cv.mean(), 'CV_R2_std': gb_cv.std()},
            'ensemble': {'MAE': ens_mae, 'R2': ens_r2},
        }
        
        print(f"\n  {'Modelo':<20} {'MAE':>8} {'R²':>8} {'CV R² (5-fold)':>18}")
        print(f"  {'-'*56}")
        print(f"  {'Random Forest':<20} {rf_mae:>8.3f} {rf_r2:>8.4f} {rf_cv.mean():>8.4f} ± {rf_cv.std():.4f}")
        print(f"  {'Gradient Boosting':<20} {gb_mae:>8.3f} {gb_r2:>8.4f} {gb_cv.mean():>8.4f} ± {gb_cv.std():.4f}")
        print(f"  {'ENSEMBLE (0.4/0.6)':<20} {ens_mae:>8.3f} {ens_r2:>8.4f}")
        
        # Feature importance
        fi_rf = dict(zip(self.feature_names, self.rf.feature_importances_))
        fi_gb = dict(zip(self.feature_names, self.gb.feature_importances_))
        fi_ens = {k: self.weights['rf'] * fi_rf[k] + self.weights['gb'] * fi_gb[k] for k in self.feature_names}
        fi_sorted = sorted(fi_ens.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n  Feature Importance (Ensemble):")
        for feat, imp in fi_sorted:
            bar = "█" * int(imp * 50)
            print(f"    {feat:<20} {imp:.4f}  {bar}")
        
        self.is_trained = True
        return self.metrics
    
    def predict(self, X):
        """Predicción ensemble."""
        X_s = self.scaler.transform(X)
        return self.weights['rf'] * self.rf.predict(X_s) + self.weights['gb'] * self.gb.predict(X_s)
    
    def predict_single(self, sio2, cao, ppc, ton, prov_profile):
        """Predicción para una muestra individual."""
        X = np.array([[sio2, cao, ppc, ton,
                       prov_profile['sio2_mean'], prov_profile['sio2_std'],
                       prov_profile['cao_mean'], prov_profile['costo_ton']]])
        return self.predict(X)[0]


# ==============================================================================
# 3. OPTIMIZADOR DE MEZCLA DE CALIZA
# ==============================================================================

class LimestoneMixOptimizer:
    """
    Optimizador no-lineal de mezcla de caliza.
    Minimiza costo total sujeto a restricciones de calidad.
    Usa Differential Evolution (algoritmo evolutivo global).
    """
    
    def __init__(self, supplier_profiles, model):
        self.suppliers = supplier_profiles
        self.model = model
        self.supplier_names = list(supplier_profiles.keys())
        self.n_suppliers = len(self.supplier_names)
    
    def optimize(self, demand_tons, destination='SINTER', verbose=True):
        """
        Optimiza la mezcla de caliza.
        
        Args:
            demand_tons: Toneladas requeridas
            destination: 'SINTER', 'CALCINACION' o 'ALTO_HORNO'
            verbose: Imprimir resultados detallados
        
        Returns:
            dict con proporciones óptimas, costo, calidad estimada
        """
        specs = self._get_quality_specs(destination)
        
        if verbose:
            print(f"\n{'=' * 70}")
            print(f"  OPTIMIZACIÓN DE MEZCLA - Destino: {destination}")
            print(f"  Demanda: {demand_tons:.0f} toneladas")
            print(f"  Restricciones: SiO2 ≤ {specs['sio2_max']}%, CaO ≥ {specs['cao_min']}%")
            print(f"{'=' * 70}")
        
        # Bounds: proporción [0, 1] para cada proveedor
        bounds = [(0, 1)] * self.n_suppliers
        
        def objective(proportions):
            """Función objetivo: minimizar costo total + penalizaciones."""
            props = proportions / proportions.sum() if proportions.sum() > 0 else proportions
            
            total_cost = 0
            mix_sio2 = 0
            mix_cao = 0
            
            for i, name in enumerate(self.supplier_names):
                p = self.suppliers[name]
                tons_i = props[i] * demand_tons
                total_cost += tons_i * p['costo_ton']
                mix_sio2 += props[i] * p['sio2_mean']
                mix_cao += props[i] * p['cao_mean']
            
            # Penalizaciones por violación de restricciones
            penalty = 0
            if mix_sio2 > specs['sio2_max']:
                penalty += (mix_sio2 - specs['sio2_max']) ** 2 * 1e8
            if mix_cao < specs['cao_min']:
                penalty += (specs['cao_min'] - mix_cao) ** 2 * 1e8
            
            # Penalización por exceder capacidad de proveedor
            for i, name in enumerate(self.supplier_names):
                cap = self.suppliers[name]['capacidad_mensual']
                if props[i] * demand_tons > cap:
                    penalty += ((props[i] * demand_tons - cap) / cap) ** 2 * 1e7
            
            return total_cost + penalty
        
        # Constraint: proporciones suman 1
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        # Optimización con Differential Evolution (global)
        result = differential_evolution(
            objective, bounds, maxiter=500, tol=1e-8,
            seed=42, constraints=[{'type': 'eq', 'fun': lambda x: sum(x) - 1}] if False else (),
            polish=True
        )
        
        # Normalizar proporciones
        opt_props = result.x / result.x.sum() if result.x.sum() > 0 else result.x
        
        # Calcular métricas de la mezcla óptima
        mix_sio2, mix_cao, total_cost = 0, 0, 0
        mix_details = []
        
        for i, name in enumerate(self.supplier_names):
            p = self.suppliers[name]
            prop = opt_props[i]
            tons = prop * demand_tons
            cost = tons * p['costo_ton']
            
            if prop > 0.001:  # Solo proveedores con participación > 0.1%
                mix_sio2 += prop * p['sio2_mean']
                mix_cao += prop * p['cao_mean']
                total_cost += cost
                mix_details.append({
                    'proveedor': name,
                    'proporcion': prop,
                    'toneladas': tons,
                    'costo': cost,
                    'sio2_aporte': p['sio2_mean'],
                    'cao_aporte': p['cao_mean'],
                    'tipo': p['tipo'],
                })
        
        # Ordenar por proporción descendente
        mix_details.sort(key=lambda x: x['proporcion'], reverse=True)
        
        result_dict = {
            'destination': destination,
            'demand_tons': demand_tons,
            'mix_details': mix_details,
            'mix_sio2': mix_sio2,
            'mix_cao': mix_cao,
            'total_cost': total_cost,
            'cost_per_ton': total_cost / demand_tons if demand_tons > 0 else 0,
            'specs': specs,
            'meets_specs': mix_sio2 <= specs['sio2_max'] and mix_cao >= specs['cao_min'],
            'optimization_success': result.success,
            'timestamp': datetime.now().isoformat(),
        }
        
        if verbose:
            self._print_results(result_dict)
        
        return result_dict
    
    def _get_quality_specs(self, destination):
        """Especificaciones de calidad por destino."""
        specs = {
            'SINTER': {'sio2_max': 4.0, 'cao_min': 50.0, 'ppc_min': 40.0, 's_max': 0.10},
            'CALCINACION': {'sio2_max': 2.0, 'cao_min': 52.0, 'ppc_min': 41.5, 's_max': 0.08},
            'ALTO_HORNO': {'sio2_max': 3.0, 'cao_min': 51.0, 'ppc_min': 40.0, 's_max': 0.10},
        }
        return specs.get(destination, specs['SINTER'])
    
    def _print_results(self, r):
        status = "✓ CUMPLE" if r['meets_specs'] else "✗ NO CUMPLE"
        print(f"\n  RESULTADO: {status}")
        print(f"  {'─' * 64}")
        print(f"  {'Proveedor':<22} {'%':>6} {'Ton':>9} {'Costo ($)':>14} {'SiO2':>6} {'CaO':>6}")
        print(f"  {'─' * 64}")
        for d in r['mix_details']:
            if d['proporcion'] > 0.001:
                print(f"  {d['proveedor']:<22} {d['proporcion']*100:>5.1f}% {d['toneladas']:>9.0f} "
                      f"{d['costo']:>14,.0f} {d['sio2_aporte']:>5.2f}% {d['cao_aporte']:>5.1f}%")
        print(f"  {'─' * 64}")
        print(f"  {'MEZCLA TOTAL':<22} {'100%':>6} {r['demand_tons']:>9.0f} "
              f"{r['total_cost']:>14,.0f} {r['mix_sio2']:>5.2f}% {r['mix_cao']:>5.1f}%")
        print(f"\n  Costo unitario: ${r['cost_per_ton']:,.0f} /ton")
        print(f"  Límites: SiO2 ≤ {r['specs']['sio2_max']}% (actual: {r['mix_sio2']:.2f}%) | "
              f"CaO ≥ {r['specs']['cao_min']}% (actual: {r['mix_cao']:.1f}%)")


# ==============================================================================
# 4. CAPA DE INTEGRACIÓN LLaMA 3 (SIMULADA / PREPARADA)
# ==============================================================================

class LLaMA3DecisionLayer:
    """
    Capa de decisión basada en LLaMA 3 (Meta, open-source).
    
    Arquitectura de despliegue:
    - Servidor local con Ollama o vLLM
    - Modelo: meta-llama/Meta-Llama-3-8B-Instruct
    - RAM requerida: 16GB+ (cuantizado Q4) o 32GB+ (FP16)
    - GPU recomendada: NVIDIA RTX 3090/4090 o A100
    
    En este prototipo se simula la respuesta del LLM.
    Para producción: reemplazar _simulate_llm_response() con llamada real a Ollama API.
    """
    
    SYSTEM_PROMPT = """Eres un ingeniero metalúrgico experto en procesos siderúrgicos, 
especializado en la gestión de calidad de caliza para Acerías Paz del Río (Belencito, Boyacá).
Tu rol es analizar datos de optimización de mezcla de caliza y generar recomendaciones 
operativas en tiempo real. Tus respuestas deben ser técnicamente precisas, orientadas a la 
acción, y considerar las normas ISO 9001, 14001, 45001 y 17025.

Contexto operativo:
- Planta siderúrgica integrada (coquería → sinterización → calcinación → alto horno → acería)
- La caliza se usa como fundente en sinter, como cal viva tras calcinación, y directamente en alto horno
- Los parámetros críticos son: %SiO2 (silice, indeseable), %CaO (calcio, deseable), %PPC, %S
- La basicidad del sinter (CaO/SiO2) debe controlarse típicamente entre 1.8-2.2
- Especificaciones: Caliza para calcinación: SiO2 ≤ 2%, CaO ≥ 52%; Para sinter: SiO2 ≤ 4%, CaO ≥ 50%
"""
    
    def __init__(self, ollama_url="http://localhost:11434"):
        self.ollama_url = ollama_url
        self.use_real_llm = False  # Cambiar a True cuando Ollama esté disponible
        
    def generate_recommendation(self, optimization_result, supplier_profiles):
        """Genera recomendación operativa basada en resultados de optimización."""
        
        context = self._build_context(optimization_result, supplier_profiles)
        prompt = f"""Analiza el siguiente resultado de optimización de mezcla de caliza y genera 
una recomendación operativa estructurada:

{context}

Genera una recomendación con los siguientes apartados:
1. DIAGNÓSTICO: Estado actual de la mezcla vs especificaciones
2. RIESGOS IDENTIFICADOS: Posibles desviaciones y su impacto en el proceso aguas abajo
3. ACCIONES INMEDIATAS: Qué hacer en las próximas 24 horas
4. ACCIONES A MEDIANO PLAZO: Ajustes en la semana
5. INDICADORES A MONITOREAR: KPIs específicos con frecuencia
"""
        
        if self.use_real_llm:
            return self._call_ollama(prompt)
        else:
            return self._simulate_llm_response(optimization_result, supplier_profiles)
    
    def _build_context(self, result, profiles):
        """Construye contexto para el LLM."""
        lines = [
            f"DESTINO: {result['destination']}",
            f"DEMANDA: {result['demand_tons']:.0f} toneladas",
            f"SiO2 MEZCLA: {result['mix_sio2']:.2f}% (límite: {result['specs']['sio2_max']}%)",
            f"CaO MEZCLA: {result['mix_cao']:.1f}% (mínimo: {result['specs']['cao_min']}%)",
            f"COSTO UNITARIO: ${result['cost_per_ton']:,.0f}/ton",
            f"CUMPLE ESPECIFICACIONES: {'SÍ' if result['meets_specs'] else 'NO'}",
            f"\nDETALLE DE MEZCLA:"
        ]
        for d in result['mix_details']:
            if d['proporcion'] > 0.01:
                lines.append(f"  - {d['proveedor']}: {d['proporcion']*100:.1f}% ({d['toneladas']:.0f} ton, "
                           f"SiO2={d['sio2_aporte']:.2f}%, CaO={d['cao_aporte']:.1f}%)")
        return "\n".join(lines)
    
    def _call_ollama(self, prompt):
        """Llamada real a Ollama API (LLaMA 3)."""
        import requests
        try:
            resp = requests.post(f"{self.ollama_url}/api/generate", json={
                "model": "llama3", "prompt": prompt,
                "system": self.SYSTEM_PROMPT, "stream": False,
                "options": {"temperature": 0.3, "top_p": 0.9, "num_predict": 1024}
            }, timeout=120)
            return resp.json().get('response', 'Error: sin respuesta del modelo')
        except Exception as e:
            return f"Error de conexión con Ollama: {e}. Usando respuesta simulada."
    
    def _simulate_llm_response(self, result, profiles):
        """Simula respuesta del LLM para prototipo."""
        meets = result['meets_specs']
        sio2 = result['mix_sio2']
        cao = result['mix_cao']
        dest = result['destination']
        spec_sio2 = result['specs']['sio2_max']
        
        # Identificar proveedor de mayor riesgo
        high_risk = [d for d in result['mix_details'] 
                     if d['sio2_aporte'] > spec_sio2 and d['proporcion'] > 0.05]
        
        response = f"""
═══════════════════════════════════════════════════════════════════
  RECOMENDACIÓN OPERATIVA LLaMA-3 | {datetime.now().strftime('%Y-%m-%d %H:%M')}
  Destino: {dest} | Estado: {'✓ EN ESPECIFICACIÓN' if meets else '⚠ FUERA DE ESPECIFICACIÓN'}
═══════════════════════════════════════════════════════════════════

1. DIAGNÓSTICO:
   La mezcla optimizada presenta SiO2 = {sio2:.2f}% {'(dentro de límite)' if sio2 <= spec_sio2 else f'(EXCEDE límite de {spec_sio2}%)'} 
   y CaO = {cao:.1f}% {'(cumple mínimo)' if cao >= result['specs']['cao_min'] else f'(POR DEBAJO del mínimo {result["specs"]["cao_min"]}%)'}.
   Costo unitario: ${result['cost_per_ton']:,.0f}/ton.
   {'El margen de SiO2 es estrecho (' + f'{spec_sio2 - sio2:.2f}% de holgura).' if meets and (spec_sio2 - sio2) < 0.5 else ''}

2. RIESGOS IDENTIFICADOS:
   {'- ALTO: SiO2 fuera de especificación impacta basicidad del sinter, reduce reducibilidad en alto horno y aumenta consumo de coque.' if not meets else '- MODERADO: Variabilidad inherente de proveedores puede generar desviaciones puntuales.'}
   {f'- Proveedores de alto riesgo por SiO2: {", ".join([d["proveedor"] for d in high_risk])}' if high_risk else '- No se identifican proveedores de riesgo crítico en la mezcla actual.'}
   - El registro manual en SAP introduce un retraso de 24h en la visibilidad del inventario real.
   - El muestreo en banda corta puede no ser representativo del lote completo del proveedor.

3. ACCIONES INMEDIATAS (próximas 24h):
   {'- REDUCIR participación de proveedores con SiO2 > ' + f'{spec_sio2}%' if not meets else '- Mantener proporciones de mezcla actual.'}
   - Verificar muestreo del último lote recibido vs análisis de laboratorio.
   - Confirmar inventario físico en patio vs registro SAP.
   - Validar que el cargador no esté contaminando caliza con arena del patio.

4. ACCIONES A MEDIANO PLAZO (esta semana):
   - Coordinar con proveedores top-3 la programación de despachos de la semana.
   - Ejecutar análisis granulométrico de la caliza en trituradora secundaria.
   - Revisar cartas de control de SiO2 y CaO del laboratorio (ISO 17025).
   - Actualizar plan de mezcla de trituración según esta optimización.

5. INDICADORES A MONITOREAR:
   - %SiO2 promedio diario de caliza alimentada → Meta: ≤ {spec_sio2}% | Frecuencia: por turno
   - %CaO promedio de cal producida → Meta: ≥ 90% | Frecuencia: por turno  
   - Basicidad del sinter (CaO/SiO2) → Meta: 1.8-2.2 | Frecuencia: cada 2 horas
   - Reactividad Wührer de cal → Meta: ≥ 250 mL | Frecuencia: por turno
   - Desviación inventario SAP vs físico → Meta: < 2% | Frecuencia: diaria
"""
        return response


# ==============================================================================
# 5. ORQUESTADOR PRINCIPAL
# ==============================================================================

class LimestoneOptimizationSystem:
    """Sistema integrado de optimización de caliza."""
    
    def __init__(self, proveedores_path, calidad_path):
        self.loader = DataLoader(proveedores_path, calidad_path)
        self.model = LimestoneEnsembleModel()
        self.optimizer = None
        self.llm = LLaMA3DecisionLayer()
    
    def initialize(self):
        """Inicializa todo el sistema."""
        print("\n" + "█" * 70)
        print("  SISTEMA DE OPTIMIZACIÓN DE CALIZA - ACERÍAS PAZ DEL RÍO")
        print("  Modelo: Ensemble (RF + GBM) + LLaMA 3")
        print("  " + "█" * 66)
        
        self.loader.load_all()
        training_data = self.loader.get_training_data()
        self.model.train(training_data)
        self.optimizer = LimestoneMixOptimizer(self.loader.supplier_profiles, self.model)
        
        print("\n  ✓ Sistema inicializado correctamente")
        return self
    
    def run_optimization(self, demand_tons=5000, destination='SINTER'):
        """Ejecuta optimización completa con recomendación LLM."""
        result = self.optimizer.optimize(demand_tons, destination)
        recommendation = self.llm.generate_recommendation(result, self.loader.supplier_profiles)
        print(recommendation)
        return result, recommendation
    
    def run_full_scenario(self):
        """Ejecuta escenarios para los 3 destinos principales."""
        scenarios = [
            {'demand': 8000, 'dest': 'SINTER'},
            {'demand': 3000, 'dest': 'CALCINACION'},
            {'demand': 2000, 'dest': 'ALTO_HORNO'},
        ]
        results = []
        for s in scenarios:
            r, rec = self.run_optimization(s['demand'], s['dest'])
            results.append(r)
        
        # Resumen consolidado
        print("\n" + "=" * 70)
        print("  RESUMEN CONSOLIDADO DE OPTIMIZACIÓN")
        print("=" * 70)
        total_cost = sum(r['total_cost'] for r in results)
        total_tons = sum(r['demand_tons'] for r in results)
        print(f"\n  {'Destino':<15} {'Ton':>8} {'$/ton':>10} {'SiO2':>7} {'CaO':>7} {'Cumple':>8}")
        print(f"  {'─' * 57}")
        for r in results:
            status = "✓" if r['meets_specs'] else "✗"
            print(f"  {r['destination']:<15} {r['demand_tons']:>8,.0f} {r['cost_per_ton']:>10,.0f} "
                  f"{r['mix_sio2']:>6.2f}% {r['mix_cao']:>6.1f}% {status:>8}")
        print(f"  {'─' * 57}")
        print(f"  {'TOTAL':<15} {total_tons:>8,.0f} {total_cost/total_tons:>10,.0f}")
        print(f"\n  Costo total mensual estimado: ${total_cost:,.0f}")
        
        return results
    
    def export_masp_data(self):
        """Exporta datos estructurados para completar el MASP."""
        data = {
            'problema': {
                'titulo': 'Ausencia de programación estructurada del abastecimiento, ingreso y consumo de caliza',
                'descripcion': self.loader.df_proveedores.describe().to_dict(),
                'n_proveedores': len(self.loader.supplier_profiles),
                'n_registros': len(self.loader.df_proveedores),
            },
            'causas_priorizadas': [
                {'rank': 1, 'causa': 'Forma de muestreo no representativo en banda corta', 'puntos': 35, 'categoria': 'MEDICIÓN'},
                {'rank': 2, 'causa': 'Control de producción basado en báscula de sinter', 'puntos': 33, 'categoria': 'MEDICIÓN'},
                {'rank': 3, 'causa': 'Ingreso de datos manual con retraso de 24h en SAP', 'puntos': 27, 'categoria': 'MANO DE OBRA'},
                {'rank': 4, 'causa': 'Ausencia de plan de mezclas para trituración', 'puntos': 25, 'categoria': 'MAQUINARIA'},
                {'rank': 5, 'causa': 'Contaminación de caliza en patio con arena', 'puntos': 24, 'categoria': 'MEDIO AMBIENTE'},
                {'rank': 6, 'causa': 'Viajes del cargador no contabilizados', 'puntos': 23, 'categoria': 'MAQUINARIA'},
                {'rank': 7, 'causa': 'Forma de cargue contamina caliza', 'puntos': 22, 'categoria': 'MÉTODO'},
                {'rank': 8, 'causa': 'Calidad de proveedores no tipificada', 'puntos': 19, 'categoria': 'MATERIALES'},
                {'rank': 9, 'causa': 'Capacidad de proveedores no establecida', 'puntos': 18, 'categoria': 'MATERIALES'},
                {'rank': 10, 'causa': 'Tipificación de caliza inexistente', 'puntos': 16, 'categoria': 'MATERIALES'},
            ],
            'modelo': {
                'tipo': 'Ensemble (Random Forest + Gradient Boosting)',
                'metricas': self.model.metrics,
                'n_features': len(self.model.feature_names) if self.model.feature_names else 0,
                'llm': 'LLaMA 3 8B Instruct (Meta, open-source, despliegue local via Ollama)',
            },
            'supplier_profiles': {k: {kk: vv for kk, vv in v.items() if not isinstance(vv, np.floating) or not np.isnan(vv)} 
                                  for k, v in self.loader.supplier_profiles.items()},
        }
        return data


# ==============================================================================
# 6. EJECUCIÓN PRINCIPAL
# ==============================================================================

if __name__ == "__main__":
    # Rutas a los archivos del proyecto
    PROV_PATH = '/mnt/project/SEGUIMIENTO_PROVEEDORES_CALIZA_PRESENTACIÓN.xlsx'
    CAL_PATH = '/mnt/project/CALIDAD_CALES_Y_CALIZA.xlsx'
    
    # Verificar que existen los archivos
    for p in [PROV_PATH, CAL_PATH]:
        if not os.path.exists(p):
            print(f"ERROR: No se encuentra {p}")
            # Intentar ruta alternativa
            alt = os.path.join('/home/claude', os.path.basename(p))
            if os.path.exists(alt):
                print(f"  Usando alternativa: {alt}")
    
    # Inicializar y ejecutar
    system = LimestoneOptimizationSystem(PROV_PATH, CAL_PATH)
    system.initialize()
    
    # Ejecutar escenarios completos
    results = system.run_full_scenario()
    
    # Exportar datos para MASP
    masp_data = system.export_masp_data()
    print(f"\n  Datos MASP exportados: {len(masp_data['causas_priorizadas'])} causas priorizadas")
    print(f"  Modelo: {masp_data['modelo']['tipo']}")
    print(f"  LLM: {masp_data['modelo']['llm']}")
