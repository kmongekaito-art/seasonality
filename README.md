# S&P 500 Hybrid Seasonal System

Aplicación Streamlit del sistema estacional híbrido de Markos Katsanos
(*Stocks & Commodities* V.40:04), con backtest y comparación contra Buy & Hold
del SPY.

## Estrategia

- **Ventana estacional**: de octubre al mes de salida configurado (agosto por
  defecto).
- **Filtro de volatilidad**: subida del VIX y del ATR respecto a sus mínimos de
  25 días, más VFI por encima del nivel crítico.
- **VFI de Katsanos** de 130 días para detectar flujo de dinero.
- **Salida estacional**: primer día hábil del mes de salida, a la apertura.
- **Salidas por pánico**: VIXUP > 2 × VIXUPMAX o cruce del VFI por debajo del
  nivel crítico con su media de 10 días descendente, ejecutadas a la apertura
  del día siguiente.

## Funcionalidad

- Selector de ticker, VIX, rango de fechas, mes de salida, umbrales de VIX/ATR,
  VFI crítico, capital, comisión y precios ajustados o no.
- Panel central de velas con hasta 6 paneles de indicadores: VFI, VIX, VIXUP,
  ATR, ATRUP, RSI y volumen.
- Comparativa completa contra Buy & Hold: CAR, Sharpe, máxima caída, Ulcer
  Index, profit factor, operaciones ganadoras y curvas de capital/drawdown.
- Tabla de operaciones con descarga CSV y Excel (incluye la serie diaria de
  indicadores).

## Datos

- Yahoo Finance (SPY y `^VIX` por defecto), sin API keys.
- Caché de 1 hora.
- Por defecto usa precios **sin ajustar por dividendos**, como el AFL original;
  puedes activar precios ajustados en la barra lateral.

## Ejecutar en local

```powershell
cd C:\Users\koldo\PROYECTO-OPENCODE\SEASONALITY
pip install -r requirements.txt
streamlit run app.py
```

También puedes usar `run.bat`.

## Desplegar en Streamlit Community Cloud

1. Sube a un repositorio público de GitHub el contenido de esta carpeta:
   `app.py`, `requirements.txt`, `README.md`, `.streamlit/config.toml`,
   `.gitignore` y, opcionalmente, `run.bat`.
2. Entra en <https://share.streamlit.io> y elige **Create app**.
3. Repositorio: el que acabas de crear. Rama: `main`.
4. Main file path: `app.py`.
5. Advanced settings: Python **3.12** (o 3.11).
6. Deploy.

## Estructura

```text
SEASONALITY/
├── app.py
├── requirements.txt
├── README.md
├── run.bat
├── .gitignore
└── .streamlit/
    └── config.toml
```

## Aviso

Herramienta educativa. No constituye asesoramiento financiero. Los resultados
son teóricos y no incluyen deslizamiento.
