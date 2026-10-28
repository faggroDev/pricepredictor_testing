from pathlib import Path
from datetime import timedelta
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from tensorflow.keras.models import load_model


app = FastAPI(
    title="Grain Price Forecasting API",
    description="API for forecasting agricultural commodity prices.",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
       
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = DATA_DIR / "modelv4"
SCALER_DIR = DATA_DIR / "gradio_scalerv4"
COMMODITY_DATA_DIR = (
    DATA_DIR / "unique_commodity_data_ceda_commodity_14c"
)

PRICE_TYPES = ["Min Price", "Max Price", "Modal Price"]
LOOK_BACK = 30
FORECAST_DAYS = 30


class ForecastRequest(BaseModel):
    district: str
    market: str
    commodity: str
    variety: str
    start_date: str

    @validator(
        "district",
        "market",
        "commodity",
        "variety",
        "start_date",
        pre=True,
    )
    def trim_strings(cls, value):
        if isinstance(value, str):
            return value.strip()
        return value


# =========================================================
# ROOT AND HEALTH ENDPOINTS
# =========================================================

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Grain Price Forecasting API.",
        "status": "running",
        "forecast_endpoint": "/forecast",
        "documentation": "/docs",
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}


# =========================================================
# LOAD COMMODITY DATA
# =========================================================

def load_commodity_data(commodity: str, folder_path: Path):
    # Restrict input to a filename rather than an arbitrary path.
    if not commodity or Path(commodity).name != commodity:
        return None

    file_path = folder_path / f"{commodity}.csv"

    if not file_path.is_file():
        print(f"Commodity CSV not found: {file_path}")
        return None

    try:
        df = pd.read_csv(file_path)
        df.dropna(inplace=True)

        required_columns = [
            "date",
            "District Name",
            "Market Name",
            "Commodity",
            "Variety",
            *PRICE_TYPES,
        ]

        missing_columns = [
            column for column in required_columns
            if column not in df.columns
        ]

        if missing_columns:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Commodity dataset is missing required columns: "
                    + ", ".join(missing_columns)
                ),
            )

        df["date"] = pd.to_datetime(
            df["date"], errors="coerce"
        )
        df.dropna(subset=["date"], inplace=True)

        return df

    except HTTPException:
        raise
    except Exception as exc:
        print(f"Error loading commodity CSV: {exc}")
        raise HTTPException(
            status_code=500,
            detail="Unable to read the commodity dataset.",
        )


# =========================================================
# LOAD MODEL AND SCALERS
# =========================================================

def load_model_and_scalers(
    district: str,
    market: str,
    commodity: str,
    variety: str,
):
    # Preserve the naming convention of the original models.
    model_filename = (
        f"model_{district}_{market}_{commodity}_{variety}.h5"
    )

    # Prevent directory traversal through request values.
    if Path(model_filename).name != model_filename:
        return None, None

    model_path = MODEL_DIR / model_filename

    if not model_path.is_file():
        print(f"Model not found: {model_path}")
        return None, None

    scalers = {}

    for price_type in PRICE_TYPES:
        scaler_filename = f"scaler_{commodity}_{price_type}.pkl"
        scaler_path = SCALER_DIR / scaler_filename

        if not scaler_path.is_file():
            print(f"Scaler not found: {scaler_path}")
            return None, None

        try:
            scalers[price_type] = joblib.load(scaler_path)
        except Exception as exc:
            print(f"Unable to load scaler {scaler_path}: {exc}")
            raise HTTPException(
                status_code=500,
                detail=f"Unable to load scaler for {price_type}.",
            )

    try:
        model = load_model(model_path, compile=False)
        print(
            f"Model loaded: {district}, {market}, "
            f"{commodity}, {variety}"
        )
        return model, scalers

    except Exception as exc:
        print(f"Unable to load model: {exc}")
        raise HTTPException(
            status_code=500,
            detail="Unable to load the forecasting model.",
        )


# =========================================================
# GET HISTORICAL DATA
# =========================================================

def get_recent_data(
    df,
    start_date,
    district,
    market,
    commodity,
    variety,
    scalers,
):
    filtered_df = df[
        (df["District Name"] == district)
        & (df["Market Name"] == market)
        & (df["Commodity"] == commodity)
        & (df["Variety"] == variety)
        & (df["date"] < start_date)
    ].copy()

    filtered_df = (
        filtered_df.sort_values("date")
        .tail(LOOK_BACK)
        .copy()
    )

    if len(filtered_df) < LOOK_BACK:
        print(
            "Insufficient historical data before "
            f"{start_date} for {district}/{market}/"
            f"{commodity}/{variety}"
        )
        return None

    for price_type in PRICE_TYPES:
        scaled_column = f"{price_type}_scaled"

        try:
            filtered_df[scaled_column] = (
                scalers[price_type].transform(
                    filtered_df[[price_type]]
                ).ravel()
            )
        except Exception as exc:
            print(f"Scaling error for {price_type}: {exc}")
            raise HTTPException(
                status_code=500,
                detail=f"Unable to scale {price_type} data.",
            )

    return filtered_df


# =========================================================
# PRICE ORDER VALIDATION
# =========================================================

def correct_price_order(min_price, max_price, modal_price):
    ordered_prices = sorted(
        [min_price, modal_price, max_price]
    )

    return (
        ordered_prices[0],
        ordered_prices[1],
        ordered_prices[2],
    )


# =========================================================
# FORECAST PRICES
# =========================================================

def forecast_prices(
    model,
    scalers,
    recent_data,
    num_days=FORECAST_DAYS,
):
    input_columns = [
        "Min Price_scaled",
        "Max Price_scaled",
        "Modal Price_scaled",
    ]

    # Expected LSTM input: (batch_size, 30 time steps, 3 features).
    input_data = recent_data[input_columns].to_numpy(
        dtype=np.float32
    )

    if input_data.shape != (LOOK_BACK, 3):
        raise HTTPException(
            status_code=500,
            detail=(
                f"Invalid model input shape: {input_data.shape}. "
                f"Expected ({LOOK_BACK}, 3)."
            ),
        )

    X_input = input_data.reshape(1, LOOK_BACK, 3)
    predictions = []

    for _ in range(num_days):
        try:
            predicted_scaled = np.asarray(
                model.predict(X_input, verbose=0)
            )

            if predicted_scaled.size != 3:
                raise ValueError(
                    "The model must predict three price values."
                )

            predicted_scaled = predicted_scaled.reshape(1, 3)

            predicted_min = float(
                scalers["Min Price"].inverse_transform(
                    predicted_scaled[:, 0:1]
                )[0, 0]
            )

            predicted_max = float(
                scalers["Max Price"].inverse_transform(
                    predicted_scaled[:, 1:2]
                )[0, 0]
            )

            predicted_modal = float(
                scalers["Modal Price"].inverse_transform(
                    predicted_scaled[:, 2:3]
                )[0, 0]
            )

            if not np.all(
                np.isfinite(
                    [predicted_min, predicted_max, predicted_modal]
                )
            ):
                raise ValueError(
                    "The model produced non-finite prices."
                )

            # Ensure Min Price <= Modal Price <= Max Price.
            corrected_min, corrected_modal, corrected_max = (
                correct_price_order(
                    predicted_min,
                    predicted_max,
                    predicted_modal,
                )
            )

            predictions.append(
                {
                    "Min Price": round(corrected_min, 2),
                    "Max Price": round(corrected_max, 2),
                    "Modal Price": round(corrected_modal, 2),
                }
            )

            # Append the predicted scaled values to the input
            # for recursive forecasting of the following day.
            X_input = np.concatenate(
                [X_input[:, 1:, :], predicted_scaled.reshape(1, 1, 3)],
                axis=1,
            )

        except HTTPException:
            raise
        except Exception as exc:
            print(f"Forecasting error: {exc}")
            raise HTTPException(
                status_code=500,
                detail="Unable to generate price forecasts.",
            )

    return predictions


# =========================================================
# CREATE FORECAST PLOT
# =========================================================

def plot_forecast(
    recent_data,
    forecast_list,
    start_forecast_date,
):
    historical_df = recent_data.copy()
    forecast_df = pd.DataFrame(forecast_list)

    forecast_df["date"] = pd.date_range(
        start=start_forecast_date,
        periods=len(forecast_df),
        freq="D",
    )

    fig = go.Figure()

    historical_series = [
        ("Min Price", "Historical Min Price", "blue"),
        ("Max Price", "Historical Max Price", "green"),
        ("Modal Price", "Historical Modal Price", "purple"),
    ]

    forecast_series = [
        ("Min Price", "Forecast Min Price", "gray"),
        ("Max Price", "Forecast Max Price", "coral"),
        ("Modal Price", "Forecast Modal Price", "brown"),
    ]

    for column, name, color in historical_series:
        fig.add_trace(
            go.Scatter(
                x=historical_df["date"],
                y=historical_df[column],
                mode="lines+markers",
                name=name,
                line=dict(color=color),
            )
        )

    for column, name, color in forecast_series:
        fig.add_trace(
            go.Scatter(
                x=forecast_df["date"],
                y=forecast_df[column],
                mode="lines+markers",
                name=name,
                line=dict(color=color, dash="dash"),
            )
        )

    all_min_prices = pd.concat(
        [historical_df["Min Price"], forecast_df["Min Price"]],
        ignore_index=True,
    )

    all_max_prices = pd.concat(
        [historical_df["Max Price"], forecast_df["Max Price"]],
        ignore_index=True,
    )

    y_min = float(all_min_prices.min())
    y_max = float(all_max_prices.max())

    padding = max((y_max - y_min) * 0.05, 1)

    fig.update_layout(
        title="Grain Price Forecasting",
        title_x=0.5,
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
        dragmode="pan",
        autosize=True,
        height=500,
        margin=dict(l=50, r=30, t=60, b=100),
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.2,
            yanchor="top",
        ),
    )

    fig.update_xaxes(
        range=[
            historical_df["date"].min() - pd.DateOffset(months=1),
            forecast_df["date"].max() + pd.DateOffset(days=1),
        ]
    )

    fig.update_yaxes(
        range=[
            max(0, y_min - padding),
            y_max + padding,
        ]
    )

    return pio.to_html(
        fig,
        full_html=True,
        include_plotlyjs="cdn",
    )



@app.post("/forecast")
def predict_for_user(request: ForecastRequest):
    district = request.district
    market = request.market
    commodity = request.commodity
    variety = request.variety

    # Validate the requested forecast date.
    try:
        start_date = pd.to_datetime(
            request.start_date,
            errors="raise",
        )

        if pd.isna(start_date):
            raise ValueError("Date is empty.")

        start_date = pd.Timestamp(start_date).normalize()

    except Exception:
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid start_date. Use YYYY-MM-DD format, "
                "for example 2026-10-27."
            ),
        )

    # Load the commodity's CSV data.
    df = load_commodity_data(
        commodity,
        COMMODITY_DATA_DIR,
    )

    if df is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Data not available for commodity: {commodity}"
            ),
        )

    # Load the trained model and the three scalers.
    model, scalers = load_model_and_scalers(
        district,
        market,
        commodity,
        variety,
    )

    if model is None or scalers is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "Model or scalers not available for the "
                "requested district, market, commodity, "
                "and variety."
            ),
        )

    # Retrieve the latest 30 historical observations.
    recent_data = get_recent_data(
        df=df,
        start_date=start_date,
        district=district,
        market=market,
        commodity=commodity,
        variety=variety,
        scalers=scalers,
    )

    if recent_data is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "At least 30 historical records before "
                "start_date are required."
            ),
        )

    # Generate 30 daily predictions.
    predictions = forecast_prices(
        model=model,
        scalers=scalers,
        recent_data=recent_data,
        num_days=FORECAST_DAYS,
    )

    # Use the requested date as the first forecast date.
    forecast_list = []

    for index, prediction in enumerate(predictions):
        forecast_date = (
            start_date + timedelta(days=index)
        ).strftime("%Y-%m-%d")

        forecast_list.append(
            {
                "date": forecast_date,
                "Min Price": prediction["Min Price"],
                "Max Price": prediction["Max Price"],
                "Modal Price": prediction["Modal Price"],
            }
        )

    # Generate the interactive chart.
    plot_html = plot_forecast(
        recent_data=recent_data,
        forecast_list=forecast_list,
        start_forecast_date=start_date,
    )

    # Convert pandas and NumPy data into JSON-compatible values.
    historical_records = recent_data[
        ["date", "Min Price", "Max Price", "Modal Price"]
    ].copy()

    historical_records["date"] = (
        historical_records["date"].dt.strftime("%Y-%m-%d")
    )

    historical_records = historical_records.to_dict(
        orient="records"
    )

    return {
        "success": True,
        "district": district,
        "market": market,
        "commodity": commodity,
        "variety": variety,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "forecast_days": len(forecast_list),
        "forecast": forecast_list,
        "recent_data": historical_records,
        "plot_html": plot_html,
    }



if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
