"""
Econometric Visualizer - Backend API
FastAPI server for econometric model analysis
"""

import io
import json
from typing import Optional
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
from scipy import stats

app = FastAPI(title="Econometric Visualizer API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    elif pd.isna(obj):
        return None
    return obj


@app.get("/")
def read_root():
    return {"status": "ok", "message": "Econometric Visualizer API"}


@app.post("/api/upload")
async def upload_csv(file: UploadFile = File(...)):
    """Upload and parse CSV file"""
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

        # Basic data info
        columns = df.columns.tolist()
        dtypes = {col: str(df[col].dtype) for col in columns}
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        # Summary statistics
        summary = df.describe().to_dict()

        # Sample data
        sample_data = df.head(10).to_dict(orient='records')

        # Missing values
        missing = df.isnull().sum().to_dict()

        return JSONResponse(content=convert_to_serializable({
            "success": True,
            "filename": file.filename,
            "rows": len(df),
            "columns": columns,
            "dtypes": dtypes,
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "summary": summary,
            "sample_data": sample_data,
            "missing_values": missing
        }))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/detect-model")
async def detect_model(file: UploadFile = File(...)):
    """Automatically detect the best econometric model based on data characteristics"""
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        recommendations = []

        # Check for panel data structure
        has_panel_structure = False
        potential_entity_cols = []
        potential_time_cols = []

        for col in categorical_cols:
            unique_ratio = df[col].nunique() / len(df)
            if 0.01 < unique_ratio < 0.5:
                potential_entity_cols.append(col)

        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['year', 'date', 'time', 'period', 'month', 'quarter']):
                potential_time_cols.append(col)

        if potential_entity_cols and potential_time_cols:
            has_panel_structure = True
            recommendations.append({
                "model": "fixed_effects",
                "reason": "Panel data structure detected with entity and time dimensions",
                "confidence": 0.8,
                "entity_candidates": potential_entity_cols,
                "time_candidates": potential_time_cols
            })
            recommendations.append({
                "model": "random_effects",
                "reason": "Alternative to fixed effects for panel data",
                "confidence": 0.7,
                "entity_candidates": potential_entity_cols,
                "time_candidates": potential_time_cols
            })

        # Always recommend OLS as baseline
        recommendations.append({
            "model": "ols",
            "reason": "Standard OLS regression - always applicable as baseline",
            "confidence": 0.9 if not has_panel_structure else 0.5
        })

        # Check for time series characteristics
        if len(numeric_cols) >= 1 and len(df) > 30:
            recommendations.append({
                "model": "time_series",
                "reason": "Sufficient observations for time series analysis",
                "confidence": 0.6
            })

        # Sort by confidence
        recommendations.sort(key=lambda x: x["confidence"], reverse=True)

        return JSONResponse(content=convert_to_serializable({
            "success": True,
            "has_panel_structure": has_panel_structure,
            "recommendations": recommendations,
            "data_characteristics": {
                "rows": len(df),
                "numeric_columns": numeric_cols,
                "categorical_columns": categorical_cols
            }
        }))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    model_type: str = Form(...),
    dependent_var: str = Form(...),
    independent_vars: str = Form(...),
    entity_col: Optional[str] = Form(None),
    time_col: Optional[str] = Form(None)
):
    """Run econometric analysis"""
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

        # Parse independent variables
        indep_vars = [v.strip() for v in independent_vars.split(',') if v.strip()]

        # Prepare data
        y = df[dependent_var].dropna()
        X = df[indep_vars].dropna()

        # Align indices
        common_idx = y.index.intersection(X.index)
        y = y.loc[common_idx]
        X = X.loc[common_idx]

        result = {}

        if model_type == "ols":
            # OLS Regression
            X_const = sm.add_constant(X)
            model = sm.OLS(y, X_const).fit()

            result = {
                "model_type": "OLS Regression",
                "dependent_variable": dependent_var,
                "independent_variables": indep_vars,
                "n_observations": int(model.nobs),
                "r_squared": float(model.rsquared),
                "r_squared_adj": float(model.rsquared_adj),
                "f_statistic": float(model.fvalue) if not np.isnan(model.fvalue) else None,
                "f_pvalue": float(model.f_pvalue) if not np.isnan(model.f_pvalue) else None,
                "aic": float(model.aic),
                "bic": float(model.bic),
                "coefficients": {
                    name: {
                        "coef": float(model.params[name]),
                        "std_err": float(model.bse[name]),
                        "t_stat": float(model.tvalues[name]),
                        "p_value": float(model.pvalues[name]),
                        "conf_int_low": float(model.conf_int().loc[name, 0]),
                        "conf_int_high": float(model.conf_int().loc[name, 1])
                    }
                    for name in model.params.index
                },
                "residuals": model.resid.tolist(),
                "fitted_values": model.fittedvalues.tolist(),
                "durbin_watson": float(durbin_watson(model.resid))
            }

            # Heteroskedasticity test
            try:
                bp_test = het_breuschpagan(model.resid, model.model.exog)
                result["breusch_pagan"] = {
                    "lm_stat": float(bp_test[0]),
                    "lm_pvalue": float(bp_test[1]),
                    "f_stat": float(bp_test[2]),
                    "f_pvalue": float(bp_test[3])
                }
            except:
                pass

        elif model_type == "fixed_effects":
            # Fixed Effects using entity dummies
            if not entity_col:
                raise HTTPException(status_code=400, detail="Entity column required for fixed effects model")

            # Create dummy variables for entities
            entity_dummies = pd.get_dummies(df[entity_col], prefix='entity', drop_first=True)
            X_fe = pd.concat([X, entity_dummies.loc[X.index]], axis=1)
            X_fe_const = sm.add_constant(X_fe)

            model = sm.OLS(y, X_fe_const).fit()

            # Get only main variable coefficients
            main_coefs = {}
            for name in ['const'] + indep_vars:
                if name in model.params.index:
                    main_coefs[name] = {
                        "coef": float(model.params[name]),
                        "std_err": float(model.bse[name]),
                        "t_stat": float(model.tvalues[name]),
                        "p_value": float(model.pvalues[name])
                    }

            result = {
                "model_type": "Fixed Effects Model",
                "entity_column": entity_col,
                "n_entities": int(df[entity_col].nunique()),
                "dependent_variable": dependent_var,
                "independent_variables": indep_vars,
                "n_observations": int(model.nobs),
                "r_squared": float(model.rsquared),
                "r_squared_adj": float(model.rsquared_adj),
                "r_squared_within": float(model.rsquared),  # Approximation
                "f_statistic": float(model.fvalue) if not np.isnan(model.fvalue) else None,
                "coefficients": main_coefs,
                "residuals": model.resid.tolist(),
                "fitted_values": model.fittedvalues.tolist()
            }

        elif model_type == "random_effects":
            # Random Effects using MixedLM
            if not entity_col:
                raise HTTPException(status_code=400, detail="Entity column required for random effects model")

            # Prepare data for mixed effects
            df_clean = df[[dependent_var] + indep_vars + [entity_col]].dropna()

            formula = f"{dependent_var} ~ " + " + ".join(indep_vars)

            model = sm.MixedLM.from_formula(
                formula,
                groups=df_clean[entity_col],
                data=df_clean
            ).fit()

            result = {
                "model_type": "Random Effects Model",
                "entity_column": entity_col,
                "n_groups": int(model.ngroups),
                "dependent_variable": dependent_var,
                "independent_variables": indep_vars,
                "n_observations": int(model.nobs),
                "log_likelihood": float(model.llf),
                "aic": float(model.aic),
                "bic": float(model.bic),
                "coefficients": {
                    name: {
                        "coef": float(model.fe_params[name]),
                        "std_err": float(model.bse_fe[name]),
                        "z_stat": float(model.tvalues[name]),
                        "p_value": float(model.pvalues[name])
                    }
                    for name in model.fe_params.index
                },
                "random_effects_variance": float(model.cov_re.iloc[0, 0]) if hasattr(model.cov_re, 'iloc') else float(model.cov_re),
                "residuals": model.resid.tolist(),
                "fitted_values": model.fittedvalues.tolist()
            }

        elif model_type == "pooled_ols":
            # Pooled OLS (same as OLS but for panel context)
            X_const = sm.add_constant(X)
            model = sm.OLS(y, X_const).fit(cov_type='cluster', cov_kwds={'groups': df.loc[common_idx, entity_col]}) if entity_col else sm.OLS(y, X_const).fit()

            result = {
                "model_type": "Pooled OLS",
                "dependent_variable": dependent_var,
                "independent_variables": indep_vars,
                "n_observations": int(model.nobs),
                "r_squared": float(model.rsquared),
                "r_squared_adj": float(model.rsquared_adj),
                "coefficients": {
                    name: {
                        "coef": float(model.params[name]),
                        "std_err": float(model.bse[name]),
                        "t_stat": float(model.tvalues[name]),
                        "p_value": float(model.pvalues[name])
                    }
                    for name in model.params.index
                },
                "residuals": model.resid.tolist(),
                "fitted_values": model.fittedvalues.tolist()
            }
        else:
            raise HTTPException(status_code=400, detail=f"Unknown model type: {model_type}")

        # Add visualization data
        result["visualizations"] = generate_visualization_data(
            y.values,
            result.get("fitted_values", []),
            result.get("residuals", []),
            indep_vars,
            X
        )

        return JSONResponse(content=convert_to_serializable(result))

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def generate_visualization_data(y_actual, fitted_values, residuals, indep_vars, X):
    """Generate data for various econometric visualizations"""
    visualizations = {}

    # 1. Actual vs Fitted scatter plot
    visualizations["actual_vs_fitted"] = {
        "actual": list(y_actual),
        "fitted": list(fitted_values),
        "title": "Actual vs Fitted Values"
    }

    # 2. Residuals vs Fitted
    visualizations["residuals_vs_fitted"] = {
        "residuals": list(residuals),
        "fitted": list(fitted_values),
        "title": "Residuals vs Fitted Values"
    }

    # 3. Histogram of residuals
    hist, bin_edges = np.histogram(residuals, bins=30)
    visualizations["residuals_histogram"] = {
        "counts": hist.tolist(),
        "bin_edges": bin_edges.tolist(),
        "title": "Distribution of Residuals"
    }

    # 4. Q-Q plot data
    sorted_residuals = np.sort(residuals)
    n = len(sorted_residuals)
    theoretical_quantiles = stats.norm.ppf(np.arange(1, n + 1) / (n + 1))
    visualizations["qq_plot"] = {
        "theoretical": theoretical_quantiles.tolist(),
        "sample": sorted_residuals.tolist(),
        "title": "Q-Q Plot of Residuals"
    }

    # 5. Residuals over observation index
    visualizations["residuals_series"] = {
        "index": list(range(len(residuals))),
        "residuals": list(residuals),
        "title": "Residuals Over Observations"
    }

    # 6. Scatter plots for each independent variable
    scatter_plots = []
    for var in indep_vars:
        if var in X.columns:
            scatter_plots.append({
                "variable": var,
                "x_values": X[var].tolist(),
                "residuals": list(residuals),
                "title": f"Residuals vs {var}"
            })
    visualizations["scatter_by_variable"] = scatter_plots

    return visualizations


@app.post("/api/correlation")
async def correlation_matrix(file: UploadFile = File(...)):
    """Calculate correlation matrix for numeric variables"""
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()

        return JSONResponse(content=convert_to_serializable({
            "success": True,
            "columns": corr_matrix.columns.tolist(),
            "correlation_matrix": corr_matrix.values.tolist()
        }))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/descriptive-stats")
async def descriptive_stats(file: UploadFile = File(...)):
    """Get detailed descriptive statistics"""
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        stats_data = {}
        for col in numeric_cols:
            series = df[col].dropna()
            stats_data[col] = {
                "count": int(series.count()),
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "q25": float(series.quantile(0.25)),
                "median": float(series.median()),
                "q75": float(series.quantile(0.75)),
                "max": float(series.max()),
                "skewness": float(series.skew()),
                "kurtosis": float(series.kurtosis()),
                "missing": int(df[col].isnull().sum())
            }

        return JSONResponse(content=convert_to_serializable({
            "success": True,
            "statistics": stats_data
        }))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
