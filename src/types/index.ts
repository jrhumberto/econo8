export interface UploadResponse {
  success: boolean;
  filename: string;
  rows: number;
  columns: string[];
  dtypes: Record<string, string>;
  numeric_columns: string[];
  categorical_columns: string[];
  summary: Record<string, Record<string, number>>;
  sample_data: Record<string, unknown>[];
  missing_values: Record<string, number>;
}

export interface ModelRecommendation {
  model: string;
  reason: string;
  confidence: number;
  entity_candidates?: string[];
  time_candidates?: string[];
}

export interface DetectModelResponse {
  success: boolean;
  has_panel_structure: boolean;
  recommendations: ModelRecommendation[];
  data_characteristics: {
    rows: number;
    numeric_columns: string[];
    categorical_columns: string[];
  };
}

export interface Coefficient {
  coef: number;
  std_err: number;
  t_stat?: number;
  z_stat?: number;
  p_value: number;
  conf_int_low?: number;
  conf_int_high?: number;
}

export interface VisualizationData {
  actual_vs_fitted: {
    actual: number[];
    fitted: number[];
    title: string;
  };
  residuals_vs_fitted: {
    residuals: number[];
    fitted: number[];
    title: string;
  };
  residuals_histogram: {
    counts: number[];
    bin_edges: number[];
    title: string;
  };
  qq_plot: {
    theoretical: number[];
    sample: number[];
    title: string;
  };
  residuals_series: {
    index: number[];
    residuals: number[];
    title: string;
  };
  scatter_by_variable: {
    variable: string;
    x_values: number[];
    residuals: number[];
    title: string;
  }[];
}

export interface AnalysisResult {
  model_type: string;
  dependent_variable: string;
  independent_variables: string[];
  n_observations: number;
  r_squared?: number;
  r_squared_adj?: number;
  r_squared_within?: number;
  f_statistic?: number;
  f_pvalue?: number;
  aic?: number;
  bic?: number;
  log_likelihood?: number;
  durbin_watson?: number;
  breusch_pagan?: {
    lm_stat: number;
    lm_pvalue: number;
    f_stat: number;
    f_pvalue: number;
  };
  coefficients: Record<string, Coefficient>;
  residuals: number[];
  fitted_values: number[];
  visualizations: VisualizationData;
  entity_column?: string;
  n_entities?: number;
  n_groups?: number;
  random_effects_variance?: number;
}

export interface CorrelationResponse {
  success: boolean;
  columns: string[];
  correlation_matrix: number[][];
}

export interface DescriptiveStats {
  count: number;
  mean: number;
  std: number;
  min: number;
  q25: number;
  median: number;
  q75: number;
  max: number;
  skewness: number;
  kurtosis: number;
  missing: number;
}

export interface DescriptiveStatsResponse {
  success: boolean;
  statistics: Record<string, DescriptiveStats>;
}

export type ModelType = 'ols' | 'fixed_effects' | 'random_effects' | 'pooled_ols';
