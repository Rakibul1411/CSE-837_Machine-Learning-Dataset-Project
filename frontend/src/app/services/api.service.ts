import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { environment } from '../../environments/environment';

export interface ModelInfo {
  name: string;
  trained: boolean;
}

export interface Metrics {
  mae: number;
  rmse: number;
  r2: number;
}

export interface Options {
  unit_names: string[];
  unit_types: string[];
  unit_name_to_type: Record<string, string>;
  year_min: number;
  year_max: number;
  target_column: string;
}

export interface PredictRequest {
  model_name: string;
  year: number;
  month_number: number;
  unit_name: string;
  unit_type: string;
}

export interface PredictResponse {
  model_name: string;
  prediction: number;
  input: PredictRequest;
}

export interface TrainRequest {
  model_name: string;
  test_size?: number | null;
}

export interface TrainResponse {
  model_name: string;
  train_size: number;
  test_size: number;
  metrics: Metrics;
}

export interface TimeSeriesTrainRequest {
  model_name: string;
  test_horizon?: number;
}

export interface TimeSeriesTrainResponse {
  model_name: string;
  train_size: number;
  test_size: number;
  metrics: Metrics;
}

export interface ForecastRequest {
  model_name: string;
  horizon?: number;
}

export interface ForecastPoint {
  date: string;
  prediction: number;
}

export interface ForecastResponse {
  model_name: string;
  forecast: ForecastPoint[];
}

@Injectable({
  providedIn: 'root',
})
export class ApiService {
  private readonly baseUrl = environment.apiUrl;
  private readonly serverRoot = environment.apiUrl.replace(/\/api\/?$/, '');

  constructor(private http: HttpClient) {}

  getModels(): Observable<ModelInfo[]> {
    return this.http.get<ModelInfo[]>(`${this.baseUrl}/models`);
  }

  getMetrics(modelName: string): Observable<Metrics> {
    return this.http.get<Metrics>(`${this.baseUrl}/metrics/${modelName}`);
  }

  getOptions(): Observable<Options> {
    return this.http.get<Options>(`${this.baseUrl}/options`);
  }

  predict(request: PredictRequest): Observable<PredictResponse> {
    return this.http.post<PredictResponse>(`${this.baseUrl}/predict`, request);
  }

  trainModel(request: TrainRequest): Observable<TrainResponse> {
    return this.http.post<TrainResponse>(`${this.baseUrl}/train`, request);
  }

  figureUrl(modelName: string, figure: 'actual_vs_predicted' | 'residuals', cacheBust?: number): string {
    const url = `${this.serverRoot}/figures/${modelName}_${figure}.png`;
    return cacheBust ? `${url}?t=${cacheBust}` : url;
  }

  getTimeSeriesModels(): Observable<ModelInfo[]> {
    return this.http.get<ModelInfo[]>(`${this.baseUrl}/timeseries/models`);
  }

  trainTimeSeriesModel(request: TimeSeriesTrainRequest): Observable<TimeSeriesTrainResponse> {
    return this.http.post<TimeSeriesTrainResponse>(`${this.baseUrl}/timeseries/train`, request);
  }

  forecastTimeSeries(request: ForecastRequest): Observable<ForecastResponse> {
    return this.http.post<ForecastResponse>(`${this.baseUrl}/timeseries/forecast`, request);
  }

  forecastFigureUrl(modelName: string, cacheBust?: number): string {
    const url = `${this.serverRoot}/figures/${modelName}_forecast.png`;
    return cacheBust ? `${url}?t=${cacheBust}` : url;
  }
}
