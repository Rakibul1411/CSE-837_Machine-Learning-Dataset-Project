import { CommonModule } from '@angular/common';
import { Component, OnInit } from '@angular/core';
import { FormsModule } from '@angular/forms';
import {
  ApiService,
  ForecastPoint,
  ModelInfo,
  TimeSeriesTrainResponse,
} from '../../services/api.service';
import { formatModelName } from '../../shared/format';

@Component({
  selector: 'app-forecast',
  imports: [CommonModule, FormsModule],
  templateUrl: './forecast.component.html',
  styleUrl: './forecast.component.css',
})
export class ForecastComponent implements OnInit {
  readonly formatModelName = formatModelName;

  models: ModelInfo[] = [];
  selectedModel = '';
  testHorizon = 12;
  forecastHorizon = 6;

  training = false;
  forecasting = false;
  error = '';

  trainResult: TimeSeriesTrainResponse | null = null;
  forecastPoints: ForecastPoint[] | null = null;
  private cacheBust = Date.now();

  constructor(private api: ApiService) {}

  ngOnInit(): void {
    this.api.getTimeSeriesModels().subscribe({
      next: (models) => {
        this.models = models;
        if (models.length) this.selectedModel = models[0].name;
      },
      error: () => (this.error = 'Could not reach the API. Is the backend running on port 8001?'),
    });
  }

  get forecastPlotUrl(): string {
    return this.api.forecastFigureUrl(this.selectedModel, this.cacheBust);
  }

  trainAndEvaluate(): void {
    if (!this.selectedModel) return;
    this.training = true;
    this.error = '';
    this.trainResult = null;
    this.api
      .trainTimeSeriesModel({ model_name: this.selectedModel, test_horizon: this.testHorizon })
      .subscribe({
        next: (res) => {
          this.trainResult = res;
          this.cacheBust = Date.now();
          this.training = false;
        },
        error: (err) => {
          this.error = err?.error?.detail ?? 'Training failed.';
          this.training = false;
        },
      });
  }

  forecast(): void {
    if (!this.selectedModel) return;
    this.forecasting = true;
    this.error = '';
    this.forecastPoints = null;
    this.api
      .forecastTimeSeries({ model_name: this.selectedModel, horizon: this.forecastHorizon })
      .subscribe({
        next: (res) => {
          this.forecastPoints = res.forecast;
          this.forecasting = false;
        },
        error: (err) => {
          this.error = err?.error?.detail ?? 'Forecast failed.';
          this.forecasting = false;
        },
      });
  }
}
