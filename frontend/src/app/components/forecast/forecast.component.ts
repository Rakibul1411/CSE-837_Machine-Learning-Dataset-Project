import { CommonModule } from '@angular/common';
import { Component, ElementRef, OnDestroy, OnInit, ViewChild } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Chart } from 'chart.js/auto';
import {
  ApiService,
  CustomRangeForecastResponse,
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
export class ForecastComponent implements OnInit, OnDestroy {
  @ViewChild('chartCanvas') chartCanvas?: ElementRef<HTMLCanvasElement>;

  readonly formatModelName = formatModelName;
  readonly months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

  models: ModelInfo[] = [];
  selectedModel = '';
  testHorizon = 12;

  startYear = 2019;
  startMonth = 1;
  endYear = 2028;
  endMonth = 12;

  includeEvaluation = true;
  forecasting = false;
  error = '';

  trainResult: TimeSeriesTrainResponse | null = null;
  customRangeResult: CustomRangeForecastResponse | null = null;

  private chart?: Chart;
  private cacheBust = Date.now();

  constructor(private api: ApiService) {}

  ngOnInit(): void {
    this.api.getTimeSeriesModels().subscribe({
      next: (models) => {
        this.models = models;
        if (models.length) {
          this.selectedModel = models[0].name;
          this.generateForecast();
        }
      },
      error: () => (this.error = 'Could not reach the API. Is the backend running on port 8001?'),
    });
  }

  ngOnDestroy(): void {
    this.destroyChart();
  }

  private destroyChart(): void {
    if (this.chart) {
      this.chart.destroy();
      this.chart = undefined;
    }
  }

  get forecastPlotUrl(): string {
    return this.api.forecastFigureUrl(this.selectedModel, this.cacheBust);
  }

  validateStartYear(): void {
    if (!this.startYear || this.startYear < 2019) {
      this.startYear = 2019;
    }
  }

  setPresetEndYear(year: number): void {
    this.endYear = year;
    this.endMonth = 12;
    this.generateForecast();
  }

  generateForecast(): void {
    if (!this.selectedModel) return;
    this.validateStartYear();
    this.forecasting = true;
    this.error = '';
    this.customRangeResult = null;

    if (this.includeEvaluation) {
      this.api
        .trainTimeSeriesModel({
          model_name: this.selectedModel,
          test_horizon: this.testHorizon,
        })
        .subscribe({
          next: (res) => {
            this.trainResult = res;
          },
          error: () => {
            this.trainResult = null;
          },
        });
    } else {
      this.trainResult = null;
    }

    this.api
      .forecastCustomRange({
        model_name: this.selectedModel,
        start_year: this.startYear,
        start_month: this.startMonth,
        end_year: this.endYear,
        end_month: this.endMonth,
      })
      .subscribe({
        next: (res) => {
          this.customRangeResult = res;
          this.cacheBust = Date.now();
          this.forecasting = false;
          setTimeout(() => this.renderChart(res), 50);
        },
        error: (err) => {
          this.error = err?.error?.detail ?? 'Custom range forecast failed.';
          this.forecasting = false;
        },
      });
  }

  private renderChart(res: CustomRangeForecastResponse): void {
    if (!this.chartCanvas) return;
    this.destroyChart();

    const historyDates = res.history.map((h) => h.date);
    const forecastDates = res.forecast.map((f) => f.date);
    const allLabels = Array.from(new Set([...historyDates, ...forecastDates])).sort();

    const historyMap = new Map(res.history.map((h) => [h.date, h.value]));
    const forecastMap = new Map(res.forecast.map((f) => [f.date, f.prediction]));

    const historyData = allLabels.map((d) => historyMap.get(d) ?? null);
    const forecastData = allLabels.map((d) => forecastMap.get(d) ?? null);

    const ctx = this.chartCanvas.nativeElement.getContext('2d');
    if (!ctx) return;

    this.chart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: allLabels,
        datasets: [
          {
            label: 'Historical Actuals',
            data: historyData,
            borderColor: '#4f7cff',
            backgroundColor: 'rgba(79, 124, 255, 0.1)',
            borderWidth: 2,
            pointRadius: 3,
            tension: 0.2,
            fill: true,
          },
          {
            label: 'Future Forecast',
            data: forecastData,
            borderColor: '#e0752d',
            backgroundColor: 'rgba(224, 117, 45, 0.1)',
            borderDash: [5, 5],
            borderWidth: 2,
            pointRadius: 4,
            pointStyle: 'rectRot',
            tension: 0.2,
            fill: true,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
          mode: 'index',
          intersect: false,
        },
        plugins: {
          legend: {
            position: 'top',
          },
          tooltip: {
            callbacks: {
              label: (context) => {
                const label = context.dataset.label || '';
                const val = context.parsed.y;
                if (val === null || val === undefined) return '';
                return `${label}: ${Math.round(val).toLocaleString()}`;
              },
            },
          },
        },
        scales: {
          x: {
            title: {
              display: true,
              text: 'Month (YYYY-MM)',
            },
            ticks: {
              maxRotation: 45,
              autoSkip: true,
              maxTicksLimit: 24,
            },
          },
          y: {
            title: {
              display: true,
              text: 'Total Cases',
            },
            beginAtZero: false,
          },
        },
      },
    });
  }
}
