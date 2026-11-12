import { CommonModule } from '@angular/common';
import { Component, OnInit } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { ApiService, Metrics, ModelInfo } from '../../services/api.service';
import { formatModelName } from '../../shared/format';

@Component({
  selector: 'app-dashboard',
  imports: [CommonModule, FormsModule],
  templateUrl: './dashboard.component.html',
  styleUrl: './dashboard.component.css',
})
export class DashboardComponent implements OnInit {
  readonly formatModelName = formatModelName;
  models: ModelInfo[] = [];
  selectedModel = '';
  metrics: Metrics | null = null;
  loading = false;
  training = false;
  testSize = 0.2;
  error = '';
  private cacheBust = Date.now();

  constructor(private api: ApiService) {}

  ngOnInit(): void {
    this.refreshModels();
  }

  refreshModels(preferredModel?: string): void {
    this.api.getModels().subscribe({
      next: (models) => {
        this.models = models;
        if (preferredModel && models.some((m) => m.name === preferredModel)) {
          this.selectedModel = preferredModel;
        } else if (!this.selectedModel && models.length) {
          const trained = models.find((m) => m.trained) || models[0];
          this.selectedModel = trained.name;
        }
        this.loadMetrics();
      },
      error: () => (this.error = 'Could not reach the API. Is the backend running on port 8001?'),
    });
  }

  onModelChange(): void {
    this.loadMetrics();
  }

  loadMetrics(): void {
    if (!this.selectedModel) return;
    this.loading = true;
    this.error = '';
    this.metrics = null;
    this.api.getMetrics(this.selectedModel).subscribe({
      next: (metrics) => {
        this.metrics = metrics;
        this.cacheBust = Date.now();
        this.loading = false;
      },
      error: () => {
        this.error = `Model '${formatModelName(this.selectedModel)}' has not been trained yet. Adjust split ratio & click 'Fit Model'.`;
        this.loading = false;
      },
    });
  }

  trainModel(): void {
    if (!this.selectedModel) return;
    this.training = true;
    this.error = '';
    this.api.trainModel({ model_name: this.selectedModel, test_size: this.testSize }).subscribe({
      next: (res) => {
        this.training = false;
        this.metrics = res.metrics;
        this.cacheBust = Date.now();
        this.refreshModels(this.selectedModel);
      },
      error: (err) => {
        this.training = false;
        this.error = err?.error?.detail ?? 'Training failed.';
      },
    });
  }

  get actualVsPredictedUrl(): string {
    return this.api.figureUrl(this.selectedModel, 'actual_vs_predicted', this.cacheBust);
  }

  get residualsUrl(): string {
    return this.api.figureUrl(this.selectedModel, 'residuals', this.cacheBust);
  }
}
