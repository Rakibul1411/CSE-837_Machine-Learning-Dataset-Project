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
  error = '';
  private cacheBust = Date.now();

  constructor(private api: ApiService) {}

  ngOnInit(): void {
    this.api.getModels().subscribe({
      next: (models) => {
        this.models = models;
        const trained = models.find((m) => m.trained);
        if (trained) {
          this.selectedModel = trained.name;
          this.loadMetrics();
        }
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
        this.error = `No trained metrics for '${this.selectedModel}' yet.`;
        this.loading = false;
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
