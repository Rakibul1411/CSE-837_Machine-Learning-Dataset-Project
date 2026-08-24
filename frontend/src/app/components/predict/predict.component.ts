import { CommonModule } from '@angular/common';
import { Component, OnInit } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { ApiService, ModelInfo, Options } from '../../services/api.service';
import { formatModelName } from '../../shared/format';

@Component({
  selector: 'app-predict',
  imports: [CommonModule, FormsModule],
  templateUrl: './predict.component.html',
  styleUrl: './predict.component.css',
})
export class PredictComponent implements OnInit {
  models: ModelInfo[] = [];
  options: Options | null = null;
  readonly formatModelName = formatModelName;

  selectedModel = '';
  year = 2026;
  monthNumber = 1;
  unitName = '';

  prediction: number | null = null;
  loading = false;
  error = '';

  constructor(private api: ApiService) {}

  ngOnInit(): void {
    this.api.getModels().subscribe((models) => {
      this.models = models;
      const trained = models.find((m) => m.trained);
      if (trained) this.selectedModel = trained.name;
    });

    this.api.getOptions().subscribe({
      next: (options) => {
        this.options = options;
        this.year = options.year_max;
        if (options.unit_names.length) this.unitName = options.unit_names[0];
      },
      error: () => (this.error = 'Could not reach the API. Is the backend running on port 8001?'),
    });
  }

  get unitType(): string {
    if (!this.options || !this.unitName) return '';
    return this.options.unit_name_to_type[this.unitName] ?? '';
  }

  submit(): void {
    if (!this.selectedModel || !this.unitName) return;
    this.loading = true;
    this.error = '';
    this.prediction = null;
    this.api
      .predict({
        model_name: this.selectedModel,
        year: this.year,
        month_number: this.monthNumber,
        unit_name: this.unitName,
        unit_type: this.unitType,
      })
      .subscribe({
        next: (res) => {
          this.prediction = res.prediction;
          this.loading = false;
        },
        error: (err) => {
          this.error = err?.error?.detail ?? 'Prediction failed.';
          this.loading = false;
        },
      });
  }
}
