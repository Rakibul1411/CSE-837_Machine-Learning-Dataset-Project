import { CommonModule } from '@angular/common';
import { Component, OnInit } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { RouterLink } from '@angular/router';
import { ApiService, ModelInfo, TrainResponse } from '../../services/api.service';
import { formatModelName } from '../../shared/format';

@Component({
  selector: 'app-train',
  imports: [CommonModule, FormsModule, RouterLink],
  templateUrl: './train.component.html',
  styleUrl: './train.component.css',
})
export class TrainComponent implements OnInit {
  models: ModelInfo[] = [];
  selectedModel = '';
  testSize = 0.2;

  training = false;
  error = '';
  result: TrainResponse | null = null;

  readonly formatModelName = formatModelName;

  constructor(private api: ApiService) {}

  ngOnInit(): void {
    this.loadModels();
  }

  loadModels(): void {
    this.api.getModels().subscribe({
      next: (models) => {
        this.models = models;
        if (!this.selectedModel && models.length) this.selectedModel = models[0].name;
      },
      error: () => (this.error = 'Could not reach the API. Is the backend running on port 8001?'),
    });
  }

  train(): void {
    if (!this.selectedModel) return;
    this.training = true;
    this.error = '';
    this.result = null;
    this.api.trainModel({ model_name: this.selectedModel, test_size: this.testSize }).subscribe({
      next: (res) => {
        this.result = res;
        this.training = false;
        this.loadModels(); // refresh "trained" flags now that this model is done
      },
      error: (err) => {
        this.error = err?.error?.detail ?? 'Training failed.';
        this.training = false;
      },
    });
  }
}
