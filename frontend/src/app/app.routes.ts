import { Routes } from '@angular/router';
import { DashboardComponent } from './components/dashboard/dashboard.component';
import { ForecastComponent } from './components/forecast/forecast.component';
import { PredictComponent } from './components/predict/predict.component';
import { TrainComponent } from './components/train/train.component';

export const routes: Routes = [
  { path: '', pathMatch: 'full', redirectTo: 'dashboard' },
  { path: 'dashboard', component: DashboardComponent },
  { path: 'train', component: TrainComponent },
  { path: 'predict', component: PredictComponent },
  { path: 'forecast', component: ForecastComponent },
];
